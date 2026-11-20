"""Bot C training-data generator: rollout labeller (#18).

Runs ordinary tournaments (same seating, rotation and stack rules as the
evaluation) with one *learner* seat: a fixed policy (a scripted bot or the
frozen Bot C v1) plus epsilon exploration. At a
sample of the learner's decisions it:

1. builds the feature vector (features.py) from what the learner can see;
2. for every legal action, plays the rest of the round K times from the
   simulator snapshot and records the learner's result;
3. labels the decision with the action of highest mean value.

Rollouts (docs/bot_c_plan.md, Section 4):
- Re-deal everything the learner cannot see, never the true future.
- Opponents' hole cards are re-dealt consistently with their actions so far:
  a candidate hand is kept only if the opponent's own act() reproduces every
  action it took this round with that hand (each player decides from public
  info + its own cards, so players are sampled independently). Falls back
  to a uniform hand after MAX_TRIES and flags the row.
- Common random numbers: rollout j uses the same re-deal for every action.
- The learner's later decisions are played by a policy drawn from cont_pool
  (no exploration); opponents play as themselves (deep copies, so their
  state is untouched) or, with probability P_SWAP, as a swap_pool bot.
- Value = learner's chip change in bb, and a round that ends on 0 chips is
  scored like the tournament does (-k * starting stack, decision D1).
- K starts at K_MIN and grows to K_MAX while the best two actions are within
  one standard error of each other.

Usage:
    python -m pokerbots.bots.ml.labeller --out data/botc --workers 16 --jobs 400
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import math
import os
import random
import time
from pathlib import Path

import numpy as np

from pokerbots.bots.ml.features import features, opponent_columns
from pokerbots.bots.simple import SCRIPTED_POOL, RandomBot
from pokerbots.core import ACTIONS, Observation
from pokerbots.core.equity import fast_equity
from pokerbots.opponents import OpponentTracker
from pokerbots.sim import simulator as sim_module
from pokerbots.sim.simulator import Simulator

K_MIN, K_MAX, K_STEP = 16, 48, 16
K_MAX_PREFLOP = 64     # v2: pre-flop labels were the noisiest
P_SWAP = 0.5           # chance an opponent continues as a random swap_pool bot
SHRINK = 20.0          # v2 feature shrinkage (features.py)
V1_DIR = Path(__file__).resolve().parents[4] / "models" / "checkpoints" / "botc_v1"
MAX_TRIES = 60
ROLLOUT_SIMS = 40   # Monte Carlo budget for bots' equity inside rollouts
BUST_K = 1.0


class LearnerBot:
    """Learner seat: one fixed policy for the whole game plus epsilon-random
    legal actions, so the data covers states the policy alone would avoid.
    Built by learner_class(); kept stochastic-flagged for the re-deal check."""


def learner_class(policy_cls, epsilon: float = 0.15):
    """The learner always has an opponent tracker (the features need one); a
    policy without its own, e.g. a scripted bot, gets one fed here."""

    def __init__(self):
        policy_cls.__init__(self)
        self.rng = random.Random(0)
        self._own_tracker = not hasattr(self, "tracker")
        if self._own_tracker:
            self.tracker = OpponentTracker()

    def new_game(self, game_id, player_id):
        policy_cls.new_game(self, game_id, player_id)
        self.rng = random.Random(game_id * 7919 + player_id)
        if self._own_tracker:
            self.tracker.reset()

    def observe_result(self, log):
        policy_cls.observe_result(self, log)
        if self._own_tracker:
            self.tracker.update(log)

    def act(self, obs):
        a = policy_cls.act(self, obs)
        if self.rng.random() < epsilon:
            a = self.rng.choice(obs.legal_actions)[0]
        return a

    return type("Learner", (LearnerBot, policy_cls),
                {"__init__": __init__, "new_game": new_game, "observe_result": observe_result, "act": act})


def _consistent_hand(bot, history, used: set[str], unseen: list[str], rng: random.Random):
    """Sample a hand for one opponent that reproduces its recorded actions."""
    stochastic = isinstance(bot, (RandomBot, LearnerBot))
    for _ in range(MAX_TRIES):
        a, b = rng.sample(unseen, 2)
        if a in used or b in used:
            continue
        if stochastic or all(
            bot.act(dataclasses.replace(o, hole_cards=(a, b))) == act for o, act in history
        ):
            return (a, b), True
    while True:
        a, b = rng.sample(unseen, 2)
        if a not in used and b not in used:
            return (a, b), False


def _with_state(bot, src, pid):
    """Give a fresh continuation bot the seat and opponent tracker of `src`."""
    bot.player_id = pid
    if hasattr(src, "tracker"):
        bot.tracker = copy.deepcopy(src.tracker)
    return bot


def label_decision(snap, obs: Observation, bots: dict, histories: dict, learner_id: int,
                   start_stack: float, rng: random.Random, cont_pool=None, swap_pool=None,
                   k_max: int = K_MAX) -> dict:
    """Mean value of every legal action from this snapshot.

    cont_pool: zero-arg factories for the learner's own continuation (one drawn per
    call; default: a copy of the learner seat itself). swap_pool: factories an
    opponent may continue as (prob P_SWAP per rollout, same for every action:
    common random numbers); None = opponents always continue as themselves."""
    actions = [a for a, _ in obs.legal_actions]
    rollout_bots = {pid: copy.deepcopy(b) for pid, b in bots.items()}
    if cont_pool:
        cont = _with_state(rng.choice(cont_pool)(), bots[learner_id], learner_id)
        rollout_bots[learner_id] = cont
    else:
        cont = rollout_bots[learner_id]
    swaps = {}
    if swap_pool:
        for pid in obs.positions:
            if pid != learner_id:
                swaps[pid] = [_with_state(f(), bots[pid], pid) for f in rng.sample(swap_pool, min(3, len(swap_pool)))]
    checker_bots = {pid: copy.deepcopy(b) for pid, b in bots.items()}

    probe = Simulator.from_snapshot(snap, rollout_bots, seed=0)
    unseen = sorted(probe.unseen_cards())   # sorted: independent of the true deal
    others = [p for p in obs.positions if p != learner_id]

    values = {a: [] for a in actions}
    fallbacks = 0
    k = 0
    while True:
        for _ in range(K_STEP):
            used: set[str] = set()
            hands = {}
            for pid in others:
                hand, ok = _consistent_hand(checker_bots[pid], histories.get(pid, ()), used, unseen, rng)
                fallbacks += not ok
                used.update(hand)
                hands[pid] = hand
            seed = rng.randrange(2**31)
            these = rollout_bots
            if swaps:
                these = dict(rollout_bots)
                for pid, opts in swaps.items():
                    if rng.random() < P_SWAP:
                        these[pid] = rng.choice(opts)
            for a in actions:
                sim = Simulator.from_snapshot(snap, these, seed=seed, hands=hands)
                sim.step(a)
                # plain play: the copy keeps LabellingSimulator's class, which must not recurse
                log = Simulator.play(sim)
                delta = log.chip_deltas[learner_id]
                end = start_stack + delta
                values[a].append(delta - BUST_K * start_stack if end <= 1e-9 else delta)
            k += 1
        means = {a: float(np.mean(v)) for a, v in values.items()}
        order = sorted(actions, key=lambda a: -means[a])
        if len(order) < 2 or k >= k_max:
            break
        diff = np.array(values[order[0]]) - np.array(values[order[1]])
        se = diff.std(ddof=1) / math.sqrt(len(diff))
        if diff.mean() > se:
            break

    means = {a: float(np.mean(v)) for a, v in values.items()}
    ses = {a: float(np.std(v, ddof=1) / math.sqrt(len(v))) for a, v in values.items()}
    order = sorted(actions, key=lambda a: -means[a])
    gap = means[order[0]] - means[order[1]] if len(order) > 1 else 0.0
    se_gap = 0.0
    if len(order) > 1:
        diff = np.array(values[order[0]]) - np.array(values[order[1]])
        se_gap = float(diff.std(ddof=1) / math.sqrt(len(diff)))
    row = {"label": order[0], "weight": min(gap, 10.0), "k": k, "gap_se": gap / max(se_gap, 1e-6),
           "cont": type(cont).__name__,
           "fallback_rate": fallbacks / max(k * len(others), 1)}
    for a in ACTIONS:
        row[f"ev_{a}"] = means.get(a, float("nan"))
        row[f"se_{a}"] = ses.get(a, float("nan"))
    return row


class LabellingSimulator(Simulator):
    """Simulator whose play() labels a sample of the learner's decisions."""

    collector: "Collector"

    def play(self):
        c = self.collector
        histories: dict[int, list] = {}
        start_stack = dict(zip(self.positions, (s / sim_module.CHIPS_PER_BB for s in self.starting_chips)))
        while not self.is_over():
            obs = self.observation()
            pid = obs.player_id
            bot = self.bots[pid]
            rate = c.sample_rate * (1.0 if obs.community_cards else c.preflop_factor)
            if pid == c.learner_id and c.rng.random() < rate:
                t = time.perf_counter()
                feats = features(obs, bot.tracker, shrink=c.shrink)
                if c.shrink > 0:   # opponent-model-off copies for dropout at training time
                    off = features(obs, bot.tracker, use_opponent_model=False, shrink=c.shrink)
                    feats.update({f"off__{k}": off[k] for k in opponent_columns(off)})
                with fast_equity(ROLLOUT_SIMS):
                    row = label_decision(self.snapshot(), obs, self.bots, histories, pid,
                                         start_stack[pid], c.rng, c.cont_pool, c.swap_pool,
                                         K_MAX if obs.community_cards else c.k_max_preflop)
                row.update(feats)
                row["learner_policy"] = c.learner_policy
                row.update(game_id=c.game_key(obs.game_id), rotation_id=obs.rotation_id,
                           round_id=obs.round_id, label_ms=(time.perf_counter() - t) * 1000)
                c.rows.append(row)
            action = bot.act(obs)
            histories.setdefault(pid, []).append((obs, action))
            self.step(action)
        return self.round_log()


@dataclasses.dataclass
class Collector:
    learner_id: int = -1
    sample_rate: float = 1.0
    preflop_factor: float = 0.25   # pre-flop is ~3/4 of decisions: label fewer of them
    worker: int = 0
    rows: list = dataclasses.field(default_factory=list)
    rng: random.Random = dataclasses.field(default_factory=random.Random)
    shrink: float = 0.0
    cont_pool: tuple = ()
    swap_pool: tuple = ()
    k_max_preflop: int = K_MAX
    learner_policy: str = ""

    def game_key(self, game_id: int) -> int:
        return self.worker * 1000 + game_id


def botc_v1():
    """Zero-arg class for the frozen Bot C v1 checkpoint (decision D5: a fixed
    model, so using it as a policy is not RL)."""
    from pokerbots.bots.ml.bot_c import MLBot

    def __init__(self):
        MLBot.__init__(self, model_dir=V1_DIR)

    return type("BotCv1", (MLBot,), {"__init__": __init__})


def _setup(worker: int, collector: "Collector"):
    """Tables: a random-policy learner and a frozen Bot C v1, scripted fill;
    random starting stack. Returns (main_bots, stack)."""
    rng = random.Random(f"v2-{worker}")
    v1 = botc_v1()
    policies = tuple(b for b in SCRIPTED_POOL if b is not RandomBot) + (v1,)
    policy = rng.choice(policies)
    collector.learner_policy = policy.__name__
    collector.shrink = SHRINK
    collector.preflop_factor = 0.6
    collector.k_max_preflop = K_MAX_PREFLOP
    collector.cont_pool = policies
    collector.swap_pool = policies
    return (learner_class(policy), v1), rng.choice((40.0, 70.0, 100.0, 100.0, 150.0, 200.0))


def run_worker(args) -> str:
    """Play `games` games with a learner seat; write one parquet file."""
    worker, games, rotations, out_dir, sample_rate = args[:5]
    import pandas as pd

    from pokerbots.tournament import Tournament, TournamentConfig
    from pokerbots.tournament import tournament as tmod

    collector = Collector(sample_rate=sample_rate, worker=worker, rng=random.Random(worker))
    LabellingSimulator.collector = collector
    tmod.Simulator = LabellingSimulator

    main_bots, stack = _setup(worker, collector)
    learner = main_bots[0]
    orig_new_game = learner.new_game

    def new_game(self, game_id, player_id):   # record the learner's ID at the start of each game
        collector.learner_id = player_id
        orig_new_game(self, game_id, player_id)

    learner.new_game = new_game
    cfg = TournamentConfig(main_bots=main_bots, simple_bots=SCRIPTED_POOL, starting_stack=stack,
                           games=games, rotations=rotations, mode="eval", base_seed=10_000 + worker)
    out = Path(out_dir) / f"part_{worker:03d}.parquet"
    t = time.perf_counter()
    try:
        Tournament(cfg).run()
    except Exception as e:  # keep what was labelled before the failure
        import traceback
        traceback.print_exc()
        if not collector.rows:
            return f"worker {worker}: FAILED with {e!r}"
    pd.DataFrame(collector.rows).to_parquet(out)
    return f"worker {worker}: {len(collector.rows)} rows in {time.perf_counter() - t:.0f}s -> {out}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/botc"))
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--jobs", type=int, default=16, help="one game per job, one parquet file per job")
    p.add_argument("--rotations", type=int, default=50)
    p.add_argument("--sample-rate", type=float, default=1.0)
    p.add_argument("--first-job", type=int, default=0)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(a.first_job + j, 1, a.rotations, str(a.out), a.sample_rate) for j in range(a.jobs)
            if not (a.out / f"part_{a.first_job + j:03d}.parquet").exists()]   # resumable
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        for msg in pool.imap_unordered(run_worker, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
