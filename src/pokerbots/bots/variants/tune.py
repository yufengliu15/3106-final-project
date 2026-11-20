"""Build the variant library: equal-budget random-search tuning + validity checks.

For every approach the same budget is spent: CANDIDATES parameter draws, each
scored over GAMES x ROTATIONS against the scripted pool (never against Bot C).
A draw enters the library only if it
  - finishes above the starting stack on average (beats the scripted pool),
  - plays only legal actions and stays under 200 ms per decision (p99),
  - agrees with every already-kept draw on < 95% of a fixed probe set of\n    non-trivial spots (post-flop, or pre-flop after action).
The best KEEP survivors per approach are saved to models/variant_library.json.

    python -m pokerbots.bots.variants.tune --workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

CANDIDATES, KEEP, GAMES, ROTATIONS = 24, 8, 16, 30
DUP_AGREE = 0.95
ORDER = ("A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "B4", "B5", "B6")


def _spec(approach, draw):
    from pokerbots.bots.variants import default_spec, sample_spec
    if draw == 0 and approach != "A6":
        return default_spec(approach)      # the reference parameters compete too
    return sample_spec(approach, draw)


def _score(job):
    approach, draw, game = job
    from pokerbots.bots.simple import SCRIPTED_POOL
    from pokerbots.tournament import Tournament, TournamentConfig
    spec = _spec(approach, draw)
    cls = spec.make()
    times = []
    orig = cls.act

    def timed(self, obs):
        t = time.perf_counter()
        a = orig(self, obs)
        times.append(time.perf_counter() - t)
        return a

    cls.act = timed
    cfg = TournamentConfig(main_bots=(cls,), simple_bots=SCRIPTED_POOL, games=1, rotations=ROTATIONS,
                           mode="eval", base_seed=900_000 + 1000 * game)   # same tables for every draw
    (res,) = Tournament(cfg).run()
    stack = next(p.final_stack for p in res.standings if p.bot == spec.name)
    return approach, draw, stack, float(np.percentile(times, 99)) if times else 0.0


def _probe_set(n=400):
    """Fixed observations from a scripted-only game, used to measure agreement."""
    from pokerbots.bots.simple import SCRIPTED_POOL
    from pokerbots.sim.simulator import Simulator
    import random
    rng = random.Random(1)
    obs_list = []
    while len(obs_list) < n:
        ids = rng.sample(range(1000, 10000), 8)
        bots = {pid: rng.choice(SCRIPTED_POOL)() for pid in ids}
        for pid, b in bots.items():
            b.new_game(0, pid)
        sim = Simulator(bots, ids, [rng.choice([30.0, 100.0, 200.0]) for _ in ids], rng.randrange(10**6), 0, 0, 0)
        while not sim.is_over():
            o = sim.observation()
            # skip trivial spots (unopened pre-flop folds) where every sane bot agrees
            if o.community_cards or any(a == "all_in" for a, _ in o.legal_actions) and len(o.action_history) > 2:
                obs_list.append(o)
            sim.step(bots[o.player_id].act(o))
    return obs_list[:n]


def _actions(spec, probes):
    bot = spec.make()()
    bot.new_game(0, probes[0].player_id)
    out = []
    for o in probes:
        bot.player_id = o.player_id
        out.append(bot.act(o))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    from multiprocessing import Pool

    from pokerbots.bots.variants import LIBRARY_FILE, save_library

    jobs = [(ap, d, g) for ap in ORDER for d in range(CANDIDATES) for g in range(GAMES)]
    t0 = time.perf_counter()
    with Pool(a.workers) as pool:
        res = pool.map(_score, jobs, chunksize=4)
    print(f"scored {len(jobs)} games in {time.perf_counter() - t0:.0f}s", flush=True)

    by = {}
    for ap, d, s, p99 in res:
        r = by.setdefault((ap, d), {"stacks": [], "p99": 0.0})
        r["stacks"].append(s)
        r["p99"] = max(r["p99"], p99)

    probes = _probe_set()
    library, report = [], {}
    for ap in ORDER:
        ranked = sorted(range(CANDIDATES), key=lambda d: -np.mean(by[(ap, d)]["stacks"]))
        kept, acts, rows = [], [], []
        for d in ranked:
            r = by[(ap, d)]
            mean = float(np.mean(r["stacks"]))
            spec = _spec(ap, d)
            reason = ""
            if mean <= 100.0:
                reason = "does not beat scripted pool"
            elif r["p99"] >= 0.2:
                reason = "too slow"
            else:
                act = _actions(spec, probes)
                agree = max((np.mean([x == y for x, y in zip(act, k)]) for k in acts), default=0.0)
                if agree >= DUP_AGREE:
                    reason = f"duplicate ({agree:.0%} agreement)"
                elif len(kept) < KEEP:
                    kept.append(spec)
                    acts.append(act)
                else:
                    reason = "budget full"
            rows.append(dict(draw=d, mean=round(mean, 1), p99_ms=round(r["p99"] * 1000, 1),
                             kept=not reason, reason=reason))
        library += kept
        report[ap] = rows
        print(f"{ap}: kept {len(kept)}/{CANDIDATES}  best {rows[0]['mean']:.1f}  "
              f"kept means {[r['mean'] for r in rows if r['kept']]}", flush=True)
    out = a.out or LIBRARY_FILE
    save_library(library, out)
    Path(out).with_name("variant_tuning_report.json").write_text(json.dumps(report, indent=1))
    print(f"saved {len(library)} variants -> {out}")


if __name__ == "__main__":
    main()
