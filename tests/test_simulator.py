import random

import pytest

from pokerbots.core import (
    ALL_IN,
    CALL,
    CHECK,
    FOLD,
    Bot,
    Observation,
    RoundLog,
)
from pokerbots.sim import IllegalActionError, Simulator


class RandomBot(Bot):
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def new_game(self, game_id: int, player_id: int) -> None:
        pass

    def act(self, obs: Observation) -> str:
        return self.rng.choice(obs.legal_actions)[0]

    def explain(self) -> str:
        return "random"

    def observe_result(self, log: RoundLog) -> None:
        pass


class FixedBot(RandomBot):
    """Always plays `action` if legal, else the first legal action."""

    def __init__(self, action: str) -> None:
        self.action = action

    def act(self, obs: Observation) -> str:
        names = [name for name, _ in obs.legal_actions]
        return self.action if self.action in names else names[0]


IDS = [1000 + 111 * i for i in range(8)]


def make_sim(stacks, seed=0, bots=None, ids=None):
    ids = ids or IDS[: len(stacks)]
    bots = bots or {pid: RandomBot(random.Random(pid)) for pid in ids}
    return Simulator(bots, ids, stacks, seed=seed, game_id=1, rotation_id=0, round_id=0)


def test_ten_thousand_random_rounds():
    """Done-when: 10,000 rounds with 8 random bots, chips conserved every round."""
    rng = random.Random(15)
    bots = {pid: RandomBot(rng) for pid in IDS}
    for seed in range(10_000):
        stacks = [rng.randint(1, 30_000) / 100 for _ in IDS]
        log = Simulator(bots, IDS, stacks, seed=seed, game_id=0, rotation_id=0, round_id=0).play()

        assert sum(log.chip_deltas.values()) == pytest.approx(0.0, abs=1e-9)
        for pid, start in zip(log.positions, log.starting_stacks):
            assert start + log.chip_deltas[pid] >= -1e-9


def test_first_observation_has_blinds_posted():
    sim = make_sim([100.0] * 8)
    obs = sim.observation()
    assert obs.player_id == IDS[2]  # first to act preflop: after the big blind
    assert obs.pot == 1.5
    assert obs.stacks[0] == 99.5 and obs.stacks[1] == 99.0
    assert obs.action_history == ()
    assert len(obs.hole_cards) == 2 and obs.community_cards == ()


def test_variable_table_size():
    sim = make_sim([100.0] * 3)
    obs = sim.observation()
    assert obs.positions == tuple(IDS[:3])
    assert len(obs.stacks) == 3
    assert obs.player_id == IDS[2]  # 3-handed: button acts first preflop


def test_check_offered_without_fold_when_free():
    # Everyone calls to the big blind, then the BB may check.
    bots = {pid: FixedBot(CALL) for pid in IDS[:3]}
    bots[IDS[1]] = FixedBot(CHECK)
    sim = make_sim([100.0] * 3, bots=bots)
    sim.step(CALL)  # button
    sim.step(CALL)  # small blind
    obs = sim.observation()
    assert obs.player_id == IDS[1]
    legal = dict(obs.legal_actions)
    assert CHECK in legal and legal[CHECK] == 0
    assert FOLD not in legal and CALL not in legal


def test_facing_bet_offers_fold_and_call_with_cost():
    sim = make_sim([100.0] * 8)
    legal = dict(sim.observation().legal_actions)
    assert legal[FOLD] == 0
    assert legal[CALL] == 1.0
    assert CHECK not in legal


def test_short_stack_sees_only_all_in_for_its_stack():
    sim = make_sim([100.0, 100.0, 0.6, 100.0])
    obs = sim.observation()
    assert obs.player_id == IDS[2]
    assert dict(obs.legal_actions) == {FOLD: 0.0, ALL_IN: 0.6}
    sim.step(ALL_IN)
    assert sim._game.players[2].in_chips == 60


def test_action_amounts_recorded():
    sim = make_sim([100.0] * 8)
    sim.step(CALL)
    sim.step(FOLD)
    obs = sim.observation()
    assert obs.action_history == ((IDS[2], CALL, 1.0), (IDS[3], FOLD, 0.0))


def test_illegal_action_raises():
    sim = make_sim([100.0] * 8)
    with pytest.raises(IllegalActionError):
        sim.step(CHECK)  # facing the big blind
    with pytest.raises(IllegalActionError):
        sim.step("raise_a_lot")


def test_same_seed_same_deal_and_result():
    def run(seed):
        bots = {pid: RandomBot(random.Random(pid)) for pid in IDS}
        sim = make_sim([100.0] * 8, seed=seed, bots=bots)
        first = sim.observation().hole_cards
        return first, sim.play()

    assert run(7) == run(7)
    assert run(7)[0] != run(8)[0]


def test_round_log_fields():
    log = make_sim([50.0, 60.0, 70.0], seed=3).play()
    assert log.positions == tuple(IDS[:3])
    assert log.starting_stacks == (50.0, 60.0, 70.0)
    assert set(log.chip_deltas) == set(IDS[:3])
    assert all(street in ("preflop", "flop", "turn", "river") for _, street, _, _ in log.actions)


def test_all_in_runs_out_full_board():
    bots = {pid: FixedBot(ALL_IN) for pid in IDS[:4]}
    log = make_sim([100.0] * 4, bots=bots).play()
    assert len(log.community_cards) == 5
    assert len(log.showdown_cards) == 4
    hole = [c for cards in log.showdown_cards.values() for c in cards]
    assert len(set(hole + list(log.community_cards))) == 13


def test_everyone_folds_means_no_showdown():
    bots = {pid: FixedBot(FOLD) for pid in IDS}
    log = make_sim([100.0] * 8, bots=bots).play()
    assert log.showdown_cards == {}
    assert log.community_cards == ()
    assert log.chip_deltas[IDS[1]] == 0.5  # big blind wins the small blind
    assert log.chip_deltas[IDS[0]] == -0.5


def test_bad_inputs_rejected():
    with pytest.raises(ValueError):
        make_sim([100.0])
    with pytest.raises(ValueError):
        make_sim([100.0, 0.0])
    with pytest.raises(ValueError):
        Simulator({}, IDS[:2], [100.0, 100.0], seed=0, game_id=0, rotation_id=0, round_id=0)


# --- Snapshots (#18) ----------------------------------------------------------


def advance_to_flop(sim):
    while sim._street() == "preflop":
        names = [n for n, _ in sim.observation().legal_actions]
        sim.step(CALL if CALL in names else CHECK)


def test_snapshot_redeals_only_unseen_cards():
    sim = make_sim([100.0] * 4, seed=11)
    advance_to_flop(sim)
    obs = sim.observation()
    snap = sim.snapshot()
    viewer = sim._game.game_pointer
    true_opp_hands = [
        [c.get_index() for c in p.hand] for i, p in enumerate(sim._game.players) if i != viewer
    ]

    changed = False
    for k in range(20):
        rollout = Simulator.from_snapshot(snap, sim.bots, seed=k)
        r_obs = rollout.observation()
        assert r_obs.hole_cards == obs.hole_cards
        assert r_obs.community_cards == obs.community_cards
        assert r_obs.action_history == obs.action_history
        opp = [
            [c.get_index() for c in p.hand]
            for i, p in enumerate(rollout._game.players)
            if i != viewer
        ]
        changed |= opp != true_opp_hands
        # no card dealt twice
        all_cards = [c.get_index() for p in rollout._game.players for c in p.hand]
        all_cards += [c.get_index() for c in rollout._game.public_cards]
        all_cards += [c.get_index() for c in rollout._game.dealer.deck]
        assert len(all_cards) == len(set(all_cards)) == 52
    assert changed


def test_snapshot_does_not_disturb_original_and_rollouts_finish():
    sim = make_sim([100.0] * 4, seed=12)
    advance_to_flop(sim)
    before = sim.observation()
    snap = sim.snapshot()

    for k in range(5):
        log = Simulator.from_snapshot(snap, sim.bots, seed=k).play()
        assert sum(log.chip_deltas.values()) == pytest.approx(0.0)

    assert sim.observation() == before
    a = Simulator.from_snapshot(snap, sim.bots, seed=99)
    b = Simulator.from_snapshot(snap, sim.bots, seed=99)
    assert a.observation() == b.observation()
