"""Tests for #18 (rollout labeller) and the pieces Bot C relies on."""

from __future__ import annotations

import random

import pytest

from pokerbots.bots.ml import labeller
from pokerbots.bots.ml.bot_c import MLBot
from pokerbots.bots.simple import (
    AlwaysAllInBot, CallingMachineBot, LooseAggressiveBot, RandomBot, TightAggressiveBot, TightPassiveBot,
)
from pokerbots.core import ACTIONS, ALL_IN, CHECK, FOLD, Observation
from pokerbots.core.situation import analyse
from pokerbots.sim.simulator import Simulator

IDS = (11, 22, 33, 44, 55, 66)


def _bots():
    return {11: LooseAggressiveBot(), 22: TightAggressiveBot(), 33: CallingMachineBot(),
            44: RandomBot(), 55: TightPassiveBot(), 66: AlwaysAllInBot()}


def _first_learner_decision(seed: int, learner: int = 55):
    bots = _bots()
    for pid, b in bots.items():
        b.new_game(0, pid)
    sim = Simulator(bots, IDS, [100.0] * len(IDS), seed, 0, 0, 0)
    histories: dict[int, list] = {}
    while not sim.is_over():
        obs = sim.observation()
        if obs.player_id == learner:
            return sim, obs, bots, histories
        a = bots[obs.player_id].act(obs)
        histories.setdefault(obs.player_id, []).append((obs, a))
        sim.step(a)
    return None


def test_labels_do_not_depend_on_the_true_future():
    """Same snapshot + same labeller RNG, different true hidden cards -> same label."""
    found = _first_learner_decision(seed=3)
    assert found is not None
    sim, obs, bots, hist = found
    snap = sim.snapshot()
    row1 = labeller.label_decision(snap, obs, bots, hist, 55, 100.0, random.Random(7))
    # change the true future: re-deal hidden cards inside the snapshot, then label again
    other = Simulator.from_snapshot(snap, bots, seed=999).snapshot()
    row2 = labeller.label_decision(other, obs, bots, hist, 55, 100.0, random.Random(7))
    assert row1["label"] == row2["label"]
    for a in ACTIONS:
        v1, v2 = row1[f"ev_{a}"], row2[f"ev_{a}"]
        assert (v1 != v1 and v2 != v2) or v1 == pytest.approx(v2)


def test_consistent_redeal_matches_recorded_actions():
    bot = TightAggressiveBot()
    found = _first_learner_decision(seed=5)
    _, obs, _, hist = found
    unseen = [c for c in __import__("pokerbots.core.cards", fromlist=["FULL_DECK"]).FULL_DECK
              if c not in obs.hole_cards]
    rng = random.Random(1)
    for pid, h in hist.items():
        if pid != 22:
            continue
        hand, ok = labeller._consistent_hand(bot, h, set(), unseen, rng)
        assert ok
        for o, act in h:
            assert bot.act(o.__class__(**{**o.__dict__, "hole_cards": hand})) == act


def test_fixed_hands_are_dealt():
    found = _first_learner_decision(seed=8)
    sim, obs, bots, _ = found
    unseen = sim.unseen_cards()
    hand = (unseen[0], unseen[1])
    rollout = Simulator.from_snapshot(sim.snapshot(), bots, seed=1, hands={22: hand})
    assert tuple(c.get_index() for c in rollout._game.players[IDS.index(22)].hand) == hand


@pytest.mark.parametrize("seed", range(40))
def test_situation_tracks_folds_and_streets(seed):
    rng = random.Random(seed)
    bots = {pid: RandomBot() for pid in IDS}
    for pid, b in bots.items():
        b.new_game(seed, pid)
    sim = Simulator(bots, IDS, [rng.choice([5.0, 40.0, 100.0]) for _ in IDS], seed, 0, 0, 0)
    while not sim.is_over():
        obs = sim.observation()
        sit = analyse(obs)
        folded = {sim.positions[i] for i, p in enumerate(sim._game.players) if p.status.name == "FOLDED"}
        assert set(sit.opponents_in_hand) == set(IDS) - folded - {obs.player_id}
        assert sit.street == sim._game.stage.value
        sim.step(bots[obs.player_id].act(obs))


def test_always_all_in_only_legal():
    bot = AlwaysAllInBot()
    obs = Observation(("SA", "HK"), (), 1.5, 1, (1, 2), (99.5, 99.0), (),
                      ((FOLD, 0.0), (ALL_IN, 99.5)), 0, 0, 0)
    assert bot.act(obs) == ALL_IN
    obs2 = Observation(("SA", "HK"), (), 1.5, 1, (1, 2), (0.0, 99.0), (), ((CHECK, 0.0),), 0, 0, 0)
    assert bot.act(obs2) == CHECK


def test_v2_mixed_continuation_ignores_true_future():
    """Mixed continuation (D5) keeps the no-peeking property."""
    found = _first_learner_decision(seed=4)
    sim, obs, bots, hist = found
    snap = sim.snapshot()
    pool = (TightAggressiveBot, LooseAggressiveBot, CallingMachineBot)
    conts = (TightAggressiveBot, MLBot)
    kw = dict(cont_pool=conts, swap_pool=pool)
    row1 = labeller.label_decision(snap, obs, bots, hist, 55, 100.0, random.Random(3), **kw)
    other = Simulator.from_snapshot(snap, bots, seed=123).snapshot()
    row2 = labeller.label_decision(other, obs, bots, hist, 55, 100.0, random.Random(3), **kw)
    assert row1["label"] == row2["label"] and row1["cont"] == row2["cont"]
