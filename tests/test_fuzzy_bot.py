"""Tests for Bot B (issue #2): fuzzification, Mamdani inference and FuzzyBot."""

from __future__ import annotations

import time
from dataclasses import replace

import pytest

from pokerbots.bots.fuzzy.bot import FuzzyBot, nearest_legal_action, score_to_action
from pokerbots.bots.fuzzy.inference import infer
from pokerbots.bots.fuzzy.membership import fuzzify
from pokerbots.bots.fuzzy.features import stack_call_price
from pokerbots.bots.fuzzy.rules import STACK_CALL_RULES, Rule
from pokerbots.bots.fuzzy.showdown import selective_showdown
from pokerbots.core import ACTIONS, ALL_IN, CALL, CHECK, FOLD, RAISE_HALF_POT, RAISE_POT, Observation, RoundLog
from pokerbots.core.situation import analyse


QUADS_BOARD = ("CA", "DA", "H2", "S5", "CK")


def _obs(hole, board=(), pot=1.5, legal=((CHECK, 0.0), (RAISE_HALF_POT, 0.75))):
    return Observation(hole, board, pot, 1, (1, 2), (99.5, 99.0), (), legal, 0, 0, 0)


def test_fuzzification_known_points():
    assert fuzzify("hand", 0.10)["thin"] == 1.0
    assert fuzzify("hand", 0.80)["value"] == 1.0
    assert fuzzify("hand", 0.92)["premium"] == 1.0
    assert fuzzify("equity", 1.0)["dominant"] == 1.0
    assert fuzzify("pot_odds", 0.0)["cheap"] == 1.0
    assert fuzzify("pot_odds", 0.80)["expensive"] == 1.0
    assert fuzzify("looseness", 0.35)["neutral"] == 1.0
    assert fuzzify("aggression", 1.5)["neutral"] == 1.0
    assert fuzzify("sample", 0.0)["large"] == 0.0
    assert fuzzify("sample", 10.0)["large"] == 0.5
    assert fuzzify("sample", 50.0)["large"] == 1.0


def test_mamdani_pipeline_uses_min_max_centroid():
    score, strengths = infer({"edge": -0.1}, STACK_CALL_RULES)
    assert score == pytest.approx(0.10)
    assert strengths["fold-to-unprofitable-stack-call"] == 1.0
    assert set(strengths) == {rule.name for rule in STACK_CALL_RULES}
    clipped = Rule("partly-fired", {"equity": "dominant", "sample": "large"}, "very-aggressive")
    fold = Rule("fold", {"edge": "poor"}, "very-passive")
    inputs = {"equity": 0.98, "sample": 10.0, "edge": -0.1}
    score, strengths = infer(inputs, (clipped,))
    assert score == pytest.approx(0.95)
    assert strengths["partly-fired"] == pytest.approx(0.5)
    assert infer(inputs, (clipped, fold))[0] == infer(inputs, (clipped, fold, replace(fold, name="duplicate")))[0]


def test_score_to_action_bands():
    assert score_to_action(0.10, 1.0) == FOLD
    assert score_to_action(0.30, 0.0) == CHECK
    assert score_to_action(0.30, 1.0) == CALL
    assert score_to_action(0.60, 0.0) == RAISE_HALF_POT
    assert score_to_action(0.80, 0.0) == RAISE_POT
    assert score_to_action(0.95, 0.0) == ALL_IN
    assert score_to_action(0.20, 0.0) == CHECK
    assert score_to_action(0.50, 0.0) == RAISE_HALF_POT
    assert score_to_action(0.75, 0.0) == RAISE_POT
    assert score_to_action(0.90, 0.0) == ALL_IN


def test_act_selects_all_in_and_explains_every_action():
    bot = FuzzyBot()
    bot.new_game(0, 1)
    legal = ((CHECK, 0.0), (RAISE_HALF_POT, 1.0), (ALL_IN, 1.5))
    obs = replace(_obs(("SA", "HA"), QUADS_BOARD, pot=2.0, legal=legal), stacks=(1.5, 2.0))
    assert bot.act(obs) == ALL_IN
    text = bot.explain()
    assert "score=0.950" in text
    assert "top rules: dominant-hand-at-short-stack=1.00" in text
    assert "action support:" in text
    assert bot.action_support() == {action: float(action == ALL_IN) for action in ACTIONS}
    assert all(f"{action}=" in text for action in ACTIONS)


def test_fallback_uses_nearest_legal_band_and_explains_change():
    assert nearest_legal_action(RAISE_HALF_POT, {FOLD: 0.0, CALL: 2.0, ALL_IN: 7.0}) == CALL
    assert nearest_legal_action(CALL, {FOLD: 0.0, ALL_IN: 2.0}) == ALL_IN
    assert nearest_legal_action(RAISE_POT, {CHECK: 0.0, RAISE_HALF_POT: 1.0, ALL_IN: 7.0}) == RAISE_HALF_POT

    bot = FuzzyBot()
    assert bot.act(_obs(("SA", "HA"), QUADS_BOARD)) == RAISE_HALF_POT
    assert "nearest legal fallback from raise_pot" in bot.explain()
    assert bot.action_support()[RAISE_POT] == 1.0


def test_preflop_equity_includes_the_limped_field():
    players = tuple(range(1, 11))
    history = tuple((pid, CALL, 1.0) for pid in players[2:]) + ((1, CALL, 0.5),)
    obs = Observation(("SA", "HA"), (), 10.0, 2, players, (99.0,) * 10, history,
                      ((CHECK, 0.0), (RAISE_HALF_POT, 5.0), (RAISE_POT, 10.0), (ALL_IN, 99.0)), 1, 0, 0)
    bot = FuzzyBot()
    bot.act(obs)
    assert "equity=0.313" in bot.explain()


def test_act_folds_weak_facing_bet():
    bot = FuzzyBot()
    bot.new_game(0, 1)
    legal = ((FOLD, 0.0), (CALL, 2.0), (RAISE_HALF_POT, 2.0), (ALL_IN, 99.0))
    obs = Observation(("S2", "H7"), ("C3", "D8", "HK"), 2.0, 1, (1, 2),
                      (99.5, 99.0), ((2, CALL, 1.0),), legal, 0, 0, 0)
    assert bot.act(obs) == FOLD


def test_stack_call_uses_contestable_pot_and_bust_cost():
    legal = ((FOLD, 0.0), (ALL_IN, 39.5))
    obs = Observation(("SK", "HQ"), (), 100.5, 1, (1, 2), (39.5, 0.0),
                      ((2, ALL_IN, 99.0),), legal, 0, 0, 0)
    # Opponent bet 100, but the caller can win only 40 of it. A loss also
    # incurs the tournament's 40 bb bust penalty.
    assert stack_call_price(obs, analyse(obs), loss_probability=0.25) == pytest.approx(49.5 / 80.0)
    bot = FuzzyBot()
    assert bot.act(obs) == FOLD
    assert bot.act(replace(obs, hole_cards=("SA", "HA"))) == ALL_IN
    assert "take-profitable-stack-call" in bot.explain()


def test_guaranteed_chop_does_not_incur_bust_cost():
    board = ("ST", "SJ", "SQ", "SK", "SA")
    legal = ((FOLD, 0.0), (ALL_IN, 39.5))
    obs = Observation(("H4", "D5"), board, 100.5, 1, (1, 2), (39.5, 0.0),
                      ((2, ALL_IN, 99.0),), legal, 0, 0, 0)
    result = selective_showdown(obs.hole_cards, board, (0.18,))
    assert result.share == pytest.approx(0.5)
    assert result.loss_probability == 0.0
    assert FuzzyBot().act(obs) == ALL_IN


def test_selective_equity_conditions_on_bets_without_changing_public_cards():
    hole = ("S9", "H9")
    board = ("C2", "D7", "D8", "CK", "SJ")
    wide = selective_showdown(hole, board, (1.0, 1.0), sims=1500)
    narrow = selective_showdown(hole, board, (0.19, 1.0), sims=1500)
    assert narrow.share < wide.share - 0.15
    assert narrow == selective_showdown(hole[::-1], board, (1.0, 0.19), sims=1500)


def test_opponent_model_flag_and_sample_default():
    on, off = FuzzyBot(use_opponent_model=True), FuzzyBot(use_opponent_model=False)
    on.new_game(0, 1)
    off.new_game(0, 1)
    legal = ((FOLD, 0.0), (CALL, 3.5), (RAISE_POT, 5.5), (ALL_IN, 99.0))
    obs = replace(_obs(("SQ", "HJ"), pot=5.5, legal=legal),
                  action_history=((1, CALL, 0.5), (2, RAISE_POT, 3.5)))
    assert on.act(obs) == off.act(obs)
    for i in range(20):
        log = RoundLog(0, i, 0, i, (1, 2), (100.0, 100.0), ((2, "preflop", RAISE_POT, 2.0),), (), {}, {})
        on.observe_result(log)
        off.observe_result(log)
    on.act(obs)
    off.act(obs)
    assert "om=0" in off.explain() and "sample=0" in off.explain()
    assert "om=1" in on.explain() and "sample=20" in on.explain()
    assert on._last.crisp["equity"] > off._last.crisp["equity"]
    on.new_game(1, 1)
    assert on.explain() == "no decision yet"
    assert on.act(obs) == off.act(obs)


def test_act_fast_and_always_legal():
    bot = FuzzyBot()
    bot.new_game(0, 1)
    cases = [
        _obs(("SA", "HA")),
        _obs(("S2", "H7"), legal=((FOLD, 0.0), (CALL, 1.0), (ALL_IN, 99.0))),
        _obs(("SK", "HQ"), ("C3", "D8", "HK"), pot=4.0,
             legal=((FOLD, 0.0), (CALL, 2.0), (RAISE_POT, 4.0), (ALL_IN, 99.0))),
    ]
    for obs in cases:
        start = time.perf_counter()
        action = bot.act(obs)
        assert time.perf_counter() - start < 0.2
        assert action in {a for a, _ in obs.legal_actions}
