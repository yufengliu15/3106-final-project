import json
from dataclasses import FrozenInstanceError, asdict

import pytest

from pokerbots.core import (
    CALL,
    CHECK,
    FOLD,
    RAISE_HALF_POT,
    Bot,
    Observation,
    RoundLog,
)

POSITIONS = (1001, 2002, 3003, 4004, 5005, 6006, 7007, 8008)


def make_observation() -> Observation:
    return Observation(
        hole_cards=("C7", "HQ"),
        community_cards=("SA", "D9", "HT"),
        pot=6.5,
        player_id=3003,
        positions=POSITIONS,
        stacks=(99.5, 98.0, 98.0, 0, None, 100.0, 100.0, 100.0),
        action_history=((1001, CHECK, 0.0), (2002, RAISE_HALF_POT, 3.25)),
        legal_actions=((FOLD, 0.0), (CALL, 3.25), (RAISE_HALF_POT, 6.5)),
        game_id=1,
        rotation_id=0,
        round_id=3,
    )


def make_round_log() -> RoundLog:
    return RoundLog(
        game_id=1,
        rotation_id=0,
        round_id=3,
        positions=POSITIONS,
        starting_stacks=(100.0,) * 7 + (None,),
        actions=(
            (3003, "preflop", CALL, 1.0),
            (1001, "flop", CHECK, 0.0),
            (2002, "flop", RAISE_HALF_POT, 3.25),
            (3003, "flop", FOLD, 0.0),
        ),
        showdown_cards={},
        chip_deltas={2002: 4.25, 3003: -1.0, 1001: -3.25},
    )


def test_build_observation():
    obs = make_observation()
    assert obs.hole_cards == ("C7", "HQ")
    assert len(obs.positions) == len(obs.stacks) == 8
    assert obs.player_id in obs.positions


def test_build_round_log():
    log = make_round_log()
    assert all(street in ("preflop", "flop", "turn", "river") for _, street, _, _ in log.actions)
    assert sum(log.chip_deltas.values()) == pytest.approx(0.0)


def test_observation_and_round_log_are_frozen():
    with pytest.raises(FrozenInstanceError):
        make_observation().pot = 0.0
    with pytest.raises(FrozenInstanceError):
        make_round_log().round_id = 5


def test_round_log_is_json_serializable():
    data = json.loads(json.dumps(asdict(make_round_log())))
    assert data["round_id"] == 3
    assert data["chip_deltas"]["2002"] == 4.25  # int keys become strings in JSON


def test_bot_is_abstract():
    with pytest.raises(TypeError):
        Bot()
