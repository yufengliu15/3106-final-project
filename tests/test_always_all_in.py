"""#8: AlwaysAllInBot returns all-in from any observation and plays 1,000
rounds through the simulator with no illegal actions."""

from __future__ import annotations

import random

from pokerbots.bots.simple import RandomBot
from pokerbots.bots.simple.always_all_in import AlwaysAllInBot
from pokerbots.core import ALL_IN, CALL, CHECK, FOLD, RAISE_POT, Observation
from pokerbots.sim.simulator import Simulator


def _obs(legal):
    return Observation(("SA", "HK"), (), 1.5, 1, (1, 2), (99.5, 99.0), (), legal, 0, 0, 0)


def test_returns_all_in_whenever_offered():
    bot = AlwaysAllInBot()
    for legal in (((FOLD, 0.0), (CALL, 0.5), (RAISE_POT, 2.0), (ALL_IN, 99.5)),
                  ((CHECK, 0.0), (ALL_IN, 99.0)),
                  ((FOLD, 0.0), (ALL_IN, 3.0))):
        assert bot.act(_obs(legal)) == ALL_IN
        assert "all_in" in bot.explain()


def test_checks_when_all_in_not_offered():
    assert AlwaysAllInBot().act(_obs(((CHECK, 0.0),))) == CHECK


def test_plays_1000_rounds_legally():
    ids = (101, 102, 103, 104, 105, 106)
    rng = random.Random(0)
    for r in range(1000):
        bots = {pid: (AlwaysAllInBot() if i % 2 == 0 else RandomBot()) for i, pid in enumerate(ids)}
        for pid, b in bots.items():
            b.new_game(r, pid)
        stacks = [rng.choice([1.0, 10.0, 100.0]) for _ in ids]
        Simulator(bots, ids, stacks, seed=r, game_id=0, rotation_id=0, round_id=r).play()
