"""Switcher (#10): changes style at exactly every 50th rotation and logs it."""

from __future__ import annotations

from pokerbots.bots.simple import SwitcherBot
from pokerbots.core import CALL, CHECK, FOLD, Observation


def _obs(rotation_id: int) -> Observation:
    legal = ((FOLD, 0.0), (CHECK, 0.0), (CALL, 1.0))
    return Observation(("SA", "HK"), (), 1.5, 1, (1, 2), (99.0, 99.0), (), legal, 0, rotation_id, 0)


def test_switches_exactly_every_50_rotations():
    bot = SwitcherBot()
    bot.new_game(3, 1)
    for r in range(200):
        for _ in range(3):          # several decisions per rotation
            assert bot.act(_obs(r)) in (FOLD, CHECK, CALL)
    assert [r for r, _ in bot.switches] == [0, 50, 100, 150]
    names = [n for _, n in bot.switches]
    assert all(a != b for a, b in zip(names, names[1:], strict=False))


def test_switch_sequence_is_seeded():
    runs = []
    for _ in range(2):
        bot = SwitcherBot()
        bot.new_game(7, 2)
        for r in range(0, 200, 10):
            bot.act(_obs(r))
        runs.append(bot.switches)
    assert runs[0] == runs[1]
