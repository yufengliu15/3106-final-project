"""Variant library: every approach plays legal, deterministic, fast poker."""

from __future__ import annotations

import dataclasses
import time

import pytest

from pokerbots.bots.simple import SCRIPTED_POOL
from pokerbots.bots.variants import APPROACHES, VariantSpec, default_spec, sample_spec
from pokerbots.sim.simulator import Simulator
from pokerbots.tournament import Tournament, TournamentConfig

ALL = sorted(APPROACHES) + ["A6"]


@pytest.mark.parametrize("approach", ALL)
@pytest.mark.parametrize("draw", [0, 1, 2])
def test_variant_plays_legal_and_fast(approach, draw):
    spec = default_spec(approach) if draw == 0 and approach != "A6" else sample_spec(approach, draw)
    cls = spec.make()
    assert cls.__name__ == spec.name
    worst = 0.0
    orig = cls.act

    def timed(self, obs):
        nonlocal worst
        t = time.perf_counter()
        a = orig(self, obs)
        worst = max(worst, time.perf_counter() - t)
        assert a in dict(obs.legal_actions)
        # deterministic: same observation, same action
        assert orig(self, dataclasses.replace(obs)) == a
        return a

    cls.act = timed
    cfg = TournamentConfig(main_bots=(cls,), simple_bots=SCRIPTED_POOL, games=1, rotations=8,
                           mode="eval", base_seed=draw)
    Tournament(cfg).run()
    assert worst < 0.2


def test_spec_json_roundtrip():
    for a in ALL:
        s = sample_spec(a, 5)
        assert VariantSpec.from_json(s.to_json()).params == s.params
