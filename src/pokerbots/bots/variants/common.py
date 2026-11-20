"""Shared plumbing for the Bot A / Bot B variant library (docs/bot_c_robust_plan.md §2).

Every variant obeys the proposal: it sees only the Observation, uses the
shared OpponentTracker (or PRIOR when the opponent model is off), is
deterministic given its inputs (so the labeller's consistent re-deal works),
and decides well under 200 ms.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

from pokerbots.core import ALL_IN, CALL, CHECK, FOLD, RAISE_HALF_POT, RAISE_POT, Bot, Observation, RoundLog
from pokerbots.core.cards import hand_class
from pokerbots.core.equity import equity, hand_percentile
from pokerbots.core.situation import Situation, analyse, legalize
from pokerbots.opponents import PRIOR, OpponentTracker


@dataclass(frozen=True)
class OppRead:
    vpip: float
    pfr: float
    aggr: float
    fold_to_raise: float
    confidence: float      # 0 = pure prior, 1 = fully trusted stats
    range_pct: float       # assumed range of the player(s) we must beat
    bettor: bool           # someone else made the last raise this street


def deterministic_unit(obs: Observation, salt: int = 0) -> float:
    """Pseudo-random number in [0, 1) fixed by the hand and decision point:
    lets bots mix (bluff some of the time) while staying a pure function of
    their inputs."""
    key = f"{obs.hole_cards}{obs.community_cards}{len(obs.action_history)}{obs.round_id}{salt}"
    return (zlib.crc32(key.encode()) & 0xFFFFFF) / float(0x1000000)


class VariantBot(Bot):
    """Base class: tracker, opponent read, equity helpers, explanation."""

    family = "?"
    approach = "?"

    def __init__(self, use_opponent_model: bool = True, params: dict | None = None) -> None:
        self.use_opponent_model = use_opponent_model
        self.p = dict(self.default_params())
        self.p.update(params or {})
        self.tracker = OpponentTracker()
        self.player_id = 0
        self._why = ""

    # -- to override -------------------------------------------------------
    @classmethod
    def default_params(cls) -> dict:
        return {}

    @classmethod
    def sample_params(cls, rng, extreme: bool = False) -> dict:
        return {}

    def decide(self, obs: Observation, sit: Situation, legal: dict) -> tuple[str, str]:
        raise NotImplementedError

    # -- Bot interface -------------------------------------------------------
    def new_game(self, game_id: int, player_id: int) -> None:
        self.player_id = player_id
        self.tracker.reset()

    def observe_result(self, log: RoundLog) -> None:
        self.tracker.update(log)

    def explain(self) -> str:
        return self._why

    def act(self, obs: Observation) -> str:
        sit = analyse(obs)
        legal = dict(obs.legal_actions)
        action, why = self.decide(obs, sit, legal)
        action = legalize(action, obs.legal_actions)
        self._why = f"{type(self).__name__}: {why} -> {action}"
        return action

    # -- helpers -----------------------------------------------------------
    def read(self, sit: Situation) -> OppRead:
        bettor = sit.last_aggressor not in (None, self.player_id)
        default_rng = 0.5 if bettor else 1.0
        if not self.use_opponent_model or not sit.opponents_in_hand:
            return OppRead(PRIOR.vpip, PRIOR.pfr, PRIOR.aggression, PRIOR.fold_to_raise, 0.0,
                           default_rng, bettor)
        focus = [sit.last_aggressor] if bettor else list(sit.opponents_in_hand)
        conf_rounds = self.p.get("conf_rounds", 20)
        vs, ps, ags, fs, ws = [], [], [], [], []
        for pid in focus:
            s = self.tracker.stats(pid)
            w = min(s.rounds / conf_rounds, 1.0) if conf_rounds > 0 else 1.0
            vs.append(w * s.vpip + (1 - w) * PRIOR.vpip)
            ps.append(w * s.pfr + (1 - w) * PRIOR.pfr)
            ags.append(w * min(s.aggression, 10.0) + (1 - w) * PRIOR.aggression)
            fs.append(w * s.fold_to_raise + (1 - w) * PRIOR.fold_to_raise)
            ws.append(w)
        vpip, pfr = max(vs), max(ps)
        conf = min(ws)
        if conf <= 0:
            rng = default_rng
        else:
            rng = max(pfr if bettor else vpip, 0.05)
            rng = conf * rng + (1 - conf) * default_rng
        return OppRead(vpip, pfr, max(ags), min(fs), conf, min(rng, 1.0), bettor)

    @staticmethod
    def n_opponents(sit: Situation) -> int:
        n = max(len(sit.opponents_in_hand), 1)
        if sit.street == 0:
            n = max(1, min(n, sit.raises_preflop + (1 if sit.to_call == 0 else 0) + 1))
        return n

    def equity_vs(self, obs: Observation, sit: Situation, rd: OppRead | None) -> float:
        rng = rd.range_pct if rd is not None and rd.range_pct < 0.99 else None
        return equity(obs.hole_cards, obs.community_cards, self.n_opponents(sit), range_pct=rng)

    @staticmethod
    def percentile(obs: Observation) -> float:
        return hand_percentile(hand_class(obs.hole_cards))

    @staticmethod
    def raises(legal: dict) -> tuple[str | None, str | None]:
        big = RAISE_POT if RAISE_POT in legal else (RAISE_HALF_POT if RAISE_HALF_POT in legal else None)
        small = RAISE_HALF_POT if RAISE_HALF_POT in legal else big
        return small, big

    @staticmethod
    def call_or_shove(legal: dict) -> str:
        return CALL if CALL in legal else ALL_IN

    @staticmethod
    def passive(legal: dict) -> str:
        return CHECK if CHECK in legal else FOLD


def jitter(rng, value: float, rel: float, lo: float | None = None, hi: float | None = None) -> float:
    v = value * (1.0 + rng.uniform(-rel, rel))
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


__all__ = ["ALL_IN", "CALL", "CHECK", "FOLD", "RAISE_HALF_POT", "RAISE_POT", "OppRead", "VariantBot",
           "deterministic_unit", "jitter"]
