"""Bot B: fuzzy rule-based poker bot (issue #2).

Mamdani fuzzy system: shared inputs are fuzzified (``membership.py``), a
human-readable rule base (``rules.py``) fires with ``min`` AND, consequents
are clipped and ``max``-aggregated, and the centroid gives an aggression
score 0-1 (``inference.py``). The score picks the action on the issue's
fixed thresholds; the nearest legal action is used when the pick is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from pokerbots.bots.fuzzy.features import estimated_equity, fold_equity, is_stack_call, opening_ratio, stack_call_price
from pokerbots.bots.fuzzy.inference import infer
from pokerbots.bots.fuzzy.membership import OUTPUT_SETS
from pokerbots.bots.fuzzy.rules import ALL_RULES, OPENING_RULES, PRICE_RULES, RERAISE_RULES, RULES, STACK_CALL_RULES
from pokerbots.core import (
    ACTIONS,
    ALL_IN,
    CALL,
    CHECK,
    FOLD,
    RAISE_HALF_POT,
    RAISE_POT,
    Bot,
    Observation,
    RoundLog,
)
from pokerbots.core.situation import Situation, analyse
from pokerbots.opponents import PRIOR, OpponentStats, OpponentTracker

# Score -> desired action bands from the issue.
FOLD_BELOW = 0.2
PASSIVE_BELOW = 0.5
HALF_POT_BELOW = 0.75
POT_BELOW = 0.9

RULE_CONSEQUENT = {rule.name: rule.consequent for rule in ALL_RULES}
_ACTION_BANDS = {FOLD: 0, CHECK: 1, CALL: 1, RAISE_HALF_POT: 2, RAISE_POT: 3, ALL_IN: 4}


def score_to_action(score: float, to_call: float) -> str:
    """Map an aggression score to the issue's action bands."""
    if score < FOLD_BELOW:
        return FOLD
    if score < PASSIVE_BELOW:
        return CHECK if to_call <= 0 else CALL
    if score < HALF_POT_BELOW:
        return RAISE_HALF_POT
    if score < POT_BELOW:
        return RAISE_POT
    return ALL_IN


def nearest_legal_action(action: str, legal: dict[str, float]) -> str:
    """Choose the nearest legal aggression band, breaking ties downward."""
    if action in legal:
        return action
    bands = _ACTION_BANDS
    if is_stack_call(legal):
        # The simulator labels a stack-capped call ALL_IN.
        bands = {**bands, ALL_IN: bands[CALL]}
    return min((a for a in ACTIONS if a in legal), key=lambda a: abs(bands[a] - bands[action]))


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


@dataclass(frozen=True)
class Decision:
    """What act() computed, kept for explain()."""

    action: str
    desired: str
    score: float
    equity: float
    to_call: float
    crisp: dict[str, float]
    strengths: dict[str, float]
    limit: str
    limit_strengths: dict[str, float]


class FuzzyBot(Bot):
    """Fuzzy-logic main bot. Only reads ``core`` + ``opponents``."""

    name = "FuzzyBot"

    def __init__(self, use_opponent_model: bool = True) -> None:
        self.use_opponent_model = use_opponent_model
        self.tracker = OpponentTracker()
        self._last: Decision | None = None

    def new_game(self, game_id: int, player_id: int) -> None:
        self.tracker.reset()
        self._last = None

    def observe_result(self, log: RoundLog) -> None:
        self.tracker.update(log)

    def _inputs(self, obs: Observation, sit: Situation, opponents: dict[int, OpponentStats],
                stack_call: bool) -> tuple[dict[str, float], float]:
        raw, estimate = estimated_equity(obs, sit, opponents, stack_call)
        weighted = estimate.share if estimate else raw
        adjusted = weighted * sqrt(max(len(opponents), 1))
        field = tuple(opponents.values()) or (PRIOR,)
        bettor = opponents.get(sit.last_aggressor)
        aggression = bettor.aggression if bettor else sum(s.aggression for s in field) / len(field)
        # Larger calls expose more of the stack to further pressure.
        risk_fraction = sit.to_call / max(sit.my_stack, 0.01)
        price = sit.pot_odds
        margin = 0.025 + 0.12 * min(1.0, 2 * risk_fraction)
        if stack_call and estimate:
            price = stack_call_price(obs, sit, estimate.loss_probability)
            margin = 0.04 * estimate.loss_probability
        crisp = {
            "hand": adjusted / (adjusted + (1.0 - weighted)),
            "equity": weighted,
            "pot_odds": _clamp(sit.pot_odds, 0.0, 1.0),
            "looseness": sum(s.vpip for s in field) / len(field),
            "aggression": _clamp(aggression, 0.0, 10.0),
            "sample": float(min(stats.rounds for stats in field)),
            "opening": opening_ratio(obs, sit) if sit.street == 0 else 0.0,
            "edge": weighted - price - margin,
            "spr": sit.spr,
            "fold_equity": fold_equity(sit, opponents) if self.use_opponent_model and not sit.to_call else 0.0,
        }
        return crisp, raw

    def act(self, obs: Observation) -> str:
        sit = analyse(obs)
        legal = dict(obs.legal_actions)
        stack_call = is_stack_call(legal)
        opponents = {pid: self.tracker.stats(pid) if self.use_opponent_model else PRIOR
                     for pid in sit.opponents_in_hand}
        crisp, raw = self._inputs(obs, sit, opponents, stack_call)
        if stack_call:
            rules = STACK_CALL_RULES
        elif sit.street > 0:
            rules = RULES
        elif sit.raises_preflop > 0:
            rules = RERAISE_RULES
        else:
            rules = OPENING_RULES
        score, strengths = infer(crisp, rules)
        limit = ""
        price_strengths = {}
        if not stack_call:
            score, price_strengths, limit = self._apply_limits(sit, legal, crisp, score)
        desired = score_to_action(score, sit.to_call)
        action = nearest_legal_action(desired, legal)
        self._last = Decision(action, desired, score, raw, sit.to_call, crisp, strengths, limit, price_strengths)
        return action

    @staticmethod
    def _apply_limits(sit: Situation, legal: dict[str, float],
                      crisp: dict[str, float], score: float) -> tuple[float, dict[str, float], str]:
        strengths = {}
        limit = ""
        if sit.to_call > 1.25 and crisp["edge"] < 0.025:
            price_score, price_strengths = infer({"edge": crisp["edge"]}, PRICE_RULES)
            if price_score < score:
                score, strengths, limit = price_score, price_strengths, "price limited aggression"

        action = nearest_legal_action(score_to_action(score, sit.to_call), legal)
        if action in (FOLD, CHECK, CALL):
            return score, strengths, limit

        # Treat raises as exposure to the effective stack. Marginal equity
        # buys smaller bets and cannot turn a missing raise into a big shove.
        allowed_share = 0.20 + 2.6 * max(0.0, crisp["equity"] - 0.48)
        budget = max(2.0, sit.my_stack * allowed_share)
        if min(legal[action], sit.effective_stack) <= budget:
            return score, strengths, limit
        small_raise = legal.get(RAISE_HALF_POT, float("inf"))
        output = "moderate" if small_raise <= budget else "passive"
        return min(score, OUTPUT_SETS[output].c), strengths, "effective-stack limit"

    def term_support(self) -> dict[str, float]:
        """Maximum firing strength per fuzzy output term."""
        if self._last is None:
            return {}
        support: dict[str, float] = {}
        for name, level in self._last.strengths.items():
            term = RULE_CONSEQUENT[name]
            support[term] = max(support.get(term, 0.0), level)
        return support

    def action_support(self) -> dict[str, float]:
        """Maximum rule strength per action band before legal fallback."""
        if self._last is None:
            return {}
        support = dict.fromkeys(ACTIONS, 0.0)
        for term, level in self.term_support().items():
            action = score_to_action(OUTPUT_SETS[term].c, self._last.to_call)
            support[action] = max(support[action], level)
        return support

    def explain(self) -> str:
        if self._last is None:
            return "no decision yet"
        d = self._last
        fired = [(name, strength) for name, strength in d.strengths.items() if strength > 0]
        top = ", ".join(f"{n}={v:.2f}" for n, v in sorted(fired, key=lambda kv: -kv[1])[:3])
        support = ", ".join(f"{a}={v:.2f}" for a, v in self.action_support().items())
        fallback = "" if d.action == d.desired else f", nearest legal fallback from {d.desired}"
        limit = f", {d.limit}" if d.limit else ""
        limited = ", ".join(f"{n}={v:.2f}" for n, v in d.limit_strengths.items() if v > 0)
        limits = f"; limit rules: {limited}" if limited else ""
        crisp = (f"equity={d.equity:.3f} range_equity={d.crisp['equity']:.3f} hand={d.crisp['hand']:.3f} "
                 f"pot_odds={d.crisp['pot_odds']:.3f} call_edge={d.crisp['edge']:+.3f} "
                 f"loose={d.crisp['looseness']:.2f} aggr={d.crisp['aggression']:.2f} "
                 f"sample={d.crisp['sample']:.0f} om={int(self.use_opponent_model)}")
        return (f"{d.action} (score={d.score:.3f}{fallback}{limit}); {crisp}; "
                f"top rules: {top or 'none'}{limits}; action support: {support}")
