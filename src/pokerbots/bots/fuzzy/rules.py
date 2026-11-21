"""Fuzzy rule base for Bot B.

Each rule is data: IF hand is X [AND pot_odds is Y ...] THEN aggression is Z.
A rule fires with strength ``min`` over its antecedents. Hand and price rules
provide default play. Opponent rules use the "large" sample membership, which
gradually turns on as the tracker accumulates observations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    """One Mamdani rule. ``antecedent`` maps variable -> term name."""

    name: str
    antecedent: dict[str, str]
    consequent: str


OPENING_RULES = (
    Rule("premium-opening-hand", {"opening": "premium"}, "aggressive"),
    Rule("position-appropriate-open", {"opening": "raise"}, "moderate"),
    Rule("marginal-defend-at-good-price", {"opening": "defend", "edge": "sound"}, "passive"),
    Rule("marginal-defend-at-bad-price", {"opening": "defend", "edge": "poor"}, "very-passive"),
    Rule("outside-opening-range", {"opening": "outside"}, "very-passive"),
)

RERAISE_RULES = (
    Rule("ordinary-hand-facing-preflop-raise", {"equity": "call"}, "passive"),
    Rule("strong-hand-facing-preflop-raise", {"equity": "reraise"}, "aggressive"),
)

RULES = (
    Rule("thin-showdown-value", {"hand": "thin", "fold_equity": "low", "equity": "ordinary"}, "passive"),
    Rule("small-bluff-against-frequent-folds",
         {"hand": "thin", "fold_equity": "high", "equity": "ordinary"}, "moderate"),
    Rule("modest-value-bet", {"hand": "value", "equity": "ordinary"}, "moderate"),
    Rule("large-value-bet", {"hand": "premium", "equity": "ordinary"}, "aggressive"),
    Rule("dominant-hand-at-deep-stack", {"equity": "dominant", "spr": "deep"}, "aggressive"),
    Rule("dominant-hand-at-short-stack", {"equity": "dominant", "spr": "short"}, "very-aggressive"),
    Rule("thin-value-against-loose-passive",
         {"hand": "thin", "looseness": "loose", "aggression": "passive", "sample": "large",
          "pot_odds": "cheap", "equity": "ordinary"}, "moderate"),
)

STACK_CALL_RULES = (
    Rule("fold-to-unprofitable-stack-call", {"edge": "poor"}, "very-passive"),
    Rule("take-profitable-stack-call", {"edge": "sound"}, "passive"),
)

PRICE_RULES = (
    Rule("give-up-at-a-bad-price", {"edge": "poor"}, "very-passive"),
    Rule("continue-at-an-affordable-price", {"edge": "sound"}, "very-aggressive"),
)

ALL_RULES = OPENING_RULES + RERAISE_RULES + RULES + STACK_CALL_RULES + PRICE_RULES
