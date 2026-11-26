"""Bot A layer 2: the rule base as data, in priority order (first match wins)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pokerbots.bots.crisp.facts import FE_HIGH, TIER_RANK, Facts

FOLD, CHECK_CALL, BLUFF, VALUE, SHOVE = "fold", "check_call", "bluff", "value", "shove"
INTENTS = (FOLD, CHECK_CALL, BLUFF, VALUE, SHOVE)


@dataclass(frozen=True)
class Rule:
    name: str
    condition: str                   # the IF part, for explain()
    when: Callable[[Facts], bool]
    then: Callable[[Facts], str]     # intent; only R1 depends on the facts


def _const(intent: str) -> Callable[[Facts], str]:
    return lambda _f: intent


RULES: tuple[Rule, ...] = (
    Rule("R1", "facing=all-in",
         lambda f: f.facing == "all-in",
         lambda f: CHECK_CALL if f.n.equity > f.n.pot_odds else FOLD),
    Rule("R2", "depth=short AND hand_tier>=playable",
         lambda f: f.depth == "short" and f.hand_tier is not None and TIER_RANK[f.hand_tier] <= TIER_RANK["playable"],
         _const(SHOVE)),
    Rule("R3", "edge=big AND villain_type=station",
         lambda f: f.edge == "big" and f.villain_type == "station",
         _const(VALUE)),
    Rule("R4", "edge=big",
         lambda f: f.edge == "big",
         _const(VALUE)),
    Rule("R5", f"edge=thin AND fold_equity>={FE_HIGH}",
         lambda f: f.edge == "thin" and f.fold_equity == "high",
         _const(BLUFF)),
    Rule("R6", "edge=thin",
         lambda f: f.edge == "thin",
         _const(CHECK_CALL)),
    Rule("R7", f"edge=negative AND fold_equity>={FE_HIGH} AND heads-up AND villain known",
         lambda f: (f.edge == "negative" and f.fold_equity == "high" and f.pot_type == "heads-up"
                    and f.villain_type != "unknown"),
         _const(BLUFF)),
    Rule("R8", "default",
         lambda _f: True,
         _const(FOLD)),
)


def fire(facts: Facts) -> tuple[Rule, str]:
    """Forward-chain over the priority list; R8 always matches."""
    rule = next(r for r in RULES if r.when(facts))
    return rule, rule.then(facts)
