"""Mamdani fuzzy inference for Bot B.

Textbook pipeline (Princeton COS436 Ch. 4.1; Kruse/Moewes lecture 7 on the
Mamdani-Assilian controller; MATLAB "Fuzzy Inference Process"):

1. Fuzzify each crisp input with its membership functions.
2. Firing strength of a rule = ``min`` over its antecedent degrees
   (fuzzy AND; degree of fulfilment).
3. Implication: clip the consequent fuzzy set at the firing strength.
4. Aggregation: pointwise ``max`` of the clipped consequents across rules.
5. Defuzzification: centroid (centre of area) of the aggregate set over a
   discrete 0-1 grid.

Returns ``(score, strengths)`` where ``strengths`` maps rule name to its
firing strength, for ``explain()``.
"""

from __future__ import annotations

from pokerbots.bots.fuzzy.membership import OUTPUT_SETS, fuzzify
from pokerbots.bots.fuzzy.rules import RULES, Rule

GRID: tuple[float, ...] = tuple(i / 100 for i in range(101))

# Each consequent's membership on GRID never changes, so evaluate it once at
# import: a decision is then ~len(rules) x 101 min/max operations.
_SHAPES: dict[str, tuple[float, ...]] = {
    term: tuple(mf.degree(x) for x in GRID) for term, mf in OUTPUT_SETS.items()
}


def infer(crisp: dict[str, float], rules: tuple[Rule, ...] = RULES) -> tuple[float, dict[str, float]]:
    """Run Mamdani inference. ``crisp`` maps variable -> value in its universe."""
    degrees = {var: fuzzify(var, x) for var, x in crisp.items()}
    strengths = {
        rule.name: min(degrees[var][term] for var, term in rule.antecedent.items())
        for rule in rules
    }

    aggregate = [0.0] * len(GRID)
    for rule in rules:
        level = strengths[rule.name]
        if level <= 0.0:
            continue
        for i, m in enumerate(_SHAPES[rule.consequent]):
            aggregate[i] = max(aggregate[i], min(level, m))

    total = sum(aggregate)
    if total <= 0.0:
        return 0.34, strengths  # check/call if no rule fires
    score = sum(x * m for x, m in zip(GRID, aggregate, strict=True)) / total
    # Keep floating-point noise from changing the action at a band boundary.
    return round(score, 12), strengths
