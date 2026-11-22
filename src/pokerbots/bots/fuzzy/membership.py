"""Fuzzy membership functions for Bot B.

All sets are triangular or trapezoidal on a named universe.

- Hand strength is showdown equity adjusted for field size. Unadjusted
  showdown equity identifies hands strong enough to commit a stack.
- pot odds 0-1 is ``to_call / (pot + to_call)`` (see ``core.situation``).
  A half-pot bet lays ~0.25, a pot bet ~0.33; above ~0.4 is expensive.
- looseness 0-1 is the tracker's mean VPIP. Centred on ``PRIOR.vpip = 0.35``.
- aggression 0-10 is the tracker's raises/calls ratio, capped at 10 exactly
  like Bot C's features. Centred on ``PRIOR.aggression = 1.5``.
- sample 0-50 is ``min(rounds)`` over opponents in hand, capped at 50 to
  match ``OpponentTracker(window=50)`` and Bot C's ``agg_rounds`` cap.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Triangular:
    """Triangle with feet ``a < b`` and peak ``c`` (``a < c < b``)."""

    a: float
    c: float
    b: float

    def degree(self, x: float) -> float:
        if x <= self.a or x >= self.b:
            return 0.0
        if x < self.c:
            return (x - self.a) / (self.c - self.a)
        return (self.b - x) / (self.b - self.c)


@dataclass(frozen=True)
class Trapezoidal:
    """0 below ``a``, ramp to 1 at ``b``, flat to ``c``, down to 0 at ``d``.

    ``a == b`` makes a left shoulder, ``c == d`` a right shoulder.
    """

    a: float
    b: float
    c: float
    d: float

    def degree(self, x: float) -> float:
        if x <= self.a:
            return 1.0 if self.a == self.b else 0.0
        if x >= self.d:
            return 1.0 if self.c == self.d else 0.0
        if x <= self.b:
            return (x - self.a) / (self.b - self.a)
        if x <= self.c:
            return 1.0
        return (self.d - x) / (self.d - self.c)


Membership = Triangular | Trapezoidal

# Input universes: variable -> term -> fuzzy set.
INPUT_SETS: dict[str, dict[str, Membership]] = {
    "hand": {
        "thin": Trapezoidal(0.0, 0.0, 0.705, 0.785),
        "value": Trapezoidal(0.715, 0.785, 0.845, 0.905),
        "premium": Trapezoidal(0.835, 0.905, 1.0, 1.0),
    },
    "equity": {
        "ordinary": Trapezoidal(0.0, 0.0, 0.95, 0.99),
        "dominant": Trapezoidal(0.95, 0.99, 1.0, 1.0),
        "call": Trapezoidal(0.0, 0.0, 0.67, 0.74),
        "reraise": Trapezoidal(0.68, 0.75, 1.0, 1.0),
    },
    "pot_odds": {
        "cheap": Trapezoidal(0.0, 0.0, 0.10, 0.25),
        "moderate": Triangular(0.10, 0.25, 0.40),
        "expensive": Trapezoidal(0.30, 0.45, 1.0, 1.0),
    },
    "looseness": {
        "tight": Trapezoidal(0.0, 0.0, 0.20, 0.35),
        "neutral": Triangular(0.20, 0.35, 0.50),
        "loose": Trapezoidal(0.35, 0.50, 1.0, 1.0),
    },
    "aggression": {
        "passive": Trapezoidal(0.0, 0.0, 0.80, 1.50),
        "neutral": Triangular(0.80, 1.50, 2.50),
        "aggressive": Trapezoidal(1.50, 2.50, 10.0, 10.0),
    },
    "sample": {
        "large": Trapezoidal(5.0, 15.0, 50.0, 50.0),
    },
    "opening": {
        "premium": Trapezoidal(0.0, 0.0, 0.64, 0.77),
        "raise": Trapezoidal(0.64, 0.77, 0.92, 1.04),
        "defend": Trapezoidal(0.92, 1.04, 1.12, 1.24),
        "outside": Trapezoidal(1.12, 1.24, 10.0, 10.0),
    },
    "edge": {
        "poor": Trapezoidal(-1.0, -1.0, -0.025, 0.025),
        "sound": Trapezoidal(-0.025, 0.025, 1.0, 1.0),
    },
    "spr": {
        "short": Trapezoidal(0.0, 0.0, 1.15, 1.85),
        "deep": Trapezoidal(1.15, 1.85, 100.0, 100.0),
    },
    "fold_equity": {
        "low": Trapezoidal(0.0, 0.0, 0.62, 0.80),
        "high": Trapezoidal(0.62, 0.80, 1.0, 1.0),
    },
}

# Each triangle is symmetric around a score inside its action band. Its
# centroid stays in that band when the term fires alone, even after clipping.
OUTPUT_SETS: dict[str, Triangular] = {
    "very-passive": Triangular(0.00, 0.10, 0.20),
    "passive": Triangular(0.18, 0.34, 0.50),
    "moderate": Triangular(0.46, 0.62, 0.78),
    "aggressive": Triangular(0.72, 0.82, 0.92),
    "very-aggressive": Triangular(0.90, 0.95, 1.00),
}


def fuzzify(variable: str, x: float) -> dict[str, float]:
    """Membership degree of crisp ``x`` in every term of ``variable``."""
    return {term: mf.degree(x) for term, mf in INPUT_SETS[variable].items()}
