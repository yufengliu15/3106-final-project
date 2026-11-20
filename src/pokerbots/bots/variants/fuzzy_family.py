"""Bot B family: fuzzy rule-based bots a teammate could plausibly have written.

B1  the reference Mamdani system (trapezoids, centroid), set edges / bands drawn
B2  zero-order Sugeno: each rule outputs a constant, weighted average
B3  held-out: Mamdani with mean-of-maxima defuzzification
B4  Gaussian membership functions and a different, larger rule base
B5  two outputs: aggression (act/fold/call/raise) and bet size (half / pot / all-in)
B6  held-out: membership widths depend on confidence in the opponent stats

All use the same inputs as Bot B: strength, edge, looseness, aggression,
position, SPR, facing a bet.
"""

from __future__ import annotations

import math

import numpy as np

from pokerbots.bots.variants.common import (
    ALL_IN, CALL, CHECK, FOLD, RAISE_HALF_POT, RAISE_POT, VariantBot,
)

UNIVERSE = np.linspace(0.0, 1.0, 101)


def trap(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoid membership: 0 below a, rises to 1 on [b, c], falls to 0 at d."""
    if x <= a or x >= d:
        return 1.0 if (b <= x <= c) else 0.0
    if x < b:
        return (x - a) / (b - a)
    if x <= c:
        return 1.0
    return (d - x) / (d - c)


# Reference (B1) Mamdani knowledge base: fuzzy sets per input, output sets for
# the 0-1 aggression score, and (antecedents ANDed, consequent) rules.
SETS = {
    "strength": {"weak": (-1, -1, 0.3, 0.5), "medium": (0.3, 0.5, 0.55, 0.72), "strong": (0.6, 0.78, 2, 2)},
    "edge": {"negative": (-2, -2, -0.08, 0.02), "small": (-0.05, 0.03, 0.08, 0.18), "large": (0.1, 0.25, 2, 2)},
    "loose": {"tight": (-1, -1, 0.2, 0.38), "loose": (0.25, 0.45, 2, 2)},
    "aggr": {"passive": (-1, -1, 1.0, 2.0), "aggressive": (1.4, 3.0, 99, 99)},
    "position": {"early": (-1, -1, 0.3, 0.6), "late": (0.4, 0.75, 2, 2)},
    "spr": {"short": (-1, -1, 1.0, 3.0), "deep": (2.0, 5.0, 999, 999)},
    "facing": {"no": (-1, -1, 0, 0.001), "yes": (0, 0.001, 2, 2)},
}
OUT = {
    "fold": (-1, -1, 0.05, 0.22),
    "passive": (0.12, 0.3, 0.4, 0.55),
    "raise": (0.45, 0.6, 0.7, 0.85),
    "allin": (0.78, 0.92, 2, 2),
}
RULES: list[tuple[tuple[tuple[str, str], ...], str]] = [
    ((("strength", "strong"),), "raise"),
    ((("strength", "strong"), ("spr", "short")), "allin"),
    ((("strength", "strong"), ("loose", "loose"), ("aggr", "passive")), "raise"),
    ((("strength", "strong"), ("aggr", "aggressive"), ("facing", "yes")), "allin"),
    ((("strength", "medium"), ("facing", "no")), "passive"),
    ((("strength", "medium"), ("facing", "no"), ("position", "late"), ("loose", "tight")), "raise"),
    ((("strength", "medium"), ("edge", "large")), "raise"),
    ((("strength", "medium"), ("edge", "small")), "passive"),
    ((("strength", "medium"), ("edge", "negative"), ("aggr", "passive")), "fold"),
    ((("strength", "medium"), ("edge", "negative"), ("aggr", "aggressive")), "passive"),
    ((("strength", "weak"), ("facing", "yes"), ("edge", "negative")), "fold"),
    ((("strength", "weak"), ("facing", "yes"), ("edge", "small")), "passive"),
    ((("strength", "weak"), ("facing", "no")), "passive"),
    ((("strength", "weak"), ("facing", "no"), ("position", "late"), ("loose", "tight"), ("aggr", "passive")), "raise"),
    ((("edge", "large"), ("spr", "short")), "allin"),
    ((("edge", "large"), ("facing", "yes")), "raise"),
    ((("edge", "negative"), ("facing", "yes")), "fold"),
    ((("edge", "small"), ("aggr", "aggressive"), ("facing", "yes")), "passive"),
]
BANDS = [(0.2, FOLD), (0.5, CALL), (0.75, RAISE_HALF_POT), (0.9, RAISE_POT), (1.01, ALL_IN)]
PREFERENCE = {  # nearest legal action if the band's action is not offered
    FOLD: (FOLD, CHECK),
    CALL: (CHECK, CALL, ALL_IN),
    RAISE_HALF_POT: (RAISE_HALF_POT, RAISE_POT, ALL_IN, CALL, CHECK),
    RAISE_POT: (RAISE_POT, RAISE_HALF_POT, ALL_IN, CALL, CHECK),
    ALL_IN: (ALL_IN, RAISE_POT, RAISE_HALF_POT, CALL, CHECK),
}
CONF_ROUNDS = 25    # rounds observed for full confidence in the stats


def gauss(x, m, s):
    return math.exp(-0.5 * ((x - m) / max(s, 1e-6)) ** 2)


def _jitter_sets(rng, sets: dict, rel: float) -> dict:
    """Shift each finite trapezoid edge by up to rel x the variable's span, keeping order."""
    out = {}
    for var, ss in sets.items():
        finite = [v for t in ss.values() for v in t if -1 < v < 50]
        span = (max(finite) - min(finite)) if finite else 1.0
        out[var] = {}
        for name, t in ss.items():
            shift = rng.uniform(-rel, rel) * span
            pts = [v + shift if -1 < v < 50 else v for v in t]
            for i in range(1, 4):
                pts[i] = max(pts[i], pts[i - 1])
            out[var][name] = tuple(pts)
    return out


def _jitter_bands(rng, bands, rel=0.05):
    """Jitter band edges; the fold edge is searched over a wider range because
    centroids of mixed fold/passive firings land around 0.2-0.3."""
    lims = [b + rng.uniform(-rel, rel) for b, _ in bands[:-1]]
    lims[0] = rng.uniform(0.18, 0.34)
    lims = sorted(min(0.98, max(0.05, b)) for b in lims)
    return [(l, a) for l, (_, a) in zip(lims, bands[:-1])] + [bands[-1]]


class FuzzyVariant(VariantBot):
    """Shared input computation (identical to Bot B) and band mapping."""

    family = "B"

    @classmethod
    def default_params(cls) -> dict:
        return dict(sets=SETS, out=OUT, rules=RULES, bands=BANDS,
                    conf_rounds=CONF_ROUNDS, preflop_scale=1.6)

    def inputs(self, obs, sit):
        rd = self.read(sit)
        if sit.street == 0 and sit.raises_preflop == 0:
            strength = 1.0 - self.percentile(obs) * self.p["preflop_scale"]
            eq = strength
        else:
            eq = self.equity_vs(obs, sit, rd)
            strength = eq
        x = {
            "strength": strength,
            "edge": eq - sit.pot_odds if sit.to_call > 0 else eq - 0.5,
            "loose": rd.vpip,
            "aggr": rd.aggr,
            "position": 1.0 - sit.seats_to_button / max(sit.n_players - 1, 1),
            "spr": sit.spr,
            "facing": 1.0 if sit.to_call > 0 else 0.0,
        }
        return x, rd

    def band_action(self, score: float, legal: dict) -> str:
        band = next(a for limit, a in self.p["bands"] if score < limit)
        for a in PREFERENCE[band]:
            if a in legal:
                return a
        return next(iter(legal))

    def mu(self, var, name, value, rd=None) -> float:
        return trap(value, *self.p["sets"][var][name])

    def strengths(self, x, rd):
        fired = []
        for ante, cons in self.p["rules"]:
            s = min(self.mu(v, n, x[v], rd) for v, n in ante)
            if s > 0:
                fired.append((s, ante, cons))
        return fired

    @staticmethod
    def describe(fired, k=3) -> str:
        top = sorted(fired, key=lambda t: -t[0])[:k]
        return "; ".join(f"{s:.2f} IF " + " AND ".join(f"{v} {n}" for v, n in a) + f" THEN {c}"
                         for s, a, c in top)


# ---------------------------------------------------------------- B1 / B3 / B6 (Mamdani)
class B1Mamdani(FuzzyVariant):
    approach = "B1"
    defuzz = "centroid"

    @classmethod
    def sample_params(cls, rng, extreme: bool = False) -> dict:
        d = cls.default_params()
        return dict(sets=_jitter_sets(rng, d["sets"], 0.05), out=d["out"], rules=d["rules"],
                    bands=_jitter_bands(rng, d["bands"]), conf_rounds=rng.choice((10, 25, 50)),
                    preflop_scale=rng.uniform(1.3, 2.0))

    def __init__(self, use_opponent_model=True, params=None):
        super().__init__(use_opponent_model, params)
        self._out = {k: np.array([trap(u, *v) for u in UNIVERSE]) for k, v in self.p["out"].items()}

    def decide(self, obs, sit, legal):
        x, rd = self.inputs(obs, sit)
        fired = self.strengths(x, rd)
        agg = np.zeros_like(UNIVERSE)
        for s, _a, cons in fired:
            agg = np.maximum(agg, np.minimum(self._out[cons], s))
        if agg.sum() == 0:
            score = 0.35
        elif self.defuzz == "mom":
            score = float(UNIVERSE[agg >= agg.max() - 1e-9].mean())
        else:
            score = float((agg * UNIVERSE).sum() / agg.sum())
        return self.band_action(score, legal), f"score {score:.2f}; {self.describe(fired)}"


class B3MeanOfMaxima(B1Mamdani):
    approach = "B3"
    defuzz = "mom"


class B6ConfidenceWidth(B1Mamdani):
    """Opponent-stat memberships widen (become vaguer) when few rounds were seen."""

    approach = "B6"
    OPP_VARS = ("loose", "aggr")

    @classmethod
    def sample_params(cls, rng, extreme: bool = False) -> dict:
        p = super().sample_params(rng, extreme)
        p["widen"] = rng.uniform(0.5, 2.0)
        return p

    def mu(self, var, name, value, rd=None):
        a, b, c, d = self.p["sets"][var][name]
        if var in self.OPP_VARS and rd is not None:
            w = (1.0 - rd.confidence) * self.p.get("widen", 1.0)
            span = 0.3 if var == "loose" else 1.5
            a, d = (a - w * span if a > -1 else a), (d + w * span if d < 50 else d)
        return trap(value, a, b, c, d)


# ---------------------------------------------------------------- B2 (Sugeno)
SUGENO_OUT = {"fold": 0.08, "passive": 0.35, "raise": 0.66, "allin": 0.95}


class B2Sugeno(FuzzyVariant):
    approach = "B2"

    @classmethod
    def default_params(cls) -> dict:
        d = super().default_params()
        d["consts"] = dict(SUGENO_OUT)
        return d

    @classmethod
    def sample_params(cls, rng, extreme: bool = False) -> dict:
        d = cls.default_params()
        consts = {k: min(1.0, max(0.0, v + rng.uniform(-0.06, 0.06))) for k, v in SUGENO_OUT.items()}
        return dict(sets=_jitter_sets(rng, d["sets"], 0.05), out=d["out"], rules=d["rules"],
                    bands=_jitter_bands(rng, d["bands"]), conf_rounds=rng.choice((10, 25, 50)),
                    preflop_scale=rng.uniform(1.3, 2.0), consts=consts)

    def decide(self, obs, sit, legal):
        x, rd = self.inputs(obs, sit)
        fired = self.strengths(x, rd)
        w = sum(s for s, _, _ in fired)
        score = sum(s * self.p["consts"][c] for s, _, c in fired) / w if w > 0 else 0.35
        return self.band_action(score, legal), f"sugeno {score:.2f}; {self.describe(fired)}"


# ---------------------------------------------------------------- B4 (Gaussian, new rule base)
G_SETS = {
    "strength": {"weak": (0.25, 0.14), "medium": (0.52, 0.1), "strong": (0.85, 0.09)},
    "edge": {"negative": (-0.2, 0.12), "small": (0.06, 0.07), "large": (0.38, 0.09)},
    "loose": {"tight": (0.15, 0.12), "loose": (0.55, 0.15)},
    "aggr": {"passive": (0.6, 0.7), "aggressive": (3.0, 1.2)},
    "position": {"early": (0.1, 0.25), "late": (0.9, 0.25)},
    "spr": {"short": (0.4, 0.7), "deep": (8.0, 4.0)},
    "facing": {"no": (0.0, 0.2), "yes": (1.0, 0.2)},
}
G_RULES = [
    ((("strength", "strong"),), "raise"),
    ((("strength", "strong"), ("spr", "short")), "allin"),
    ((("strength", "strong"), ("facing", "yes"), ("aggr", "aggressive")), "allin"),
    ((("strength", "strong"), ("loose", "loose")), "raise"),
    ((("strength", "medium"), ("facing", "no"), ("position", "late")), "raise"),
    ((("strength", "medium"), ("facing", "no"), ("position", "early")), "passive"),
    ((("strength", "medium"), ("facing", "yes"), ("edge", "large")), "raise"),
    ((("strength", "medium"), ("facing", "yes"), ("edge", "small")), "passive"),
    ((("strength", "medium"), ("facing", "yes"), ("edge", "negative")), "fold"),
    ((("strength", "medium"), ("aggr", "aggressive"), ("edge", "small")), "passive"),
    ((("strength", "medium"), ("spr", "short"), ("edge", "large")), "allin"),
    ((("strength", "weak"), ("facing", "yes")), "fold"),
    ((("strength", "weak"), ("facing", "yes"), ("edge", "large")), "passive"),
    ((("strength", "weak"), ("facing", "no")), "passive"),
    ((("strength", "weak"), ("facing", "no"), ("position", "late"), ("loose", "tight")), "raise"),
    ((("strength", "weak"), ("facing", "no"), ("position", "early")), "passive"),
    ((("edge", "large"), ("facing", "yes")), "raise"),
    ((("edge", "large"), ("spr", "short")), "allin"),
    ((("edge", "negative"), ("facing", "yes"), ("aggr", "passive")), "fold"),
    ((("edge", "negative"), ("facing", "yes"), ("aggr", "aggressive")), "passive"),
    ((("edge", "small"), ("loose", "loose"), ("facing", "no")), "raise"),
    ((("spr", "deep"), ("strength", "medium"), ("facing", "yes")), "passive"),
    ((("loose", "loose"), ("aggr", "passive"), ("strength", "strong")), "raise"),
    ((("loose", "tight"), ("aggr", "aggressive"), ("facing", "yes"), ("strength", "medium")), "fold"),
    ((("loose", "tight"), ("aggr", "passive"), ("facing", "no"), ("position", "late")), "raise"),
    ((("position", "late"), ("facing", "no"), ("edge", "small")), "raise"),
    ((("position", "early"), ("facing", "yes"), ("edge", "small")), "passive"),
    ((("spr", "short"), ("edge", "small"), ("facing", "yes")), "passive"),
]
G_OUT = {"fold": (0.05, 0.08), "passive": (0.35, 0.1), "raise": (0.65, 0.1), "allin": (0.95, 0.07)}


class B4Gaussian(FuzzyVariant):
    approach = "B4"

    @classmethod
    def default_params(cls) -> dict:
        d = super().default_params()
        d.update(sets=G_SETS, rules=G_RULES, out=G_OUT,
                 bands=[(0.3, FOLD), (0.55, CALL), (0.75, RAISE_HALF_POT), (0.9, RAISE_POT), (1.01, ALL_IN)])
        return d

    @classmethod
    def sample_params(cls, rng, extreme: bool = False) -> dict:
        d = cls.default_params()
        sets = {v: {n: (m + rng.uniform(-0.3, 0.3) * s, s * rng.uniform(0.85, 1.2))
                    for n, (m, s) in ss.items()} for v, ss in G_SETS.items()}
        return dict(sets=sets, rules=G_RULES, out=G_OUT, bands=_jitter_bands(rng, d["bands"]),
                    conf_rounds=rng.choice((10, 25, 50)), preflop_scale=rng.uniform(1.3, 2.0))

    def __init__(self, use_opponent_model=True, params=None):
        super().__init__(use_opponent_model, params)
        self._out = {k: np.exp(-0.5 * ((UNIVERSE - m) / s) ** 2) for k, (m, s) in self.p["out"].items()}

    def mu(self, var, name, value, rd=None):
        return gauss(value, *self.p["sets"][var][name])

    def strengths(self, x, rd):   # product t-norm, drop negligible firings
        fired = []
        for ante, cons in self.p["rules"]:
            s = 1.0
            for v, n in ante:
                s *= self.mu(v, n, x[v], rd)
            if s > 0.01:
                fired.append((s, ante, cons))
        return fired

    def decide(self, obs, sit, legal):
        x, rd = self.inputs(obs, sit)
        fired = self.strengths(x, rd)
        agg = np.zeros_like(UNIVERSE)
        for s, _a, cons in fired:
            agg = np.maximum(agg, self._out[cons] * s)      # product implication
        score = float((agg * UNIVERSE).sum() / agg.sum()) if agg.sum() > 0 else 0.35
        return self.band_action(score, legal), f"gauss {score:.2f}; {self.describe(fired)}"


# ---------------------------------------------------------------- B5 (two outputs)
SIZE_RULES = [
    ((("strength", "strong"), ("spr", "short")), 0.95),
    ((("strength", "strong"), ("spr", "deep")), 0.55),
    ((("strength", "strong"), ("loose", "loose")), 0.75),
    ((("strength", "medium"),), 0.3),
    ((("edge", "large"), ("spr", "short")), 0.9),
    ((("strength", "weak"),), 0.25),
]


class B5TwoOutput(FuzzyVariant):
    approach = "B5"

    @classmethod
    def default_params(cls) -> dict:
        d = super().default_params()
        d.update(act_consts=dict(SUGENO_OUT), size_cuts=(0.45, 0.85), raise_at=0.55, fold_at=0.2)
        return d

    @classmethod
    def sample_params(cls, rng, extreme: bool = False) -> dict:
        d = cls.default_params()
        lo = rng.uniform(0.35, 0.55)
        return dict(sets=_jitter_sets(rng, d["sets"], 0.05), out=d["out"], rules=d["rules"],
                    bands=d["bands"], conf_rounds=rng.choice((10, 25, 50)),
                    preflop_scale=rng.uniform(1.3, 2.0), act_consts=d["act_consts"],
                    size_cuts=(lo, rng.uniform(lo + 0.25, 0.95)), raise_at=rng.uniform(0.48, 0.62),
                    fold_at=rng.uniform(0.15, 0.27))

    def decide(self, obs, sit, legal):
        x, rd = self.inputs(obs, sit)
        fired = self.strengths(x, rd)
        w = sum(s for s, _, _ in fired)
        act = sum(s * self.p["act_consts"][c] for s, _, c in fired) / w if w > 0 else 0.35
        sw = ss = 0.0
        for ante, val in SIZE_RULES:
            s = min(self.mu(v, n, x[v], rd) for v, n in ante)
            sw += s
            ss += s * val
        size = ss / sw if sw > 0 else 0.4
        why = f"act {act:.2f}, size {size:.2f}; {self.describe(fired, 2)}"
        if act < self.p["fold_at"]:
            return (FOLD if FOLD in legal else CHECK), why
        if act < self.p["raise_at"]:
            return (CHECK if CHECK in legal else CALL), why
        lo, hi = self.p["size_cuts"]
        want = RAISE_HALF_POT if size < lo else (RAISE_POT if size < hi else ALL_IN)
        for a in PREFERENCE[want]:
            if a in legal:
                return a, why
        return next(iter(legal)), why
