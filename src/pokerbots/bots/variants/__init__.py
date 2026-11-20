"""Variant library of plausible Bot A / Bot B implementations (robust plan §2).

    spec = VariantSpec("A2", 3, params)          # approach, draw number, parameters
    cls = spec.make(use_opponent_model=True)     # zero-arg class named "A2_d3"

Approaches are split into TRAIN and HELD_OUT: held-out ones are never used for
labelling or model selection, only for the final evaluation.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from pokerbots.bots.variants.crisp_family import (
    A6_BASES, A1Chart, A2Threshold, A3ChenMade, A4TypeTable, A5DepthTree,
)
from pokerbots.bots.variants.fuzzy_family import (
    B1Mamdani, B2Sugeno, B3MeanOfMaxima, B4Gaussian, B5TwoOutput, B6ConfidenceWidth,
)

APPROACHES = {
    "A1": A1Chart, "A2": A2Threshold, "A3": A3ChenMade, "A4": A4TypeTable, "A5": A5DepthTree,
    "B1": B1Mamdani, "B2": B2Sugeno, "B3": B3MeanOfMaxima, "B4": B4Gaussian, "B5": B5TwoOutput,
    "B6": B6ConfidenceWidth,
}
TRAIN = ("A1", "A2", "A4", "A5", "B1", "B2", "B4", "B5")
HELD_OUT = ("A3", "A6", "B3", "B6")
LIBRARY_FILE = Path(__file__).resolve().parents[4] / "models" / "variant_library.json"


@dataclass(frozen=True)
class VariantSpec:
    approach: str            # A1..A6, B1..B6
    draw: int
    params: dict = field(default_factory=dict, hash=False, compare=False)
    base: str = ""           # A6 only: the approach whose code it runs

    @property
    def name(self) -> str:
        return f"{self.approach}_d{self.draw}"

    @property
    def impl(self):
        return APPROACHES[self.base or self.approach]

    def make(self, use_opponent_model: bool = True):
        impl, params = self.impl, self.params
        name = self.name

        def __init__(self, _impl=impl, _p=params, _om=use_opponent_model):
            _impl.__init__(self, use_opponent_model=_om, params=_p)

        return type(name, (impl,), {"__init__": __init__, "name": name, "__module__": __name__})

    def to_json(self) -> dict:
        return {"approach": self.approach, "draw": self.draw, "base": self.base,
                "params": _jsonable(self.params)}

    @staticmethod
    def from_json(d: dict) -> "VariantSpec":
        return VariantSpec(d["approach"], d["draw"], _unjson(d["params"]), d.get("base", ""))


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return {"__tuple__": [_jsonable(v) for v in x]} if isinstance(x, tuple) else [_jsonable(v) for v in x]
    return x


def _unjson(x):
    if isinstance(x, dict):
        if set(x) == {"__tuple__"}:
            return tuple(_unjson(v) for v in x["__tuple__"])
        return {k: _unjson(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_unjson(v) for v in x]
    return x


def sample_spec(approach: str, draw: int, seed: int = 0) -> VariantSpec:
    rng = random.Random(f"{approach}-{draw}-{seed}")
    if approach == "A6":
        base = rng.choice(A6_BASES)
        return VariantSpec("A6", draw, base.sample_params(rng, extreme=True), base.approach)
    return VariantSpec(approach, draw, APPROACHES[approach].sample_params(rng))


def default_spec(approach: str) -> VariantSpec:
    return VariantSpec(approach, 0, {})


def load_library(path: Path = LIBRARY_FILE) -> list[VariantSpec]:
    return [VariantSpec.from_json(d) for d in json.loads(Path(path).read_text())]


def save_library(specs: list[VariantSpec], path: Path = LIBRARY_FILE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([s.to_json() for s in specs], separators=(",", ":")) + "\n")
