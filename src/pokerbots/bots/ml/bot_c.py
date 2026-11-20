"""Bot C: supervised machine-learning agent (#3).

An ensemble of LightGBM models, trained by train.py on #18's rollout labels,
scores the six actions from the shared feature vector; Bot C plays the
highest-scoring legal action. The shipped model (models/bot_c, kind "ev") has
one regressor per action predicting its rollout EV in bb; kind "clf" is a
multiclass classifier per member. No reinforcement learning: the
model is fixed at play time and only the shared opponent tracker changes.

explain(): exact TreeSHAP attributions (LightGBM pred_contrib) for the
chosen action -- the features that pushed the model toward it the most.
"""

from __future__ import annotations

import json
import lzma
from pathlib import Path

import lightgbm as lgb
import numpy as np

from pokerbots.bots.ml.features import features
from pokerbots.core import ACTIONS, Bot, Observation, RoundLog
from pokerbots.opponents import OpponentTracker

MODEL_DIR = Path(__file__).resolve().parents[4] / "models" / "bot_c"


class MLBot(Bot):
    name = "BotC"
    _cache: dict = {}

    def __init__(self, use_opponent_model: bool = True, model_dir: Path | None = None) -> None:
        self.use_opponent_model = use_opponent_model
        self.tracker = OpponentTracker()
        self.player_id = 0
        d = Path(model_dir or MODEL_DIR)
        key = str(d)
        if key not in MLBot._cache:  # load once per process, shared by every seat
            MLBot._cache[key] = _load(d)
        self.kind, self.members, self.cols, self.shrink = MLBot._cache[key]
        self._last: tuple | None = None

    def new_game(self, game_id: int, player_id: int) -> None:
        self.player_id = player_id
        self.tracker.reset()

    def observe_result(self, log: RoundLog) -> None:
        self.tracker.update(log)

    def act(self, obs: Observation) -> str:
        f = features(obs, self.tracker, self.use_opponent_model, shrink=self.shrink)
        x = np.array([[f[c] for c in self.cols]], dtype=float)
        proba = self.scores(x)
        legal = {a for a, _ in obs.legal_actions}
        scores = [(proba[i] if a in legal else -np.inf) for i, a in enumerate(ACTIONS)]
        best = int(np.argmax(scores))
        self._last = (x, best, proba, f)
        return ACTIONS[best]

    def attributions(self, top: int = 5) -> list[tuple[str, float, float]]:
        """[(feature, value, SHAP contribution toward the chosen action)]."""
        if self._last is None:
            return []
        x, best, _, f = self._last
        n = len(self.cols)
        if self.kind == "ev":    # one regressor per action: its own SHAP values
            c = np.mean([m[best].predict(x, pred_contrib=True)[0][:n] for m in self.members], axis=0)
        else:                    # multiclass: per-class slice of width n + 1 (features + bias)
            c = np.mean([m.predict(x, pred_contrib=True)[0][best * (n + 1): best * (n + 1) + n]
                         for m in self.members], axis=0)
        order = np.argsort(-np.abs(c))[:top]
        return [(self.cols[i], f[self.cols[i]], float(c[i])) for i in order]

    def explain(self) -> str:
        if self._last is None:
            return "no decision yet"
        _, best, proba, _ = self._last
        parts = [f"{n}={v:.2f} ({'+' if s >= 0 else ''}{s:.2f})" for n, v, s in self.attributions()]
        what = f"EV={proba[best]:+.2f} bb" if self.kind == "ev" else f"p={proba[best]:.2f}"
        return f"{ACTIONS[best]} ({what}); top SHAP: " + ", ".join(parts)

    def scores(self, x: np.ndarray) -> np.ndarray:
        """Ensemble-averaged class probabilities ("clf") or action EVs ("ev")."""
        if self.kind == "ev":
            return np.mean([[m[i].predict(x)[0] for i in range(len(ACTIONS))] for m in self.members], axis=0)
        return np.mean([m.predict(x)[0] for m in self.members], axis=0)


# Model files are stored as xz-compressed LightGBM text with the fields that
# neither prediction nor TreeSHAP reads removed (split gains, hessian sums, the
# tree-size index) and leaf values rounded to 7 significant digits: about 5x
# smaller, predictions change by < 1e-6.
_DROP = ("split_gain=", "internal_weight=", "leaf_weight=", "tree_sizes=")


def slim_model(text: str) -> str:
    out = []
    for line in text.splitlines():
        if line.startswith(_DROP):
            continue
        if line.startswith(("leaf_value=", "internal_value=")):
            k, v = line.split("=", 1)
            line = k + "=" + " ".join(f"{float(x):.7g}" for x in v.split())
        out.append(line)
    return "\n".join(out) + "\n"


def save_model(booster: lgb.Booster, path: Path, num_iteration: int | None = None) -> None:
    """Write `path` (name ending .txt.xz) in the compact format above."""
    path.write_bytes(lzma.compress(slim_model(booster.model_to_string(num_iteration=num_iteration)).encode()))


def _booster(path: Path) -> lgb.Booster:
    if path.suffix == ".xz":
        return lgb.Booster(model_str=lzma.decompress(path.read_bytes()).decode())
    return lgb.Booster(model_file=str(path))


def _load(d: Path):
    """bot_c_features.json plus bot_c_meta.json (kind, member files, shrink)."""
    cols = json.loads((d / "bot_c_features.json").read_text())
    meta = json.loads((d / "bot_c_meta.json").read_text())
    if meta["kind"] == "ev":
        members = [[_booster(d / f) for f in files] for files in meta["members"]]
    else:
        members = [_booster(d / f) for f in meta["members"]]
    return meta["kind"], members, cols, float(meta.get("shrink", 0.0))
