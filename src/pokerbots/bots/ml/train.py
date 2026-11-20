"""Train Bot C's classifier on the #18 rollout-labelled dataset.

    python -m pokerbots.bots.ml.train --data data/botc --out models/checkpoints/botc_v1

- Train / validation / test split by game_id (70/15/15), never by round.
- LightGBM multiclass over the six actions; sample weight = chip gap between
  the best and second-best action (+ a small floor), so near-ties count less.
- At prediction, actions not legal in the row are masked out.
- Reports accuracy and mean regret (bb lost vs the best action) per street.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from pokerbots.bots.ml.features import feature_names
from pokerbots.core import ACTIONS

WEIGHT_FLOOR = 0.05


def load(data_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(data_dir / "**" / "*.parquet"), recursive=True))
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    # game_id is unique per worker file only within a run directory
    df["game_key"] = df.groupby(["game_id"]).ngroup() if "run" not in df else \
        df.groupby(["run", "game_id"]).ngroup()
    return df


def split(df: pd.DataFrame, seed: int = 0):
    games = df["game_key"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    n = len(games)
    tr, va = set(games[: int(0.7 * n)]), set(games[int(0.7 * n): int(0.85 * n)])
    m = df["game_key"]
    return df[m.isin(tr)], df[m.isin(va)], df[~m.isin(tr | va)]


def masked_predict(model, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict(X)
    legal = X[[f"legal_{a}" for a in ACTIONS]].to_numpy() > 0.5
    return np.where(legal, proba, -1.0).argmax(axis=1)


def evaluate(df: pd.DataFrame, pred: np.ndarray) -> dict:
    ev = df[[f"ev_{a}" for a in ACTIONS]].to_numpy()
    best = np.nanmax(ev, axis=1)
    chosen = ev[np.arange(len(df)), pred]
    regret = best - chosen
    y = df["label"].map(ACTIONS.index).to_numpy()
    out = {"n": int(len(df)), "accuracy": float((pred == y).mean()), "mean_regret_bb": float(np.nanmean(regret))}
    for s in range(4):
        m = (df["street"] == s).to_numpy()
        if m.any():
            out[f"street{s}"] = {"n": int(m.sum()), "accuracy": float((pred[m] == y[m]).mean()),
                                 "mean_regret_bb": float(np.nanmean(regret[m]))}
    return out


def baseline_regret(df: pd.DataFrame, action_col: str | None = None) -> float:
    """Regret of always picking the label of a reference policy, for context."""
    ev = df[[f"ev_{a}" for a in ACTIONS]].to_numpy()
    return float(np.nanmean(np.nanmax(ev, axis=1) - ev[np.arange(len(df)), df[action_col].to_numpy()]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/botc"))
    p.add_argument("--out", type=Path, default=Path("models"))
    p.add_argument("--rounds", type=int, default=2000)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--leaves", type=int, default=63)
    p.add_argument("--min-child", type=int, default=40)
    a = p.parse_args()

    df = load(a.data)
    cols = feature_names()
    tr, va, te = split(df)
    print(f"rows: train {len(tr)}, val {len(va)}, test {len(te)}; games {df.game_key.nunique()}")

    def ds(d, ref=None):
        return lgb.Dataset(d[cols], label=d["label"].map(ACTIONS.index),
                           weight=d["weight"].clip(lower=0) + WEIGHT_FLOOR, reference=ref)

    params = dict(objective="multiclass", num_class=len(ACTIONS), learning_rate=a.lr,
                  num_leaves=a.leaves, min_data_in_leaf=a.min_child, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0)
    dtr = ds(tr)
    model = lgb.train(params, dtr, a.rounds, valid_sets=[ds(va, dtr)],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])

    report = {"best_iteration": model.best_iteration}
    for name, d in (("validation", va), ("test", te)):
        report[name] = evaluate(d, masked_predict(model, d[cols]))
    # reference: what Bot A's rules would have picked is not stored; compare
    # with always taking the cheapest passive action (check, else fold)
    passive = np.where(te["legal_check"] > 0.5, ACTIONS.index("check"), ACTIONS.index("fold"))
    report["test_passive_baseline_regret_bb"] = evaluate(te, passive)["mean_regret_bb"]
    imp = sorted(zip(cols, model.feature_importance("gain")), key=lambda t: -t[1])[:15]
    report["top_features_gain"] = [(c, round(float(g), 1)) for c, g in imp]

    a.out.mkdir(parents=True, exist_ok=True)
    from pokerbots.bots.ml.bot_c import save_model
    save_model(model, a.out / "bot_c.txt.xz", model.best_iteration)
    (a.out / "bot_c_features.json").write_text(json.dumps(cols))
    (a.out / "bot_c_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
