"""Train Bot C v2 candidates.

    python -m pokerbots.bots.ml.train2 --data data/v2 --out models/cand_clf --kind clf
    python -m pokerbots.bots.ml.train2 --data data/v2 --out models/cand_ev  --kind ev

- Same game-level 70/15/15 split as train.py.
- Opponent-model dropout: on a random DROPOUT share of training rows, the
  opponent columns are replaced by their opponent-model-off copies (off__*),
  so the model is sensible with the model on and off.
- Ensemble of SEEDS members, each trained on a game-level bootstrap.
- kind "clf": weighted multiclass (v1 objective); kind "ev": one regressor per
  action predicting its rollout EV (argmax over legal actions at play time).
- Reports accuracy and regret with the opponent model on and off.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from pokerbots.bots.ml.bot_c import save_model
from pokerbots.bots.ml.features import feature_names, opponent_columns
from pokerbots.bots.ml.labeller import SHRINK
from pokerbots.bots.ml.train import WEIGHT_FLOOR, evaluate, load, split
from pokerbots.core import ACTIONS

DROPOUT = 0.3


def with_om_off(d: pd.DataFrame, cols, mask: np.ndarray | None = None) -> pd.DataFrame:
    d = d.copy()
    m = np.ones(len(d), bool) if mask is None else mask
    has = d["off__eq_range"].notna().to_numpy() if "off__eq_range" in d else np.zeros(len(d), bool)
    m = m & has          # v1 rows have no opponent-model-off copies: left as they are
    for c in opponent_columns(cols):
        if f"off__{c}" in d:
            d.loc[m, c] = d.loc[m, f"off__{c}"].to_numpy()
    return d


def scores(members, kind, X) -> np.ndarray:
    if kind == "ev":
        return np.mean([np.column_stack([m.predict(X) for m in mem]) for mem in members], axis=0)
    return np.mean([m.predict(X) for m in members], axis=0)


def pick(s: np.ndarray, X: pd.DataFrame) -> np.ndarray:
    legal = X[[f"legal_{a}" for a in ACTIONS]].to_numpy() > 0.5
    return np.where(legal, s, -np.inf).argmax(axis=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, nargs="+", default=[Path("data/v2")])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--kind", choices=("clf", "ev"), default="clf")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--dropout", type=float, default=DROPOUT)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--leaves", type=int, default=63)
    p.add_argument("--min-child", type=int, default=60)
    p.add_argument("--l2", type=float, default=3.0)
    p.add_argument("--rounds", type=int, default=3000)
    a = p.parse_args()

    df = pd.concat([load(d).assign(run=i) for i, d in enumerate(a.data)], ignore_index=True)
    df["game_key"] = df.groupby(["run", "game_id"]).ngroup()
    # v1 rows: opponent model was on; their stats are unshrunk (small mismatch, accepted)
    df["om"] = df["om"].fillna(1.0) if "om" in df else 1.0
    cols = feature_names(shrink=SHRINK)
    tr, va, te = split(df)
    print(f"rows: train {len(tr)}, val {len(va)}, test {len(te)}", flush=True)
    rng = np.random.default_rng(0)
    va_mixed = with_om_off(va, cols, rng.random(len(va)) < a.dropout)

    base = dict(learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=a.min_child,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=a.l2, verbose=-1)
    members, files, iters = [], [], []
    games = tr["game_key"].unique()
    for s in range(a.seeds):
        r = np.random.default_rng(100 + s)
        boot = r.choice(games, size=len(games), replace=True)
        counts = pd.Series(boot).value_counts()
        d = tr[tr.game_key.isin(counts.index)]
        d = d.loc[d.index.repeat(d.game_key.map(counts).to_numpy())]
        d = with_om_off(d, cols, r.random(len(d)) < a.dropout)
        if a.kind == "clf":
            prm = dict(base, objective="multiclass", num_class=len(ACTIONS), seed=s)
            dtr = lgb.Dataset(d[cols], d["label"].map(ACTIONS.index), weight=d["weight"].clip(lower=0) + WEIGHT_FLOOR)
            dva = lgb.Dataset(va_mixed[cols], va_mixed["label"].map(ACTIONS.index),
                              weight=va_mixed["weight"].clip(lower=0) + WEIGHT_FLOOR, reference=dtr)
            m = lgb.train(prm, dtr, a.rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
            name = f"bot_c_{s}.txt.xz"
            a.out.mkdir(parents=True, exist_ok=True)
            save_model(m, a.out / name, m.best_iteration)
            members.append(m)
            files.append(name)
            iters.append(m.best_iteration)
        else:
            prm = dict(base, objective="huber", alpha=5.0, seed=s)
            mem, names = [], []
            for act in ACTIONS:
                ok, okv = d[f"ev_{act}"].notna(), va_mixed[f"ev_{act}"].notna()
                dtr = lgb.Dataset(d.loc[ok, cols], d.loc[ok, f"ev_{act}"])
                dva = lgb.Dataset(va_mixed.loc[okv, cols], va_mixed.loc[okv, f"ev_{act}"], reference=dtr)
                m = lgb.train(prm, dtr, a.rounds, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(100, verbose=False)])
                a.out.mkdir(parents=True, exist_ok=True)
                name = f"bot_c_{s}_{act}.txt.xz"
                save_model(m, a.out / name, m.best_iteration)
                mem.append(m)
                names.append(name)
                iters.append(m.best_iteration)
            members.append(mem)
            files.append(names)
        print(f"member {s} done", flush=True)

    report = {"kind": a.kind, "args": {k: str(v) for k, v in vars(a).items()}, "iterations": iters}
    for name, d in (("validation", va), ("test", te)):
        for om in ("on", "off"):
            dd = d if om == "on" else with_om_off(d, cols)
            report[f"{name}_om_{om}"] = evaluate(dd, pick(scores(members, a.kind, dd[cols]), dd[cols]))
    passive = np.where(te["legal_check"] > 0.5, ACTIONS.index("check"), ACTIONS.index("fold"))
    report["test_passive_baseline_regret_bb"] = evaluate(te, passive)["mean_regret_bb"]

    (a.out / "bot_c_features.json").write_text(json.dumps(cols))
    (a.out / "bot_c_meta.json").write_text(json.dumps({"kind": a.kind, "members": files, "shrink": SHRINK}))
    (a.out / "bot_c_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "iterations"}, indent=1))


if __name__ == "__main__":
    main()
