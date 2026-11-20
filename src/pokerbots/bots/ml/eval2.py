"""Duplicate head-to-head evaluation of Bot C against the variant library
(docs/bot_c_robust_plan.md §5-6).

For every (variant v, deck seed) two games are played with the same seed:
(C, v) and (v, C) as the main bots. The tournament draws scripted fillers,
player IDs and card seeds from base_seed alone, so swapping the order swaps
C's and v's seats and cards: luck largely cancels in

    d = [(C - v) in game 1 + (C - v) in game 2] / 2     (final stacks, bb)

Per variant: mean d, 95% bootstrap CI, one-sided t-test p (H1: d > 0). Holm
correction across all (variant, opponent-model condition) tests. R(C) =
min over variants of the CI lower bound (the robust selection criterion).

    python -m pokerbots.bots.ml.eval2 --pool train   --models models/cand_clf models/cand_ev --seeds 24
    python -m pokerbots.bots.ml.eval2 --pool heldout --models models/final --seeds 96 --om both
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def _c_class(model_dir: str, use_opp: bool):
    from pokerbots.bots.ml.bot_c import MLBot

    def __init__(self):
        MLBot.__init__(self, use_opponent_model=use_opp, model_dir=Path(model_dir))

    return type("BotC", (MLBot,), {"__init__": __init__})


def _ref_class(name: str, use_opp: bool):
    """BotA_ref / BotB_ref (the library's A1 / B1 defaults, stand-ins for Bots
    A and B) or any "module:Class" bot, e.g. pokerbots.bots.crisp:CrispBot."""
    if ":" in name:
        import importlib
        mod, cls = name.split(":")
        base = getattr(importlib.import_module(mod), cls)

        def __init__(self):
            try:
                base.__init__(self, use_opponent_model=use_opp)
            except TypeError:        # bot without the opponent-model switch
                base.__init__(self)

        return type(cls, (base,), {"__init__": __init__})
    from pokerbots.bots.variants import APPROACHES
    base = APPROACHES[{"BotA_ref": "A1", "BotB_ref": "B1"}[name]]

    def __init__(self):
        base.__init__(self, use_opponent_model=use_opp, params={})

    return type(name, (base,), {"__init__": __init__})


def _play(job):
    model_dir, vjson, seed, rotations, use_opp = job
    from pokerbots.bots.simple import SCRIPTED_POOL
    from pokerbots.bots.variants import VariantSpec
    from pokerbots.tournament import Tournament, TournamentConfig

    C = _c_class(model_dir, use_opp)
    V = _ref_class(vjson, use_opp) if isinstance(vjson, str) else VariantSpec.from_json(vjson).make(use_opp)
    out = []
    for order in ((C, V), (V, C)):
        cfg = TournamentConfig(main_bots=order, simple_bots=SCRIPTED_POOL, games=1, rotations=rotations,
                               mode="eval", base_seed=seed)
        (g,) = Tournament(cfg).run()
        st = {p.bot: p.final_stack for p in g.standings}
        out.append(st["BotC"] - st[V.__name__])
    return model_dir, (vjson if isinstance(vjson, str) else VariantSpec.from_json(vjson).name), use_opp, \
        seed, float(np.mean(out))


def bootstrap_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(pvals: dict, alpha=0.05) -> dict:
    order = sorted(pvals, key=pvals.get)
    m, out, alive = len(order), {}, True
    for i, k in enumerate(order):
        alive = alive and pvals[k] <= alpha / (m - i)
        out[k] = alive
    return out


def main() -> None:
    from scipy import stats

    from pokerbots.bots.variants import HELD_OUT, TRAIN, load_library

    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--pool", choices=("train", "heldout", "refs"), default="train")
    p.add_argument("--vs", nargs="+", metavar="MODULE:CLASS",
                   help="play these bots instead of a pool, e.g. pokerbots.bots.crisp:CrispBot")
    p.add_argument("--seeds", type=int, default=24)
    p.add_argument("--rotations", type=int, default=50)
    p.add_argument("--om", choices=("on", "off", "both"), default="on")
    p.add_argument("--first-seed", type=int, default=700_000)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    lib = load_library()
    if a.vs:
        opps = list(a.vs)
    elif a.pool == "refs":
        opps = ["BotA_ref", "BotB_ref"]
    else:
        keep = TRAIN if a.pool == "train" else HELD_OUT
        opps = [s.to_json() for s in lib if s.approach in keep]
    oms = {"on": [True], "off": [False], "both": [True, False]}[a.om]
    # seeds disjoint between pools so model selection never sees held-out tables
    base = a.first_seed + (0 if a.pool != "heldout" else 5_000_000)
    jobs = [(m, v, base + s, a.rotations, om) for m in a.models for v in opps for om in oms
            for s in range(a.seeds)]
    print(f"{len(jobs)} duplicate pairs", flush=True)
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        res = pool.map(_play, jobs, chunksize=1)

    d = defaultdict(list)
    for m, v, om, _s, x in res:
        d[(m, v, om)].append(x)
    report = {}
    for m in a.models:
        rows, pv = {}, {}
        for (mm, v, om), xs in d.items():
            if mm != m:
                continue
            x = np.array(xs)
            lo, hi = bootstrap_ci(x)
            pval = float(stats.ttest_1samp(x, 0.0, alternative="greater").pvalue) if x.std() > 0 else \
                (0.0 if x.mean() > 0 else 1.0)
            key = f"{v}|om_{'on' if om else 'off'}"
            rows[key] = dict(mean=float(x.mean()), lo=lo, hi=hi, p=pval, n=len(x))
            pv[key] = pval
        sig = holm(pv)
        for k in rows:
            rows[k]["holm_win"] = sig[k]
        r_c = min(r["lo"] for r in rows.values())
        p10 = float(np.percentile([r["mean"] for r in rows.values()], 10))
        report[m] = dict(R=r_c, p10_mean=p10, all_holm_wins=all(sig.values()),
                         wins=sum(sig.values()), tests=len(sig), rows=rows)
        print(f"\n{m}: R(C) = {r_c:+.1f} bb   p10 mean {p10:+.1f}   Holm wins {sum(sig.values())}/{len(sig)}")
        for k, r in sorted(rows.items(), key=lambda t: t[1]["lo"])[:8]:
            print(f"   worst {k:<22} {r['mean']:+7.1f}  CI [{r['lo']:+6.1f}, {r['hi']:+6.1f}]  "
                  f"p={r['p']:.1e} {'WIN' if r['holm_win'] else '---'}")
    out = a.out or Path("models/results") / f"eval2_{a.pool}_{a.om}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, separators=(",", ":")) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
