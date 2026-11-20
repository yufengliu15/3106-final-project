"""Is bot X better than Bot C, or did it just get better cards?

Uses the tournament harness, but plays every game twice with the same seed:
once with Bot C listed first and once with the other bot listed first. That
swaps their seats and hole cards while everything else stays the same, so
lucky cards mostly cancel out. Many seeds are played in parallel.

For each opponent it prints Bot C's average chip lead in big blinds (positive
means Bot C won), a 95% confidence interval, and whether the lead is
statistically significant (one-sided t-test, Holm-corrected when several
opponents or settings are tested).

    python -m pokerbots.bots.ml.eval2 --models models/bot_c --vs your.module:YourBot --om both
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


def _opp_class(name: str, use_opp: bool):
    """Any "module:Class" bot, e.g. pokerbots.bots.simple:RandomBot."""
    import importlib
    mod, cls = name.split(":")
    base = getattr(importlib.import_module(mod), cls)

    def __init__(self):
        try:
            base.__init__(self, use_opponent_model=use_opp)
        except TypeError:        # bot without the opponent-model switch
            base.__init__(self)

    return type(cls, (base,), {"__init__": __init__})


def _play(job):
    model_dir, vname, seed, rotations, use_opp = job
    from pokerbots.bots.simple import SCRIPTED_POOL
    from pokerbots.tournament import Tournament, TournamentConfig

    C = _c_class(model_dir, use_opp)
    V = _opp_class(vname, use_opp)
    out = []
    for order in ((C, V), (V, C)):
        cfg = TournamentConfig(main_bots=order, simple_bots=SCRIPTED_POOL, games=1, rotations=rotations,
                               mode="eval", base_seed=seed)
        (g,) = Tournament(cfg).run()
        st = {p.bot: p.final_stack for p in g.standings}
        out.append(st["BotC"] - st[V.__name__])
    return model_dir, vname, use_opp, seed, float(np.mean(out))


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

    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--vs", nargs="+", metavar="MODULE:CLASS", required=True,
                   help="opponent bots, e.g. pokerbots.bots.simple:RandomBot")
    p.add_argument("--seeds", type=int, default=24)
    p.add_argument("--rotations", type=int, default=50)
    p.add_argument("--om", choices=("on", "off", "both"), default="on")
    p.add_argument("--first-seed", type=int, default=700_000)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    oms = {"on": [True], "off": [False], "both": [True, False]}[a.om]
    jobs = [(m, v, a.first_seed + s, a.rotations, om) for m in a.models for v in a.vs for om in oms
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
    out = a.out or Path("models/results") / f"eval2_vs_{a.om}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, separators=(",", ":")) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
