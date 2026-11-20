# Bot C: evaluation results

All numbers come from duplicate head-to-head games (`eval2`) of 50 rotations,
in bb, with 95% bootstrap CIs. A "win" means a one-sided t-test that is still
significant after Holm correction across all tests. R(C) is the lowest CI
lower bound across every opponent and condition, which is the robust
selection criterion from `bot_c_robust_plan.md`. Raw JSON is in
`models/results/`.

The opponent variants and reference stand-ins used here were a development
tool and are not part of this repository. To reproduce comparisons, run
`eval2 --vs` against the bots you want to test.

## Model selection (training variants, opponent model on, 24 seeds)

| Candidate | R(C) | Holm wins | p10 mean |
|---|---|---|---|
| **cand_ev (shipped)** | **+12.3** | **42/42** | **+82.2** |
| botc_v1 (checkpoint) | −75.7 | 0/42 | −15.0 |
| cand_clf_full | −85.2 | 0/42 | −33.0 |
| cand_clf | −86.5 | 0/42 | −46.3 |

- **cand_ev** is the per-action EV regressor ensemble, and it is the only
  candidate that beats every training variant.
- **Worst match-up:** A5_d10, at +44.3 [+12.3, +78.3].
- **Test-set regret:** 2.39 bb with the opponent model on and 2.41 bb with it
  off. The passive baseline is 4.85 bb.

## Held-out variants (never used for labels or selection, 48 seeds, opponent model on and off)

| Measure | Result |
|---|---|
| R(C) | +41.4 bb |
| Holm wins | 36/36 |
| p10 mean | +65.7 bb |

The worst match-up is A6_d2 with the opponent model off: +68.2 [+41.4, +95.6].
Every one of the eight weakest match-ups has the opponent model off.

## Reference stand-ins (A1/B1 defaults, 48 seeds)

| Opponent | Opponent model on | Opponent model off |
|---|---|---|
| BotA_ref | +97.3 [+63.4, +130.8] | +72.4 [+45.5, +99.6] |
| BotB_ref | +141.6 [+116.9, +166.8] | +122.2 [+94.6, +147.8] |

The shipped model wins all 4 tests, with R(C) = +45.5. The v1 checkpoint wins
3 of 4, with R(C) = −25.2. Its result against BotA_ref with the opponent model
on is not significant.

## Deviation from the plan

The held-out and reference runs used **48 seeds instead of the planned 96**
because of compute time. That makes the CIs about \(\sqrt{2} \approx 1.4\times\)
wider than planned. Every lower bound is still well above zero.

## Model storage

Models are stored as xz-compressed LightGBM text. Fields that prediction and
TreeSHAP never read are dropped, and leaf values are rounded to 7 significant
digits. That makes them about 5× smaller, and predictions drift by less than
\(4 \times 10^{-6}\).
