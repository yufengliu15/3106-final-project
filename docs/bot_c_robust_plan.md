# Bot C v2: a plan to beat any reasonable Bot A / Bot B

Goal: Bot C should beat **every** plausible implementation of Bot A (crisp rules) and Bot B (fuzzy logic), and the result should be statistically reliable. The size of the margin is secondary. All of the proposal rules from `bot_c_plan.md` §0 still hold:

- same inputs and the same opponent tracker for every bot;
- at most 200 ms per decision;
- equal tuning budget for every bot;
- no reinforcement learning;
- Bot C is a supervised classifier with SHAP explanations;
- labels come from rollouts, and only self-generated data is used.

Baseline (checkpoint `botc-v1`, files in `models/checkpoints/botc_v1/`):

| Opponent model | C − A | C − B | Comment |
|---|---|---|---|
| on | +24.6 bb, CI [−0.4, +49.8] | +93.0 bb | the lead over A is **not** significant |
| off | +86.8 bb | +142.0 bb | C is *better* without the opponent model, which is a red flag |

## 1. Why v1 is fragile

1. **Single-opponent overfit.** The labels assume a Bot A continuation, and every table contains exactly one A and one B. C has learned to exploit *these* implementations. A teammate's A or B, with different thresholds or rules, is an unseen opponent.
2. **Distribution shift in the opponent features.** Every training row had the opponent model on. Opponent-model-off rows (PRIOR stats) were never seen. The fact that C does better with the model off suggests the opponent features hurt more than they help. Likely causes are noisy stats early in a game and over-trust in the stats.
3. **Biased labels.** The continuation policy is Bot A everywhere, for both us and the opponents. An action's value is therefore "the value if everyone afterwards plays like A". That estimate is biased against opponents who play unlike A.
4. **Pre-flop is weak** (52% accuracy, the highest regret), and 34% of the test rows are pre-flop.
5. **The evaluation is noisy.** 128 games gave a ±25 bb CI. Card luck is not paired between the bots.

## 2. Opponent families: many Bot A and Bot B implementations

Build a **library of variants** that a reasonable teammate could plausibly have written. Every variant obeys the proposal: the same tracker, the same inputs, and less than 200 ms. Each family covers different approaches, and each approach also gets random parameters.

**Bot A family (crisp rules):**

| ID | Approach |
|---|---|
| A1 | current `CrispBot`: pre-flop chart plus equity vs range, pot odds, bluffs |
| A2 | pure equity thresholds, with no chart (equity bands map to actions) |
| A3 | a Sklansky/Chen-style hand score pre-flop, made-hand categories post-flop (no Monte Carlo) |
| A4 | an opponent-type table: a separate rule set for each of tight/loose × passive/aggressive |
| A5 | decision-tree style nested if/else over SPR, position and to_call (stack-depth driven) |
| A6 | a "nit" or "maniac" extreme of A1–A5 (parameters at the edges of their sane ranges) |

**Bot B family (fuzzy):**

| ID | Approach |
|---|---|
| B1 | current Mamdani with centroid defuzzification |
| B2 | Sugeno (zero order), with a weighted-average output |
| B3 | Mamdani with mean-of-maxima defuzzification (more decisive, less mushy) |
| B4 | Gaussian membership functions, with a different rule base (about 30 rules) |
| B5 | fuzzy aggression output plus a fuzzy bet size, mapped to the nearest legal action |
| B6 | type-2-ish: membership widths depend on the tracker's confidence |

For each approach, sample **N = 8 parameter draws** from sane ranges: tightness, bluff frequency, how much it trusts the opponent model, and how membership functions are placed. Each draw is tuned with the **same tuning budget** (decision D2: equal self-play hands, using hill-climbing or CMA-ES against the scripted pool, *not* against C). This produces about 96 opponents.

**Hold-out discipline (this is what makes the result rigorous):** split by *approach*, not by draw.

- **Train approaches:** A1, A2, A4, A5, B1, B2, B4, B5.
- **Held-out approaches:** A3, A6, B3, B6. C never sees them during labelling or model selection.

The final claim is only valid on the held-out approaches. That tests "regardless of implementation" rather than "against what we trained on".

Validity check for each variant before it enters the library:

- it beats the scripted pool;
- it makes no illegal actions;
- p99 decision time is under 200 ms;
- it is not a duplicate of another variant (their action agreement on a fixed set of situations is below 90%).

A variant that loses to the scripted pool is not a "reasonable" teammate implementation and is dropped.

## 3. Labeller changes (all still supervised)

1. **Mixed continuation policy.** After the decision being labelled, each player continues with a policy drawn from the opponent library (train approaches only), rather than always Bot A. Our own continuation is drawn from {A1, B1, current C-v1}. Using a frozen, earlier C is not RL: the model is fixed and nothing is learned from rewards during play.
   - Decision **D5**: you chose "Bot A, one pass" earlier. This changes that choice to "a mix of fixed policies, one pass". It needs your OK.
2. **Table randomization.** For each game, seat 2–4 variants from the library, the scripted bots, and possibly frozen copies of C-v1. Also randomize the starting stacks (20–200 bb) and the number of live players.
3. **Action-consistent re-deal.** This stays as it is, using the *actual* seated variant's `act()`.
4. **Pre-flop.** Raise the pre-flop sampling factor back to 0.6 and raise K_MAX to 64 for pre-flop. A better option is a pre-flop label cache keyed by (hand class, position bucket, action-sequence bucket, stack bucket), averaged over many rollouts. That gives low-variance labels cheaply.
5. **Label noise control.** Keep the adaptive K. Add `label_confidence` = gap / SE as a column. Train with weight = min(gap, cap) × clip(confidence).

Target: about 400–600k rows. v1 produced 93k rows in about an hour of STRIX time; this estimate is from the log timings and has not been measured precisely. Mixed continuations cost about the same, so the target needs roughly 5–7 STRIX hours, run overnight in resumable jobs.

## 4. Model changes

1. **Opponent-model dropout.** On 30% of the rows, replace the opponent features with PRIOR values, plus a flag `om_available` = 0. On a further 20% of the rows, shrink the stats toward PRIOR by the factor rounds / (rounds + 20). This fixes the distribution shift, and C then plays sensibly in both experimental conditions.
2. **Opponent-model feature audit.** Use SHAP plus an ablation that drops each group of features. Keep a feature only if it improves the robust objective defined in §5.
3. **Monotone constraints** in LightGBM. For example, the scores for raise and all-in are non-decreasing in equity and edge. This stops odd extrapolation on unseen opponents.
4. **Seed ensemble.** Average the probabilities of 5 LightGBM models trained on different bootstraps or seeds. Predicting 5 models costs about 5 × 2 ms, well under 200 ms. SHAP explanations average across members (TreeSHAP values add up).
5. **Candidate objective, experiment E1:** regress the per-action EV (one model per action, or a multi-output NN) and take the argmax. The alternative is the current weighted classifier. Both are supervised and both are labelled by rollouts. Keep whichever wins on the robust objective.
6. **Hyperparameters.** Tune with Optuna. Model selection uses a **robust validation set**: rows from held-out *games* played against the train approaches. The held-out approaches are only touched in §6.

## 5. Robust selection criterion

We care about **reliably** beating A and B. For a candidate C, run duplicate tournaments (§6) against every train-approach variant, then compute:

$$
R(C) = \min_{v \in \text{train variants}} \ \mathrm{LCB}_{95}\big(\text{stack}_C - \text{stack}_v\big)
$$

\( R(C) \) is the worst lower confidence bound across variants. Choose the candidate with the largest \( R(C) \), not the largest mean margin. Break ties by the 10th percentile of the margin.

This is a maximin choice. It deliberately accepts a smaller average margin in exchange for no losing match-up.

## 6. Evaluation protocol (pre-registered)

- **Duplicate poker for variance reduction.** Every deck seed is played several times with the seats rotated. Each bot then gets the same cards in the same position across replays, which cancels most of the card luck. This could make the CIs about 2–4× narrower. That is a common estimate for duplicate formats that I have not verified for 10-handed play, so measure it on a pilot.
- **Power analysis.** Use a pilot of 64 duplicate games per match-up to estimate the variance. Then pick the number of games so that a +10 bb edge is detected with 80% power.
- **Pairing.** Use a paired bootstrap by deck seed, with 10k resamples.
- **Multiple comparisons.** Apply Holm–Bonferroni across all held-out variants (about 32 comparisons). That way "beats every variant" is a familywise claim at α = 0.05.
- **Conditions.** Test every held-out variant with the opponent model on *and* off, as the proposal requires for the ablation. Also run a mixed table where all of A1–A6 and B1–B6 play at once.
- **Success criterion, fixed before looking at the results:** after the Holm correction, C − v > 0 for **every** held-out variant in **both** conditions, and C's bust rate is at most that of every variant.

Secondary report: the mean margin, a per-street regret breakdown, and the decision-time distribution (p50, p99, max).

## 7. Schedule and compute (all CPU, $0 GPU expected)

| Step | Work | Where | Time (est.) |
|---|---|---|---|
| 1 | Variant library (12 approaches), validity checks | laptop | 1–2 days of coding |
| 2 | Equal-budget tuning of 96 variants | STRIX | about 4 h |
| 3 | Labeller v2 (mixed continuation, table randomization, pre-flop cache) and tests | laptop | 0.5 day |
| 4 | Generate 500k rows | STRIX, overnight | 5–7 h |
| 5 | Train candidates (dropout, monotone, ensemble, EV regression), Optuna | STRIX | 2–3 h |
| 6 | Robust selection R(C) on train variants | STRIX | 2 h |
| 7 | Held-out duplicate evaluation, once (no peeking and re-tuning) | STRIX | 3 h |

The $10 GPU budget is only worth using if the multi-output NN in E1 wins. A small MLP trains fine on CPU, so I expect to spend $0.

## 8. If the criterion fails on a held-out variant

1. Analyse why: look at the streets and spots where C loses chips to that variant. Use SHAP on those decisions.
2. Fix it only through *general* levers: more coverage of that *style* by train variants, more data or features. Never train on the held-out variant itself, because that would invalidate the claim.
3. Re-run the held-out evaluation **once**, and report it as a second attempt.

## 9. Decisions for you

- **D5:** mixed fixed-policy continuation, including a frozen C-v1, instead of Bot A alone (§3.1).
- **D6:** seating frozen copies of C-v1 at training tables. It is not RL, but it might *look* like self-play to a grader.
- **D7:** whether to share the variant library with the A and B owners. It could double as their tuning baseline, but then the "equal tuning budget" accounting has to cover it.
