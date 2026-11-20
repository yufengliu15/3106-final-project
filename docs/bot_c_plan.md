# Bot C plan: supervised rollout-labelled agent

Design for issues #3 (Bot C) and #18 (training-data generator). Drafted
2026-11-18. The reference for what is allowed is the proposal
(`docs/COMP3106-Project-Proposal.pdf`); the environment is described in
`docs/simulator.md` and `docs/tournament.md`.

## 0. Rules Bot C must keep (from the proposal)

| Rule | How this plan keeps it |
|---|---|
| Supervised classifier (gradient-boosted trees or small NN) on the shared feature vector | LightGBM multiclass classifier; one optional small-MLP comparison |
| Training data = logged hands vs a rotating pool of A, B and the scripted bots | Section 3: seat pool drawn per game exactly like the tournament |
| Label = replay each decision with every action, take the action that won the most chips | Section 4: rollouts from the simulator snapshot, argmax of mean chips |
| No reinforcement learning | One labelling pass; rollouts continue with Bot A's rules, never with Bot C itself |
| Explanations via feature attributions (SHAP) | TreeSHAP from LightGBM (`pred_contrib`), Section 6 |
| Same observations and the same opponent-model module as A and B | Features use only `Observation` + the shared #17 tracker |
| 200 ms per decision | Tree inference < 1 ms; equity cached/precomputed, Section 5 |
| No external dataset | All data self-generated; no outside hand histories or pretrained poker models |
| Opponent-model ablation (experiment 3) | `use_opponent_model=False` replaces stat features with population priors |

Decisions taken on 2026-11-18:
- **Re-deal consistent with actions** (labelling only): rollouts re-deal opponents'
  hidden cards only where each opponent's own policy would have taken the actions it
  actually took. This uses opponent policies **only to compute labels**, never at play time.
- **Weighted classifier:** label = argmax action; sample weight = chip gap between the
  best and second-best action, so near-ties do not add noise.
- **Single pass, Bot A continuation** for the labelled player's later decisions.
- **Self-generated data only.**

## 1. Environment facts that shape the design

- Table is now **10 seats** (proposal said 8). `positions` length varies as players
  bust, so every positional feature is relative (seats to the button, players left to
  act), never an absolute index.
- **Stacks carry between rotations** by averaging, and a round that ends on 0 counts as
  **-k*S** (k = 1). Busting therefore costs about twice the stack in the score.
  See open decision D1.
- Actions are discrete: fold / check / call / raise_half_pot / raise_pot / all_in, with
  RLCard quirks (an all-in always reopens the action).
- The simulator already provides `snapshot()` and `from_snapshot(snap, bots, seed)`,
  which re-deals all unseen cards. #18 extends this with action-consistent re-dealing.

## 2. Shared prerequisites (built first, used by all three bots)

These are not Bot C-specific. Where an issue is owned by a teammate, the local version is
a stand-in that is **not pushed**.

1. **Opponent stats tracker (#17)**: VPIP, aggression factor, fold-to-raise and rounds
   observed over a 50-round sliding window. Extra counters that are cheap and useful to
   all bots: PFR (pre-flop raise %), all-in frequency, went-to-showdown %.
   Proposing extras to #17 keeps the "same opponent-model module" rule.
2. **Equity calculator** (proposal milestone, Nov 9-15):
   - Pre-flop: a 169-class x (1..9 opponents) equity table, precomputed once by Monte
     Carlo with `treys` (runs on STRIX in minutes), shipped as a small `.npy`.
   - Post-flop: Monte Carlo against N random hands, capped at a time budget (about
     30 ms), with an LRU cache keyed by (hole, board, n_opponents).
   - Optional **range-weighted equity**: sample opponent hands from the top-VPIP% of the
     169-hand ordering. This is offered to all three bots so no bot gains information.
3. **Scripted bots (#4-#10)**: TAG, TP, LAG, calling machine, always-all-in (#8, mine),
   random (done), switcher. Hand ranking = the 169-hand pre-flop equity order;
   "top x%" = the first x% of combos in that order. Post-flop, "always raises / always
   calls" applies to hands they entered with.

## 3. Bots A and B (local stand-ins, not pushed)

The goal is **strong, honest** versions of each paradigm: a weak A or B would make
the comparison meaningless, and Bot C also uses Bot A for its rollouts.

**Bot A (crisp)**: rules fire in priority order:
1. Pre-flop: chart by (equity percentile, seats to button, raises faced); open/3-bet/
   call/fold thresholds.
2. Post-flop value: equity vs players in hand above a threshold -> raise (pot if strong
   and opponents are calling stations, 1/2 pot otherwise).
3. Pot-odds rule: call iff equity > pot odds + margin; the margin shrinks against
   aggressive opponents (they bluff more) and grows against tight-passive ones.
4. Bluff/steal: late position, only passive or tight players left, fold-to-raise > 60%
   -> raise 1/2 pot.
5. Opponent labels from crisp thresholds (VPIP > 35% loose, AF > 2 aggressive, with a
   minimum of 15 rounds observed, else "unknown" -> default rules).
6. Short-stack / all-in rule: stack-to-pot < 1 with enough equity -> all-in; facing an
   all-in -> call only with equity > needed odds.
   `explain()` returns the chain of rules that fired.

**Bot B (fuzzy)**: as in #2: triangular/trapezoid memberships on equity, pot odds,
position, SPR, opponent looseness and aggression and a sample-size set; Mamdani (min
AND, max aggregation, centroid) -> aggression score -> action bands from #2.
About 25-35 rules mirroring Bot A's knowledge, so crisp vs fuzzy differ only in the
boundaries.

**Tuning budget (both):** the same number of self-play hands against the practice pool
as Bot C's logged hands (see D2), tuned by coordinate search on thresholds/membership
breakpoints. Tuning runs on STRIX.

## 4. Training-data generator (#18)

### 4.1 Logging pass
- Tournaments with the same seating rules as evaluation: A, B and one **learner seat**
  plus 7 scripted bots drawn from the pool each game, including the switcher.
- The learner seat is played by **Bot A with 15% exploration** (a random legal action),
  so the dataset covers states that Bot C may reach and A would avoid. Rollouts still
  continue with plain Bot A.
- At every learner decision, save: the feature vector (Section 5), `game_id`, the street,
  and a simulator snapshot.

### 4.2 Labelling each snapshot
For every legal action *a*, play the rest of the round **K** times and record the
learner's result:
- **Common random numbers:** rollout *j* uses the same re-deal for every action, so the
  differences between actions are far less noisy than independent samples.
- **Action-consistent re-deal:** draw opponents' hole cards, then keep the sample only if
  replaying each opponent's own `act()` on the hand's history so far reproduces its real
  actions. Scripted bots are deterministic range checks (cheap). A and B are re-run on
  history prefixes (cached). Cap at 200 attempts, then fall back to a uniform re-deal
  and flag the row.
- **Adaptive K:** start with K = 20; if the top two actions are within one standard error,
  add batches up to K = 100.
- **Value per rollout** = learner's chip change in bb, adjusted by D1.
- Store: mean and standard error per action, the label (argmax), the weight
  (best - second best, clipped at 10 bb), and the flags.
- The test from #18: same snapshot with a different true future -> labels identical.

### 4.3 Volume and compute
Rough estimate, to be measured in week 1: about 20 ms per rollout x about 4 legal
actions x 20 rollouts ~= 1.6 s per decision per core. STRIX (16 threads, `multiprocessing`)
~= 10 decisions/s -> **100k labelled decisions in about 3 hours**. The laptop is
used only for development and smoke runs.
Target: 150-300k decisions (pre-flop is about 60% of decisions, so post-flop streets
are oversampled to keep at least 15k rows each).

### 4.4 Output
Parquet: one row per decision with `game_id`, `street`, the features, `label`,
`weight`, `ev_<action>` and `se_<action>` columns, and flags. Committed as a compressed
sample; the full set is shared via release/Drive (too big for git).

## 5. Feature vector (shared inputs only)

All of it is computed from `Observation` + the #17 tracker:

| Group | Features |
|---|---|
| Hand | pre-flop equity (table), post-flop equity vs N random, equity vs VPIP-weighted ranges, made-hand class (treys rank class), flush/straight draw flags, pocket pair, suited, gap |
| Board | street, paired board, monotone/two-tone, connectedness, high-card rank |
| Pot / price | pot (bb), amount to call, pot odds, call / stack, SPR, cost of each raise option / pot |
| Position | seats to button (relative), players left to act, players in hand, active players at table |
| Action | raises this street, we were the last aggressor, aggressor's position, our invested chips, facing an all-in |
| Stacks | own stack, effective stack vs largest opponent still in, our rank in chips |
| Opponents (#17) | for the last aggressor and for the mean/max of players still in: VPIP, PFR, AF, fold-to-raise, all-in freq, rounds observed (as log(1 + n)) |

`use_opponent_model=False` -> opponent features set to population priors from the
training set (the same constant for every opponent), so the ablation is equivalent for A,
B and C.

## 6. Model, inference and explanations

- **LightGBM** multiclass (6 classes), `sample_weight` = chip-gap weight. Split
  train/validation/test **by game_id** (70/15/15). Hyperparameters tuned by a small random
  search on the laptop/STRIX (CPU is enough for trees; no GPU needed).
- **Metrics:** accuracy and **mean regret** = average (best action EV - chosen action EV)
  on the test set, which matches the goal better than accuracy. Report both, per street.
- **Play time:** predict the probabilities, mask the actions not in `obs.legal_actions`,
  take the argmax. Inference < 1 ms; total including equity < 50 ms, checked by a timing
  test with a 200 ms assertion.
- **Explain:** LightGBM `predict(..., pred_contrib=True)` gives exact TreeSHAP values
  per class; `explain()` reports the top 3-5 features pushing toward the chosen action
  (e.g. "equity 0.71 (+), aggressor VPIP 0.48 (+), pot odds 0.25 (-)"). No `shap`
  dependency needed (to verify against the LightGBM docs before relying on it).
- Model file: `models/bot_c.txt` (LightGBM text format, a few MB) loaded at start-up.
- Optional comparison: a 2-layer MLP (scikit-learn) on the same data, reported in the
  report's results; the tree model stays the bot.

## 7. Compute and budget

| Resource | Use |
|---|---|
| Laptop | development, unit tests, smoke runs (< 1k decisions) |
| STRIX | equity tables, A/B tuning, label generation (hours), LightGBM training, experiments 1-3 |
| gpu.ai ($10 cap) | **not planned**: rollouts and tree training are CPU-bound. Kept in reserve only for the optional MLP if it turns out slow (expected cost < $1) |

STRIX has no git: the code is copied over SSH as a zip; results come back the same way.

## 8. Evaluation (how "wide margin" is checked)

- Experiment 1 table with A, B, C + 7 scripted, eval mode, at least 20 games x 100
  rotations; report the mean final stack per bot with a 95% bootstrap CI over games.
- Per-opponent profit (experiment 2) and hands-to-recover after a switcher change.
- Ablation (experiment 3) with `use_opponent_model=False` for all three.
- Development target: C > A and C > B with non-overlapping CIs. If not reached, the main
  levers are, in order: more labelled data on the streets where regret is highest, better
  features, larger K on close decisions.

## 9. Schedule (by the proposal's milestones)

| Week | Bot C work |
|---|---|
| Nov 16-22 | tracker + equity + scripted bots (local), Bot A/B stand-ins, #18 generator with tests, first 10k labelled decisions, Bot C v0 plays legal games |
| Nov 23-29 | full data generation on STRIX, LightGBM tuning, explain(), timing test, ablation flag |
| Nov 30-Dec 6 | experiments 1-3, figures for Bot C, agreement/disagreement hands for the qualitative analysis |
| Dec 7-11 | clean-up, README section, report text for Bot C |

## 10. Open decisions (need team agreement)

- **D1. Value of a bust in labels.** The tournament scores a round ending on 0 as -k*S,
  so pure "chips won" undervalues the bust risk. Recommended: rollout value = the
  tournament's own per-round score (bust -> -k*S), which still is "the chips the
  action wins" under our scoring. Alternative: plain chip delta, as literally written.
- **D2. Equal tuning budget.** Proposed definition: the number of self-play hands each
  bot may observe against the practice pool (e.g. 20k rotations). Rollouts are
  labelling compute on those hands, not extra observed hands; this must be stated in
  the report.
- **D3. Bot A/B dependency.** Bot C's labels depend on Bot A (rollouts) and on A/B as
  opponents. If the teammates' final A/B differ from the local stand-ins, labels must be
  regenerated (about 3 STRIX hours), so this is planned for after A and B freeze
  (target Nov 26).
- **D4. Proposal drift.** 10 seats and carried stacks differ from the proposal text; the
  report should state the final environment.
