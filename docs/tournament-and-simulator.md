# Tournament and simulator design

Design for issue #15 (simulator wrapper over RLCard) and the tournament rules it
must support. Status: rotation/stack model agreed 2026-11-08; per-action details

## Responsibilities

- **Simulator** (`pokerbots.sim`) plays exactly **one round** (one hand). It
  calls `bot.act()` for every decision and returns a `RoundLog`.
- **Tournament** (`pokerbots.tournament`) owns everything that carries between
  rounds: stacks, seat order, seed, busted players, and the game / rotation /
  round IDs. It builds a **new simulator for every round** and passes that state in.

```
tournament --(positions, stacks, seed, IDs)--> simulator (one round)
simulator  --(Observation)--> bot.act() --(action)--> simulator
simulator  --(RoundLog)--> tournament --> bot.observe_result()
```

## Positions

- `positions` lists **live players only**; its length varies as players bust.
- The only hard-coded indices: **0 = small blind, 1 = big blind**. The last
  index is the button (last to act after the flop); "index 7 = button" only holds at a
  full 8-player table.
- RLCard player index `i` is `positions[i]`. The simulator sets
  `dealer_id = len(positions) - 1`, so RLCard's own blind logic
  (`(dealer_id + 1) % n`, `(dealer_id + 2) % n`) puts the blinds on 0 and 1.
- `stacks[i]` belongs to `positions[i]`; same length as `positions`. No `None`
  entries (busted players are simply not listed).

## Rotations

Let **x** = number of live players at the start of a rotation.

1. **Before round 1:** each player is given their starting stack **S** for this
   rotation (the default stack, 100 bb, in the first rotation; otherwise the
   result of the previous rotation's averaging).
2. **Rounds 1..x:** every round starts from the **same starting stacks** and the
   **same seed** (same deck order). Only the positions shift by one each round,
   so each player plays every position exactly once on the same deal.
   Each player's ending stack for the round is recorded.
3. **After round x:** for each player, average their x ending stacks, with one
   substitution: **an ending stack of 0 counts as −k·S**, with **k = 1** (so a
   bust counts as minus the player's starting stack for that rotation).
   This average is the player's starting stack for the next rotation.
4. **Elimination:** a player whose average is **≤ 0** is busted, is reported as
   such for that rotation, and never re-enters. The next rotation has fewer players.

Busting is only checked between rotations, so the table size (and the RLCard
env's player count) changes only between rotations, never mid-rotation.

### Why k = 1

Assuming a player roughly breaks even in the rounds they survive, they are
eliminated when `b ≥ x / (1 + k)` of their x rounds end in a bust. With x = 8:

| k | busts out of 8 needed |
|---|---|
| 0.5 | 6 |
| **1** | **4** |
| 2 | 3 |

k = 1 eliminates a bot that busts in about half the seats. Rounds in a rotation
share a deal, so busts are correlated; revisit k after measuring the bust rate
in the 10,000-round random-bot run.

## Chip units

RLCard works in whole chips: blinds are `small_blind = 1`, `big_blind = 2`
(`game.py`), half-pot raises use `int(pot / 2)`, and split pots use `divmod`.

- Set `small_blind = 50`, `big_blind = 100`, so **1 bb = 100 chips**
  (0.01 bb resolution). Convert chips ↔ bb at the simulator boundary; bots and
  logs use bb.
- Rotation averages are **truncated** to whole chips.

## Chip conservation

- **Within every round:** total chips at the end equal total chips at the start.
  Checked by the simulator / tests.
- **Between rotations:** not conserved. The bust penalty and truncation both
  shrink the table total; this is by design.

## RLCard hooks used

`Game.init_game()` (run on every `env.reset()`) reads these attributes, which
the simulator sets before dealing:

| Attribute | Set to |
|---|---|
| `init_chips` | per-player list of starting stacks in chips |
| `dealer_id` | `len(positions) - 1` |
| `small_blind`, `big_blind` | 50, 100 |
| `num_players` | fixed when the env is built; a new env is built per round anyway |

## Busting spike (do first)

Biggest risk in the issue. Pass/fail test for staying on RLCard:

- Start 8 players, force one to 0, rebuild the env with 7 players.
- Assert the blinds are posted by indices 0 and 1, and chips are conserved
  within the round.
- If this can't be made reliable, evaluate PokerKit (30-minute spike).

**Result (2026-11-08): passed.** `tests/test_rlcard_spike.py` covers hooks at
8/7/3/2 players, a rebuild after a bust, chip conservation over 300 random
rounds per table size (most with 2+ all-ins, so side pots), same-seed deals,
and the short-stack `CHECK_CALL` all-in. Staying on RLCard.

Known deviation: heads-up (2 players) RLCard puts the small blind on index 0
and the big blind on the button (index 1); real heads-up has the button post
the small blind.

## Spec changes to #14 types

- `Observation.positions` / `RoundLog.positions`: variable length, live players only.
- `Observation.stacks` / `RoundLog.starting_stacks`: `tuple[float, ...]`, no `None`.
- Update docstrings that say "8 player IDs ... index 7 = button".
- How a bust is reported (e.g. a list of eliminated player IDs) belongs to the
  tournament's rotation result, since elimination happens between rotations.

## Issue #15 done-when (revised)

- 10,000 rounds with 8 random bots run without crashing.
- Total chips are conserved **within** every round.
- Same seed → identical deal.
- A player eliminated after a rotation never appears in later `positions`.

## Action translation

The simulator computes every legal action's cost **before** the bot acts, from
RLCard's round state. With `diff = max(raised) - raised[me]`, `pot` read fresh
(RLCard only refreshes `dealer.pot` inside `get_state()`), and `stack =
remained_chips`:

| Our action | Offered when | Cost | Sent to RLCard as |
|---|---|---|---|
| `fold` | `diff > 0` (hidden when checking is free, though RLCard allows it) | 0 | `FOLD` |
| `check` | `diff == 0` | 0 | `CHECK_CALL` |
| `call` | `0 < diff < stack` | `diff` | `CHECK_CALL` |
| `raise_half_pot` | RLCard offers `RAISE_HALF_POT` | `int(pot / 2)` | `RAISE_HALF_POT` |
| `raise_pot` | RLCard offers `RAISE_POT` | `pot` | `RAISE_POT` |
| `all_in` | always when the player has chips | `stack` | `ALL_IN`, **or `CHECK_CALL` when `diff >= stack`** (RLCard removes `ALL_IN` then; `CHECK_CALL` bets `min(diff, stack)`) |

Bots always see one `all_in` with its chip cost, whether it is a short call,
a short raise, or a full shove.

**Recorded amounts:** `action_history` / `RoundLog.actions` store the
pre-computed cost. The simulator also measures the player's `in_chips` before
and after each step and asserts it equals that cost, so a formula bug fails
loudly.

**Known RLCard deviations (accepted for now):**
- Every `ALL_IN` counts as a full raise and reopens action, even when it is
  smaller than a legal raise (real no-limit would not reopen).
- On a short all-in call RLCard sets `raised[me] = max(raised)`, overstating
  the bet; payouts use `in_chips`, so results are correct.
- Side pots are split correctly in layers (`split_pots_among_players`).

## Seeding

A new RLCard env is built for every round with `config={'seed': rotation_seed}`.
RLCard shares one `np_random` between env and game, and each `reset()` builds a
fresh shuffled `Dealer`, so a fresh env per round gives every round of a
rotation the same deck order. RLCard index `i` always receives the same hole
cards; whoever is at `positions[i]` gets them.

## Logging and illegal actions

- Every finished round is logged (`RoundLog`).
- If a bot returns an action not in its `legal_actions`, the simulator raises
  an error and the run **ends** (fail fast). RLCard's own fallback
  (`NolimitholdemEnv._decode_action` silently converts to check/fold) is
  bypassed: the simulator validates the action itself and calls
  `env.game.step(...)` directly.

## Snapshots (for #18)

- #18 decides **when** to snapshot and **keeps** the snapshot.
- The **simulator builds** the snapshot (a deep copy of the RLCard game state)
  and hands it back as an opaque object; #18 never imports RLCard.
- The simulator continues a round from a snapshot. Proposed: continuation
  always re-deals every card the acting player can't see (opponents' hole
  cards + remaining board) with a fresh RNG, so replaying a snapshot never
  leaks the true future.
