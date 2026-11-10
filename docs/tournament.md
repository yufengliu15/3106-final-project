# Tournament design

Design for issue #16: the tournament harness (`pokerbots.tournament`). Rotation
and stack rules were agreed 2026-11-08 (during #15); the rest 2026-11-09.
The simulator it drives is described in `docs/simulator.md`.

The tournament owns everything that carries between rounds: stacks, seat
order, seeds, eliminations and IDs. It builds a new `Simulator` for every
round and gets a `RoundLog` back.

Code: `pokerbots.tournament` (`TournamentConfig`, `Tournament`).

## Games
- A run plays `games` games (default 5). Each game starts every player at
  `starting_stack` (default 100 bb) and plays rotations (rules below) until
  `rotations` (default 100) have been played **or one player is left**.

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

## The bust penalty k

`k` sets how much a bust hurts. When a round ends with a player on 0 chips,
that round counts as **−k × S** in their rotation average instead of 0, where
S is their starting stack for the rotation. With k = 0 a bust would just count
as 0; with k = 1 it cancels out a round where they doubled up.

k = 1 because then a player who busts in about half their seats is eliminated.

## Chip units and conservation

- Stacks are tracked in whole chips (1 bb = 100 chips, from the simulator).
- Rotation averages are **truncated** to whole chips.
- Chips are **not** conserved between rotations: the bust penalty and
  truncation both shrink the table total. This is by design.

## Seating
- `seats` (default 8) per game. Every class in `main_bots` gets one seat; the
  rest are drawn at random, with replacement, from `simple_bots`, fresh each game.
- Fresh player IDs (1000-9999, unique) each game, and a random initial seat
  order. Every bot gets `new_game()` at the start of a game, and
  `observe_result()` after **every** round, including eliminated bots.
- Within a rotation, player `i` in seat order sits at index `(i + r) % x` in round `r`.

## Seeds
- One `random.Random(base_seed)` produces, in a fixed order: simple-bot draws,
  player IDs, seat order and (eval mode) card seeds. Same config → identical results.
- **dev mode:** card seed = `game_id * 1000 + rotation_id` (zero-padded join),
  so the cards are the same in every run regardless of `base_seed`.
  `games` and `rotations` are capped at 999 so seeds never collide.
- **eval mode:** card seeds are drawn from the base-seed RNG; games do not repeat cards.
- Every `RoundLog` records its `seed`.

## Leaderboards
- **Per game:** final stack (0 if eliminated), ties broken by later
  elimination rotation; survivors rank above every eliminated player.
- **Overall:** grouped by bot class, averaged over every seat it held:
  mean final stack, tie-break mean elimination rotation (a survivor counts as
  `rotations_played + 1`).

## Logs
One file per tournament run: `roundlogs/<date>_<time>.jsonl` in the repo root
(e.g. `roundlogs/2026-11-09_15-04-12.jsonl`; logs are committed. The folder is
created on first run). JSONL = one JSON object per line. For each game, in order:
1. `{"type": "game", "players": {id: bot class}, "seat_order": [...], ...}`
2. `{"type": "round", ...RoundLog fields}` for every round
3. `{"type": "rotation", "seed", "stacks", "eliminated"}` after every rotation
4. `{"type": "result", ...GameResult}` at the end

JSON object keys are strings; convert player IDs back to `int` when loading.
