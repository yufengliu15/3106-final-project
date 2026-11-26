# Evaluation of 3 poker bot strategies
## COMP3106 Final Project
Yufeng Liu 101258905, Thomas Nolasque 101266644, Teja Chilukuri 101261938

Adaptive poker bots: crisp rules vs fuzzy logic vs learning, compared at an
10-seat no-limit hold'em table (see `docs/COMP3106-Project-Proposal.pdf`).

## Install

```bash
uv venv
uv pip install -r requirements.txt
```

Run the commands below from the project root. They use the Python executable
in `.venv`, so activation is optional on macOS and Linux.

## Run tests

```sh
.venv/bin/python -m pytest
```

`pyproject.toml` puts `src/` on pytest's import path, so tests can
`import pokerbots` without installing the package.

### Demo TUI
```bash
PYTHONPATH=src .venv/bin/python -m pokerbots.sim.demo --vs-bots --seed 0
```

Play one seat against Bot B, Bot C, and Calling Machine. Pick an action by
number. The demo shows the bots' last decisions after each hand. Omit
`--vs-bots` to control all four seats.

### Tournament
```bash
PYTHONPATH=src .venv/bin/python -m pokerbots.tournament --games 1 --rotations 20 --no-logs
```

The default tournament seats Bot A, Bot B and Bot C with seven scripted opponents.
The leaderboard shows final stacks in big blinds. Run without the flags for
five full games and JSONL round logs.
Every finished run (tournament command or live viewer) adds its overall
standings to the top of [`LEADERBOARD.md`](LEADERBOARD.md), headed with the number of games.

### Live viewer (watch games in the browser)

1. Start the viewer server from the project root:
   ```bash
   PYTHONPATH=src .venv/bin/python -m pokerbots.web --games 1 --rotations 20
   ```
   It prints `Live viewer: http://localhost:8765/` and starts playing.
2. Open http://localhost:8765/ in any browser.
3. Press Space to play or pause. Stop the server with Ctrl+C.
4. To report a bug at a step, click **Copy location** in the "You are here" panel.
   It gives the game, rotation, round and action IDs from the match log in `roundlogs/`.

Open the page through the server, not by double-clicking
`src/pokerbots/web/live.html`: the page gets its data from the server.
Full guide (controls, flags, troubleshooting): `docs/live_viewer.md`.

To compare Bot B and Bot C with seats swapped for each seed, run

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m pokerbots.bots.ml.eval2 \
  --models models/bot_c --vs pokerbots.bots.fuzzy:FuzzyBot \
  --om both --seeds 12 --rotations 50 --workers 4
```

Positive results are Bot C's average final-stack lead. Negative results
are Bot B's. The opponent-model flag applies to both bots.

Bot A is a crisp forward-chaining expert system in four layers: layer 0
computes equity against each opponent's range, response probabilities and EV
per action; layer 1 turns them into crisp facts (position, edge, villain type,
fold equity); layer 2 fires the first matching rule (R1-R8) to set an intent
and its investment cap; layer 3 plays the best-EV legal action under the cap,
with a seeded roll among actions within 0.5bb of the best. `explain()` prints
the facts, the fired rule, the EVs and the roll.

Bot B keeps separate Mamdani rules for opening, responding to a preflop
raise, betting after the flop, and calling for its stack. Its range model
uses the current bettor's observed style and always leaves some probability
for off-range hands. Its stack-call rules price only contestable chips and
charge the tournament's default bust penalty in proportion to the chance of
losing outright.
[Mamdani and Assilian, 1975](https://www.sciencedirect.com/science/article/pii/S0020737375800022)
[Billings et al., 1999](https://cdn.aaai.org/Symposia/Spring/1999/SS-99-07/SS99-07-003.pdf)

## Layout

```
src/pokerbots/
  core/          # Observation, Bot interface, RoundLog, card helpers, equity calculator
  sim/           # RLCard wrapper: plays one hand, translates RLCard state into an Observation
  tournament/    # seeding, seat rotation, seat filling, chip totals, logging, leaderboard
  opponents/     # shared opponent model / stats tracker (updated via observe_result())
  bots/
    crisp/       # Bot A: crisp rule-based expert system
    fuzzy/       # Bot B: fuzzy rule-based system
    ml/          # Bot C: machine-learning agent
    simple/    # simple opponents (TAG, TP, LAG, ...)
  analysis/      # report figures, log interpretation, experiment scripts
  web/           # demo server + HTML view of a table
tests/
```

Bots should import from `core/` (and `opponents/`), never from
`rlcard` or `sim/` directly.
