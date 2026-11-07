# 3106-final-project
Yufeng Liu 101258905, Thomas Nolasque 101266644, Teja Chilukuri 101261938

Adaptive poker bots: crisp rules vs fuzzy logic vs learning, compared at an
8-seat no-limit hold'em table (see `docs/COMP3106-Project-Proposal.pdf`).

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run tests

```bash
pytest
```

`pyproject.toml` puts `src/` on pytest's import path, so tests can
`import pokerbots` without installing the package.

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
