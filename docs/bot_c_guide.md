# Bot C: guide for teammates

Bot C (`pokerbots.bots.ml.bot_c:MLBot`) is the supervised ML agent. It needs
no Bot A or Bot B code, so you can play against it and measure your own bot
against it right away.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

## Playing Bot C

```powershell
.venv\Scripts\python.exe -m pokerbots.tournament
```

The default config seats Bot C against the scripted opponent pool. To put your
bot at the table, pass it in `main_bots` of a `TournamentConfig`, as
`pokerbots/tournament/__main__.py` does.

## Measuring your bot against Bot C

```powershell
.venv\Scripts\python.exe -m pokerbots.bots.ml.eval2 --models models/bot_c --vs your.module:YourBot --om both
```

This runs duplicate games. Each seed is played twice with the seats and cards
swapped, so card luck mostly cancels.

- **Output:** for each condition you get Bot C's mean edge in bb (positive
  means Bot C wins), a 95% bootstrap CI and a one-sided p-value with Holm
  correction. The JSON goes to `models/results/`.
- **Options:** `--om both` tests with the opponent model on and off. Raise
  `--seeds` (default 24) to get tighter CIs. Each seed plays 50 rotations
  (`--rotations`).
- **Quick check:** with no bot of your own yet, try
  `--vs pokerbots.bots.simple:RandomBot`.

## What your bot must implement

Subclass `pokerbots.core.Bot` and implement:

| Method | What it does |
|---|---|
| `new_game(game_id, player_id)` | Called once at the start of each game |
| `act(obs) -> str` | Returns one legal action, at most 200 ms per decision |
| `observe_result(log)` | Receives the round log, to feed the opponent tracker |

If `__init__` accepts `use_opponent_model: bool`, `eval2` tests both settings.
If it doesn't, the bot is built with no arguments. Use only the shared
`Observation` and `OpponentTracker`, the same inputs Bot C gets.

## Reading Bot C's decisions

After `act`, `bot.explain()` returns the chosen action, its predicted EV and
the top TreeSHAP features behind it. Use this to find where Bot C's decisions
depend on features your bot can exploit.

## Leave these alone

- **`models/bot_c/`** holds Bot C's trained model: LightGBM trees, stored as
  compressed text. Bot C can't play without it.
- **Retraining** uses `bots/ml/labeller.py` (rollout labels) and then
  `bots/ml/train.py`. The labeller's tables use the scripted bots and the
  current Bot C as policies.
