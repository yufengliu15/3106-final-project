# Live viewer

Watch a tournament in the browser, action by action, while it runs.

## Run it

From the project root (after the install steps in the README):

```bash
PYTHONPATH=src .venv/bin/python -m pokerbots.web --games 1 --rotations 20
```

The terminal prints:

```
Live viewer: http://localhost:8765/  (Ctrl+C to stop)
```

Open http://localhost:8765/ in a browser. The tournament starts as soon as
the server starts; the page catches up on everything played so far.

Stop with Ctrl+C in the terminal. Each start plays a fresh run.

The viewer also writes the tournament's JSONL match log to
`roundlogs/<date>_<time>.jsonl` (or `--log-file PATH`). `--no-logs` turns it off.
When the run finishes, its overall standings go to the top of `LEADERBOARD.md`
(see [tournament.md](tournament.md#leaderboardmd)).

Do not open `src/pokerbots/web/live.html` straight from disk: the page gets
its data from the server and shows only a "start the server" message without it.

## Flags

Same as `python -m pokerbots.tournament`, except `--games` defaults to 1 here
(the tournament command defaults to 5). Also `--port`, `--log-file` and `--no-logs`.

| Flag | Default | Meaning |
|---|---|---|
| `--games` | 1 | Games to play |
| `--rotations` | 100 | Maximum rotations per game |
| `--seats` | 10 | Players per table |
| `--stack` | 100 | Starting stack in bb |
| `--k` | 1 | Bust penalty |
| `--mode` | eval | `dev` = same cards every run |
| `--seed` | 0 | Base seed |
| `--port` | 8765 | Port for the page |
| `--log-file` | `roundlogs/<date>_<time>.jsonl` | Match log path, same JSONL format as the tournament command |
| `--no-logs` | off | Do not write a match log |

The bots are the tournament defaults: Bot A (CrispBot), Bot B (FuzzyBot),
Bot C (MLBot) and scripted opponents. Change them in
`src/pokerbots/tournament/__main__.py` (`DEFAULT_CONFIG`).

The browser keeps every step, so long runs (many games x 100 rotations) use a
lot of memory. Start small: `--games 1 --rotations 20`.

## Reporting a spot

The **You are here** panel (top right) describes the step on screen with the
same IDs the match log uses. **Copy location** copies it for a bug report:

- `game_id`, `rotation_id`, `round_id` (0-based, as in the log; the header shows
  round `round_id + 1`) and the action index (0-based into `actions`; the header
  shows that plus 1);
- the acting player and position, the legal options, and the bot's `explain()`
  text, plus pot, board and stacks before the action and everyone's hole cards;
- the log path and a `jq` line that prints the same round record from it.

Run the `find:` line from the project root. It needs `jq` (`brew install jq`):

```bash
jq 'select(.type=="round" and .game_id==1 and .rotation_id==1 and .round_id==1) | .actions[2]' /private/tmp/loc-smoke.jsonl
```

The log stores each action as `[player_id, street, action, amount_bb]`. The
legal options and `explain()` text exist only in the viewer, so the copied text
is the only place they are kept.

## What the page shows

- **Header:** game, rotation, round, street, action number, card seed, run status.
- **Table:** each seat's bot, player ID, position (SB, BB, UTG ... BTN), stack,
  bet this street, last action and hole cards. Gold border = just acted;
  dashed = acts next; green = won chips. The board and pot are in the middle.
- **Decision:** the acting bot's legal options (chosen one highlighted) and its
  `explain()` text. At the end of a hand: every player's chip change.
- **Round actions:** every action this hand, by street. Click a line to jump to it.
- **Leaderboard, this game:** stack at the start of the rotation, stack in
  this hand, who is out and when.
- **Leaderboard, overall:** mean final stack per bot class over finished games.
  The same table is saved to `LEADERBOARD.md` when the run finishes.
- **Rotation grid:** ending stack of each round in this rotation, and the average
  that becomes the next starting stack.
- **Rotation history:** starting stacks over the last 12 rotations.

## Controls

| Key / button | Action |
|---|---|
| Space / Play | Play or pause |
| ← / → | Step one action |
| Shift+← / Shift+→ / ⏮ Round / Round ⏭ | Previous / next round |
| [ / ] / ⏮ Rotation / Rotation ⏭ | Previous / next rotation, landing on its first round. Mid-rotation, previous goes to this rotation's start first. Crosses into the next or previous game |
| L / Live | Jump to the newest step and follow it |
| H / Hole cards | Show all hole cards, or only at showdown |
| Speed | Playback speed, 0.5 to 200 actions per second |
| Slider | Scrub through every step received so far |

## Troubleshooting

| Problem | Fix |
|---|---|
| `Address already in use` | Another viewer is running. Stop it, or add `--port 8766` and open that port. |
| `ModuleNotFoundError: phevaluator` (or another package) | Run the install steps: `uv pip install -r requirements.txt` |
| `No module named pokerbots` | Run from the project root with `PYTHONPATH=src` |
| Page says "connecting" | The server stopped; restart it and reload the page. |
| Status shows "error" | A bot crashed; the message is in the header and the terminal. |

## How it works

`pokerbots.web` (`src/pokerbots/web/__main__.py`) runs the tournament in a
background thread with an observer (`Tournament(config, observer=...)`). The
observer receives every log record plus two viewer-only records: `round_start`
(every player's hole cards) and `action` (the acting bot's observation, choice
and `explain()`). The server streams them to the page over server-sent events
at `/events`. Round logs on disk are unchanged.
