# Live viewer

Watch a tournament in the browser, action by action, while it runs, or play
against the bots yourself.

## Run it

From the project root (after the install steps in the README):

```bash
PYTHONPATH=src .venv/bin/python -m pokerbots.web --games 1 --rotations 20
```

The terminal prints:

```
Live viewer: http://localhost:8765/  (Ctrl+C to stop)
```

Open http://localhost:8765/ in a browser. Nothing is dealt until you pick a
mode in the popup:

| Choice | What runs | Rotations | Writes to disk |
|---|---|---|---|
| **Watch the simulation** | The tournament ([tournament.md](tournament.md)) | Yes: same cards, shifted seats | Match log and `LEADERBOARD.md` |
| **Join as a player** | A sandbox game with you at the table ([below](#play-as-a-human)) | No: fresh cards every hand | Nothing |

The first choice decides the run; a page opened later catches up on everything
played so far. Stop with Ctrl+C in the terminal. Each start is a fresh run with a new choice.

In watch mode the viewer writes the tournament's JSONL match log to
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
| `--rotations` | 100 | Maximum rotations per game (play mode: maximum hands per game) |
| `--seats` | 10 | Players per table |
| `--stack` | 100 | Starting stack in bb |
| `--k` | 1 | Bust penalty |
| `--mode` | eval | `dev` = same cards every run (watch mode only) |
| `--seed` | 0 | Base seed (play mode: random unless given) |
| `--port` | 8765 | Port for the page |
| `--log-file` | `roundlogs/<date>_<time>.jsonl` | Match log path in watch mode, same JSONL format as the tournament command |
| `--no-logs` | off | Do not write a match log in watch mode |

The bots are the tournament defaults: Bot A (CrispBot), Bot B (FuzzyBot),
Bot C (MLBot) and scripted opponents. Change them in
`src/pokerbots/tournament/__main__.py` (`DEFAULT_CONFIG`).

The browser keeps every step, so long runs (many games x 100 rotations) use a
lot of memory. Start small: `--games 1 --rotations 20`.

## Play as a human

1. In the start popup, type a name and click **Join as a player**. Tick
   **Hide bot names** to play without knowing which bot is in which seat.
2. You sit at the table with the bots for the first deal. The bar under the
   controls shows your status.
3. On your turn the bar turns gold and shows one button per legal action with its
   cost in bb. Click one, or press its number key (1-6).
4. Every hand ends on a result box over the table: who won, how much, and the
   winning hand's name ("Full house, kings full of sevens"). The five cards that
   make it are outlined in green; unused board cards are dimmed.
5. **Leave table** ends your play: your remaining decisions this game are
   check/fold, and you are not dealt into later games.

Play mode is a sandbox, not the tournament:

- Every hand is a fresh deal. There are no rotations: the button moves one seat
  each hand and stacks carry over from hand to hand.
- A player with no chips is out of the game at once. A game ends when one
  player is left or after `--rotations` hands (default 100); then the next game
  (`--games`) starts with fresh stacks.
- Nothing is written: no match log, no `LEADERBOARD.md` entry. The header shows
  **Hand** instead of Rotation, and the rotation grid is hidden.

Details:

- Any number of humans can join, up to 23 players at the table (the deck limit
  with 10 bots is 13 humans). Click **Add another human player** (top right) to
  add a seat from the same tab, or open the page in another tab or window and
  join there. New humans are dealt in at the next hand. Each tab acts only for
  the humans it joined.
- Humans cannot join a watch-mode run; restart the server to play.
- Joining switches **Hole cards** to **yours only**: you see your own cards, and
  everyone's cards at showdown. Press H to see every player's cards again.
- While a human is seated in the tab, **Live** replays the bots' actions at the
  Speed setting instead of jumping to the newest one, and holds each hand's
  result for 3 seconds. Your action buttons appear once the page has caught up.
- The game waits for you; bots do not act while it is your turn.
- Hidden hole cards are still in the page's event stream; this is a demo, not
  a cheat-proof game.
- Restarting the server starts a fresh run; the page drops humans from the old one.

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

- **Header:** game, rotation, round, street, action number, card seed, run status,
  and the **Join as a human player?** button.
- **Human bar** (after joining): each human this tab plays, their seat, and the
  action buttons on their turn.
- **Table:** each seat's bot, player ID, position (SB, BB, UTG ... BTN), stack,
  bet this street, last action and hole cards. Gold border = just acted;
  dashed = acts next; green = won chips; pink ring and YOU = a human this tab plays.
  The board and pot are in the middle.
- **Result box** (end of every hand): the winners, their chip gain and hand
  name, the other hands shown at showdown, and your own result. "Won without
  showdown" means everyone else folded.
- **Decision:** the acting bot's legal options (chosen one highlighted) and its
  `explain()` text. At the end of a hand: every player's hand at showdown and chip change.
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
| H / Hole cards | Show every player's hole cards, or only yours (humans this tab plays) plus showdown hands |
| N / Bot names | Show or hide which bot is in each seat. Hidden: every bot reads "Bot" in the same colour, its `explain()` text and the overall leaderboard are hidden too (they name the bot), and copied locations use "Bot". Humans keep their names. Works in both modes |
| 1-6 | Your action, in the order of the buttons (only on your turn) |
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

`pokerbots.web` (`src/pokerbots/web/__main__.py`) starts a `Session` that waits
for `POST /start` with `{"mode": "watch"}` or `{"mode": "play", "name": ...}`.
It then runs the game in a background thread with an observer: watch runs
`Tournament(config, observer=...)`; play runs `Sandbox` (`src/pokerbots/web/sandbox.py`,
a `Tournament` subclass with its own hand loop and `log_file=None`). The
observer receives every log record plus two viewer-only records: `round_start`
(every player's hole cards) and `action` (the acting bot's observation, choice
and `explain()`). The server streams them to the page over server-sent events
at `/events`, after a first `hello` record that tells the page to show the
start popup until a `meta` record (with `mode`) arrives.

Play mode reuses the tournament's record types so the page renders it
unchanged: `rotation_id` is the hand number, `round_id` is 0, a `rotation`
record after every hand holds the stacks and who busted, and a `join` record
marks a human seated mid-game.

Human players (`src/pokerbots/web/humans.py`): the page posts to `/human/join`,
`/human/act` and `/human/leave`. Each human is a `HumanSeat` bot; the sandbox
gets them through `Sandbox(..., extra_players=lobby.players)` and seats anyone
new before every hand. `HumanSeat.act()` publishes a `prompt` record (legal
actions) and blocks the game thread until the page answers. A `seat` record
tells the page which player ID each human has in each game. The server adds
`hands` (hand name and best five cards per player, `src/pokerbots/web/hands.py`)
to showdown `round` records it streams.
