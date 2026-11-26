"""Run a tournament from the command line.

    python -m pokerbots.tournament                      # default config, logs to roundlogs/<date>_<time>.jsonl
    python -m pokerbots.tournament --mode dev --games 2 --rotations 20

Every finished run adds its overall standings to the top of LEADERBOARD.md.

Edit DEFAULT_CONFIG below to change which bots sit at the table.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from datetime import datetime
from pathlib import Path

from pokerbots.bots.crisp import CrispBot
from pokerbots.bots.fuzzy import FuzzyBot
from pokerbots.bots.ml.bot_c import MLBot
from pokerbots.bots.simple import SCRIPTED_POOL
from pokerbots.tournament import Tournament, TournamentConfig, overall_leaderboard

# Repo root (src/pokerbots/tournament/__main__.py -> up 3), so logs land in
# <root>/roundlogs/ wherever the command is run from.
ROUNDLOGS_DIR = Path(__file__).resolve().parents[3] / "roundlogs"
LEADERBOARD_PATH = ROUNDLOGS_DIR.parent / "LEADERBOARD.md"
LEADERBOARD_TITLE = "# Leaderboard\n\nNewest run first. Written by `pokerbots.tournament` and `pokerbots.web`.\n"


def default_log_path(explicit: Path | None = None) -> Path:
    """The --log-file path, or a new timestamped file in roundlogs/."""
    return explicit or ROUNDLOGS_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.jsonl"


DEFAULT_CONFIG = TournamentConfig(
    main_bots=(CrispBot, FuzzyBot, MLBot),
    simple_bots=SCRIPTED_POOL,
)


def _bb(x: float) -> str:
    """Same rounding as the viewer's fmt(): to 0.01 first, then no decimals from 100 bb up, else one."""
    x = round(x, 2)
    return f"{x:.0f}" if abs(x) >= 100 else f"{x:.1f}"


def _repo_relative(path: Path) -> str:
    path = Path(path).resolve()
    root = ROUNDLOGS_DIR.parent
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def prepend_leaderboard(results, config: TournamentConfig, path: Path = LEADERBOARD_PATH) -> None:
    """Put this run's overall standings at the top of LEADERBOARD.md."""
    when = f"{datetime.now():%Y-%m-%d %H:%M}"
    lines = [
        f"## Overall: {len(results)} game(s), {when}",
        "",
        f"Rotations {config.rotations} · seats {config.seats} · stack {config.starting_stack:g} bb"
        f" · k {config.k:g} · mode {config.mode} · seed {config.base_seed}"
        + (f" · log `{_repo_relative(config.log_file)}`" if config.log_file else ""),
        "",
        "| # | Bot | Seats | Mean stack | Mean elim rot |",
        "|---:|---|---:|---:|---:|",
    ]
    for place, row in enumerate(overall_leaderboard(results), 1):
        lines.append(f"| {place} | {row.bot} | {row.appearances} | {_bb(row.mean_final_stack)}"
                     f" | {row.mean_eliminated_in:.1f} |")
    old = path.read_text() if path.exists() else ""
    old = old.removeprefix(LEADERBOARD_TITLE).lstrip("\n")
    path.write_text(LEADERBOARD_TITLE + "\n" + "\n".join(lines) + "\n" + ("\n" + old if old else ""))


def print_results(results) -> None:
    for game in results:
        print(f"\nGame {game.game_id} ({game.rotations_played} rotations)")
        for place, p in enumerate(game.standings, 1):
            status = "alive" if p.eliminated_in is None else f"out after rotation {p.eliminated_in}"
            print(f"  {place}. {p.bot:<14} #{p.player_id}  {p.final_stack:>10.2f} bb  {status}")
    print("\nOverall")
    for place, row in enumerate(overall_leaderboard(results), 1):
        print(f"  {place}. {row.bot:<14} seats={row.appearances:<3} mean stack={row.mean_final_stack:>10.2f} bb"
              f"  mean elim rotation={row.mean_eliminated_in:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int)
    parser.add_argument("--rotations", type=int)
    parser.add_argument("--seats", type=int)
    parser.add_argument("--stack", type=float, help="starting stack in bb")
    parser.add_argument("--k", type=float)
    parser.add_argument("--mode", choices=["dev", "eval"])
    parser.add_argument("--seed", type=int, help="base seed")
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--no-logs", action="store_true")
    args = parser.parse_args()

    overrides = {
        "games": args.games, "rotations": args.rotations, "seats": args.seats,
        "starting_stack": args.stack, "k": args.k, "mode": args.mode, "base_seed": args.seed,
    }
    config = dataclasses.replace(DEFAULT_CONFIG, **{k: v for k, v in overrides.items() if v is not None})

    if not args.no_logs:
        config = dataclasses.replace(config, log_file=default_log_path(args.log_file))

    start = time.perf_counter()
    results = Tournament(config).run()
    elapsed = time.perf_counter() - start
    print_results(results)
    total = sum(g.rotations_played for g in results)
    print(f"\n{len(results)} games, {total} rotations in {elapsed:.2f} s"
          f" ({elapsed / max(total, 1) * 1000:.1f} ms per rotation)")
    if config.log_file:
        print(f"Logs: {config.log_file}")
    prepend_leaderboard(results, config)
    print(f"Leaderboard: {LEADERBOARD_PATH}")


if __name__ == "__main__":
    main()
