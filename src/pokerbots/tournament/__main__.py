"""Run a tournament from the command line.

    python -m pokerbots.tournament                      # default config, logs to roundlogs/<date>_<time>.jsonl
    python -m pokerbots.tournament --mode dev --games 2 --rotations 20

Edit DEFAULT_CONFIG below to change which bots sit at the table.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from datetime import datetime
from pathlib import Path

from pokerbots.bots.simple import RandomBot
from pokerbots.tournament import Tournament, TournamentConfig, overall_leaderboard

# Repo root (src/pokerbots/tournament/__main__.py -> up 3), so logs land in
# <root>/roundlogs/ wherever the command is run from.
ROUNDLOGS_DIR = Path(__file__).resolve().parents[3] / "roundlogs"

DEFAULT_CONFIG = TournamentConfig(
    main_bots=(),  # add CrispBot, FuzzyBot, MLBot here as they are ready
    simple_bots=(RandomBot,),
)


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
        log_file = args.log_file or ROUNDLOGS_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.jsonl"
        config = dataclasses.replace(config, log_file=log_file)

    start = time.perf_counter()
    results = Tournament(config).run()
    elapsed = time.perf_counter() - start
    print_results(results)
    total = sum(g.rotations_played for g in results)
    print(f"\n{len(results)} games, {total} rotations in {elapsed:.2f} s"
          f" ({elapsed / max(total, 1) * 1000:.1f} ms per rotation)")
    if config.log_file:
        print(f"Logs: {config.log_file}")


if __name__ == "__main__":
    main()
