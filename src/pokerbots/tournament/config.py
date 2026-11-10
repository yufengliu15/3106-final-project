"""Tournament configuration.

A config is a plain Python object that names bot classes directly:

    from pokerbots.bots.simple import RandomBot
    from pokerbots.tournament import TournamentConfig

    config = TournamentConfig(main_bots=(CrispBot, FuzzyBot), simple_bots=(RandomBot,))

Seating each game: every class in `main_bots` gets one seat; the remaining
seats are filled with classes drawn at random (with replacement) from
`simple_bots`. Every bot class must have a no-argument constructor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pokerbots.bots.simple import RandomBot
from pokerbots.core import Bot

Mode = Literal["dev", "eval"]

# Dev seeds are game_id * 1000 + rotation_id, so both must stay below 1000.
MAX_ID = 999


@dataclass(frozen=True)
class TournamentConfig:
    main_bots: tuple[type[Bot], ...] = ()
    """Always seated, one seat each. Add the crisp / fuzzy / ML bots here as
    they become ready."""

    simple_bots: tuple[type[Bot], ...] = (RandomBot,)
    """Pool that randomly fills the seats left after the main bots."""

    seats: int = 10
    """Players at the table at the start of every game."""

    games: int = 5

    rotations: int = 100
    """Maximum rotations per game; a game also ends when one player is left."""

    starting_stack: float = 100.0
    """Stack in bb every player starts each game with."""

    k: float = 1.0
    """Bust penalty: a round ending on 0 counts as -k * S in the rotation average."""

    mode: Mode = "eval"
    """dev: card seed = game_id * 1000 + rotation_id (same cards every run).
    eval: every card seed is drawn from base_seed."""

    base_seed: int = 0
    """Seeds player IDs, simple-bot draws, seating order, and (eval) card seeds."""

    log_file: Path | None = None
    """Write the whole run here, one JSON object per line (JSONL); parent
    folders are created. None = don't write logs. The CLI uses
    roundlogs/<date>_<time>.jsonl."""

    def validate(self) -> None:
        if self.seats < 2:
            raise ValueError("need at least 2 seats")
        if self.seats > 9000:
            raise ValueError("at most 9000 seats (player IDs are 1000-9999)")
        if len(self.main_bots) > self.seats:
            raise ValueError(f"{len(self.main_bots)} main bots but only {self.seats} seats")
        if len(self.main_bots) < self.seats and not self.simple_bots:
            raise ValueError("simple_bots is empty but there are seats left to fill")
        if not 1 <= self.games <= MAX_ID:
            raise ValueError(f"games must be 1-{MAX_ID}")
        if not 1 <= self.rotations <= MAX_ID:
            raise ValueError(f"rotations must be 1-{MAX_ID}")
        if self.starting_stack <= 0:
            raise ValueError("starting_stack must be positive")
        if self.k < 0:
            raise ValueError("k must be >= 0")
        if self.mode not in ("dev", "eval"):
            raise ValueError("mode must be 'dev' or 'eval'")
