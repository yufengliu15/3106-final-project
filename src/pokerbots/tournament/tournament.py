"""Runs games and keeps score.

Loop: game -> rotation -> round. The simulator plays one round; everything
that carries between rounds (stacks, seating, seeds, eliminations) lives here.

Rotation rules (docs/tournament.md):
    - x = live players. The rotation has x rounds, all from the same card seed
      and the same starting stacks.
    - Player i (in seat order) sits at index (i + r) % x in round r, so every
      player plays every position exactly once on the same deal.
    - Next starting stack = truncated average of the x ending stacks, where an
      ending stack of 0 counts as -k * S (S = this rotation's starting stack).
    - Average <= 0 -> eliminated for the rest of the game.

All randomness (player IDs, simple-bot draws, seating, eval-mode card seeds)
comes from one random.Random(base_seed), drawn in a fixed order, so the same
config always gives the same results.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO

from pokerbots.core import Bot, RoundLog
from pokerbots.sim import CHIPS_PER_BB, Simulator
from pokerbots.tournament.config import TournamentConfig

SEED_RANGE = 2**31


@dataclass(frozen=True)
class PlayerResult:
    player_id: int
    bot: str
    """Bot class name."""
    final_stack: float
    """bb at the end of the game; 0 if eliminated."""
    eliminated_in: int | None
    """Rotation the player was eliminated after; None if still alive."""


@dataclass(frozen=True)
class GameResult:
    game_id: int
    rotations_played: int
    standings: tuple[PlayerResult, ...]
    """Best first: by final stack, then later elimination (alive beats any)."""


@dataclass(frozen=True)
class OverallRow:
    bot: str
    appearances: int
    mean_final_stack: float
    mean_eliminated_in: float
    """Survivors count as rotations_played + 1 for that game."""


def rotation_average(start_chips: int, ending_chips: list[int], k: float) -> int:
    """Next starting stack in chips. A bust (0) counts as -k * start."""
    values = [e if e > 0 else -k * start_chips for e in ending_chips]
    return math.trunc(sum(values) / len(values))


def rank(players: list[PlayerResult], rotations_played: int) -> tuple[PlayerResult, ...]:
    def key(p: PlayerResult):
        survived_until = p.eliminated_in if p.eliminated_in is not None else rotations_played + 1
        return (-p.final_stack, -survived_until, p.player_id)

    return tuple(sorted(players, key=key))


def overall_leaderboard(results: list[GameResult]) -> tuple[OverallRow, ...]:
    stacks: dict[str, list[float]] = defaultdict(list)
    elims: dict[str, list[int]] = defaultdict(list)
    for game in results:
        for p in game.standings:
            stacks[p.bot].append(p.final_stack)
            elims[p.bot].append(p.eliminated_in if p.eliminated_in is not None else game.rotations_played + 1)
    rows = [
        OverallRow(bot, len(stacks[bot]), sum(stacks[bot]) / len(stacks[bot]), sum(elims[bot]) / len(elims[bot]))
        for bot in stacks
    ]
    return tuple(sorted(rows, key=lambda r: (-r.mean_final_stack, -r.mean_eliminated_in, r.bot)))


class Tournament:
    def __init__(self, config: TournamentConfig) -> None:
        config.validate()
        self.config = config
        self.rng = random.Random(config.base_seed)
        self.results: list[GameResult] = []
        self._out: IO[str] | None = None

    def run(self) -> list[GameResult]:
        """Play every game, writing all of them to config.log_file (if set)."""
        self._out = self._open_log()
        try:
            for game_id in range(1, self.config.games + 1):
                self.results.append(self.play_game(game_id))
        finally:
            if self._out is not None:
                self._out.close()
                self._out = None
        return self.results

    def play_game(self, game_id: int) -> GameResult:
        cfg, rng = self.config, self.rng

        # --- Seating: main bots + random simple bots, fresh IDs, random order.
        classes = list(cfg.main_bots) + [
            rng.choice(cfg.simple_bots) for _ in range(cfg.seats - len(cfg.main_bots))
        ]
        ids = rng.sample(range(1000, 10000), cfg.seats)
        bots: dict[int, Bot] = {pid: cls() for pid, cls in zip(ids, classes)}
        names = {pid: type(bot).__name__ for pid, bot in bots.items()}
        seat_order = list(ids)
        rng.shuffle(seat_order)
        for pid, bot in bots.items():
            bot.new_game(game_id, pid)

        stacks = {pid: round(cfg.starting_stack * CHIPS_PER_BB) for pid in ids}
        live = seat_order
        eliminated_in: dict[int, int] = {}
        rotations_played = 0

        out = self._out
        self._write(out, {"type": "game", "game_id": game_id, "players": names,
                          "seat_order": seat_order, "mode": cfg.mode})
        for rotation_id in range(1, cfg.rotations + 1):
            if len(live) < 2:
                break
            seed = self._card_seed(game_id, rotation_id)
            endings = self._play_rotation(bots, live, stacks, seed, game_id, rotation_id, out)
            rotations_played = rotation_id

            still_live = []
            for pid in live:
                avg = rotation_average(stacks[pid], endings[pid], cfg.k)
                if avg <= 0:
                    stacks[pid] = 0
                    eliminated_in[pid] = rotation_id
                else:
                    stacks[pid] = avg
                    still_live.append(pid)
            self._write(out, {"type": "rotation", "game_id": game_id, "rotation_id": rotation_id,
                              "seed": seed, "stacks": {p: stacks[p] / CHIPS_PER_BB for p in live},
                              "eliminated": [p for p in live if p not in still_live]})
            live = still_live

        standings = rank(
            [PlayerResult(pid, names[pid], stacks[pid] / CHIPS_PER_BB, eliminated_in.get(pid)) for pid in ids],
            rotations_played,
        )
        result = GameResult(game_id, rotations_played, standings)
        self._write(out, {"type": "result", **asdict(result)})
        return result

    # --- Internals -------------------------------------------------------------

    def _play_rotation(self, bots, live, stacks, seed, game_id, rotation_id, out) -> dict[int, list[int]]:
        x = len(live)
        endings: dict[int, list[int]] = {pid: [] for pid in live}
        for round_id in range(x):
            positions = [0] * x
            for i, pid in enumerate(live):
                positions[(i + round_id) % x] = pid
            start_bb = [stacks[pid] / CHIPS_PER_BB for pid in positions]
            log = Simulator(bots, positions, start_bb, seed, game_id, rotation_id, round_id).play()
            self._write(out, {"type": "round", **asdict(log)})
            for bot in bots.values():
                bot.observe_result(log)
            for pid in positions:
                endings[pid].append(stacks[pid] + round(log.chip_deltas[pid] * CHIPS_PER_BB))
        return endings

    def _card_seed(self, game_id: int, rotation_id: int) -> int:
        if self.config.mode == "dev":
            return game_id * 1000 + rotation_id
        return self.rng.randrange(SEED_RANGE)

    def _open_log(self) -> IO[str] | None:
        if self.config.log_file is None:
            return None
        path = Path(self.config.log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        return open(path, "w")

    @staticmethod
    def _write(out: IO[str] | None, record: dict) -> None:
        if out is not None:
            out.write(json.dumps(record) + "\n")
