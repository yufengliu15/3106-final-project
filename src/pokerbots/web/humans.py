"""Human players for the live viewer.

A HumanSeat is a Bot whose act() blocks the tournament thread until the
browser posts a choice (POST /human/act). The Lobby holds everyone who joined;
the tournament asks it for extra players at the start of every game and
rotation (Tournament(extra_players=lobby.players)), so a human who joins
mid-game is dealt in at the next rotation.

Records published to the viewer's feed (never to the match log):
    {"type": "seat", "human_id", "game_id", "player_id"}  -- seated in a game
    {"type": "prompt", "human_id", "prompt_id", "game_id", "rotation_id",
     "round_id", "player_id", "legal"}                     -- waiting for a choice
"""

from __future__ import annotations

import secrets
import threading
from typing import Callable

from pokerbots.core import CHECK, FOLD, Bot, Observation, RoundLog

MAX_TABLE = 23
"""RLCard deals 2 cards per player plus 5 board cards from one deck: 2 * 23 + 5 = 51."""
MAX_NAME = 24


class HumanSeat(Bot):
    def __init__(self, human_id: int, name: str, publish: Callable[[dict], None]) -> None:
        self.human_id = human_id
        self.name = name
        self.token = secrets.token_hex(16)
        self._publish = publish
        self._cond = threading.Condition()
        self._pending: tuple[int, dict[str, float]] | None = None  # (prompt_id, legal actions)
        self._choice: str | None = None
        self._prompts = 0
        self._explain = ""
        self.left = False

    def new_game(self, game_id: int, player_id: int) -> None:
        self._publish({"type": "seat", "human_id": self.human_id, "game_id": game_id, "player_id": player_id})

    def act(self, obs: Observation) -> str:
        legal = dict(obs.legal_actions)
        with self._cond:
            if self.left:
                return self._auto(legal)
            self._prompts += 1
            self._pending = (self._prompts, legal)
            self._choice = None
            prompt_id = self._prompts
        self._publish({"type": "prompt", "human_id": self.human_id, "prompt_id": prompt_id,
                       "game_id": obs.game_id, "rotation_id": obs.rotation_id, "round_id": obs.round_id,
                       "player_id": obs.player_id, "legal": [list(a) for a in obs.legal_actions]})
        with self._cond:
            while self._choice is None and not self.left:
                self._cond.wait()
            choice, self._choice, self._pending = self._choice, None, None
        if choice is None:
            return self._auto(legal)
        self._explain = f"{self.name} chose this"
        return choice

    def choose(self, prompt_id: object, action: str) -> str | None:
        """Answer the pending prompt (HTTP thread). Returns an error, or None if accepted."""
        with self._cond:
            if self._pending is None or self._pending[0] != prompt_id:
                return "not your turn: that decision is no longer open"
            if action not in self._pending[1]:
                return f"{action!r} is not a legal action now"
            self._choice = action
            self._cond.notify_all()
        return None

    def leave(self) -> None:
        """Check or fold every remaining decision; also answers an open prompt."""
        with self._cond:
            self.left = True
            self._cond.notify_all()

    def _auto(self, legal: dict[str, float]) -> str:
        self._explain = f"{self.name} left the table: auto check/fold"
        return CHECK if CHECK in legal else FOLD

    def explain(self) -> str:
        return self._explain

    def observe_result(self, log: RoundLog) -> None:
        pass


class Lobby:
    """Every human who joined this run, by token. Thread-safe."""

    def __init__(self, publish: Callable[[dict], None], bot_seats: int) -> None:
        self._publish = publish
        self._max_humans = MAX_TABLE - bot_seats
        self._lock = threading.Lock()
        self._seats: dict[str, HumanSeat] = {}  # by token

    def join(self, name: str) -> HumanSeat:
        """Add a human; seated at the next game or rotation start. Raises ValueError when full."""
        name = " ".join(name.split())[:MAX_NAME] or "Player"
        with self._lock:
            if sum(not s.left for s in self._seats.values()) >= self._max_humans:
                raise ValueError(f"the table is full ({MAX_TABLE} seats at most)")
            taken = {s.name for s in self._seats.values()}
            label, n = f"{name} (human)", 2
            while label in taken:
                label, n = f"{name} {n} (human)", n + 1
            seat = HumanSeat(len(self._seats) + 1, label, self._publish)
            self._seats[seat.token] = seat
        return seat

    def get(self, token: str) -> HumanSeat | None:
        with self._lock:
            return self._seats.get(token)

    def players(self) -> list[tuple[str, Bot]]:
        """Humans still playing, as Tournament extra players."""
        with self._lock:
            return [(s.name, s) for s in self._seats.values() if not s.left]
