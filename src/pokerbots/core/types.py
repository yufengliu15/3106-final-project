"""Shared formats every component uses to talk to each other.

Units of play:
    Game     -- thousands of rotations; player IDs stay fixed.
    Rotation -- 8 rounds from the same seed; each player plays every position once.
    Round    -- one hand, from dealing the hole cards until the pot is awarded.

Cards use RLCard's native format (e.g. "H4", "SK", "CT"). Chip amounts are in big blinds (bb).

Observation and RoundLog are frozen snapshots: the simulator builds a new
one for every decision / round, and bots must never modify them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

# --- Action names -----------------------------------------------------------
# Bots return one of these strings from act(). Use the constants, not literals.

FOLD = "fold"
CHECK = "check"
CALL = "call"
RAISE_HALF_POT = "raise_half_pot"
RAISE_POT = "raise_pot"
ALL_IN = "all_in"

ACTIONS = (FOLD, CHECK, CALL, RAISE_HALF_POT, RAISE_POT, ALL_IN)

# --- Streets ----------------------------------------------------------------

Street = Literal["preflop", "flop", "turn", "river"]
STREETS: tuple[Street, ...] = ("preflop", "flop", "turn", "river")


@dataclass(frozen=True)
class Observation:
    """Everything a bot sees when it must act. Built by the simulator."""

    hole_cards: tuple[str, str]
    """This bot's two private cards, e.g. ("C7", "HQ")."""

    community_cards: tuple[str, ...]
    """Board cards dealt so far; length 0, 3, 4 or 5."""

    pot: float
    """Total chips in the pot, in bb."""

    player_id: int
    """This bot's ID: random 1000-9999, fixed for the whole game."""

    positions: tuple[int, ...]
    """The 8 player IDs by seat: index 0 = small blind ... index 7 = button."""

    stacks: tuple[float | None, ...]
    """Chips remaining in bb; stacks[i] belongs to positions[i].
    0 means all-in, None means busted."""

    action_history: tuple[tuple[int, str, float], ...]
    """Actions so far in this round only, as (player_id, action, amount in bb)."""

    legal_actions: tuple[tuple[str, float], ...]
    """Actions this bot may take now, as (action, cost in bb). Cost is the
    additional chips the action puts in, not the total bet.
    - If there is nothing to call, CHECK is offered instead of CALL and FOLD
      is not offered.
    - If calling would cost the whole stack, only ALL_IN is offered, not CALL."""

    game_id: int
    """ID of the current game."""

    rotation_id: int
    """ID of the current rotation within the game."""

    round_id: int
    """Round index within the rotation, 0-7."""


@dataclass(frozen=True)
class RoundLog:
    """Full record of one finished round. Passed to every bot after every
    round (including rounds it folded), and written to JSON by the tournament.

    Note: frozen stops fields being reassigned, but the dict fields can still
    be mutated in place -- bots must treat them as read-only.
    JSON turns int dict keys into strings; convert back when loading.
    """

    game_id: int
    """ID of the game this round belongs to."""

    rotation_id: int
    """ID of the rotation within the game."""

    round_id: int
    """Round index within the rotation, 0-7."""

    positions: tuple[int, ...]
    """The 8 player IDs by seat for this round: index 0 = small blind ... 7 = button."""

    starting_stacks: tuple[float | None, ...]
    """Stacks in bb at the start of the round; starting_stacks[i] belongs to
    positions[i]. None means busted."""

    actions: tuple[tuple[int, Street, str, float], ...]
    """Every action in the round, in order, as (player_id, street, action,
    amount in bb). street is needed to compute stats such as VPIP."""

    showdown_cards: dict[int, tuple[str, str]]
    """Hole cards of players who showed at showdown, keyed by player ID.
    Players who folded or did not show are absent."""

    chip_deltas: dict[int, float]
    """Chips won (positive) or lost (negative) this round in bb, keyed by player ID."""


class Bot(ABC):
    """Interface every bot implements. Bots depend only on pokerbots.core
    (and opponents), never on RLCard or the simulator."""

    @abstractmethod
    def new_game(self, game_id: int, player_id: int) -> None:
        """Start a new game: remember our ID and clear opponent memory."""

    @abstractmethod
    def act(self, obs: Observation) -> str:
        """Return the name of one action from obs.legal_actions."""

    @abstractmethod
    def explain(self) -> str:
        """Return a human-readable reason for the last action taken."""

    @abstractmethod
    def observe_result(self, log: RoundLog) -> None:
        """Called after EVERY round, including ones this bot folded."""
