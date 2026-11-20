"""Scripted opponents from proposal Section 4.

"Plays the top x% of starting hands" = hole cards in equity.top_range(x).
A playable hand is played all the way: aggressive bots raise whenever they
can, passive bots check/call. Unplayable hands check when free, else fold.
Decisions depend only on the observation (no hidden state), except for the
Switcher's current style.
"""

from __future__ import annotations

import random

from pokerbots.core import (
    ALL_IN, CALL, CHECK, FOLD, RAISE_HALF_POT, RAISE_POT, Bot, Observation, RoundLog,
)
from pokerbots.bots.simple.always_all_in import AlwaysAllInBot
from pokerbots.core.cards import hand_class
from pokerbots.core.equity import top_range


class ScriptedBot(Bot):
    range_pct: float = 1.0
    aggressive: bool = False
    name = "Scripted"

    def __init__(self) -> None:
        self._last = ""

    def new_game(self, game_id: int, player_id: int) -> None:
        pass

    def playable(self, obs: Observation) -> bool:
        return hand_class(obs.hole_cards) in top_range(self.range_pct)

    def act(self, obs: Observation) -> str:
        legal = dict(obs.legal_actions)
        if not self.playable(obs):
            choice = CHECK if CHECK in legal else FOLD
        elif self.aggressive:
            raise_pref = (RAISE_POT, RAISE_HALF_POT) if not obs.community_cards else (RAISE_HALF_POT, RAISE_POT)
            choice = next((a for a in (*raise_pref, ALL_IN) if a in legal), CHECK)
        else:
            choice = next(a for a in (CHECK, CALL, ALL_IN) if a in legal)
        self._last = choice
        return choice

    def explain(self) -> str:
        return f"{self.name}: {self._last}"

    def observe_result(self, log: RoundLog) -> None:
        pass


class TightAggressiveBot(ScriptedBot):
    range_pct, aggressive, name = 0.15, True, "TAG"


class TightPassiveBot(ScriptedBot):
    range_pct, aggressive, name = 0.10, False, "TP"


class LooseAggressiveBot(ScriptedBot):
    range_pct, aggressive, name = 0.40, True, "LAG"


class CallingMachineBot(ScriptedBot):
    range_pct, aggressive, name = 0.50, False, "CallingMachine"


class SwitcherBot(Bot):
    """Switches between the five non-random styles every `period` rounds."""

    STYLES = (TightAggressiveBot, TightPassiveBot, LooseAggressiveBot,
              CallingMachineBot, AlwaysAllInBot)

    def __init__(self, period: int = 50) -> None:
        self.period = period
        self.rng = random.Random(0)
        self.rounds = 0
        self.style: Bot = self.STYLES[0]()

    def new_game(self, game_id: int, player_id: int) -> None:
        self.rng = random.Random(game_id * 10_000 + player_id)
        self.rounds = 0
        self.style = self.rng.choice(self.STYLES)()

    def act(self, obs: Observation) -> str:
        return self.style.act(obs)

    def explain(self) -> str:
        return f"Switcher as {self.style.explain()}"

    def observe_result(self, log: RoundLog) -> None:
        self.rounds += 1
        if self.rounds % self.period == 0:
            options = [s for s in self.STYLES if not isinstance(self.style, s)]
            self.style = self.rng.choice(options)()
