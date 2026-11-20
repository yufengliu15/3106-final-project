"""AlwaysAllInBot (#8): goes all-in every time it can act.

all_in is offered whenever the player has chips to put in. In the rare spot
where it is not offered (nothing left to bet), it checks, or calls.
"""

from __future__ import annotations

from pokerbots.core import ALL_IN, CALL, CHECK, Bot, Observation, RoundLog


class AlwaysAllInBot(Bot):
    def __init__(self) -> None:
        self._last = ALL_IN

    def new_game(self, game_id: int, player_id: int) -> None:
        pass

    def act(self, obs: Observation) -> str:
        legal = {a for a, _ in obs.legal_actions}
        if ALL_IN in legal:
            self._last = ALL_IN
        elif CHECK in legal:
            self._last = CHECK
        elif CALL in legal:
            self._last = CALL
        else:
            self._last = obs.legal_actions[0][0]
        return self._last

    def explain(self) -> str:
        return f"AlwaysAllInBot: {self._last} (goes all-in whenever it can)"

    def observe_result(self, log: RoundLog) -> None:
        pass
