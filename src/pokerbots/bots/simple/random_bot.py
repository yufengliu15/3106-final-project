"""RandomBot: picks a uniformly random legal action.

Seeded from (game_id, player_id) in new_game(), so a tournament with the same
config and base seed plays identically.
"""

from __future__ import annotations

import random

from pokerbots.core import Bot, Observation, RoundLog


class RandomBot(Bot):
    def __init__(self) -> None:
        self.rng = random.Random(0)

    def new_game(self, game_id: int, player_id: int) -> None:
        self.rng = random.Random(game_id * 10_000 + player_id)

    def act(self, obs: Observation) -> str:
        return self.rng.choice(obs.legal_actions)[0]

    def explain(self) -> str:
        return "Picked a random legal action."

    def observe_result(self, log: RoundLog) -> None:
        pass
