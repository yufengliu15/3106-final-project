"""Play mode for the live viewer: humans against the tournament bots.

Plain hold'em, not the tournament format (docs/tournament.md):
    - every hand is dealt from a new seed (no rotations, no repeated deals);
    - the button moves one seat each hand and stacks carry over;
    - a player on 0 chips is out at once;
    - a game ends when one player is left or after config.rotations hands.

Humans (or any extra players) are seated at the start of the next hand after
they join. Nothing is written to disk: run it with config.log_file = None.

Records reuse the tournament's types so the page renders them unchanged:
rotation_id = hand number, round_id = 0, and a "rotation" record after every
hand holds the stacks and who busted.
"""

from __future__ import annotations

from typing import Callable, Sequence

from pokerbots.core import Bot
from pokerbots.sim import CHIPS_PER_BB, Simulator
from pokerbots.tournament.tournament import SEED_RANGE, GameResult, Tournament

ExtraPlayers = Callable[[], Sequence[tuple[str, Bot]]]


class Sandbox(Tournament):
    def __init__(self, config, observer: Callable[[dict], None] | None = None,
                 extra_players: ExtraPlayers | None = None) -> None:
        if config.log_file is not None:
            raise ValueError("play mode writes no logs; set log_file=None")
        super().__init__(config, observer)
        self.extra_players = extra_players

    def play_game(self, game_id: int) -> GameResult:
        cfg = self.config
        ids, bots, names, live = self._seat_bots(game_id)
        stacks = {pid: round(cfg.starting_stack * CHIPS_PER_BB) for pid in ids}
        ids += self._seat_extras(game_id, bots, names, stacks, live)
        eliminated_in: dict[int, int] = {}
        self._emit({"type": "game", "game_id": game_id, "players": names, "seat_order": list(live), "mode": "sandbox"})

        hand = 0
        while hand < cfg.rotations:
            for pid in self._seat_extras(game_id, bots, names, stacks, live):
                ids.append(pid)
                self._emit({"type": "join", "game_id": game_id, "rotation_id": hand + 1,
                            "player_id": pid, "name": names[pid], "stack": cfg.starting_stack})
            if len(live) < 2:
                break
            hand += 1
            button = (hand - 1) % len(live)
            positions = live[button:] + live[:button]  # index 0 posts the small blind
            seed = self.rng.randrange(SEED_RANGE)
            sim = Simulator(bots, positions, [stacks[p] / CHIPS_PER_BB for p in positions], seed, game_id, hand, 0)
            log = self._play_hand(sim, bots)
            for pid in positions:
                stacks[pid] += round(log.chip_deltas[pid] * CHIPS_PER_BB)
            busted = [pid for pid in positions if stacks[pid] <= 0]
            for pid in busted:
                eliminated_in[pid] = hand
            live = [pid for pid in live if pid not in busted]
            self._emit({"type": "rotation", "game_id": game_id, "rotation_id": hand, "seed": seed,
                        "stacks": {p: stacks[p] / CHIPS_PER_BB for p in positions}, "eliminated": busted})
        return self._finish_game(game_id, ids, names, stacks, eliminated_in, hand)

    def _seat_extras(self, game_id, bots, names, stacks, live) -> list[int]:
        """Seat every extra player not yet at this game's table (busted players
        count as at the table) at the end of the seat order. Returns their IDs."""
        if self.extra_players is None:
            return []
        seated = []
        for name, bot in self.extra_players():
            if any(b is bot for b in bots.values()):
                continue
            pid = self.rng.randrange(1000, 10000)
            while pid in bots:
                pid = self.rng.randrange(1000, 10000)
            bots[pid] = bot
            names[pid] = name
            stacks[pid] = round(self.config.starting_stack * CHIPS_PER_BB)
            live.append(pid)
            bot.new_game(game_id, pid)
            seated.append(pid)
        return seated
