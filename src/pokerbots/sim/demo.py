"""Terminal demo: play no-limit hold'em by hand or against the bots.

Run:
    python -m pokerbots.sim.demo            # seed 0
    python -m pokerbots.sim.demo --vs-bots --seed 42

On every turn the full Observation for the player to act is printed, then you
pick one of its legal actions (by number or by name). After each round the
RoundLog is printed. Every round starts at 100 bb per player; positions rotate
by one each round, like a rotation.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from pokerbots.core import Bot, Observation, RoundLog
from pokerbots.sim import IllegalActionError, Simulator

NUM_PLAYERS = 4
STARTING_STACK = 100.0
PLAYER_IDS = [1001, 2002, 3003, 4004]


def show(title: str, obj) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))
    for field in dataclasses.fields(obj):
        print(f"  {field.name:16} = {getattr(obj, field.name)!r}")


class HumanBot(Bot):
    """Prints the observation and asks the terminal for an action."""

    def new_game(self, game_id: int, player_id: int) -> None:
        pass

    def act(self, obs: Observation) -> str:
        seat = obs.positions.index(obs.player_id)
        show(f"Observation: player {obs.player_id} (index {seat})", obs)
        names = [name for name, _ in obs.legal_actions]
        print()
        for i, (name, cost) in enumerate(obs.legal_actions, start=1):
            print(f"  [{i}] {name:15} cost {cost:g} bb")
        while True:
            choice = input(f"player {obs.player_id} > ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(names):
                return names[int(choice) - 1]
            if choice in names:
                return choice
            if choice in ("q", "quit", "exit"):
                raise SystemExit(0)
            # Anything else is passed through so you can test illegal actions.
            if choice:
                return choice

    def explain(self) -> str:
        return "human"

    def observe_result(self, log: RoundLog) -> None:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0, help="seed of the first round")
    parser.add_argument("--vs-bots", action="store_true", help="play one seat against Bot B, Bot C, and CallingMachine")
    args = parser.parse_args(argv)

    human = HumanBot()
    bots: dict[int, Bot] = dict.fromkeys(PLAYER_IDS, human)
    if args.vs_bots:
        from pokerbots.bots.fuzzy import FuzzyBot
        from pokerbots.bots.ml.bot_c import MLBot
        from pokerbots.bots.simple import CallingMachineBot

        bots = dict(zip(PLAYER_IDS, (human, FuzzyBot(), MLBot(), CallingMachineBot()), strict=True))
    opponents = {pid: bot for pid, bot in bots.items() if bot is not human}
    for pid, bot in opponents.items():
        bot.new_game(0, pid)
    positions = list(PLAYER_IDS)
    print("Pick an action by number or name. Type anything else to test an illegal action; 'q' quits.")

    round_id = 0
    while True:
        print(f"\n##### Round {round_id}  seed {args.seed + round_id}  positions {positions} #####")
        sim = Simulator(
            bots,
            positions,
            [STARTING_STACK] * NUM_PLAYERS,
            seed=args.seed + round_id,
            game_id=0,
            rotation_id=0,
            round_id=round_id,
        )
        try:
            log = sim.play()
        except IllegalActionError as err:
            print(f"\nIllegalActionError: {err}")
            print("(In a real run this ends the game.)")
            return 1
        show("RoundLog", log)
        actors = {pid for pid, _, _, _ in log.actions}
        for pid, bot in opponents.items():
            if pid in actors:
                print(f"{type(bot).__name__} ({pid}), last decision: {bot.explain()}")
            bot.observe_result(log)

        if input("\nAnother round? [Y/n] ").strip().lower() in ("n", "no", "q"):
            return 0
        positions = positions[1:] + positions[:1]
        round_id += 1


if __name__ == "__main__":
    sys.exit(main())
