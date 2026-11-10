"""Plays exactly one round (one hand) of no-limit hold'em on RLCard.

The tournament builds a new Simulator for every round and passes in everything
that carries between rounds (positions, stacks, seed, IDs). The simulator asks
each bot to act, translates actions to and from RLCard, and returns a RoundLog.

Design: docs/simulator.md

Units: bots and logs use big blinds (bb); RLCard uses whole chips with
1 bb = CHIPS_PER_BB chips. Conversion happens only in this module.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import rlcard
from rlcard.games.limitholdem import PlayerStatus
from rlcard.games.nolimitholdem.round import Action

from pokerbots.core import (
    ALL_IN,
    CALL,
    CHECK,
    FOLD,
    RAISE_HALF_POT,
    RAISE_POT,
    STREETS,
    Bot,
    Observation,
    RoundLog,
    Street,
)

CHIPS_PER_BB = 100
SMALL_BLIND_CHIPS = CHIPS_PER_BB // 2
BIG_BLIND_CHIPS = CHIPS_PER_BB


class IllegalActionError(Exception):
    """A bot returned an action that was not in its legal_actions.
    Not caught by the simulator: the run is meant to end."""


def bb_to_chips(bb: float) -> int:
    return int(round(bb * CHIPS_PER_BB))


def chips_to_bb(chips: float) -> float:
    return float(chips) / CHIPS_PER_BB


@dataclass(frozen=True)
class Snapshot:
    """Opaque saved state at a decision point (for #18).

    Holders keep it and pass it back to Simulator.from_snapshot(); they must
    not look inside -- it contains RLCard objects.
    """

    _sim: "Simulator"


class Simulator:
    """One round of play.

    Typical use (tournament):
        log = Simulator(bots, positions, stacks, seed, game_id, rotation_id, round_id).play()

    Step-by-step use (#18 rollouts):
        sim = Simulator(...)
        while not sim.is_over():
            obs = sim.observation()
            snap = sim.snapshot()          # whenever #18 wants one
            sim.step(bots[obs.player_id].act(obs))
        ...
        rollout = Simulator.from_snapshot(snap, bots, seed=k)
        log = rollout.play()
    """

    def __init__(
        self,
        bots: Mapping[int, Bot],
        positions: Sequence[int],
        stacks: Sequence[float],
        seed: int,
        game_id: int,
        rotation_id: int,
        round_id: int,
    ) -> None:
        if len(positions) < 2:
            raise ValueError("need at least 2 players")
        if len(positions) != len(stacks):
            raise ValueError("positions and stacks must have the same length")
        if len(set(positions)) != len(positions):
            raise ValueError("duplicate player IDs in positions")
        missing = [pid for pid in positions if pid not in bots]
        if missing:
            raise ValueError(f"no bot for player IDs {missing}")

        chips = [bb_to_chips(s) for s in stacks]
        if any(c <= 0 for c in chips):
            raise ValueError(f"every stack must be at least 1 chip, got {list(stacks)}")

        self.bots = bots
        self.positions: tuple[int, ...] = tuple(positions)
        self.starting_chips: tuple[int, ...] = tuple(chips)
        self.game_id = game_id
        self.rotation_id = rotation_id
        self.round_id = round_id
        self.seed = seed
        # (player_id, street, action, chips) for every action, in order
        self._actions: list[tuple[int, Street, str, int]] = []

        n = len(self.positions)
        env = rlcard.make("no-limit-holdem", config={"game_num_players": n, "seed": seed})
        game = env.game
        # init_game() (run by reset) reads these, so set them first.
        game.small_blind = SMALL_BLIND_CHIPS
        game.big_blind = BIG_BLIND_CHIPS
        game.init_chips = list(chips)
        game.dealer_id = n - 1  # last index = button, so blinds land on 0 and 1
        env.reset()
        self._game = game
        self._skip_all_in_players()

    # --- Queries -------------------------------------------------------------

    def is_over(self) -> bool:
        return bool(self._game.is_over())

    def current_player(self) -> int:
        return self.positions[self._game.game_pointer]

    def observation(self) -> Observation:
        """What the player to act sees right now."""
        self._require_not_over()
        game = self._game
        seat = game.game_pointer
        return Observation(
            hole_cards=tuple(c.get_index() for c in game.players[seat].hand),
            community_cards=tuple(c.get_index() for c in game.public_cards),
            pot=chips_to_bb(self._pot()),
            player_id=self.positions[seat],
            positions=self.positions,
            stacks=tuple(chips_to_bb(p.remained_chips) for p in game.players),
            action_history=tuple(
                (pid, action, chips_to_bb(amount)) for pid, _, action, amount in self._actions
            ),
            legal_actions=tuple(
                (name, chips_to_bb(cost)) for name, (cost, _) in self._legal().items()
            ),
            game_id=self.game_id,
            rotation_id=self.rotation_id,
            round_id=self.round_id,
        )

    # --- Playing ---------------------------------------------------------------

    def step(self, action: str) -> None:
        """Apply one action for the current player."""
        self._require_not_over()
        game = self._game
        seat = game.game_pointer
        pid = self.positions[seat]
        legal = self._legal()
        if action not in legal:
            raise IllegalActionError(
                f"player {pid} chose {action!r}; legal actions were "
                f"{ {name: chips_to_bb(cost) for name, (cost, _) in legal.items()} } "
                f"(game {self.game_id}, rotation {self.rotation_id}, round {self.round_id})"
            )

        cost, rl_action = legal[action]
        street = self._street()
        before = game.players[seat].in_chips
        game.step(rl_action)
        put_in = game.players[seat].in_chips - before
        assert put_in == cost, f"{action}: expected {cost} chips, RLCard took {put_in}"

        self._actions.append((pid, street, action, cost))
        self._skip_all_in_players()

    def play(self) -> RoundLog:
        """Ask the bots to act until the round ends, then return its log."""
        while not self.is_over():
            obs = self.observation()
            self.step(self.bots[obs.player_id].act(obs))
        return self.round_log()

    def round_log(self) -> RoundLog:
        if not self.is_over():
            raise RuntimeError("round is not finished")
        game = self._game
        payoffs = game.get_payoffs()
        live = [i for i, p in enumerate(game.players) if p.status != PlayerStatus.FOLDED]
        showdown = (
            {self.positions[i]: tuple(c.get_index() for c in game.players[i].hand) for i in live}
            if len(live) >= 2
            else {}
        )
        return RoundLog(
            game_id=self.game_id,
            rotation_id=self.rotation_id,
            round_id=self.round_id,
            seed=self.seed,
            positions=self.positions,
            starting_stacks=tuple(chips_to_bb(c) for c in self.starting_chips),
            actions=tuple(
                (pid, street, action, chips_to_bb(amount))
                for pid, street, action, amount in self._actions
            ),
            community_cards=tuple(c.get_index() for c in game.public_cards),
            showdown_cards=showdown,
            chip_deltas={self.positions[i]: chips_to_bb(payoffs[i]) for i in range(len(self.positions))},
        )

    # --- Snapshots (#18) -------------------------------------------------------

    def snapshot(self) -> Snapshot:
        """Save the current decision point. Does not change this simulator."""
        self._require_not_over()
        saved = copy.copy(self)
        saved.bots = {}
        saved._actions = list(self._actions)
        saved._game = copy.deepcopy(self._game)
        return Snapshot(saved)

    @classmethod
    def from_snapshot(cls, snapshot: Snapshot, bots: Mapping[int, Bot], seed: int) -> "Simulator":
        """Continue a round from a snapshot with new bots.

        Every card the player to act at the snapshot cannot see -- all other
        players' hole cards and the undealt deck (future board) -- is shuffled
        and re-dealt with `seed`, so a rollout never sees the true future.
        """
        sim = copy.copy(snapshot._sim)
        sim.bots = bots
        sim._actions = list(snapshot._sim._actions)
        sim._game = copy.deepcopy(snapshot._sim._game)
        missing = [pid for pid in sim.positions if pid not in bots]
        if missing:
            raise ValueError(f"no bot for player IDs {missing}")
        sim._redeal_unseen(seed)
        return sim

    # --- Internals -------------------------------------------------------------

    def _legal(self) -> dict[str, tuple[int, Action]]:
        """Our legal actions for the player to act: name -> (cost in chips, RLCard action)."""
        game = self._game
        seat = game.game_pointer
        rnd = game.round
        player = game.players[seat]
        stack = player.remained_chips
        diff = max(rnd.raised) - rnd.raised[seat]
        pot = self._pot()
        rl_legal = set(game.get_legal_actions())

        legal: dict[str, tuple[int, Action]] = {}
        if diff > 0:
            legal[FOLD] = (0, Action.FOLD)
        if diff == 0:
            legal[CHECK] = (0, Action.CHECK_CALL)
        elif diff < stack:
            legal[CALL] = (diff, Action.CHECK_CALL)
        if Action.RAISE_HALF_POT in rl_legal:
            legal[RAISE_HALF_POT] = (int(pot / 2), Action.RAISE_HALF_POT)
        if Action.RAISE_POT in rl_legal:
            legal[RAISE_POT] = (pot, Action.RAISE_POT)
        if diff >= stack:
            # Short all-in call: RLCard removes ALL_IN and caps CHECK_CALL at the stack.
            legal[ALL_IN] = (stack, Action.CHECK_CALL)
        elif Action.ALL_IN in rl_legal:
            legal[ALL_IN] = (stack, Action.ALL_IN)
        return legal

    def _pot(self) -> int:
        # RLCard only refreshes dealer.pot inside get_state(); compute it directly.
        return int(sum(p.in_chips for p in self._game.players))

    def _street(self) -> Street:
        return STREETS[self._game.stage.value]

    def _skip_all_in_players(self) -> None:
        """RLCard's turn order only skips folded players, so it can hand the turn
        to an all-in player. They have no decision: pass for them (0 chips),
        without asking the bot or logging an action."""
        game = self._game
        while not game.is_over() and game.players[game.game_pointer].status == PlayerStatus.ALLIN:
            player = game.players[game.game_pointer]
            before = player.in_chips
            game.step(Action.CHECK_CALL)
            assert player.in_chips == before, "all-in player was charged chips"

    def _redeal_unseen(self, seed: int) -> None:
        game = self._game
        viewer = game.game_pointer
        rng = np.random.RandomState(seed)
        others = [i for i in range(len(game.players)) if i != viewer]

        pool = list(game.dealer.deck)
        for i in others:
            pool.extend(game.players[i].hand)
        order = rng.permutation(len(pool))
        pool = [pool[j] for j in order]

        for i in others:
            game.players[i].hand = [pool.pop(), pool.pop()]
        game.dealer.deck = pool

        # Fresh randomness everywhere RLCard keeps a generator (e.g. split-pot remainders).
        game.np_random = rng
        game.dealer.np_random = rng
        game.judger.np_random = rng
        game.round.np_random = rng
        for p in game.players:
            p.np_random = rng

    def _require_not_over(self) -> None:
        if self.is_over():
            raise RuntimeError("round is over")
