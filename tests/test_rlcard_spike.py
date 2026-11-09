"""Issue #15 spike: can RLCard support our table-state design?

Uses RLCard directly (no pokerbots code). Checks the hooks the simulator will
rely on -- see docs/tournament-and-simulator.md:

- blinds/stacks/dealer set per round via game attributes before reset()
- rebuilding with fewer players after a bust
- chip conservation within a round
- same seed -> same deal
- short-stack all-in via CHECK_CALL
"""

from __future__ import annotations

import random

import pytest
import rlcard
from rlcard.games.limitholdem import PlayerStatus
from rlcard.games.nolimitholdem.round import Action

SMALL_BLIND = 50  # 1 bb = 100 chips
BIG_BLIND = 100


def make_table(stacks: list[int], seed: int):
    """Build a fresh env for one round, the way the simulator will.

    The attributes must be set before reset(): init_game() reads them there.
    """
    n = len(stacks)
    env = rlcard.make("no-limit-holdem", config={"game_num_players": n, "seed": seed})
    game = env.game
    game.small_blind = SMALL_BLIND
    game.big_blind = BIG_BLIND
    game.init_chips = list(stacks)
    game.dealer_id = n - 1  # last index is the button -> blinds on 0 and 1
    env.reset()
    return env, game


def total_chips(game) -> int:
    return sum(p.remained_chips + p.in_chips for p in game.players)


def card_ids(cards) -> list[str]:
    return [c.get_index() for c in cards]


@pytest.mark.parametrize("n", [8, 7, 3, 2])
def test_hooks_set_blinds_and_stacks(n):
    stacks = [10_000 + 500 * i for i in range(n)]  # uneven on purpose
    _, game = make_table(stacks, seed=1)

    assert game.players[0].in_chips == SMALL_BLIND
    assert game.players[1].in_chips == BIG_BLIND
    for i, p in enumerate(game.players):
        assert p.remained_chips + p.in_chips == stacks[i]
    # first to act preflop is the seat after the big blind
    assert game.game_pointer == 2 % n


def test_rebuild_after_bust_keeps_mapping():
    """Seat 3 busts at 8 players; the next round is built with the 7 survivors."""
    ids = [1000 + i for i in range(8)]
    stacks = {pid: 10_000 for pid in ids}
    busted = ids[3]

    positions = [pid for pid in ids if pid != busted]
    positions = positions[1:] + positions[:1]  # rotate: next round's order
    _, game = make_table([stacks[pid] for pid in positions], seed=2)

    assert game.num_players == 7
    assert len(game.players) == 7
    assert game.players[0].in_chips == SMALL_BLIND  # positions[0] posts SB
    assert game.players[1].in_chips == BIG_BLIND  # positions[1] posts BB
    assert total_chips(game) == 7 * 10_000


def play_random_round(game, rng: random.Random) -> None:
    while not game.is_over():
        game.step(rng.choice(game.get_legal_actions()))


@pytest.mark.parametrize("n", [8, 5, 2])
def test_chips_conserved_over_many_rounds(n):
    rng = random.Random(0)
    for seed in range(300):
        stacks = [rng.randrange(150, 30_000) for _ in range(n)]  # >= SB+BB-ish, uneven
        stacks[0] = max(stacks[0], SMALL_BLIND)
        stacks[1] = max(stacks[1], BIG_BLIND)
        _, game = make_table(stacks, seed=seed)
        before = total_chips(game)

        play_random_round(game, rng)
        payoffs = game.get_payoffs()

        assert total_chips(game) == before
        assert sum(payoffs) == 0
        final = [stacks[i] + payoffs[i] for i in range(n)]
        assert all(s >= 0 for s in final)
        assert sum(final) == sum(stacks)


def test_same_seed_same_deal():
    stacks = [10_000] * 8
    _, a = make_table(stacks, seed=1234)
    _, b = make_table(stacks, seed=1234)
    _, c = make_table(stacks, seed=4321)

    hands_a = [card_ids(p.hand) for p in a.players]
    hands_b = [card_ids(p.hand) for p in b.players]
    assert hands_a == hands_b
    assert card_ids(a.dealer.deck) == card_ids(b.dealer.deck)  # same future board
    assert hands_a != [card_ids(p.hand) for p in c.players]


def test_short_stack_all_in_is_check_call():
    """Stack below the amount to call: RLCard offers only FOLD / CHECK_CALL,
    and CHECK_CALL puts in exactly the stack."""
    stacks = [10_000, 10_000, 60, 10_000]  # index 2 acts first, faces a 100-chip BB
    _, game = make_table(stacks, seed=3)
    assert game.game_pointer == 2

    legal = game.get_legal_actions()
    assert set(legal) == {Action.FOLD, Action.CHECK_CALL}

    game.step(Action.CHECK_CALL)
    short = game.players[2]
    assert short.in_chips == 60
    assert short.remained_chips == 0
    assert short.status == PlayerStatus.ALLIN
