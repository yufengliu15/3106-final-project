"""Selective showdown sampling for Bot B's opponent model."""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import ceil

from phevaluator import evaluate_cards

from pokerbots.core.cards import FULL_DECK, hand_class
from pokerbots.core.equity import MAX_OPP, _SIMS_OVERRIDE, to_ph, top_range

RANGE_TAIL = 0.25


@dataclass(frozen=True)
class ShowdownEstimate:
    share: float
    loss_probability: float


def selective_showdown(hole: tuple[str, str], board: tuple[str, ...], ranges: tuple[float, ...],
                       sims: int = 256) -> ShowdownEstimate:
    """Showdown share and outright loss probability against individual ranges.

    A fraction of 1.0 represents an unconditioned opponent. After the flop
    the preferred range is based on made-hand rank on the visible board.
    Draw future cards only after dealing all opponents without replacement.
    """
    fractions = tuple(sorted(round(min(max(fraction, 0.05), 1.0), 2) for fraction in ranges[:MAX_OPP]))
    if not fractions:
        return ShowdownEstimate(1.0, 0.0)
    # Honor the simulator's existing fast_equity context during rollouts.
    if _SIMS_OVERRIDE:
        sims = min(sims, _SIMS_OVERRIDE[-1])
    return _selective_showdown_cached(tuple(sorted(hole)), tuple(sorted(board)), fractions, sims)


@lru_cache(maxsize=32_768)
def _selective_showdown_cached(hole: tuple[str, str], board: tuple[str, ...],
                               fractions: tuple[float, ...], sims: int) -> ShowdownEstimate:
    hero = (to_ph(hole[0]), to_ph(hole[1]))
    public = tuple(map(to_ph, board))
    known = {*hero, *public}
    deck = tuple(card for card in range(52) if card not in known)
    pools = _selective_pools(deck, public, fractions)
    if len(pools) == 1 and len(public) == 5:
        return _river_showdown(hero, public, deck, frozenset(pools[0]))
    rng = random.Random(zlib.crc32(repr((hole, board, fractions)).encode()))
    total = 0.0
    losses = 0
    for _ in range(sims):
        available = list(deck)
        opponents = []
        for pool in pools:
            pocket = _sample_pocket(rng, available, pool)
            available.remove(pocket[0])
            available.remove(pocket[1])
            opponents.append(pocket)
        runout = [*public, *rng.sample(available, 5 - len(public))]
        share = _showdown_share(hero, runout, opponents)
        total += share
        losses += share == 0.0
    return ShowdownEstimate(total / sims, losses / sims)


def _selective_pools(deck: tuple[int, ...], board: tuple[int, ...],
                     fractions: tuple[float, ...]) -> tuple[tuple[tuple[int, int], ...], ...]:
    available = frozenset(deck)
    pools: dict[float, tuple[tuple[int, int], ...]] = {1.0: ()}
    for fraction in set(fractions) - pools.keys():
        if board:
            ranked = _made_hand_ranks(deck, board)
            boundary = ranked[ceil(fraction * len(ranked)) - 1][0]
            pools[fraction] = tuple(pocket for rank, pocket in ranked if rank <= boundary)
        else:
            pools[fraction] = tuple(
                pocket for pocket in _opening_pockets(fraction) if available.issuperset(pocket)
            )
    return tuple(pools[fraction] for fraction in fractions)


@lru_cache(maxsize=256)
def _opening_pockets(fraction: float) -> tuple[tuple[int, int], ...]:
    classes = top_range(fraction)
    return tuple((to_ph(first), to_ph(second)) for first, second in combinations(FULL_DECK, 2)
                 if hand_class((first, second)) in classes)


@lru_cache(maxsize=128)
def _made_hand_ranks(deck: tuple[int, ...], board: tuple[int, ...]) -> tuple[tuple[int, tuple[int, int]], ...]:
    return tuple(sorted((evaluate_cards(*pocket, *board), pocket) for pocket in combinations(deck, 2)))


def _sample_pocket(rng: random.Random, available: list[int], pool: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    # Leave a wide tail for bluffs and draws outside the preferred range.
    if pool and rng.random() >= RANGE_TAIL:
        for _ in range(12):
            pocket = rng.choice(pool)
            if pocket[0] in available and pocket[1] in available:
                return pocket
    first, second = rng.sample(available, 2)
    return first, second


def _showdown_share(hero: tuple[int, int], board: list[int], opponents: list[tuple[int, int]]) -> float:
    rank = evaluate_cards(*hero, *board)
    ties = 1
    for opponent in opponents:
        opponent_rank = evaluate_cards(*opponent, *board)
        if opponent_rank < rank:
            return 0.0
        ties += opponent_rank == rank
    return 1.0 / ties


def _river_showdown(hero: tuple[int, int], board: tuple[int, ...], deck: tuple[int, ...],
                    pool: frozenset[tuple[int, int]]) -> ShowdownEstimate:
    """Exact heads-up river equity under the same wide-tail mixture."""
    hero_rank = evaluate_cards(*hero, *board)
    ranked = _made_hand_ranks(deck, board)
    unweighted = RANGE_TAIL / len(ranked) if pool else 1.0 / len(ranked)
    focused = (1.0 - RANGE_TAIL) / len(pool) if pool else 0.0
    share = loss = 0.0
    for rank, pocket in ranked:
        weight = unweighted + focused * (pocket in pool)
        if rank < hero_rank:
            loss += weight
        else:
            share += weight if rank > hero_rank else weight / 2
    return ShowdownEstimate(min(1.0, share), min(1.0, loss))
