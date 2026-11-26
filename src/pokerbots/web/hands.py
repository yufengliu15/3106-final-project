"""Showdown hand names for the live viewer's result screen.

best_hand() picks the best five of a player's seven cards and names it the way
a dealer would ("Kings full of sevens"), so a human can see why a hand won.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Sequence

from treys import Evaluator

from pokerbots.core.cards import rank_of, to_treys

_EVALUATOR = Evaluator()
_ONE = ("two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "jack", "queen", "king", "ace")
_MANY = ("twos", "threes", "fours", "fives", "sixes", "sevens", "eights", "nines", "tens",
         "jacks", "queens", "kings", "aces")
_ACE, _FIVE = 12, 3


def best_hand(cards: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    """Name and cards of the best five-card hand in `cards` (5-7 RLCard cards).
    The five cards are ordered by group size, then rank ("KKK77", "AKQ95")."""
    score, five = min((_EVALUATOR.evaluate([], [to_treys(c) for c in five]), five)
                      for five in combinations(cards, 5))
    counts = Counter(rank_of(c) for c in five)
    ordered = tuple(sorted(five, key=lambda c: (-counts[rank_of(c)], -rank_of(c))))
    return _describe(_EVALUATOR.get_rank_class(score), ordered, counts), ordered


def _describe(rank_class: int, ordered: tuple[str, ...], counts: Counter) -> str:
    """rank_class is treys' (0 royal flush .. 9 high card)."""
    top = rank_of(ordered[0])
    second = rank_of(ordered[counts[top]])  # first card after the top group
    straight_high = _FIVE if set(counts) == {_ACE, 0, 1, 2, 3} else top  # the wheel is five-high
    names = {
        0: "Royal flush",
        1: f"Straight flush, {_ONE[straight_high]} high",
        2: f"Four {_MANY[top]}",
        3: f"Full house, {_MANY[top]} full of {_MANY[second]}",
        4: f"Flush, {_ONE[top]} high",
        5: f"Straight, {_ONE[straight_high]} high",
        6: f"Three {_MANY[top]}",
        7: f"Two pair, {_MANY[top]} and {_MANY[second]}",
        8: f"Pair of {_MANY[top]}",
    }
    return names.get(rank_class, f"{_ONE[top].capitalize()} high")


def showdown_hands(community: Sequence[str], shown: dict[int, Sequence[str]]) -> dict[int, dict]:
    """{player_id: {"name", "best"}} for every player who showed down. Empty if
    the board was not run out (the simulator always runs it out at showdown)."""
    if len(community) < 3:
        return {}
    out = {}
    for pid, hole in shown.items():
        name, best = best_hand([*hole, *community])
        out[pid] = {"name": name, "best": list(best)}
    return out
