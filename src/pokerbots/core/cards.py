"""Card helpers shared by every bot.

Cards arrive in RLCard's format: suit letter then rank, e.g. "H4", "SK", "CT".
treys uses rank then lowercase suit ("4h", "Ks", "Tc").
"""

from __future__ import annotations

from functools import lru_cache

from treys import Card

RANKS = "23456789TJQKA"
SUITS = "SHDC"
FULL_DECK: tuple[str, ...] = tuple(s + r for s in SUITS for r in RANKS)


def rank_of(card: str) -> int:
    """0 (deuce) .. 12 (ace)."""
    return RANKS.index(card[1])


def suit_of(card: str) -> str:
    return card[0]


@lru_cache(maxsize=None)
def to_treys(card: str) -> int:
    return Card.new(card[1] + card[0].lower())


def hand_class(hole: tuple[str, str]) -> str:
    """One of the 169 starting-hand classes, e.g. "AKs", "T9o", "77"."""
    a, b = sorted(hole, key=rank_of, reverse=True)
    ra, rb = a[1], b[1]
    if ra == rb:
        return ra + rb
    return ra + rb + ("s" if a[0] == b[0] else "o")


def all_hand_classes() -> list[str]:
    out = []
    for i in range(12, -1, -1):
        for j in range(i, -1, -1):
            if i == j:
                out.append(RANKS[i] * 2)
            else:
                out.append(RANKS[i] + RANKS[j] + "s")
                out.append(RANKS[i] + RANKS[j] + "o")
    return out


def class_combos(cls: str) -> int:
    """Number of two-card combos in a class: pair 6, suited 4, offsuit 12."""
    if len(cls) == 2:
        return 6
    return 4 if cls[2] == "s" else 12
