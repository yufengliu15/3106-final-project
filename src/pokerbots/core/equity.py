"""Equity calculator shared by every bot (proposal milestone Nov 9-15).

- preflop_equity(hole, n_opp): table lookup, 169 classes x 1..9 opponents,
  precomputed by Monte Carlo (build_preflop_table) and stored in data/.
- equity(hole, board, n_opp, ...): Monte Carlo against n random hands, or
  against hands drawn from a top-x% range (range-weighted equity).
- top_range(pct): the set of starting-hand classes making up the best pct of
  combos, ordered by pre-flop equity vs 2 random opponents. The scripted bots
  use this as "plays the top x% of starting hands".

Equity = P(win) + P(tie) / (number of players tied), against all opponents.
"""

from __future__ import annotations

import json
import random
import zlib
from functools import lru_cache
from pathlib import Path

from phevaluator import Card as _PhCard
from phevaluator import evaluate_cards as _ph_eval  # C evaluator, ~6x faster than treys; lower = better

from pokerbots.core.cards import (
    FULL_DECK,
    all_hand_classes,
    class_combos,
    hand_class,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
PREFLOP_FILE = DATA_DIR / "preflop_equity.json"
MAX_OPP = 9


@lru_cache(maxsize=None)
def to_ph(card: str) -> int:
    """RLCard "HA" -> phevaluator integer id."""
    return _PhCard(card[1] + card[0].lower()).id_


def simulate(hole: tuple[str, str], board: tuple[str, ...], n_opp: int, sims: int,
             rng: random.Random, opp_classes: frozenset[str] | None = None) -> float:
    """Monte Carlo equity of `hole` on `board` vs `n_opp` opponents."""
    if n_opp <= 0:
        return 1.0
    known = {to_ph(c) for c in (*hole, *board)}
    deck = [i for i in range(52) if i not in known]
    h1, h2 = (to_ph(c) for c in hole)
    board_t = [to_ph(c) for c in board]
    need_board = 5 - len(board)
    combos = None
    if opp_classes is not None:
        combos = [c for c in _range_combos(opp_classes) if c[0] not in known and c[1] not in known]
        if not combos:
            combos = None
    ev = _ph_eval
    total = 0.0
    done = 0
    attempts = 0
    while done < sims and attempts < sims * 20:
        attempts += 1
        if combos is None:
            draw = rng.sample(deck, 2 * n_opp + need_board)
            opps = [draw[2 * i:2 * i + 2] for i in range(n_opp)]
            rest = draw[2 * n_opp:]
        else:
            used: set[int] = set()
            opps = []
            for _ in range(n_opp):
                for _try in range(20):
                    c = combos[rng.randrange(len(combos))]
                    if c[0] not in used and c[1] not in used:
                        break
                else:
                    break
                used.update(c)
                opps.append(c)
            if len(opps) < n_opp:
                continue
            free = [c for c in deck if c not in used]
            rest = rng.sample(free, need_board)
        b1, b2, b3, b4, b5 = board_t + rest if need_board else board_t
        mine = ev(h1, h2, b1, b2, b3, b4, b5)
        best = mine
        ties = 1
        for o in opps:
            r = ev(o[0], o[1], b1, b2, b3, b4, b5)
            if r < best:
                best = r
                ties = 0
                break
            if r == best:
                ties += 1
        if best == mine and ties > 0:
            total += 1.0 / ties
        done += 1
    return total / max(done, 1)


@lru_cache(maxsize=256)
def _range_combos(classes: frozenset[str]) -> tuple[tuple[int, int], ...]:
    out = []
    for i, a in enumerate(FULL_DECK):
        for b in FULL_DECK[i + 1:]:
            if hand_class((a, b)) in classes:
                out.append((to_ph(a), to_ph(b)))
    return tuple(out)


# --- Pre-flop table -----------------------------------------------------------

def build_preflop_table(sims: int = 3000, seed: int = 0, processes: int = 1) -> dict[str, list[float]]:
    """{class: [equity vs 1 opp, ..., vs 9 opps]} by Monte Carlo."""
    classes = all_hand_classes()
    jobs = [(cls, sims, seed) for cls in classes]
    if processes > 1:
        from multiprocessing import Pool
        with Pool(processes) as pool:
            rows = pool.map(_class_row, jobs)
    else:
        rows = [_class_row(j) for j in jobs]
    return dict(zip(classes, rows))


def _example_hand(cls: str) -> tuple[str, str]:
    r1, r2 = cls[0], cls[1]
    if len(cls) == 2:
        return ("S" + r1, "H" + r2)
    return ("S" + r1, ("S" if cls[2] == "s" else "H") + r2)


def _class_row(job) -> list[float]:
    cls, sims, seed = job
    rng = random.Random(f"{seed}-{cls}")
    hole = _example_hand(cls)
    return [round(simulate(hole, (), n, sims, rng), 4) for n in range(1, MAX_OPP + 1)]


@lru_cache(maxsize=1)
def _preflop_table() -> dict[str, list[float]]:
    if not PREFLOP_FILE.exists():
        raise FileNotFoundError(
            f"{PREFLOP_FILE} missing; build it with `python -m pokerbots.core.equity`")
    return json.loads(PREFLOP_FILE.read_text())


def preflop_equity(hole: tuple[str, str], n_opp: int) -> float:
    n = min(max(n_opp, 1), MAX_OPP)
    return _preflop_table()[hand_class(hole)][n - 1]


@lru_cache(maxsize=1)
def hand_order() -> tuple[str, ...]:
    """The 169 classes, strongest first (equity vs 2 random opponents)."""
    table = _preflop_table()
    return tuple(sorted(table, key=lambda c: -table[c][1]))


@lru_cache(maxsize=None)
def top_range(pct: float) -> frozenset[str]:
    """Classes making up the best `pct` (0-1) of the 1326 combos."""
    target = pct * 1326
    out, total = [], 0
    for cls in hand_order():
        if total >= target:
            break
        out.append(cls)
        total += class_combos(cls)
    return frozenset(out)


@lru_cache(maxsize=None)
def hand_percentile(cls: str) -> float:
    """Fraction of combos at least as strong as this class: AA ~ 0.005, 72o = 1.0."""
    total = 0
    for c in hand_order():
        total += class_combos(c)
        if c == cls:
            return total / 1326
    raise KeyError(cls)


# --- Post-flop (cached) ---------------------------------------------------------

@lru_cache(maxsize=200_000)
def _equity_cached(hole, board, n_opp, sims, range_pct) -> float:
    rng = random.Random(zlib.crc32(repr((hole, board, n_opp, range_pct)).encode()))
    classes = top_range(range_pct) if range_pct is not None and range_pct < 1.0 else None
    return simulate(hole, board, n_opp, sims, rng, classes)


_SIMS_OVERRIDE: list[int] = []


class fast_equity:
    """Context manager: cap Monte Carlo sims (used inside #18 rollouts, where
    thousands of simulated decisions would otherwise dominate the run time).

        with fast_equity(40): ...
    """

    def __init__(self, sims: int) -> None:
        self.sims = sims

    def __enter__(self):
        _SIMS_OVERRIDE.append(self.sims)

    def __exit__(self, *exc):
        _SIMS_OVERRIDE.pop()


def equity(hole: tuple[str, str], board: tuple[str, ...], n_opp: int,
           sims: int = 200, range_pct: float | None = None) -> float:
    """Equity vs n_opp opponents. Pre-flop without a range uses the table.

    range_pct: if given (0-1), opponents hold hands from the top range_pct of
    starting hands (e.g. their VPIP), otherwise any two cards. Deterministic
    for the same arguments, so bots behave the same in rollouts and replays.
    """
    n_opp = min(n_opp, MAX_OPP)
    if n_opp <= 0:
        return 1.0
    if _SIMS_OVERRIDE:
        sims = min(sims, _SIMS_OVERRIDE[-1])
    hole = tuple(sorted(hole))
    if not board and range_pct is None:
        return preflop_equity(hole, n_opp)
    if range_pct is not None:
        range_pct = round(min(max(range_pct, 0.05), 1.0), 2)
        if range_pct >= 1.0:
            range_pct = None
            if not board:
                return preflop_equity(hole, n_opp)
    return _equity_cached(hole, tuple(sorted(board[:3])) + tuple(board[3:]), n_opp, sims, range_pct)


if __name__ == "__main__":
    import argparse
    import os

    p = argparse.ArgumentParser(description="Build the pre-flop equity table")
    p.add_argument("--sims", type=int, default=3000)
    p.add_argument("--processes", type=int, default=os.cpu_count())
    a = p.parse_args()
    DATA_DIR.mkdir(exist_ok=True)
    table = build_preflop_table(a.sims, processes=a.processes)
    PREFLOP_FILE.write_text(json.dumps(table))
    print(f"wrote {PREFLOP_FILE} ({len(table)} classes)")
