"""Derived facts about the current decision, shared by every main bot.

Observation.action_history has no street markers, so this module replays the
betting (same order rules as RLCard) to recover who folded, who is all-in,
street boundaries, raises per street and the last aggressor. Every bot may
use it, so no bot knows more than another.
"""

from __future__ import annotations

from dataclasses import dataclass

from pokerbots.core import ALL_IN, CALL, FOLD, RAISE_HALF_POT, RAISE_POT, Observation

AGGRESSIVE = (RAISE_HALF_POT, RAISE_POT, ALL_IN)
STREET_OF_BOARD = {0: 0, 3: 1, 4: 2, 5: 3}


@dataclass(frozen=True)
class Situation:
    street: int                 # 0 pre-flop .. 3 river
    n_players: int              # live players at the table
    seat: int                   # our index in positions (0 = SB, last = button)
    seats_to_button: int        # 0 = we are the button
    opponents_in_hand: tuple[int, ...]  # not folded, excluding us
    allin_opponents: int
    left_to_act: int            # opponents still to act after us this street
    pot: float
    to_call: float
    pot_odds: float             # to_call / (pot + to_call)
    my_stack: float
    effective_stack: float      # min(our stack, biggest stack of an opponent in hand)
    spr: float                  # effective stack / pot
    raises_this_street: int
    raises_preflop: int
    last_aggressor: int | None  # player ID of the last raiser this street
    preflop_aggressor: int | None
    i_was_aggressor: bool       # we made the last raise of the previous street
    facing_allin: bool
    invested: float             # our chips in the pot this hand


_FALLBACK = {
    "check": ("check", "call", ALL_IN, FOLD),
    CALL: (CALL, ALL_IN, "check", FOLD),
    RAISE_HALF_POT: (RAISE_HALF_POT, RAISE_POT, ALL_IN, CALL, "check"),
    RAISE_POT: (RAISE_POT, RAISE_HALF_POT, ALL_IN, CALL, "check"),
    ALL_IN: (ALL_IN, CALL, "check"),
    FOLD: (FOLD, "check"),
}


def legalize(action: str, legal) -> str:
    """Closest legal action to `action` (safety net for rare spots, e.g. a
    0-chip player facing a bet is offered only fold / all-in for 0)."""
    names = {a for a, _ in legal}
    if action in names:
        return action
    for a in _FALLBACK.get(action, ()):
        if a in names:
            return a
    return next(iter(names))


def analyse(obs: Observation) -> Situation:
    pos = obs.positions
    n = len(pos)
    me = obs.player_id
    seat = pos.index(me)
    stacks_now = dict(zip(pos, obs.stacks))
    contrib_total = {p: 0.0 for p in pos}
    for pid, _a, amt in obs.action_history:
        contrib_total[pid] += amt
    # blinds (not in the history); a blind can be short if the stack was tiny
    blinds = {pos[0]: 0.5, pos[1]: 1.0} if n > 1 else {}
    start = {p: stacks_now[p] + contrib_total[p] + blinds.get(p, 0.0) for p in pos}
    for p, b in blinds.items():
        blinds[p] = min(b, start[p])

    folded: set[int] = set()
    allin: set[int] = {p for p in pos if start[p] - blinds.get(p, 0.0) <= 1e-9}
    street = 0
    bet = dict(blinds)                      # chips put in this street
    acted: set[int] = set()
    raises = {0: 0, 1: 0, 2: 0, 3: 0}
    last_agg: dict[int, int | None] = {0: None, 1: None, 2: None, 3: None}

    def street_done() -> bool:
        active = [p for p in pos if p not in folded and p not in allin]
        if len([p for p in pos if p not in folded]) <= 1:
            return True
        top = max(bet.get(p, 0.0) for p in pos)
        return all(p in acted and abs(bet.get(p, 0.0) - top) < 1e-9 for p in active)

    for pid, action, amt in obs.action_history:
        bet[pid] = bet.get(pid, 0.0) + amt
        if action == FOLD:
            folded.add(pid)
        elif action in AGGRESSIVE:
            top_before = max(v for q, v in bet.items() if q != pid) if len(bet) > 1 else 0.0
            if bet[pid] > top_before + 1e-9:
                raises[street] += 1
                last_agg[street] = pid
                acted = set()
            if action == ALL_IN:
                allin.add(pid)
        acted.add(pid)
        if street_done() and street < 3:
            street += 1
            bet, acted = {}, set()
    street = STREET_OF_BOARD.get(len(obs.community_cards), street)

    in_hand = tuple(p for p in pos if p not in folded and p != me)
    # players acting after us this street: postflop order is index 0..n-1,
    # pre-flop starts at index 2 and the blinds act last
    order = list(range(n)) if street > 0 else [(2 + i) % n for i in range(n)]
    my_turn = order.index(seat)
    after = [pos[i] for i in order[my_turn + 1:]]
    top = max(bet.values()) if bet else 0.0
    left = sum(1 for p in in_hand if p not in allin and (p in after or abs(bet.get(p, 0.0) - top) > 1e-9))

    to_call = 0.0
    for a, cost in obs.legal_actions:
        if a == CALL:
            to_call = cost
        elif a == ALL_IN and not any(x == CALL for x, _ in obs.legal_actions) \
                and not any(x == "check" for x, _ in obs.legal_actions):
            to_call = cost  # short all-in call
    my_stack = stacks_now[me]
    opp_max = max((stacks_now[p] + bet.get(p, 0.0) for p in in_hand), default=0.0)
    eff = min(my_stack, max(opp_max - bet.get(me, 0.0), 0.0)) if in_hand else my_stack
    pot = max(obs.pot, 1e-9)
    facing_allin = bool(obs.action_history) and any(
        p in allin and p != me and p in in_hand for p in pos) and to_call > 0
    prev = street - 1
    return Situation(
        street=street,
        n_players=n,
        seat=seat,
        seats_to_button=n - 1 - seat,
        opponents_in_hand=in_hand,
        allin_opponents=sum(1 for p in in_hand if p in allin),
        left_to_act=left,
        pot=obs.pot,
        to_call=to_call,
        pot_odds=to_call / (obs.pot + to_call) if to_call > 0 else 0.0,
        my_stack=my_stack,
        effective_stack=eff,
        spr=eff / pot,
        raises_this_street=raises[street],
        raises_preflop=raises[0],
        last_aggressor=last_agg[street],
        preflop_aggressor=last_agg[0],
        i_was_aggressor=prev >= 0 and last_agg[prev] == me,
        facing_allin=facing_allin,
        invested=contrib_total[me] + blinds.get(me, 0.0),
    )
