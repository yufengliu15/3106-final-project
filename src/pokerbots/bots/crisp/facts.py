"""Bot A layers 0 and 1 (private/bot_a design, sections 4-5).

Layer 0 is plain math: pot odds, equity vs ranges, response probabilities,
equity realization, bust penalty, EV per legal action and the seeded roll.
Layer 1 turns those numbers into crisp facts: every fact has one value, and
every threshold test is fully true or fully false.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, fields, replace
from math import inf, prod

from pokerbots.core import ALL_IN, CALL, CHECK, FOLD, RAISE_HALF_POT, RAISE_POT, Observation
from pokerbots.core.cards import hand_class
from pokerbots.core.equity import equity, hand_percentile
from pokerbots.core.situation import Situation
from pokerbots.opponents import PRIOR, OpponentStats

STREET_NAMES = ("preflop", "flop", "turn", "river")
POSITION_FROM_BUTTON = ("BTN", "CO", "HJ", "LJ", "MP", "UTG+2", "UTG+1", "UTG")
TIERS = (("premium", 0.05), ("strong", 0.10), ("playable", 0.20), ("speculative", 0.50))
TIER_RANK = {"premium": 0, "strong": 1, "playable": 2, "speculative": 3, "trash": 4}

SHRINK_K = 16.0            # sample shrinkage: w = n / (n + k)
MIN_ROUNDS = 20            # villain_type is "unknown" below this many tracked rounds
SHORT_BB, DEEP_BB = 15.0, 60.0
EDGE_BIG = 0.15
EDGE_BIG_VS_STATION = 0.08  # stations pay off thinner value
FE_HIGH, FE_LOW = 0.6, 0.2
OPEN_VALUE_SHARE = 1 / 3    # preflop unopened: top third of the opening range is "big"
BUST_K = 1.0                # TournamentConfig.k default; the observation does not expose k
EQUITY_SIMS = 200


@dataclass(frozen=True)
class Numbers:
    """Layer 0 output."""

    pot: float
    to_call: float
    pot_odds: float
    price: float          # pot odds when facing a bet, fair share (1 / players) when not
    equity: float         # W: vs the villain's range
    calling_equity: float  # Wc: vs the part of the range that continues
    reraise_equity: float  # vs the villain's re-raising range
    contestants: float    # expected opponents who reach showdown: acted ones count 1, yet-to-act ones their vpip
    behind: float         # expected callers among opponents yet to act this hand
    realization: float    # eta
    fold_all: float       # P(every opponent folds to a raise)
    call_p: float         # P(the raise is called), given not everyone folds
    raise_p: float        # P(the raise is re-raised)
    percentile: float     # starting-hand percentile (0 = AA)
    open_range: float     # preflop opening range at this seat
    roll: float           # seeded uniform [0, 1) for this decision
    ev: dict[str, float]  # EV in bb of every legal action


@dataclass(frozen=True)
class Facts:
    """Layer 1 output: the working memory the rules read."""

    street: str
    position: str
    facing: str           # nothing / bet / raise / reraise / all-in
    pot_type: str         # heads-up / multiway
    depth: str            # short / medium / deep
    initiative: bool
    hand_tier: str | None  # preflop only
    edge: str             # big / thin / negative
    edge_value: float
    edge_cut: float
    villain: int | None
    villain_type: str
    villain_range: float
    fold_equity: str      # high / medium / low
    n: Numbers


def decision_roll(obs: Observation) -> float:
    """Same spot, same roll: replays and rollouts stay reproducible."""
    key = repr((obs.game_id, obs.rotation_id, obs.round_id, obs.player_id, len(obs.action_history)))
    return zlib.crc32(key.encode()) / 2**32


def shrink(s: OpponentStats) -> OpponentStats:
    """Blend every rate toward the prior; five hands of 'always folds' is noise."""
    w = s.rounds / (s.rounds + SHRINK_K)
    mixed = {f.name: w * getattr(s, f.name) + (1 - w) * getattr(PRIOR, f.name)
             for f in fields(OpponentStats) if f.name != "rounds"}
    return replace(s, **mixed)


# villain_type rows in priority order (design section 5); the first true row names the type.
VILLAIN_TYPES = (
    ("maniac", lambda s: s.allin_rate > 0.3),
    ("nit", lambda s: s.vpip < 0.15),
    ("TAG", lambda s: s.vpip < 0.25 and s.aggression > 1.5),
    ("LAG", lambda s: s.vpip >= 0.25 and s.aggression > 1.5),
    ("station", lambda s: s.vpip >= 0.25 and s.call_to_raise > 0.6),
)


def villain_type(raw: OpponentStats) -> str:
    """The sample gate replaces shrinkage here: shrunk rates could never reach
    the nit / station cut-offs inside the tracker's 50-round window."""
    if raw.rounds < MIN_ROUNDS:
        return "unknown"
    return next((name for name, test in VILLAIN_TYPES if test(raw)), "regular")


def position_name(sit: Situation) -> str:
    if sit.seat == 0:
        return "SB"
    if sit.seat == 1 and sit.n_players > 2:
        return "BB"
    return POSITION_FROM_BUTTON[min(sit.seats_to_button, len(POSITION_FROM_BUTTON) - 1)]


def opening_range(sit: Situation) -> float:
    """About top 10% UTG to top 40% BTN; blinds defend wide."""
    if sit.seat == 0:
        return 0.30
    if sit.seat == 1:
        return 0.40
    return 0.40 - 0.30 * min(sit.seats_to_button, 7) / 7


def facing(obs: Observation, sit: Situation) -> str:
    if sit.to_call <= 0:
        return "nothing"
    # "all-in" only when the shove set the price we face. A shove smaller than the
    # current bet (e.g. 0.01bb) is just a cheap side stake; the decision stays an
    # ordinary bet / raise against the players still behind.
    if sit.facing_allin and _shove_set_price(obs, sit):
        return "all-in"
    # analyse() counts the first post-flop bet as a raise; pre-flop the blind is the bet.
    raises = sit.raises_this_street - (0 if sit.street == 0 else 1)
    return "bet" if raises <= 0 else "raise" if raises == 1 else "reraise"


def _shove_set_price(obs: Observation, sit: Situation) -> bool:
    agg = sit.last_aggressor
    last = next((a for p, a, _ in reversed(obs.action_history) if p == agg), None)
    return agg is not None and last == ALL_IN


def in_position(obs: Observation, sit: Situation) -> bool:
    """Post-flop order is seat 0..n-1, so we act last if every opponent sits before us."""
    return all(obs.positions.index(p) < sit.seat for p in sit.opponents_in_hand)


def realization(obs: Observation, sit: Situation) -> float:
    if sit.street == 3 or in_position(obs, sit):
        return 1.0
    return 0.75 if len(sit.opponents_in_hand) > 1 else 0.85


def pick_villain(sit: Situation, shrunk: dict[int, OpponentStats]) -> int | None:
    """The bettor when facing a bet; otherwise the opponent most likely to pay us off."""
    if not shrunk:
        return None
    if sit.to_call > 0 and sit.last_aggressor in shrunk:
        return sit.last_aggressor
    return max(sorted(shrunk), key=lambda p: shrunk[p].call_to_raise)


def villain_range(sit: Situation, villain: int | None, shrunk: dict[int, OpponentStats]) -> float:
    if villain is None:
        return 1.0
    s = shrunk[villain]
    raised = villain in (sit.last_aggressor, sit.preflop_aggressor)
    return min(max(s.pfr if raised else s.vpip, 0.05), 1.0)


def _bust_cost(sit: Situation, risked: float, lose_p: float) -> float:
    """Extra loss when an all-in loses and the opponent covers us: k * S (docs/tournament.md)."""
    if risked < sit.my_stack - 1e-9 or sit.effective_stack < sit.my_stack - 1e-9:
        return 0.0
    return lose_p * BUST_K * (sit.my_stack + sit.invested)


def _raise_ev(sit: Situation, n: Numbers, b: float) -> float:
    """EV(bet b) = F*P + C*(eta*Wc*(P + b + v) - b) + R*V_rr, b = chips we add now."""
    pot, c, eta = n.pot, n.to_call, n.realization
    v = min(max(b - c, 0.0), max(sit.effective_stack - c, 0.0))   # what a caller adds
    win_c = eta * n.calling_equity
    k = max(1.0, n.contestants)                                   # expected callers, each adds v
    called = win_c * (pot + b + k * v) - b - _bust_cost(sit, b, 1 - win_c)
    x = min(pot + b + v, max(sit.effective_stack - b, 0.0))       # pot-sized re-raise, capped
    call_p, raise_p = n.call_p, n.raise_p
    if x <= 1e-9:
        call_p, raise_p = call_p + raise_p, 0.0
    win_r = eta * n.reraise_equity
    call_rr = win_r * (pot + b + v + 2 * x) - x - _bust_cost(sit, b + x, 1 - win_r)
    v_rr = -b + max(0.0, call_rr)
    return n.fold_all * pot + call_p * called + raise_p * v_rr


def action_ev(sit: Situation, n: Numbers, action: str, cost: float) -> float:
    if action == FOLD:
        return 0.0
    if action == CHECK:
        return n.realization * n.equity * n.pot
    if action == CALL or (action == ALL_IN and cost <= n.to_call + 1e-9):
        win = n.realization * n.equity
        return win * (n.pot + cost + n.behind * cost) - cost - _bust_cost(sit, cost, 1 - win)
    return _raise_ev(sit, n, cost)


def _numbers(obs: Observation, sit: Situation, shrunk: dict[int, OpponentStats],
             villain: int | None, v_range: float) -> Numbers:
    # Folded players are already out of opponents_in_hand. Players who have not acted yet
    # mostly fold, so each counts as their vpip, not as a sure showdown opponent.
    acted = {p for p, _a, _amt in obs.action_history}
    stays = {p: 1.0 if p in acted else shrunk[p].vpip for p in sit.opponents_in_hand}
    contestants = sum(stays.values())
    behind = sum(v for p, v in stays.items() if p not in acted)
    n_opp = max(1, round(contestants)) if stays else 0
    board = obs.community_cards
    w = equity(obs.hole_cards, board, n_opp, EQUITY_SIMS, v_range)
    legal = dict(obs.legal_actions)
    can_raise = any(a in legal and legal[a] > sit.to_call + 1e-9 for a in (RAISE_HALF_POT, RAISE_POT, ALL_IN))
    vs = shrunk[villain] if villain is not None else PRIOR
    if can_raise and n_opp:
        wc = equity(obs.hole_cards, board, n_opp, EQUITY_SIMS, v_range * (1 - vs.fold_to_raise))
        wr = equity(obs.hole_cards, board, n_opp, EQUITY_SIMS, v_range * vs.raise_to_raise)
    else:
        wc = wr = w
    fold_all = 0.0 if sit.allin_opponents else prod(shrunk[p].fold_to_raise for p in sit.opponents_in_hand)
    cont = vs.call_to_raise + vs.raise_to_raise
    share = vs.call_to_raise / cont if cont > 0 else 1.0
    pct = hand_percentile(hand_class(obs.hole_cards))
    n = Numbers(
        pot=sit.pot, to_call=sit.to_call, pot_odds=sit.pot_odds,
        price=sit.pot_odds if sit.to_call > 0 else 1 / (n_opp + 1),
        equity=w, calling_equity=wc, reraise_equity=wr, realization=realization(obs, sit),
        fold_all=fold_all, call_p=(1 - fold_all) * share, raise_p=(1 - fold_all) * (1 - share),
        contestants=contestants, behind=behind,
        percentile=pct, open_range=opening_range(sit), roll=decision_roll(obs), ev={},
    )
    return replace(n, ev={a: action_ev(sit, n, a, cost) for a, cost in obs.legal_actions})


def _edge(sit: Situation, n: Numbers, v_type: str) -> tuple[str, float, float]:
    """(class, value, big cut-off). Pre-flop unopened compares the hand with the opening range."""
    if sit.street == 0 and sit.raises_preflop == 0:
        value = n.open_range - n.percentile
        cut = n.open_range * (1 - OPEN_VALUE_SHARE)
    else:
        value = n.realization * n.equity - n.price
        cut = EDGE_BIG_VS_STATION if v_type == "station" else EDGE_BIG
    return ("big" if value > cut else "thin" if value > 0 else "negative"), value, cut


def build_facts(obs: Observation, sit: Situation, stats: dict[int, OpponentStats]) -> Facts:
    """stats: raw tracker stats for every opponent still in the hand."""
    shrunk = {p: shrink(s) for p, s in stats.items()}
    villain = pick_villain(sit, shrunk)
    v_range = villain_range(sit, villain, shrunk)
    v_type = villain_type(stats[villain]) if villain is not None else "unknown"
    n = _numbers(obs, sit, shrunk, villain, v_range)
    edge, edge_value, edge_cut = _edge(sit, n, v_type)
    tier = None
    if sit.street == 0:
        tier = next((name for name, top in TIERS if n.percentile <= top), "trash")
    depth = "short" if sit.effective_stack < SHORT_BB else "deep" if sit.effective_stack > DEEP_BB else "medium"
    fe = "high" if n.fold_all >= FE_HIGH else "low" if n.fold_all < FE_LOW else "medium"
    return Facts(
        street=STREET_NAMES[sit.street], position=position_name(sit), facing=facing(obs, sit),
        pot_type="heads-up" if len(sit.opponents_in_hand) <= 1 else "multiway",
        depth=depth, initiative=sit.i_was_aggressor, hand_tier=tier,
        edge=edge, edge_value=edge_value, edge_cut=edge_cut,
        villain=villain, villain_type=v_type, villain_range=v_range, fold_equity=fe, n=n,
    )


def cap_for(intent: str, sit: Situation, legal: dict[str, float]) -> float:
    """max_invest for an intent (design section 6)."""
    c, pot = sit.to_call, sit.pot
    if intent == "fold":
        return 0.0
    if intent == "check_call":
        return c
    if intent == "bluff":
        return legal.get(RAISE_HALF_POT, c + (pot + c) / 2)
    if intent == "value":
        return legal.get(RAISE_POT, c + (pot + c))
    return inf  # shove: the whole stack
