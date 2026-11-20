"""Bot C feature vector (#3), built only from the Observation and the shared
opponent tracker, so Bot C knows exactly what Bots A and B know.

use_opponent_model=False replaces every opponent statistic with the
population prior (experiment 3 ablation).
"""

from __future__ import annotations

from collections import Counter

from treys import Evaluator

from pokerbots.core import ACTIONS, Observation
from pokerbots.core.cards import hand_class, rank_of, suit_of, to_treys
from pokerbots.core.equity import equity, hand_percentile
from pokerbots.core.situation import analyse
from pokerbots.opponents import PRIOR, OpponentStats, OpponentTracker

_EVAL = Evaluator()

STAT_FIELDS = ("vpip", "pfr", "aggression", "fold_to_raise", "allin_rate", "showdown_rate")


def _stat_vec(s: OpponentStats) -> list[float]:
    return [s.vpip, s.pfr, min(s.aggression, 10.0), s.fold_to_raise, s.allin_rate, s.showdown_rate]


def _board_texture(board: tuple[str, ...]) -> dict[str, float]:
    if not board:
        return dict(board_paired=0, board_flush=0, board_conn=0, board_high=0)
    ranks = sorted({rank_of(c) for c in board})
    suits = Counter(suit_of(c) for c in board)
    conn = max(sum(1 for r in ranks if lo <= r < lo + 5) for lo in range(-1, 10))
    return dict(
        board_paired=float(len(ranks) < len(board)),
        board_flush=float(max(suits.values())),         # cards of the most common suit
        board_conn=float(conn),                          # most board ranks in a 5-rank window
        board_high=float(max(ranks)) / 12,
    )


def _hand_features(hole, board) -> dict[str, float]:
    r1, r2 = sorted((rank_of(c) for c in hole), reverse=True)
    out = dict(pair=float(r1 == r2), suited=float(hole[0][0] == hole[1][0]),
               high=r1 / 12, low=r2 / 12, gap=float(r1 - r2),
               pct=hand_percentile(hand_class(hole)))
    if board:
        cls = _EVAL.get_rank_class(_EVAL.evaluate([to_treys(c) for c in hole], [to_treys(c) for c in board]))
        out["made_class"] = float(cls)          # 1 straight flush .. 9 high card
        cards = hole + board
        suits = Counter(suit_of(c) for c in cards)
        out["flush_draw"] = float(max(suits.values()) == 4 and len(board) < 5)
        ranks = {rank_of(c) for c in cards} | ({-1} if any(rank_of(c) == 12 for c in cards) else set())
        out["straight_draw"] = float(len(board) < 5 and max(
            sum(1 for r in ranks if lo <= r < lo + 5) for lo in range(-1, 9)) == 4)
        top_board = max(rank_of(c) for c in board)
        out["top_pair_plus"] = float(cls <= 8 and (cls < 8 or any(rank_of(c) >= top_board for c in hole)))
    else:
        out.update(made_class=10.0, flush_draw=0.0, straight_draw=0.0, top_pair_plus=0.0)
    return out


def _shrunk(s: OpponentStats, k: float) -> OpponentStats:
    """Blend stats toward the population prior by sample size: w = n / (n + k)."""
    if k <= 0 or s.rounds == 0:
        return s
    w = s.rounds / (s.rounds + k)
    mix = lambda a, b: w * a + (1 - w) * b  # noqa: E731
    return OpponentStats(mix(s.vpip, PRIOR.vpip), mix(s.pfr, PRIOR.pfr),
                         mix(min(s.aggression, 10.0), PRIOR.aggression),
                         mix(s.fold_to_raise, PRIOR.fold_to_raise), mix(s.allin_rate, PRIOR.allin_rate),
                         mix(s.showdown_rate, PRIOR.showdown_rate), s.rounds)


# columns that change when the opponent model is switched off (stored twice by the
# labeller so training can apply opponent-model dropout)
def opponent_columns(cols) -> list[str]:
    return [c for c in cols if c.startswith(("agg_", "field_")) or c in (
        "assumed_range", "eq_range", "edge", "om")]


def features(obs: Observation, tracker: OpponentTracker, use_opponent_model: bool = True,
             shrink: float = 0.0) -> dict[str, float]:
    """shrink > 0 (Bot C v2): stats are blended toward PRIOR by n / (n + shrink) and
    an `om` flag column says whether the opponent model is on."""
    sit = analyse(obs)
    me = obs.player_id

    def stats(pid) -> OpponentStats:
        return _shrunk(tracker.stats(pid), shrink) if use_opponent_model else PRIOR

    n_in = max(len(sit.opponents_in_hand), 1)
    n_eq = n_in if sit.street > 0 else max(1, min(n_in, sit.raises_preflop + 1))
    agg = sit.last_aggressor if sit.last_aggressor not in (None, me) else None
    agg_stats = stats(agg) if agg is not None else PRIOR
    field = [stats(p) for p in sit.opponents_in_hand] or [PRIOR]
    # assumed opponent range: the bettor's raising range, else the field's VPIP
    rng = max(agg_stats.pfr, 0.05) if agg is not None else max(max(s.vpip for s in field), 0.05)
    eq_rand = equity(obs.hole_cards, obs.community_cards, n_eq)
    eq_range = equity(obs.hole_cards, obs.community_cards, n_eq, range_pct=rng if rng < 0.99 else None)

    legal = dict(obs.legal_actions)
    pot = max(obs.pot, 1e-9)
    f: dict[str, float] = dict(
        street=float(sit.street),
        eq_rand=eq_rand,
        eq_range=eq_range,
        assumed_range=rng,
        pot=obs.pot,
        to_call=sit.to_call,
        pot_odds=sit.pot_odds,
        edge=eq_range - sit.pot_odds,
        my_stack=sit.my_stack,
        eff_stack=sit.effective_stack,
        spr=min(sit.spr, 50.0),
        invested_frac=sit.invested / pot,
        n_players=float(sit.n_players),
        opp_in_hand=float(len(sit.opponents_in_hand)),
        allin_opp=float(sit.allin_opponents),
        left_to_act=float(sit.left_to_act),
        seats_to_button=float(sit.seats_to_button),
        rel_position=1.0 - sit.seats_to_button / max(sit.n_players - 1, 1),
        raises_street=float(sit.raises_this_street),
        raises_preflop=float(sit.raises_preflop),
        was_aggressor=float(sit.i_was_aggressor),
        facing_allin=float(sit.facing_allin),
        stack_rank=float(sorted(obs.stacks, reverse=True).index(sit.my_stack)) / max(len(obs.stacks) - 1, 1),
        has_agg=float(agg is not None),
    )
    for a in ACTIONS:
        f[f"legal_{a}"] = float(a in legal)
        f[f"cost_{a}"] = legal.get(a, 0.0) / pot
    f.update(_hand_features(obs.hole_cards, obs.community_cards))
    f.update(_board_texture(obs.community_cards))
    for name, v in zip(STAT_FIELDS, _stat_vec(agg_stats)):
        f[f"agg_{name}"] = v
    f["agg_rounds"] = float(min(agg_stats.rounds, 50))
    for name, i in zip(STAT_FIELDS, range(6)):
        vals = [_stat_vec(s)[i] for s in field]
        f[f"field_mean_{name}"] = sum(vals) / len(vals)
        f[f"field_max_{name}"] = max(vals)
    f["field_min_rounds"] = float(min(min(s.rounds for s in field), 50))
    if shrink > 0:
        f["om"] = float(use_opponent_model)
    return f


def feature_names(shrink: float = 0.0) -> list[str]:
    """Stable column order (taken from a dummy observation)."""
    from pokerbots.core import CALL, FOLD
    obs = Observation(("SA", "HK"), (), 1.5, 1, (1, 2), (99.5, 99.0), (),
                      ((FOLD, 0.0), (CALL, 0.5)), 0, 0, 0)
    return list(features(obs, OpponentTracker(), shrink=shrink).keys())
