"""Observable betting context and research-inspired equity inputs for Bot B."""

from __future__ import annotations

from math import prod

from pokerbots.bots.fuzzy.showdown import ShowdownEstimate, selective_showdown
from pokerbots.core import ALL_IN, FOLD, Observation
from pokerbots.core.cards import hand_class
from pokerbots.core.equity import equity, hand_percentile
from pokerbots.core.situation import AGGRESSIVE, Situation
from pokerbots.opponents import PRIOR, OpponentStats


def _with_prior(value: float, prior: float, rounds: int) -> float:
    return (rounds * value + 16 * prior) / (rounds + 16)


def estimated_equity(obs: Observation, sit: Situation, opponents: dict[int, OpponentStats],
                     stack_call: bool) -> tuple[float, ShowdownEstimate | None]:
    """Unconditional and action-conditioned shares of the showdown pot."""
    count = len(opponents)
    raw_equity = equity(obs.hole_cards, obs.community_cards, count)
    bettor = sit.last_aggressor
    if sit.to_call <= 0 or bettor not in opponents:
        estimate = (selective_showdown(obs.hole_cards, obs.community_cards, (1.0,) * count)
                    if stack_call else None)
        return raw_equity, estimate

    stats = opponents[bettor]
    if sit.street == 0:
        fraction = _with_prior(stats.pfr, PRIOR.pfr, stats.rounds)
    else:
        aggression = _with_prior(stats.aggression, PRIOR.aggression, stats.rounds)
        # A larger raise/call ratio leaves more room for thin value and bluffs.
        fraction = min(0.75, 0.22 + 0.09 * aggression)
    last_aggressive = next(
        (action for pid, action, _ in reversed(obs.action_history) if pid == bettor and action in AGGRESSIVE), None
    )
    if last_aggressive == ALL_IN:
        frequency = _with_prior(stats.allin_rate, PRIOR.allin_rate, stats.rounds)
        fraction = max(fraction, 1.6 * frequency)
    fraction = min(1.0, max(0.12, 1.5 * fraction))
    ranges = (fraction,) + (1.0,) * (count - 1)
    weighted = selective_showdown(obs.hole_cards, obs.community_cards, ranges)
    return raw_equity, weighted


def opening_ratio(obs: Observation, sit: Situation) -> float:
    """Starting-hand percentile divided by the opening range at this seat."""
    opponents = len(sit.opponents_in_hand)
    players_behind = sum(pid in sit.opponents_in_hand
                         for pid in obs.positions[sit.seat + 1:] + obs.positions[:min(sit.seat, 2)])
    if sit.seat < 2:
        players_behind = int(sit.seat == 0 and obs.positions[1] in sit.opponents_in_hand)
    late = 1.0 - players_behind / max(1, opponents)
    # The tighter practice profile shrank the position-based range by 9%.
    opening_range = (0.14 + 0.18 * late + 0.09 * max(0, 4 - opponents)) / 1.10
    return hand_percentile(hand_class(obs.hole_cards)) / opening_range


def stack_call_price(obs: Observation, sit: Situation, loss_probability: float) -> float:
    """Price of a stack-capped call, with the bust penalty weighted by loss."""

    # Only matched contributions belong to pots the caller can win.
    contributions = dict.fromkeys(obs.positions, 0.0)
    for pid, _action, amount in obs.action_history:
        contributions[pid] += amount
    blinds_in_pot = min(1.5, max(0.0, obs.pot - sum(contributions.values())))
    contributions[obs.positions[0]] += min(0.5, blinds_in_pot)
    contributions[obs.positions[1]] += max(0.0, blinds_in_pot - 0.5)
    cap = contributions[obs.player_id] + sit.to_call
    unmatchable = sum(max(0.0, amount - cap) for amount in contributions.values())
    contestable = max(0.0, obs.pot - unmatchable)

    # TournamentConfig.k defaults to one starting stack of additional loss
    # when a hand ends in a bust. The observation exposes no custom k.
    bust_cost = (sit.my_stack + sit.invested) * loss_probability
    return (sit.to_call + bust_cost) / max(contestable + sit.to_call, 1e-9)


def fold_equity(sit: Situation, opponents: dict[int, OpponentStats]) -> float:
    """Estimated chance that a small raise wins the pot immediately."""
    if not opponents or sit.allin_opponents or len(opponents) > 2:
        return 0.0
    return prod(_with_prior(stats.fold_to_raise, PRIOR.fold_to_raise, stats.rounds)
                for stats in opponents.values())


def is_stack_call(legal: dict[str, float]) -> bool:
    return legal.keys() == {FOLD, ALL_IN}
