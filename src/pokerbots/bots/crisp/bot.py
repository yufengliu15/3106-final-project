"""Bot A: crisp rule-based expert system (design: private/bot_a).

Four forward-chaining layers per decision:
    0. compute  -- numbers (facts.py)
    1. classify -- crisp facts (facts.py)
    2. rules    -- first matching rule sets an intent and its max_invest cap (rules.py)
    3. act      -- EV picks among legal actions under the cap; a seeded roll
                   breaks near-ties (within DELTA of the best EV)
"""

from __future__ import annotations

from dataclasses import dataclass

from pokerbots.bots.crisp.facts import Facts, build_facts, cap_for
from pokerbots.bots.crisp.rules import Rule, fire
from pokerbots.core import ACTIONS, CHECK, FOLD, Bot, Observation, RoundLog
from pokerbots.core.situation import analyse
from pokerbots.opponents import PRIOR, OpponentTracker

DELTA = 0.5  # bb: actions this close to the best EV count as a tie


@dataclass(frozen=True)
class Decision:
    """What act() computed, kept for explain()."""

    action: str
    facts: Facts
    rule: Rule
    intent: str
    cap: float
    candidates: tuple[str, ...]
    survivors: tuple[str, ...]


def pick(candidates: dict[str, float], roll: float) -> tuple[str, tuple[str, ...]]:
    """Keep actions within DELTA of the best EV; the roll chooses among them."""
    best = max(candidates.values())
    # Fold is exactly 0 EV, so rolling it against a +EV action only gives chips away.
    survivors = tuple(a for a in ACTIONS if a in candidates and candidates[a] >= best - DELTA
                      and not (a == FOLD and best > 0))
    return survivors[min(int(roll * len(survivors)), len(survivors) - 1)], survivors


class CrispBot(Bot):
    """Crisp rule-based main bot. Only reads ``core`` + ``opponents``."""

    name = "CrispBot"

    def __init__(self, use_opponent_model: bool = True) -> None:
        self.use_opponent_model = use_opponent_model
        self.tracker = OpponentTracker()
        self._last: Decision | None = None

    def new_game(self, game_id: int, player_id: int) -> None:
        self.tracker.reset()
        self._last = None

    def observe_result(self, log: RoundLog) -> None:
        self.tracker.update(log)

    def act(self, obs: Observation) -> str:
        sit = analyse(obs)
        legal = dict(obs.legal_actions)
        stats = {p: self.tracker.stats(p) if self.use_opponent_model else PRIOR for p in sit.opponents_in_hand}
        facts = build_facts(obs, sit, stats)
        rule, intent = fire(facts)
        cap = cap_for(intent, sit, legal)
        allowed = {a: facts.n.ev[a] for a, cost in legal.items() if cost <= cap + 1e-9 or a == CHECK}
        if not allowed:  # e.g. a 0-chip player offered only fold / all-in for 0
            cheapest = min(legal, key=legal.get)
            allowed = {cheapest: facts.n.ev[cheapest]}
        action, survivors = pick(allowed, facts.n.roll)
        self._last = Decision(action, facts, rule, intent, cap, tuple(allowed), survivors)
        return action

    def explain(self) -> str:
        if self._last is None:
            return "no decision yet"
        d = self._last
        f, n = d.facts, d.facts.n
        tier = f.hand_tier or "-"
        villain = f"#{f.villain}" if f.villain is not None else "-"
        facts = (f"facts: street={f.street} position={f.position} facing={f.facing} pot={f.pot_type} "
                 f"depth={f.depth} initiative={'yes' if f.initiative else 'no'} hand_tier={tier} "
                 f"villain={villain} {f.villain_type} range={f.villain_range:.0%} "
                 f"fold_equity={f.fold_equity} ({n.fold_all:.2f}) om={int(self.use_opponent_model)}")
        if f.street == "preflop" and f.facing in ("nothing", "bet"):
            edge = (f"edge={f.edge} (open range {n.open_range:.0%} - hand {n.percentile:.0%} "
                    f"= {f.edge_value:+.2f}, big > {f.edge_cut:.2f})")
        else:
            edge = (f"edge={f.edge} (eta {n.realization:.2f} x equity {n.equity:.2f} - price {n.price:.2f} "
                    f"= {f.edge_value:+.2f}, big > {f.edge_cut:.2f})")
        cap = "stack" if d.cap == float("inf") else f"{d.cap:.1f}bb"
        rule = f"{d.rule.name} {d.rule.condition} -> {d.intent}, cap {cap}"
        evs = " | ".join(f"{a} {n.ev[a]:+.2f}" for a in sorted(d.candidates, key=lambda a: -n.ev[a]))
        if len(d.survivors) > 1:
            choice = f"within delta {', '.join(d.survivors)} -> roll {n.roll:.2f} -> {d.action}"
        else:
            choice = f"best EV -> {d.action}"
        return f"{d.action}; {facts}; {edge}; {rule}; EV: {evs}; {choice}"
