"""Bot A family: crisp rule-based bots a teammate could plausibly have written.

A1  reference chart bot (chart + equity vs range + pot odds), parameters drawn
A2  pure equity thresholds, no chart
A3  Chen-style hand score pre-flop, made-hand categories post-flop (no Monte Carlo)
A4  opponent-type table: one threshold set per tight/loose x passive/aggressive
A5  stack-depth decision tree (SPR bucket -> position -> price)
A6  held-out: extreme (nit / maniac) draws of A1, A2 or A5
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from pokerbots.bots.ml.features import _hand_features
from pokerbots.bots.variants.common import (
    ALL_IN, CALL, CHECK, FOLD, VariantBot, deterministic_unit, jitter,
)
from pokerbots.core import Bot, Observation, RoundLog
from pokerbots.core.cards import hand_class, rank_of
from pokerbots.core.equity import equity, hand_percentile
from pokerbots.core.situation import Situation, analyse, legalize
from pokerbots.opponents import PRIOR, OpponentStats, OpponentTracker

EXTREME_SIGN = "extreme_sign"   # +1 nit, -1 maniac


def _ext(rng, extreme: bool) -> float:
    """Spread multiplier: normal draws stay near the defaults, extreme ones
    sit at the edge of the sane range."""
    return 2.2 if extreme else 1.0


# ---------------------------------------------------------------- A1 ----
# Priority-ordered IF-THEN rules over crisp facts; opponent labels
# (loose/tight x passive/aggressive) shift the thresholds.
MIN_ROUNDS = 12          # fewer observed rounds -> opponent is "unknown"
LOOSE_VPIP = 0.35
AGGRESSIVE_AF = 2.0


@dataclass
class Read:
    label: str                 # e.g. "loose-aggressive", "unknown"
    range_pct: float           # assumed hand range of the relevant opponent(s)
    loose: bool
    aggressive: bool
    folds_often: bool


@dataclass
class Params:
    open_pct: tuple[float, float, float] = (0.12, 0.20, 0.30)   # early, middle, late
    value_eq: float = 0.62
    nut_eq: float = 0.80
    call_margin: float = 0.04
    shove_spr: float = 1.5
    steal_fold: float = 0.6
    raise_war_eq: float = 0.72


class A1Chart(Bot):
    family, approach = "A", "A1"

    def __init__(self, use_opponent_model: bool = True, params: dict | None = None) -> None:
        params = dict(params or {})
        self.p = Params(**{f.name: params[f.name] for f in fields(Params) if f.name in params})
        self.use_opponent_model = use_opponent_model
        self.tracker = OpponentTracker()
        self.player_id = 0
        self._trace: list[str] = []

    def new_game(self, game_id: int, player_id: int) -> None:
        self.player_id = player_id
        self.tracker.reset()

    def observe_result(self, log: RoundLog) -> None:
        self.tracker.update(log)

    def explain(self) -> str:
        return " -> ".join(self._trace)

    def act(self, obs: Observation) -> str:
        sit = analyse(obs)
        legal = dict(obs.legal_actions)
        read = self.read_opponents(sit)
        eq = self.hand_equity(obs, sit, read)
        self._trace = [f"equity {eq:.2f} vs {read.label} (range {read.range_pct:.0%})",
                       f"pot odds {sit.pot_odds:.2f}, SPR {sit.spr:.1f}"]
        action, rule = self.fire_rules(obs, sit, read, eq, legal)
        action = legalize(action, obs.legal_actions)
        self._trace.append(f"rule {rule} -> {action}")
        return action

    def stats(self, pid: int) -> OpponentStats:
        if not self.use_opponent_model:
            return PRIOR
        s = self.tracker.stats(pid)
        return s if s.rounds >= MIN_ROUNDS else PRIOR

    def read_opponents(self, sit: Situation) -> Read:
        if not self.use_opponent_model or not sit.opponents_in_hand:
            return Read("unknown", 1.0 if sit.to_call == 0 else 0.5, False, False, False)
        bettor = sit.last_aggressor not in (None, self.player_id)
        focus = [sit.last_aggressor] if bettor else list(sit.opponents_in_hand)
        known = [s for s in (self.stats(p) for p in focus) if s.rounds >= MIN_ROUNDS]
        if not known:
            return Read("unknown", 0.5 if bettor else 1.0, False, False, False)
        loose = max(s.vpip for s in known) > LOOSE_VPIP
        aggressive = max(s.aggression for s in known) > AGGRESSIVE_AF
        folds = min(s.fold_to_raise for s in known) > self.p.steal_fold
        rng = max(max(s.pfr for s in known) if bettor else max(s.vpip for s in known), 0.05)
        label = ("loose" if loose else "tight") + "-" + ("aggressive" if aggressive else "passive")
        return Read(label, min(rng, 1.0), loose, aggressive, folds)

    def hand_equity(self, obs: Observation, sit: Situation, read: Read) -> float:
        rng = read.range_pct if read.range_pct < 0.99 else None
        return equity(obs.hole_cards, obs.community_cards, VariantBot.n_opponents(sit), range_pct=rng)

    def fire_rules(self, obs, sit: Situation, read: Read, eq: float, legal) -> tuple[str, str]:
        p = self.p
        can = legal.__contains__
        call_or_shove = CALL if can(CALL) else ALL_IN
        small, big = VariantBot.raises(legal)

        if sit.street == 0 and sit.raises_preflop == 0:     # opening chart
            pct = hand_percentile(hand_class(obs.hole_cards))
            band = 2 if sit.seats_to_button <= 1 else (1 if sit.seats_to_button <= 3 else 0)
            limit = p.open_pct[band] * (1.3 if read.folds_often else 1.0)
            if pct <= limit and big:
                return big, "R1 open raise (hand in position chart)"
            if pct <= limit * 1.5 and can(CHECK):
                return CHECK, "R2 big blind check-option"
            if pct <= limit * 1.3 and can(CALL) and sit.to_call <= 1.0 and read.loose:
                return CALL, "R3 complete vs loose table"
            return (CHECK, "R4 check (free)") if can(CHECK) else (FOLD, "R5 fold outside chart")

        if eq >= p.nut_eq:                                   # R6 nuts
            if sit.spr <= 2.5 and can(ALL_IN):
                return ALL_IN, "R6 nuts + low SPR -> all-in"
            if big:
                return big, "R6 nuts -> raise pot"
            return (call_or_shove if can(FOLD) else CHECK), "R6 nuts (no raise) -> call"

        if sit.spr < p.shove_spr and eq >= max(0.5, sit.pot_odds + 0.05) and can(ALL_IN):
            return ALL_IN, "R7 short SPR + good equity -> all-in"

        # R8 value: thinner vs loose-passive, thicker vs tight-aggressive
        value_eq = p.value_eq - (0.05 if read.loose and not read.aggressive else 0.0) \
                   + (0.05 if read.aggressive and not read.loose else 0.0)
        if eq >= value_eq and small:
            if sit.raises_this_street >= 2 and eq < p.raise_war_eq:
                if can(FOLD):
                    return call_or_shove, "R8b value but raise war -> call"
            else:
                pick = big if (read.loose and not read.aggressive) else small
                return pick, "R8 value raise" + (" (pot vs calling station)" if pick == big else "")

        if can(FOLD):                                        # R9/R10 pot odds
            margin = p.call_margin
            if read.aggressive:
                margin -= 0.04
            elif not read.loose and read.label != "unknown":
                margin += 0.05
            if eq > sit.pot_odds + margin:
                return call_or_shove, f"R9 equity {eq:.2f} > pot odds {sit.pot_odds:.2f} + {margin:.2f}"
            return FOLD, "R10 fold: price too high"

        if read.folds_often and len(sit.opponents_in_hand) <= 2 and sit.left_to_act <= 1 and small:
            return small, "R11 bluff vs frequent folders"
        return CHECK, "R12 check"

    @classmethod
    def sample_params(cls, rng, extreme: bool = False) -> dict:
        d = Params()
        s = _ext(rng, extreme)
        sign = rng.choice((-1, 1)) if extreme else 0
        widen = {1: 0.55, -1: 2.2}.get(sign) or rng.uniform(0.7, 1.35)
        return dict(
            open_pct=tuple(min(0.9, x * widen) for x in d.open_pct),
            value_eq=jitter(rng, d.value_eq + 0.08 * sign, 0.08 * s, 0.5, 0.85),
            nut_eq=jitter(rng, d.nut_eq + 0.05 * sign, 0.05 * s, 0.7, 0.92),
            call_margin=d.call_margin + sign * 0.06 + rng.uniform(-0.04, 0.04),
            shove_spr=jitter(rng, d.shove_spr, 0.4 * s, 0.5, 4.0),
            steal_fold=jitter(rng, d.steal_fold, 0.2, 0.3, 0.9),
            raise_war_eq=jitter(rng, d.raise_war_eq + 0.05 * sign, 0.06, 0.6, 0.9),
            **{EXTREME_SIGN: sign},
        )


# ---------------------------------------------------------------- A2 ----
class A2Threshold(VariantBot):
    family, approach = "A", "A2"

    @classmethod
    def default_params(cls) -> dict:
        return dict(t_open=0.20, call_margin=0.03, t_raise=0.62, t_pot=0.72, t_allin=0.85,
                    shove_spr=1.2, bluff=0.08, opp_shift=0.06, conf_rounds=20)

    @classmethod
    def sample_params(cls, rng, extreme: bool = False) -> dict:
        d = cls.default_params()
        sign = rng.choice((-1, 1)) if extreme else 0
        k = 0.1 if extreme else 0.0
        return dict(
            t_open=min(0.6, max(0.05, d["t_open"] * (rng.uniform(0.6, 1.6) if not extreme
                                                      else (0.45 if sign > 0 else 2.5)))),
            call_margin=d["call_margin"] + sign * 0.08 + rng.uniform(-0.04, 0.04),
            t_raise=min(0.85, max(0.5, d["t_raise"] + sign * k + rng.uniform(-0.06, 0.06))),
            t_pot=min(0.9, max(0.55, d["t_pot"] + sign * k + rng.uniform(-0.06, 0.06))),
            t_allin=min(0.95, max(0.65, d["t_allin"] + sign * k + rng.uniform(-0.05, 0.05))),
            shove_spr=rng.uniform(0.6, 2.5),
            bluff=max(0.0, (0.0 if sign > 0 else 0.3 if sign < 0 else d["bluff"]) + rng.uniform(-0.05, 0.05)),
            opp_shift=rng.uniform(0.0, 0.12),
            conf_rounds=rng.choice((10, 20, 40)),
            **{EXTREME_SIGN: sign},
        )

    def decide(self, obs, sit, legal):
        p = self.p
        rd = self.read(sit)
        small, big = self.raises(legal)
        if sit.street == 0 and sit.raises_preflop == 0:
            pct = self.percentile(obs)
            late = sit.seats_to_button <= 2
            limit = p["t_open"] * (1.5 if late else 1.0) * (1.25 if rd.fold_to_raise > 0.6 else 1.0)
            if pct <= limit and big:
                return big, f"open: top {pct:.0%} <= {limit:.0%}"
            return self.passive(legal), f"no open: top {pct:.0%}"
        eq = self.equity_vs(obs, sit, rd)
        # loose-passive opponents pay off: value-bet thinner; aggressive ones bluff: call lighter
        shift = p["opp_shift"] * rd.confidence
        loose = rd.vpip > 0.35
        aggressive = rd.aggr > 2.0
        t_raise = p["t_raise"] - (shift if loose and not aggressive else 0.0)
        margin = p["call_margin"] - (shift if aggressive else 0.0)
        if eq >= p["t_allin"] or (sit.spr < p["shove_spr"] and eq >= max(0.5, sit.pot_odds + 0.05)):
            if ALL_IN in legal:
                return ALL_IN, f"eq {eq:.2f} all-in band"
        if eq >= p["t_pot"] and big:
            return big, f"eq {eq:.2f} >= pot band"
        if eq >= t_raise and small:
            return small, f"eq {eq:.2f} >= raise band"
        if FOLD in legal:
            if eq > sit.pot_odds + margin:
                return self.call_or_shove(legal), f"eq {eq:.2f} > odds {sit.pot_odds:.2f}+{margin:.2f}"
            return FOLD, f"eq {eq:.2f} below price"
        if small and len(sit.opponents_in_hand) <= 2 and deterministic_unit(obs) < p["bluff"] * (
                1.5 if rd.fold_to_raise > 0.6 else 1.0):
            return small, "bluff"
        return CHECK, f"eq {eq:.2f} check"


# ---------------------------------------------------------------- A3 ----
def chen_score(hole) -> float:
    """Bill Chen's pre-flop formula (approximation, 0..20)."""
    vals = {12: 10, 11: 8, 10: 7, 9: 6}
    r = sorted((rank_of(c) for c in hole), reverse=True)
    hi, lo = r
    score = vals.get(hi, (hi + 2) / 2)
    if hi == lo:
        return max(5.0, score * 2)
    if hole[0][0] == hole[1][0]:
        score += 2
    gap = hi - lo - 1
    score -= (0, 1, 2, 4, 5)[min(gap, 4)]
    if gap <= 1 and hi < 10:
        score += 1
    return score


class A3ChenMade(VariantBot):
    family, approach = "A", "A3"

    @classmethod
    def default_params(cls) -> dict:
        return dict(open_chen=8.0, call_chen=9.5, shove_chen=12.0, value_class=8, top_pair_calls=True,
                    draw_odds=0.3, bet_made=True, bluff=0.05, conf_rounds=20)

    @classmethod
    def sample_params(cls, rng, extreme: bool = False) -> dict:
        sign = rng.choice((-1, 1)) if extreme else 0
        return dict(open_chen=rng.uniform(6.0, 10.0) + sign * 2.0, call_chen=rng.uniform(8.0, 11.0) + sign * 1.5,
                    shove_chen=rng.uniform(10.0, 14.0), value_class=rng.choice((7, 8, 8)),
                    top_pair_calls=rng.random() < 0.8, draw_odds=rng.uniform(0.2, 0.4),
                    bet_made=rng.random() < 0.8, bluff=rng.uniform(0.0, 0.12),
                    conf_rounds=rng.choice((10, 20, 40)), **{EXTREME_SIGN: sign})

    def decide(self, obs, sit, legal):
        p = self.p
        rd = self.read(sit)
        small, big = self.raises(legal)
        if sit.street == 0:
            c = chen_score(obs.hole_cards)
            late = sit.seats_to_button <= 2
            opener = p["open_chen"] - (1.0 if late else 0.0) - (1.0 if rd.fold_to_raise > 0.6 else 0.0)
            if c >= p["shove_chen"] and (sit.raises_preflop >= 2 or sit.spr < 3) and ALL_IN in legal:
                return ALL_IN, f"chen {c:.0f} premium shove"
            if sit.raises_preflop == 0:
                if c >= opener and big:
                    return big, f"chen {c:.0f} open"
                return self.passive(legal), f"chen {c:.0f} too weak"
            need = p["call_chen"] + (sit.raises_preflop - 1) * 1.5 - (1.0 if rd.vpip > 0.4 else 0.0)
            if c >= need + 2 and small:
                return small, f"chen {c:.0f} re-raise"
            if c >= need:
                return self.call_or_shove(legal) if FOLD in legal else CHECK, f"chen {c:.0f} call"
            return self.passive(legal), f"chen {c:.0f} fold to raise"
        h = _hand_features(obs.hole_cards, obs.community_cards)
        made = int(h["made_class"])  # 1 straight flush .. 9 high card
        draw = h["flush_draw"] or h["straight_draw"]
        if made <= 4:  # straight or better
            if sit.spr < 3 and ALL_IN in legal:
                return ALL_IN, f"class {made} strong, low SPR"
            return (big or self.call_or_shove(legal)), f"class {made} strong"
        if made <= 6 or (made <= p["value_class"] and h["top_pair_plus"]):  # trips/two pair or top pair
            if FOLD in legal:
                if sit.raises_this_street >= 2 and made >= 7:
                    return FOLD, "top pair vs raise war"
                return (self.call_or_shove(legal) if p["top_pair_calls"] or made <= 6 else FOLD), \
                    f"class {made} call"
            return (small if p["bet_made"] and small else CHECK), f"class {made} bet"
        if draw and FOLD in legal:
            ok = sit.pot_odds <= p["draw_odds"]
            return (self.call_or_shove(legal), "draw, price ok") if ok else (FOLD, "draw, price too high")
        if FOLD in legal:
            return FOLD, f"class {made} nothing"
        if small and deterministic_unit(obs) < p["bluff"]:
            return small, "bluff"
        return CHECK, f"class {made} check"


# ---------------------------------------------------------------- A4 ----
TYPES = ("unknown", "tight-passive", "tight-aggressive", "loose-passive", "loose-aggressive")


class A4TypeTable(VariantBot):
    family, approach = "A", "A4"

    @classmethod
    def default_params(cls) -> dict:
        base = dict(open=0.2, margin=0.03, raise_=0.64, allin=0.85, bluff=0.05)
        table = {t: dict(base) for t in TYPES}
        table["tight-passive"].update(margin=0.08, bluff=0.12)
        table["tight-aggressive"].update(margin=0.04, raise_=0.68)
        table["loose-passive"].update(raise_=0.57, bluff=0.0, margin=0.02)
        table["loose-aggressive"].update(margin=-0.02, raise_=0.66, allin=0.8)
        return dict(table=table, min_rounds=12, loose_vpip=0.35, aggr_af=2.0, conf_rounds=0)

    @classmethod
    def sample_params(cls, rng, extreme: bool = False) -> dict:
        d = cls.default_params()
        table = {t: {k: (v + rng.uniform(-0.04, 0.04) if k != "open" else v * rng.uniform(0.7, 1.4))
                     for k, v in row.items()} for t, row in d["table"].items()}
        for row in table.values():
            row["bluff"] = max(0.0, row["bluff"])
        return dict(table=table, min_rounds=rng.choice((8, 12, 20)), loose_vpip=rng.uniform(0.3, 0.42),
                    aggr_af=rng.uniform(1.6, 2.6), conf_rounds=0)

    def opp_type(self, sit) -> str:
        if not self.use_opponent_model or not sit.opponents_in_hand:
            return "unknown"
        bettor = sit.last_aggressor not in (None, self.player_id)
        focus = [sit.last_aggressor] if bettor else list(sit.opponents_in_hand)
        st = [self.tracker.stats(pid) for pid in focus]
        st = [s for s in st if s.rounds >= self.p["min_rounds"]]
        if not st:
            return "unknown"
        loose = max(s.vpip for s in st) > self.p["loose_vpip"]
        aggr = max(s.aggression for s in st) > self.p["aggr_af"]
        return ("loose" if loose else "tight") + "-" + ("aggressive" if aggr else "passive")

    def decide(self, obs, sit, legal):
        t = self.opp_type(sit)
        row = self.p["table"][t]
        rd = self.read(sit) if t != "unknown" else None
        small, big = self.raises(legal)
        if sit.street == 0 and sit.raises_preflop == 0:
            pct = self.percentile(obs)
            limit = row["open"] * (1.5 if sit.seats_to_button <= 2 else 1.0)
            if pct <= limit and big:
                return big, f"[{t}] open top {pct:.0%}"
            return self.passive(legal), f"[{t}] no open"
        eq = self.equity_vs(obs, sit, rd)
        if eq >= row["allin"] and ALL_IN in legal:
            return ALL_IN, f"[{t}] eq {eq:.2f} all-in"
        if eq >= row["raise_"] and small:
            return (big if t == "loose-passive" and big else small), f"[{t}] eq {eq:.2f} raise"
        if FOLD in legal:
            if eq > sit.pot_odds + row["margin"]:
                return self.call_or_shove(legal), f"[{t}] eq {eq:.2f} call"
            return FOLD, f"[{t}] eq {eq:.2f} fold"
        if small and deterministic_unit(obs) < row["bluff"]:
            return small, f"[{t}] bluff"
        return CHECK, f"[{t}] check"


# ---------------------------------------------------------------- A5 ----
class A5DepthTree(VariantBot):
    family, approach = "A", "A5"

    @classmethod
    def default_params(cls) -> dict:
        return dict(short_spr=2.0, deep_spr=6.0, short_eq=0.52, mid_value=0.66, deep_value=0.72,
                    pos_bonus=0.04, cheap=0.2, cheap_eq=0.3, open_eq=0.18, conf_rounds=20)

    @classmethod
    def sample_params(cls, rng, extreme: bool = False) -> dict:
        sign = rng.choice((-1, 1)) if extreme else 0
        s = 0.1 * sign
        return dict(short_spr=rng.uniform(1.2, 3.0), deep_spr=rng.uniform(4.0, 9.0),
                    short_eq=min(0.75, max(0.4, rng.uniform(0.46, 0.58) + s)),
                    mid_value=min(0.85, max(0.5, rng.uniform(0.6, 0.72) + s)),
                    deep_value=min(0.9, max(0.55, rng.uniform(0.66, 0.78) + s)),
                    pos_bonus=rng.uniform(0.0, 0.08), cheap=rng.uniform(0.1, 0.3),
                    cheap_eq=min(0.5, max(0.15, rng.uniform(0.25, 0.38) + s)),
                    open_eq=min(0.5, max(0.05, rng.uniform(0.12, 0.26) * (0.5 if sign > 0 else 2.2 if sign < 0 else 1))),
                    conf_rounds=rng.choice((10, 20, 40)), **{EXTREME_SIGN: sign})

    def decide(self, obs, sit, legal):
        p = self.p
        rd = self.read(sit)
        small, big = self.raises(legal)
        in_pos = sit.seats_to_button <= 1 or sit.left_to_act == 0
        if sit.street == 0 and sit.raises_preflop == 0:
            pct = self.percentile(obs)
            lim = p["open_eq"] * (1.6 if in_pos else 1.0)
            return (big, "open") if pct <= lim and big else (self.passive(legal), "fold/check pre")
        eq = self.equity_vs(obs, sit, rd) + (p["pos_bonus"] if in_pos else 0.0)
        facing = FOLD in legal
        if sit.spr <= p["short_spr"]:                       # SHORT: commit or fold
            if eq >= max(p["short_eq"], sit.pot_odds + 0.03) and ALL_IN in legal:
                return ALL_IN, f"short SPR {sit.spr:.1f}, eq {eq:.2f} -> commit"
            return (FOLD if facing else CHECK), f"short SPR {sit.spr:.1f}, eq {eq:.2f} -> give up"
        value = p["mid_value"] if sit.spr <= p["deep_spr"] else p["deep_value"]
        if eq >= value:                                     # value
            if sit.raises_this_street >= 2 and eq < value + 0.1 and facing:
                return self.call_or_shove(legal), "value, but 3-bet pot -> call"
            return ((big if in_pos else small) or self.call_or_shove(legal)), f"eq {eq:.2f} value"
        if facing:                                          # price
            price = sit.to_call / max(sit.pot, 1e-9)
            if price <= p["cheap"] and eq >= p["cheap_eq"]:
                return self.call_or_shove(legal), f"cheap ({price:.2f} pot), eq {eq:.2f}"
            if eq > sit.pot_odds + 0.02:
                return self.call_or_shove(legal), f"eq {eq:.2f} > odds"
            return FOLD, f"eq {eq:.2f} fold"
        if in_pos and small and rd.fold_to_raise > 0.55 and len(sit.opponents_in_hand) == 1:
            return small, "position stab"
        return CHECK, "check"


# ---------------------------------------------------------------- A6 ----
A6_BASES = (A1Chart, A2Threshold, A5DepthTree)
