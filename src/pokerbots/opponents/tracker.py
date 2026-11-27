"""Opponent stats tracker (#17): shared counting code for every main bot.

Counts per player ID over a sliding window of the last `window` rounds that
player was dealt into. Bots read the stats their own way; the tracker only
counts. Update it from observe_result(); reset() on new_game().

Stats (all fractions 0-1 except aggression):
    vpip           -- voluntarily put chips in pre-flop (call/raise/all-in, not the BB check)
    pfr            -- raised (or shoved) pre-flop
    aggression     -- (raises + all-ins) / calls over all streets
    fold_to_raise  -- folds / times the player acted facing a raise on that street
    call_to_raise  -- calls / times the player acted facing a raise on that street
    raise_to_raise -- re-raises (incl. all-in) / times the player acted facing a raise
    allin_rate     -- rounds with an all-in
    showdown_rate  -- rounds reaching showdown / rounds with VPIP
    rounds         -- rounds observed (sample size)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from pokerbots.core import ALL_IN, CALL, FOLD, RAISE_HALF_POT, RAISE_POT, RoundLog

AGGRESSIVE = (RAISE_HALF_POT, RAISE_POT, ALL_IN)


@dataclass(frozen=True)
class OpponentStats:
    vpip: float
    pfr: float
    aggression: float
    fold_to_raise: float
    allin_rate: float
    showdown_rate: float
    rounds: int
    # After `rounds` with defaults so positional construction (bots/ml) keeps working.
    call_to_raise: float = 0.35
    raise_to_raise: float = 0.15


# Population priors: used for unseen players and by bots with the opponent
# model switched off. Roughly an average table of the scripted pool.
# fold_to_raise + call_to_raise + raise_to_raise = 1.
PRIOR = OpponentStats(vpip=0.35, pfr=0.2, aggression=1.5, fold_to_raise=0.5,
                      allin_rate=0.1, showdown_rate=0.4, rounds=0,
                      call_to_raise=0.35, raise_to_raise=0.15)


@dataclass(frozen=True)
class _RoundRecord:
    vpip: bool
    pfr: bool
    raises: int
    calls: int
    faced_raise: int
    folded_to_raise: int
    called_raise: int
    reraised: int
    allin: bool
    showdown: bool


class OpponentTracker:
    def __init__(self, window: int = 50) -> None:
        self.window = window
        self._records: dict[int, deque[_RoundRecord]] = {}

    def reset(self) -> None:
        self._records.clear()

    def update(self, log: RoundLog) -> None:
        per: dict[int, dict] = {
            pid: dict(vpip=False, pfr=False, raises=0, calls=0, faced=0, ftr=0, ctr=0, rtr=0, allin=False)
            for pid in log.positions
        }
        street = None
        raised_by: set[int] = set()
        for pid, st, action, _amount in log.actions:
            if st != street:
                street, raised_by = st, set()
            r = per.setdefault(pid, dict(vpip=False, pfr=False, raises=0, calls=0,
                                         faced=0, ftr=0, ctr=0, rtr=0, allin=False))
            facing = bool(raised_by - {pid})
            if facing:
                r["faced"] += 1
                response = "ftr" if action == FOLD else "ctr" if action == CALL else "rtr" if action in AGGRESSIVE else None
                if response:
                    r[response] += 1
            if action in AGGRESSIVE:
                r["raises"] += 1
                raised_by.add(pid)
                if action == ALL_IN:
                    r["allin"] = True
                if st == "preflop":
                    r["vpip"] = r["pfr"] = True
            elif action == CALL:
                r["calls"] += 1
                if st == "preflop":
                    r["vpip"] = True
        for pid, r in per.items():
            rec = _RoundRecord(r["vpip"], r["pfr"], r["raises"], r["calls"], r["faced"],
                               r["ftr"], r["ctr"], r["rtr"], r["allin"], pid in log.showdown_cards)
            self._records.setdefault(pid, deque(maxlen=self.window)).append(rec)

    def stats(self, player_id: int) -> OpponentStats:
        recs = self._records.get(player_id)
        if not recs:
            return PRIOR
        n = len(recs)
        vp = sum(r.vpip for r in recs)
        raises = sum(r.raises for r in recs)
        calls = sum(r.calls for r in recs)
        faced = sum(r.faced_raise for r in recs)
        return OpponentStats(
            vpip=vp / n,
            pfr=sum(r.pfr for r in recs) / n,
            aggression=raises / calls if calls else float(min(raises, 10)),
            fold_to_raise=sum(r.folded_to_raise for r in recs) / faced if faced else PRIOR.fold_to_raise,
            allin_rate=sum(r.allin for r in recs) / n,
            showdown_rate=sum(r.showdown for r in recs) / vp if vp else 0.0,
            rounds=n,
            call_to_raise=sum(r.called_raise for r in recs) / faced if faced else PRIOR.call_to_raise,
            raise_to_raise=sum(r.reraised for r in recs) / faced if faced else PRIOR.raise_to_raise,
        )
