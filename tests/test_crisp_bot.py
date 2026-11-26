from pokerbots.bots.crisp.bot import CrispBot, pick
from pokerbots.core import Observation

IDS = tuple(range(1001, 1011))  # 10 seats; seat 2 = UTG shoves, we are seat 3


def _aces_after_utg_shove(shove: float, round_id: int) -> Observation:
    stacks = [99.5, 99.0, 0.0] + [100.0] * 7
    pot = 1.5 + shove
    to_call = max(shove, 1.0)
    legal = (("fold", 0.0), ("call", to_call), ("raise_half_pot", to_call + (pot + to_call) / 2),
             ("raise_pot", to_call + pot + to_call), ("all_in", 100.0))
    return Observation(hole_cards=("SA", "HA"), community_cards=(), pot=pot, player_id=IDS[3],
                       positions=IDS, stacks=tuple(stacks), action_history=((IDS[2], "all_in", shove),),
                       legal_actions=legal, game_id=1, rotation_id=0, round_id=round_id)


def test_aces_never_fold_to_a_shove_smaller_than_the_blind():
    for r in range(20):
        assert CrispBot().act(_aces_after_utg_shove(0.01, r)) != "fold"


def test_a_real_shove_still_goes_through_the_all_in_rule():
    bot = CrispBot()
    bot.act(_aces_after_utg_shove(20.0, 0))
    assert bot._last.facts.facing == "all-in"


def test_fold_is_never_rolled_against_a_positive_ev_action():
    for roll in (0.0, 0.3, 0.5, 0.99):
        assert pick({"fold": 0.0, "call": 0.01}, roll)[0] == "call"
