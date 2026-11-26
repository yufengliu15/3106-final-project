import threading

import pytest

from pokerbots.bots.simple import RandomBot
from pokerbots.core import CALL, CHECK, FOLD, RAISE_POT, Observation
from pokerbots.tournament import TournamentConfig
from pokerbots.web.__main__ import Session, join_request, post_request
from pokerbots.web.hands import best_hand
from pokerbots.web.humans import MAX_TABLE, HumanSeat, Lobby
from pokerbots.web.sandbox import Sandbox


def obs(legal):
    return Observation(
        hole_cards=("SA", "HA"), community_cards=(), pot=1.5, player_id=1234, positions=(1234, 5678),
        stacks=(99.5, 99.0), action_history=(), legal_actions=tuple(legal), game_id=1, rotation_id=1, round_id=0,
    )


def act_in_thread(seat, observation):
    """Run seat.act() like the tournament thread does; returns (thread, result list)."""
    out = []
    thread = threading.Thread(target=lambda: out.append(seat.act(observation)), daemon=True)
    thread.start()
    return thread, out


def wait_for_prompt(events, timeout=2.0):
    for _ in range(int(timeout / 0.01)):
        prompts = [e for e in events if e["type"] == "prompt"]
        if prompts:
            return prompts[-1]
        threading.Event().wait(0.01)
    raise AssertionError("no prompt published")


@pytest.mark.parametrize(("cards", "name", "best"), [
    (["SA", "S2", "H3", "D4", "C5", "HK", "DQ"], "Straight, five high", {"SA", "S2", "H3", "D4", "C5"}),
    (["SK", "HK", "DK", "S7", "H7", "C2", "D3"], "Full house, kings full of sevens", {"SK", "HK", "DK", "S7", "H7"}),
    (["ST", "SJ", "SQ", "SK", "SA", "H2", "D3"], "Royal flush", {"ST", "SJ", "SQ", "SK", "SA"}),
    (["S9", "H9", "D4", "C7", "HJ", "S2", "D3"], "Pair of nines", {"S9", "H9", "HJ", "C7", "D4"}),
    (["S9", "H9", "D4", "C6", "HJ", "SJ", "D3"], "Two pair, jacks and nines", {"HJ", "SJ", "S9", "H9", "C6"}),
])
def test_best_hand_names_the_winning_five(cards, name, best):
    got_name, got_best = best_hand(cards)
    assert got_name == name
    assert len(got_best) == 5 and set(got_best) == best


def test_human_seat_waits_for_a_legal_choice_on_the_open_prompt():
    events = []
    seat = HumanSeat(1, "Ann (human)", events.append)
    thread, out = act_in_thread(seat, obs([(FOLD, 0.0), (CALL, 0.5), (RAISE_POT, 2.0)]))
    prompt = wait_for_prompt(events)
    assert prompt["legal"] == [[FOLD, 0.0], [CALL, 0.5], [RAISE_POT, 2.0]]

    assert seat.choose(prompt["prompt_id"] + 1, CALL) is not None  # stale prompt
    assert seat.choose(prompt["prompt_id"], CHECK) is not None     # not legal here
    assert not out
    assert seat.choose(prompt["prompt_id"], CALL) is None
    thread.join(2)
    assert out == [CALL]
    assert seat.choose(prompt["prompt_id"], CALL) is not None      # already answered


def test_leaving_answers_the_open_prompt_with_check_or_fold():
    events = []
    seat = HumanSeat(1, "Ann (human)", events.append)
    thread, out = act_in_thread(seat, obs([(FOLD, 0.0), (CALL, 0.5)]))
    wait_for_prompt(events)
    seat.leave()
    thread.join(2)
    assert out == [FOLD]
    assert seat.act(obs([(CHECK, 0.0), (RAISE_POT, 2.0)])) == CHECK  # no prompt after leaving
    assert len([e for e in events if e["type"] == "prompt"]) == 1


def test_lobby_names_humans_uniquely_and_caps_the_table():
    lobby = Lobby(lambda record: None, bot_seats=MAX_TABLE - 2)
    a, b = lobby.join(" Ann "), lobby.join("Ann")
    assert (a.name, b.name) == ("Ann (human)", "Ann 2 (human)")
    with pytest.raises(ValueError, match="full"):
        lobby.join("Cy")
    b.leave()
    assert [name for name, _ in lobby.players()] == ["Ann (human)"]
    assert lobby.join("Cy").name == "Cy (human)"  # a seat freed up


def test_humans_join_only_a_play_mode_run():
    watch = Session(TournamentConfig(seats=4), play_seed=1)
    assert watch.choose("watch") == (True, None)
    assert watch.choose("watch") == (False, None)  # a second tab picking the same mode is fine
    assert watch.choose("play")[1] is not None
    assert join_request(watch, {"name": "Ann"})[0] == 409
    assert post_request(watch, "/start", {"mode": "both"})[0] == 400


def test_play_mode_requests_need_a_known_token():
    session = Session(TournamentConfig(seats=4), play_seed=1)
    session.choose("play")
    status, joined = join_request(session, {"name": "Ann"})
    assert status == 200 and joined["name"] == "Ann (human)"
    assert post_request(session, "/human/act", {"token": "nope", "action": CALL})[0] == 404
    assert post_request(session, "/human/act", {"token": joined["token"], "prompt_id": 1, "action": CALL})[0] == 409
    assert post_request(session, "/human/leave", {"token": joined["token"]}) == (200, {})
    assert post_request(session, "/human/join", None)[0] == 400
    session.feed.finish()
    assert post_request(session, "/human/join", {"name": "Bo"})[0] == 409


def test_sandbox_deals_fresh_cards_carries_stacks_and_seats_joiners():
    late = RandomBot()
    wanted: list = []
    asks = 0

    def extra_players():
        nonlocal asks
        asks += 1
        if asks == 3:  # asked at game start, before hand 1, before hand 2 -> joins for hand 2
            wanted.append(("Late (human)", late))
        return list(wanted)

    events: list[dict] = []
    Sandbox(TournamentConfig(seats=4, games=1, rotations=40, base_seed=3),
            observer=events.append, extra_players=extra_players).run()
    rounds = [e for e in events if e["type"] == "round"]
    assert len(rounds) > 2
    assert len({r["seed"] for r in rounds}) == len(rounds)  # a new deal every hand
    assert [r["rotation_id"] for r in rounds] == list(range(1, len(rounds) + 1))

    out: set[int] = set()
    for prev, nxt in zip(rounds, rounds[1:], strict=False):
        ended = {p: s + prev["chip_deltas"][p] for p, s in zip(prev["positions"], prev["starting_stacks"], strict=True)}
        out |= {p for p, s in ended.items() if s <= 0}
        assert not out & set(nxt["positions"])  # busted players are gone for good
        for p, s in zip(nxt["positions"], nxt["starting_stacks"], strict=True):
            assert s == pytest.approx(ended.get(p, 100.0))  # stacks carry over; joiners start at 100

    joins = [e for e in events if e["type"] == "join"]
    assert [(j["rotation_id"], j["name"]) for j in joins] == [(2, "Late (human)")]
    assert joins[0]["player_id"] not in rounds[0]["positions"]
    assert joins[0]["player_id"] in rounds[1]["positions"]


def test_sandbox_refuses_to_write_logs(tmp_path):
    with pytest.raises(ValueError, match="no logs"):
        Sandbox(TournamentConfig(log_file=tmp_path / "run.jsonl"))
