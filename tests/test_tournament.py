import json
import math
from collections import defaultdict

import pytest

from pokerbots.bots.simple import RandomBot
from pokerbots.core import RoundLog
from pokerbots.tournament import (
    PlayerResult,
    Tournament,
    TournamentConfig,
    overall_leaderboard,
    rank,
    rotation_average,
)


class RecordingBot(RandomBot):
    """Random play, but remembers every new_game / observe_result call."""

    def __init__(self) -> None:
        super().__init__()
        self.games: list[tuple[int, int]] = []
        self.logs: list[RoundLog] = []

    def new_game(self, game_id: int, player_id: int) -> None:
        super().new_game(game_id, player_id)
        self.games.append((game_id, player_id))

    def observe_result(self, log: RoundLog) -> None:
        self.logs.append(log)


class OtherBot(RandomBot):
    pass


LOG = "run.jsonl"


def read_rounds(log_dir):
    lines = (log_dir / LOG).read_text().splitlines()
    return [r for r in map(json.loads, lines) if r["type"] == "round"]


def small_config(tmp_path, **kw):
    defaults = dict(games=3, rotations=15, base_seed=7, log_file=tmp_path / LOG)
    defaults.update(kw)
    return TournamentConfig(**defaults)


def test_same_config_same_results(tmp_path):
    a = Tournament(small_config(tmp_path / "a")).run()
    b = Tournament(small_config(tmp_path / "b")).run()
    assert a == b
    assert (tmp_path / "a" / LOG).read_text() == (tmp_path / "b" / LOG).read_text()


def test_different_base_seed_different_results(tmp_path):
    a = Tournament(small_config(tmp_path / "a", base_seed=1)).run()
    b = Tournament(small_config(tmp_path / "b", base_seed=2)).run()
    assert a != b


def test_each_player_plays_each_position_once_per_rotation(tmp_path):
    Tournament(small_config(tmp_path)).run()
    by_rotation = defaultdict(list)
    for r in read_rounds(tmp_path):
        by_rotation[(r["game_id"], r["rotation_id"])].append(r)
    assert by_rotation
    for rounds in by_rotation.values():
        players = set(rounds[0]["positions"])
        x = len(players)
        assert len(rounds) == x
        assert sorted(r["round_id"] for r in rounds) == list(range(x))
        for idx in range(x):
            assert {r["positions"][idx] for r in rounds} == players
        # Same deal and same starting stacks per player in every round.
        assert len({r["seed"] for r in rounds}) == 1
        start = {}
        for r in rounds:
            for pid, s in zip(r["positions"], r["starting_stacks"]):
                assert start.setdefault(pid, s) == s


def test_eliminated_players_never_return(tmp_path):
    results = Tournament(small_config(tmp_path, rotations=40)).run()
    rounds = read_rounds(tmp_path)
    assert any(p.eliminated_in for g in results for p in g.standings)
    for game in results:
        for p in game.standings:
            if p.eliminated_in is None:
                continue
            assert p.final_stack == 0
            later = [r for r in rounds if r["game_id"] == game.game_id and r["rotation_id"] > p.eliminated_in]
            assert all(p.player_id not in r["positions"] for r in later)


def test_game_ends_when_one_player_left(tmp_path):
    results = Tournament(small_config(tmp_path, seats=2, rotations=999, games=2)).run()
    for game in results:
        alive = [p for p in game.standings if p.eliminated_in is None]
        assert len(alive) == 1 or game.rotations_played == 999


def test_every_bot_sees_every_round_and_new_game(tmp_path):
    created: list[RecordingBot] = []

    class Tracked(RecordingBot):
        def __init__(self):
            super().__init__()
            created.append(self)

    config = small_config(tmp_path, main_bots=(Tracked,), simple_bots=(Tracked,), games=1, rotations=30)
    Tournament(config).run()
    rounds = read_rounds(tmp_path)
    assert len(created) == config.seats
    ids = set()
    for bot in created:
        assert len(bot.games) == 1 and bot.games[0][0] == 1
        ids.add(bot.games[0][1])
        # Every bot, eliminated or not, sees every round's result.
        assert len(bot.logs) == len(rounds)
    assert len(ids) == config.seats
    assert all(1000 <= pid <= 9999 for pid in ids)


def test_main_bots_always_seated_rest_from_pool(tmp_path):
    config = small_config(tmp_path, main_bots=(RecordingBot,), simple_bots=(OtherBot,), rotations=1)
    for game in Tournament(config).run():
        names = sorted(p.bot for p in game.standings)
        assert names == ["OtherBot"] * 7 + ["RecordingBot"]


def test_player_ids_fresh_each_game(tmp_path):
    results = Tournament(small_config(tmp_path, rotations=1)).run()
    id_sets = [frozenset(p.player_id for p in g.standings) for g in results]
    assert len(set(id_sets)) == len(id_sets)


def test_dev_mode_seed_is_game_and_rotation(tmp_path):
    Tournament(small_config(tmp_path, mode="dev")).run()
    for r in read_rounds(tmp_path):
        assert r["seed"] == r["game_id"] * 1000 + r["rotation_id"]


def test_dev_mode_same_cards_across_base_seeds(tmp_path):
    Tournament(small_config(tmp_path / "a", mode="dev", base_seed=1, rotations=1)).run()
    Tournament(small_config(tmp_path / "b", mode="dev", base_seed=2, rotations=1)).run()
    seeds_a = [r["seed"] for r in read_rounds(tmp_path / "a")]
    seeds_b = [r["seed"] for r in read_rounds(tmp_path / "b")]
    assert seeds_a[0] == seeds_b[0]


def test_eval_mode_seeds_differ_between_games(tmp_path):
    Tournament(small_config(tmp_path, rotations=1)).run()
    seeds = {r["game_id"]: r["seed"] for r in read_rounds(tmp_path)}
    assert len(set(seeds.values())) == len(seeds)


def test_jsonl_header_maps_ids_to_bots(tmp_path):
    results = Tournament(small_config(tmp_path, games=1, rotations=1)).run()
    header = json.loads((tmp_path / LOG).read_text().splitlines()[0])
    assert header["type"] == "game"
    assert {int(k): v for k, v in header["players"].items()} == {p.player_id: p.bot for p in results[0].standings}


def test_rotation_average_counts_bust_as_minus_k_start():
    assert rotation_average(100, [0, 300], k=1) == 100
    assert rotation_average(100, [0, 0, 150], k=1) == math.trunc((-100 - 100 + 150) / 3)
    assert rotation_average(100, [0, 100], k=1) == 0  # -> eliminated
    assert rotation_average(100, [101, 100, 100], k=1) == 100  # truncated


def test_rank_by_stack_then_later_elimination():
    players = [
        PlayerResult(1, "A", 0.0, 2),
        PlayerResult(2, "B", 50.0, None),
        PlayerResult(3, "C", 0.0, 9),
        PlayerResult(4, "D", 120.0, None),
    ]
    assert [p.player_id for p in rank(players, rotations_played=10)] == [4, 2, 3, 1]


def test_overall_leaderboard_groups_by_bot_class(tmp_path):
    results = Tournament(small_config(tmp_path, main_bots=(OtherBot,), rotations=5)).run()
    rows = {r.bot: r for r in overall_leaderboard(results)}
    assert rows["OtherBot"].appearances == 3
    assert rows["RandomBot"].appearances == 21


@pytest.mark.parametrize(
    "kw",
    [
        dict(seats=1),
        dict(main_bots=(RandomBot,) * 9),
        dict(simple_bots=()),
        dict(games=1000),
        dict(rotations=0),
        dict(rotations=1000),
        dict(starting_stack=0),
        dict(k=-1),
        dict(mode="prod"),
    ],
)
def test_bad_configs_rejected(kw):
    with pytest.raises(ValueError):
        Tournament(TournamentConfig(**kw))
