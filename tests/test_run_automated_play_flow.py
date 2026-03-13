import asyncio

import scripts.play as play_script


class _FakeLoopBrowser:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.initialize_calls = 0
        self.navigate_calls = 0
        self.setup_calls = []
        self.restart_calls = []
        self.close_calls = 0
        self.num_simulations = kwargs.get("num_simulations", 1)
        self.batch_size = kwargs.get("batch_size", 16)
        self.player_color = kwargs.get("player_color", 1)
        self.game_data = []
        self.current_move_idx = 0
        type(self).instances.append(self)

    async def initialize(self):
        self.initialize_calls += 1

    async def navigate_to_game(self):
        self.navigate_calls += 1

    async def setup_game(self, **kwargs):
        self.setup_calls.append(kwargs)

    async def restart_game(self, **kwargs):
        self.restart_calls.append(kwargs)

    async def close(self):
        self.close_calls += 1


def test_run_automated_play_reuses_browser_between_games(monkeypatch):
    _FakeLoopBrowser.instances = []
    results = iter(
        [
            {"result": 1, "samples": 3},
            {"result": -1, "samples": 2},
        ]
    )

    async def _fake_play_game_with_data(*args, **kwargs):
        return next(results)

    async def _fake_sleep(_seconds):
        return None

    monkeypatch.setattr(play_script, "XiangqiBrowser", _FakeLoopBrowser)
    monkeypatch.setattr(play_script, "play_game_with_data", _fake_play_game_with_data)
    monkeypatch.setattr(play_script.asyncio, "sleep", _fake_sleep)

    asyncio.run(
        play_script.run_automated_play(
            model_path=None,
            num_games=2,
            player_color=1,
            restart_after_game=True,
        )
    )

    assert len(_FakeLoopBrowser.instances) == 1
    browser = _FakeLoopBrowser.instances[0]
    assert browser.initialize_calls == 1
    assert browser.navigate_calls == 1
    assert browser.setup_calls == [
        {"difficulty": 1, "player_color": 1, "red_first": True}
    ]
    assert browser.restart_calls == [
        {"difficulty": 1, "player_color": 1, "red_first": True}
    ]
    assert browser.close_calls == 1


def test_run_automated_play_uses_resolved_random_color_for_stats(monkeypatch, capsys):
    _FakeLoopBrowser.instances = []

    async def _fake_play_game_with_data(*args, **kwargs):
        browser = args[1]
        browser.player_color = -1
        return {"result": -1, "samples": 1}

    async def _fake_sleep(_seconds):
        return None

    monkeypatch.setattr(play_script, "XiangqiBrowser", _FakeLoopBrowser)
    monkeypatch.setattr(play_script, "play_game_with_data", _fake_play_game_with_data)
    monkeypatch.setattr(play_script.asyncio, "sleep", _fake_sleep)

    asyncio.run(
        play_script.run_automated_play(
            model_path=None,
            num_games=1,
            player_color=0,
        )
    )

    out = capsys.readouterr().out
    assert "胜: 1" in out
    assert "负: 0" in out
