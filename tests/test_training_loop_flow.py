import asyncio

import scripts.training_loop as training_loop_script


class _FakeBrowser:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def initialize(self):
        return None

    async def navigate_to_game(self):
        return None

    async def setup_game(self, **kwargs):
        return None

    async def close(self):
        return None


def test_training_loop_selfplay_forwards_reward_tuning(monkeypatch, tmp_path):
    captured = {}

    class _FakeSelfPlay:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def generate_dataset(self, num_games=100, temperature=1.0, save_dir="data"):
            return []

    monkeypatch.setattr(training_loop_script, "SelfPlay", _FakeSelfPlay)

    loop = training_loop_script.TrainingLoop(
        data_dir=str(tmp_path / "data"),
        model_dir=str(tmp_path / "models"),
        log_dir=str(tmp_path / "logs"),
        speed_bonus_max=0.45,
        draw_penalty=0.22,
    )
    loop.model = object()

    loop.selfplay()

    assert captured["init"]["speed_bonus_max"] == 0.45
    assert captured["init"]["draw_penalty"] == 0.22


def test_training_loop_selfplay_forwards_batch_size(monkeypatch, tmp_path):
    captured = {}

    class _FakeSelfPlay:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def generate_dataset(self, num_games=100, temperature=1.0, save_dir="data"):
            return []

    monkeypatch.setattr(training_loop_script, "SelfPlay", _FakeSelfPlay)

    loop = training_loop_script.TrainingLoop(
        data_dir=str(tmp_path / "data"),
        model_dir=str(tmp_path / "models"),
        log_dir=str(tmp_path / "logs"),
        sp_batch_size=64,
    )
    loop.model = object()

    loop.selfplay()

    assert captured["init"]["batch_size"] == 64


def test_training_loop_online_play_forwards_reward_tuning(monkeypatch, tmp_path):
    captured = {}

    async def _fake_play_game_with_data(
        model,
        browser,
        save_data=True,
        data_dir="data",
        device="cpu",
        batch_size=16,
        wait_timeout_ms=45000,
        speed_bonus_max=0.3,
        draw_penalty=0.1,
    ):
        captured["args"] = {
            "speed_bonus_max": speed_bonus_max,
            "draw_penalty": draw_penalty,
        }
        return {"result": 0, "samples": 1}

    async def _fake_sleep(_seconds):
        return None

    monkeypatch.setattr(training_loop_script, "XiangqiBrowser", _FakeBrowser)
    monkeypatch.setattr(training_loop_script.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr("scripts.play.play_game_with_data", _fake_play_game_with_data)

    loop = training_loop_script.TrainingLoop(
        data_dir=str(tmp_path / "data"),
        model_dir=str(tmp_path / "models"),
        log_dir=str(tmp_path / "logs"),
        online_games=1,
        speed_bonus_max=0.45,
        draw_penalty=0.22,
    )
    loop.model = object()

    result = asyncio.run(loop.online_play())

    assert captured["args"]["speed_bonus_max"] == 0.45
    assert captured["args"]["draw_penalty"] == 0.22
    assert result["samples"] == 1
