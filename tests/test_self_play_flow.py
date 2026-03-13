import scripts.self_play as self_play_script
import numpy as np


class _FakeModel:
    def load(self, path: str):
        return None

    def set_training(self, training: bool):
        return None


def test_run_selfplay_forwards_temperature_to_dataset(monkeypatch):
    captured = {}

    class _FakeSelfPlay:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def generate_dataset(self, num_games=100, temperature=1.0, save_dir="data"):
            captured["generate"] = {
                "num_games": num_games,
                "temperature": temperature,
                "save_dir": save_dir,
            }
            return []

    monkeypatch.setattr(self_play_script, "SelfPlay", _FakeSelfPlay)
    monkeypatch.setattr(self_play_script, "create_model", lambda cfg: _FakeModel())

    self_play_script.run_selfplay(
        model_path=None,
        num_games=3,
        num_simulations=5,
        temperature=0.37,
        device="cpu",
    )

    assert captured["generate"]["num_games"] == 3
    assert captured["generate"]["temperature"] == 0.37


def test_play_game_normalizes_none_result_to_draw(monkeypatch):
    class _ImmediateUnknownGameState:
        def __init__(self, **kwargs):
            self.current_player = 1
            self.board = []
            self.move_history = []
            self.captured_by = {1: {}, -1: {}}

        def is_game_over(self):
            return True

        def get_game_result(self):
            return None

        def get_capture_reward(self, player):
            return 0.0

    monkeypatch.setattr(self_play_script, "GameState", _ImmediateUnknownGameState)

    sp = self_play_script.SelfPlay(
        model=_FakeModel(),
        num_simulations=1,
        temperature=1.0,
        device="cpu",
    )

    result, move_data = sp.play_game(temperature=1.0, record_moves=True)

    assert result == 0
    assert move_data == []


def test_compute_repeat_penalty_by_player_detects_back_and_forth_streak():
    moves = [
        {"player": 1, "move": (0, 9, 0, 8)},
        {"player": -1, "move": (8, 0, 8, 1)},
        {"player": 1, "move": (0, 8, 0, 9)},
        {"player": -1, "move": (8, 1, 8, 0)},
        {"player": 1, "move": (0, 9, 0, 8)},
        {"player": -1, "move": (8, 0, 8, 1)},
        {"player": 1, "move": (0, 8, 0, 9)},
    ]

    penalties = self_play_script.compute_repeat_penalty_by_player(
        moves,
        threshold=2,
        penalty_unit=0.2,
    )

    assert penalties[1] > 0.0
    assert penalties[-1] > 0.0


def test_build_repetition_key_distinguishes_side_to_move():
    board = [["" for _ in range(9)] for _ in range(10)]

    key_red = self_play_script.build_repetition_key(board, 1)
    key_black = self_play_script.build_repetition_key(board, -1)

    assert key_red != key_black


def test_play_game_resigns_when_value_below_threshold(monkeypatch):
    class _NeverEndingGameState:
        def __init__(self, **kwargs):
            self.current_player = 1
            self.board = [["" for _ in range(9)] for _ in range(10)]
            self.move_history = []
            self.captured_by = {1: {}, -1: {}}

        def is_game_over(self):
            return False

        def get_game_result(self):
            return None

        def to_numpy(self):
            return np.zeros((15, 10, 9), dtype=np.float32)

        def get_capture_reward(self, player):
            return 0.0

    class _FailIfMCTSRuns:
        def __init__(self, **kwargs):
            return None

        def get_move(self, state, temperature=None):
            raise AssertionError("resign should happen before MCTS move selection")

        def get_policy(self, state, temperature=None):
            return np.zeros(8010, dtype=np.float32)

        def reset(self):
            return None

    class _LowValueModel(_FakeModel):
        def predict(self, board):
            return np.zeros(8010, dtype=np.float32), -0.95

    monkeypatch.setattr(self_play_script, "GameState", _NeverEndingGameState)
    monkeypatch.setattr(self_play_script, "MCTSPlayer", _FailIfMCTSRuns)

    sp = self_play_script.SelfPlay(
        model=_LowValueModel(),
        num_simulations=1,
        temperature=1.0,
        resign_threshold=-0.9,
        min_resign_moves=0,
        device="cpu",
    )

    result, move_data = sp.play_game(temperature=1.0, record_moves=True)

    assert result == -1
    assert move_data == []


def test_play_game_does_not_resign_when_value_above_threshold(monkeypatch):
    class _OneMoveGameState:
        do_move_calls = 0

        def __init__(self, **kwargs):
            self.current_player = 1
            self.board = [["" for _ in range(9)] for _ in range(10)]
            self.move_history = []
            self.captured_by = {1: {}, -1: {}}
            self._game_over = False
            self._result = None

        def is_game_over(self):
            return self._game_over

        def get_game_result(self):
            return self._result

        def to_numpy(self):
            return np.zeros((15, 10, 9), dtype=np.float32)

        def do_move(self, move):
            type(self).do_move_calls += 1
            self._game_over = True
            self._result = 1
            self.current_player = -1
            self.move_history.append(move)

        def get_capture_reward(self, player):
            return 0.0

    class _FakeMCTSPlayer:
        def __init__(self, **kwargs):
            return None

        def get_move(self, state, temperature=None):
            return (0, 0, 0, 1)

        def get_policy(self, state, temperature=None):
            policy = np.zeros(8010, dtype=np.float32)
            policy[0] = 1.0
            return policy

        def reset(self):
            return None

    class _HigherValueModel(_FakeModel):
        def predict(self, board):
            return np.zeros(8010, dtype=np.float32), -0.2

    monkeypatch.setattr(self_play_script, "GameState", _OneMoveGameState)
    monkeypatch.setattr(self_play_script, "MCTSPlayer", _FakeMCTSPlayer)

    sp = self_play_script.SelfPlay(
        model=_HigherValueModel(),
        num_simulations=1,
        temperature=1.0,
        resign_threshold=-0.9,
        min_resign_moves=0,
        device="cpu",
    )

    result, _ = sp.play_game(temperature=1.0, record_moves=False)

    assert result == 1
    assert _OneMoveGameState.do_move_calls == 1


def test_run_selfplay_forwards_resign_settings(monkeypatch):
    captured = {}

    class _FakeSelfPlay:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def generate_dataset(self, num_games=100, temperature=1.0, save_dir="data"):
            return []

    monkeypatch.setattr(self_play_script, "SelfPlay", _FakeSelfPlay)
    monkeypatch.setattr(self_play_script, "create_model", lambda cfg: _FakeModel())

    self_play_script.run_selfplay(
        model_path=None,
        num_games=2,
        num_simulations=3,
        temperature=0.5,
        resign_threshold=-0.88,
        min_resign_moves=24,
        device="cpu",
    )

    assert captured["init"]["resign_threshold"] == -0.88
    assert captured["init"]["min_resign_moves"] == 24


def test_run_selfplay_forwards_speed_bonus_max(monkeypatch):
    captured = {}

    class _FakeSelfPlay:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def generate_dataset(self, num_games=100, temperature=1.0, save_dir="data"):
            return []

    monkeypatch.setattr(self_play_script, "SelfPlay", _FakeSelfPlay)
    monkeypatch.setattr(self_play_script, "create_model", lambda cfg: _FakeModel())

    self_play_script.run_selfplay(
        model_path=None,
        num_games=2,
        num_simulations=3,
        temperature=0.5,
        speed_bonus_max=0.42,
        device="cpu",
    )

    assert captured["init"]["speed_bonus_max"] == 0.42


def test_run_selfplay_forwards_draw_penalty(monkeypatch):
    captured = {}

    class _FakeSelfPlay:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def generate_dataset(self, num_games=100, temperature=1.0, save_dir="data"):
            return []

    monkeypatch.setattr(self_play_script, "SelfPlay", _FakeSelfPlay)
    monkeypatch.setattr(self_play_script, "create_model", lambda cfg: _FakeModel())

    self_play_script.run_selfplay(
        model_path=None,
        num_games=2,
        num_simulations=3,
        temperature=0.5,
        draw_penalty=0.25,
        device="cpu",
    )

    assert captured["init"]["draw_penalty"] == 0.25


def test_play_game_applies_speed_bonus_to_quick_winner(monkeypatch):
    class _OneMoveWinGameState:
        def __init__(self, **kwargs):
            self.current_player = 1
            self.board = [["" for _ in range(9)] for _ in range(10)]
            self.move_history = []
            self.captured_by = {1: {}, -1: {}}
            self._game_over = False
            self._result = None

        def is_game_over(self):
            return self._game_over

        def get_game_result(self):
            return self._result

        def to_numpy(self):
            return np.zeros((15, 10, 9), dtype=np.float32)

        def do_move(self, move):
            self._game_over = True
            self._result = 1
            self.current_player = -1
            self.move_history.append(move)
            return True

        def get_capture_reward(self, player):
            return 0.0

    class _SingleMoveMCTSPlayer:
        def __init__(self, **kwargs):
            return None

        def get_move(self, state, temperature=None):
            return (0, 0, 0, 1)

        def get_policy(self, state, temperature=None):
            policy = np.zeros(8010, dtype=np.float32)
            policy[0] = 1.0
            return policy

        def reset(self):
            return None

    monkeypatch.setattr(self_play_script, "GameState", _OneMoveWinGameState)
    monkeypatch.setattr(self_play_script, "MCTSPlayer", _SingleMoveMCTSPlayer)

    sp = self_play_script.SelfPlay(
        model=_FakeModel(),
        num_simulations=1,
        temperature=1.0,
        max_moves=100,
        resign_threshold=None,
        device="cpu",
    )

    result, move_data = sp.play_game(temperature=1.0, record_moves=True)

    assert result == 1
    assert len(move_data) == 1
    assert move_data[0]["value"] > 1.0


def test_play_game_logs_each_step(monkeypatch, capsys):
    class _OneMoveWinGameState:
        def __init__(self, **kwargs):
            self.current_player = 1
            self.board = [["" for _ in range(9)] for _ in range(10)]
            self.move_history = []
            self.captured_by = {1: {}, -1: {}}
            self._game_over = False
            self._result = None

        def is_game_over(self):
            return self._game_over

        def get_game_result(self):
            return self._result

        def to_numpy(self):
            return np.zeros((15, 10, 9), dtype=np.float32)

        def do_move(self, move):
            self._game_over = True
            self._result = 1
            self.current_player = -1
            self.move_history.append(move)
            return True

        def get_capture_reward(self, player):
            return 0.0

    class _SingleMoveMCTSPlayer:
        def __init__(self, **kwargs):
            return None

        def get_move(self, state, temperature=None):
            return (0, 0, 0, 1)

        def get_policy(self, state, temperature=None):
            policy = np.zeros(8010, dtype=np.float32)
            policy[0] = 1.0
            return policy

        def reset(self):
            return None

    monkeypatch.setattr(self_play_script, "GameState", _OneMoveWinGameState)
    monkeypatch.setattr(self_play_script, "MCTSPlayer", _SingleMoveMCTSPlayer)

    sp = self_play_script.SelfPlay(
        model=_FakeModel(),
        num_simulations=1,
        temperature=1.0,
        max_moves=100,
        resign_threshold=None,
        device="cpu",
    )

    sp.play_game(temperature=1.0, record_moves=True)

    out = capsys.readouterr().out
    assert "--- 第 1 步 ---" in out
    assert "当前玩家: 红方" in out
    assert "选择落子: (0,0) -> (0,1)" in out


def test_play_game_stops_when_mcts_returns_illegal_move(monkeypatch, capsys):
    class _IllegalMoveGameState:
        def __init__(self, **kwargs):
            self.current_player = 1
            self.board = [["" for _ in range(9)] for _ in range(10)]
            self.move_history = []
            self.captured_by = {1: {}, -1: {}}

        def is_game_over(self):
            return False

        def get_game_result(self):
            return None

        def to_numpy(self):
            return np.zeros((15, 10, 9), dtype=np.float32)

        def do_move(self, move):
            return False

        def get_capture_reward(self, player):
            return 0.0

    class _IllegalMoveMCTSPlayer:
        def __init__(self, **kwargs):
            return None

        def get_move(self, state, temperature=None):
            return (0, 0, 0, 1)

        def get_policy(self, state, temperature=None):
            policy = np.zeros(8010, dtype=np.float32)
            policy[0] = 1.0
            return policy

        def reset(self):
            return None

    monkeypatch.setattr(self_play_script, "GameState", _IllegalMoveGameState)
    monkeypatch.setattr(self_play_script, "MCTSPlayer", _IllegalMoveMCTSPlayer)

    sp = self_play_script.SelfPlay(
        model=_FakeModel(),
        num_simulations=1,
        temperature=1.0,
        max_moves=20,
        resign_threshold=None,
        device="cpu",
    )

    result, move_data = sp.play_game(temperature=1.0, record_moves=True)

    out = capsys.readouterr().out
    assert result == 0
    assert len(move_data) == 0
    assert "非法落子" in out
    assert out.count("当前玩家: 红方") == 1
