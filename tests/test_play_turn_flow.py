import asyncio
import re

import numpy as np

import scripts.play as play_script
from browser.automate import XiangqiBrowser
from game.engine import GameState


class _FakePlayBrowser(XiangqiBrowser):
    def __init__(self, initial_board, board_after_opponent):
        super().__init__(player_color=1)
        self.initial_board = dict(initial_board)
        self.board_after_opponent = dict(board_after_opponent)
        self.return_opponent_board = False
        self.game_over_responses = iter([False, False, False, True])
        self.wait_baselines = []
        self.wait_timeouts = []
        self.executed_moves = []
        self.num_simulations = 1

    async def read_board(self):
        if self.return_opponent_board:
            return dict(self.board_after_opponent)
        return dict(self.initial_board)

    async def execute_move(self, x1: int, y1: int, x2: int, y2: int):
        self.executed_moves.append((x1, y1, x2, y2))
        return True

    async def wait_for_opponent_move(self, timeout: int = 30000, baseline_board=None):
        self.wait_timeouts.append(timeout)
        self.wait_baselines.append(dict(baseline_board or {}))
        self.return_opponent_board = True
        return True

    async def is_game_over(self):
        try:
            return next(self.game_over_responses)
        except StopIteration:
            return True

    async def get_game_result_text(self):
        return "unknown"

    async def get_my_game_outcome(self):
        return "unknown"


class _FakeTimeoutRecoverPlayBrowser(XiangqiBrowser):
    def __init__(self, initial_board):
        super().__init__(player_color=1)
        self.initial_board = dict(initial_board)
        self.game_over_responses = iter([False, False, False, True])
        self.executed_moves = []
        self.num_simulations = 1
        self.hint_probe_calls = 0
        self.wait_calls = 0

    async def read_board(self):
        return dict(self.initial_board)

    async def execute_move(self, x1: int, y1: int, x2: int, y2: int):
        self.executed_moves.append((x1, y1, x2, y2))
        return True

    async def wait_for_opponent_move(self, timeout: int = 30000, baseline_board=None):
        self.wait_calls += 1
        return False

    async def detect_our_turn_from_hints(self, board_state=None, max_pieces=6):
        self.hint_probe_calls += 1
        return True

    async def is_game_over(self):
        try:
            return next(self.game_over_responses)
        except StopIteration:
            return True

    async def get_game_result_text(self):
        return "unknown"

    async def get_my_game_outcome(self):
        return "unknown"


class _FakeExecuteFailBrowser(XiangqiBrowser):
    def __init__(self, initial_board):
        super().__init__(player_color=1)
        self.initial_board = dict(initial_board)
        self.game_over_responses = iter([False, False, True])
        self.wait_calls = 0
        self.num_simulations = 1

    async def read_board(self):
        return dict(self.initial_board)

    async def execute_move(self, x1: int, y1: int, x2: int, y2: int):
        return False

    async def wait_for_opponent_move(self, timeout: int = 30000, baseline_board=None):
        self.wait_calls += 1
        return False

    async def is_game_over(self):
        try:
            return next(self.game_over_responses)
        except StopIteration:
            return True

    async def get_game_result_text(self):
        return "unknown"

    async def get_my_game_outcome(self):
        return "unknown"


class _FakeRetryMoveBrowser(XiangqiBrowser):
    def __init__(self, initial_board, successful_move):
        super().__init__(player_color=1)
        self.initial_board = dict(initial_board)
        self.successful_move = successful_move
        self.game_over_responses = iter([False, True])
        self.executed_moves = []
        self.wait_calls = 0
        self.num_simulations = 1

    async def read_board(self):
        return dict(self.initial_board)

    async def execute_move(self, x1: int, y1: int, x2: int, y2: int):
        move = (x1, y1, x2, y2)
        self.executed_moves.append(move)
        return move == self.successful_move

    async def wait_for_opponent_move(self, timeout: int = 30000, baseline_board=None):
        self.wait_calls += 1
        return False

    async def is_game_over(self):
        try:
            return next(self.game_over_responses)
        except StopIteration:
            return True

    async def get_game_result_text(self):
        return "unknown"

    async def get_my_game_outcome(self):
        return "unknown"


def test_red_first_second_step_stays_our_turn_after_wait(monkeypatch, capsys):
    start_state = GameState()
    move1 = (0, 9, 0, 8)
    move2 = (0, 8, 0, 7)
    opponent_move = (0, 3, 0, 4)

    state_after_our_first = start_state.copy()
    assert state_after_our_first.do_move(move1) is True
    baseline_after_our_first = XiangqiBrowser.game_state_to_board_dict(state_after_our_first)

    state_after_opponent = state_after_our_first.copy()
    assert state_after_opponent.do_move(opponent_move) is True
    board_after_opponent = XiangqiBrowser.game_state_to_board_dict(state_after_opponent)

    fake_browser = _FakePlayBrowser(
        initial_board=XiangqiBrowser.game_state_to_board_dict(start_state),
        board_after_opponent=board_after_opponent,
    )

    async def _fake_get_ai_move_with_policy(game_state, model, num_simulations, device, batch_size):
        move = move1 if len(fake_browser.executed_moves) == 0 else move2
        return move, np.zeros(2086, dtype=np.float32)

    monkeypatch.setattr(play_script, "_get_ai_move_with_policy", _fake_get_ai_move_with_policy)

    result = asyncio.run(
        play_script.play_game_with_data(
            model=None,
            browser=fake_browser,
            save_data=False,
            device="cpu",
        )
    )

    out = capsys.readouterr().out

    assert re.search(r"--- 第 2 步 ---.*?是否我方回合: True", out, re.S)
    assert fake_browser.wait_baselines
    assert fake_browser.wait_baselines[0] == baseline_after_our_first
    assert (0, 4) not in fake_browser.wait_baselines[0]
    assert result["samples"] == 2


def test_timeout_can_recover_to_our_turn_via_hint_probe(monkeypatch, capsys):
    start_state = GameState()
    move1 = (0, 9, 0, 8)
    move2 = (0, 8, 0, 7)

    fake_browser = _FakeTimeoutRecoverPlayBrowser(
        initial_board=XiangqiBrowser.game_state_to_board_dict(start_state)
    )

    async def _fake_get_ai_move_with_policy(game_state, model, num_simulations, device, batch_size):
        move = move1 if len(fake_browser.executed_moves) == 0 else move2
        return move, np.zeros(2086, dtype=np.float32)

    monkeypatch.setattr(play_script, "_get_ai_move_with_policy", _fake_get_ai_move_with_policy)

    asyncio.run(
        play_script.play_game_with_data(
            model=None,
            browser=fake_browser,
            save_data=False,
            device="cpu",
        )
    )

    out = capsys.readouterr().out
    assert re.search(r"--- 第 2 步 ---.*?是否我方回合: True", out, re.S)
    assert fake_browser.wait_calls == 1
    assert fake_browser.hint_probe_calls >= 1


def test_play_game_with_data_forwards_wait_timeout(monkeypatch):
    start_state = GameState()
    move1 = (0, 9, 0, 8)
    opponent_move = (0, 3, 0, 4)

    state_after_our_first = start_state.copy()
    assert state_after_our_first.do_move(move1) is True
    state_after_opponent = state_after_our_first.copy()
    assert state_after_opponent.do_move(opponent_move) is True

    fake_browser = _FakePlayBrowser(
        initial_board=XiangqiBrowser.game_state_to_board_dict(start_state),
        board_after_opponent=XiangqiBrowser.game_state_to_board_dict(state_after_opponent),
    )
    fake_browser.game_over_responses = iter([False, False, True])

    async def _fake_get_ai_move_with_policy(game_state, model, num_simulations, device, batch_size):
        return move1, np.zeros(2086, dtype=np.float32)

    monkeypatch.setattr(play_script, "_get_ai_move_with_policy", _fake_get_ai_move_with_policy)

    asyncio.run(
        play_script.play_game_with_data(
            model=None,
            browser=fake_browser,
            save_data=False,
            device="cpu",
            wait_timeout_ms=45678,
        )
    )

    assert fake_browser.wait_timeouts == [45678]


def test_play_game_with_data_forwards_speed_bonus_max(monkeypatch):
    start_state = GameState()
    move1 = (0, 9, 0, 8)
    captured = {}

    fake_browser = _FakePlayBrowser(
        initial_board=XiangqiBrowser.game_state_to_board_dict(start_state),
        board_after_opponent=XiangqiBrowser.game_state_to_board_dict(start_state),
    )
    fake_browser.game_over_responses = iter([False, True])

    async def _fake_get_ai_move_with_policy(game_state, model, num_simulations, device, batch_size):
        return move1, np.zeros(2086, dtype=np.float32)

    def _fake_speed_bonus(result, total_moves, max_moves, max_bonus=0.3):
        captured["args"] = {
            "result": result,
            "total_moves": total_moves,
            "max_moves": max_moves,
            "max_bonus": max_bonus,
        }
        return {1: 0.0, -1: 0.0}

    monkeypatch.setattr(play_script, "_get_ai_move_with_policy", _fake_get_ai_move_with_policy)
    monkeypatch.setattr(play_script, "compute_speed_bonus_by_player", _fake_speed_bonus)

    asyncio.run(
        play_script.play_game_with_data(
            model=None,
            browser=fake_browser,
            save_data=False,
            device="cpu",
            speed_bonus_max=0.42,
        )
    )

    assert captured["args"]["max_bonus"] == 0.42


def test_play_game_with_data_forwards_draw_penalty(monkeypatch):
    start_state = GameState()
    move1 = (0, 9, 0, 8)
    captured = {}

    fake_browser = _FakePlayBrowser(
        initial_board=XiangqiBrowser.game_state_to_board_dict(start_state),
        board_after_opponent=XiangqiBrowser.game_state_to_board_dict(start_state),
    )
    fake_browser.game_over_responses = iter([False, True])

    async def _fake_get_ai_move_with_policy(game_state, model, num_simulations, device, batch_size):
        return move1, np.zeros(2086, dtype=np.float32)

    def _fake_draw_penalty(result, penalty=0.1):
        captured["args"] = {
            "result": result,
            "penalty": penalty,
        }
        return {1: 0.0, -1: 0.0}

    monkeypatch.setattr(play_script, "_get_ai_move_with_policy", _fake_get_ai_move_with_policy)
    monkeypatch.setattr(play_script, "compute_draw_penalty_by_player", _fake_draw_penalty)

    asyncio.run(
        play_script.play_game_with_data(
            model=None,
            browser=fake_browser,
            save_data=False,
            device="cpu",
            draw_penalty=0.25,
        )
    )

    assert captured["args"]["penalty"] == 0.25


def test_execute_move_failure_does_not_wait_opponent(monkeypatch):
    start_state = GameState()
    move1 = (0, 9, 0, 8)

    fake_browser = _FakeExecuteFailBrowser(
        initial_board=XiangqiBrowser.game_state_to_board_dict(start_state)
    )

    async def _fake_get_ai_move_with_policy(game_state, model, num_simulations, device, batch_size):
        return move1, np.zeros(2086, dtype=np.float32)

    monkeypatch.setattr(play_script, "_get_ai_move_with_policy", _fake_get_ai_move_with_policy)

    result = asyncio.run(
        play_script.play_game_with_data(
            model=None,
            browser=fake_browser,
            save_data=False,
            device="cpu",
        )
    )

    assert fake_browser.wait_calls == 0
    assert result["samples"] == 0


def test_primary_move_failure_retries_alternative_candidate(monkeypatch):
    start_state = GameState()
    primary_move = (0, 9, 0, 8)
    alternative_move = (1, 9, 2, 7)

    fake_browser = _FakeRetryMoveBrowser(
        initial_board=XiangqiBrowser.game_state_to_board_dict(start_state),
        successful_move=alternative_move,
    )

    async def _fake_get_ai_move_with_policy(game_state, model, num_simulations, device, batch_size):
        policy = np.zeros(8010, dtype=np.float32)
        policy[play_script._encode_move(alternative_move)] = 1.0
        return primary_move, policy

    monkeypatch.setattr(play_script, "_get_ai_move_with_policy", _fake_get_ai_move_with_policy)

    result = asyncio.run(
        play_script.play_game_with_data(
            model=None,
            browser=fake_browser,
            save_data=False,
            device="cpu",
        )
    )

    assert fake_browser.executed_moves[:2] == [primary_move, alternative_move]
    assert fake_browser.wait_calls == 0
    assert result["samples"] == 1


def test_retry_prefers_different_source_piece_after_failure(monkeypatch):
    start_state = GameState()
    primary_move = (0, 9, 0, 8)
    same_source_alt = (0, 9, 0, 7)
    different_source_alt = (1, 9, 2, 7)

    fake_browser = _FakeRetryMoveBrowser(
        initial_board=XiangqiBrowser.game_state_to_board_dict(start_state),
        successful_move=different_source_alt,
    )

    async def _fake_get_ai_move_with_policy(game_state, model, num_simulations, device, batch_size):
        policy = np.zeros(8010, dtype=np.float32)
        policy[play_script._encode_move(same_source_alt)] = 1.0
        policy[play_script._encode_move(different_source_alt)] = 0.9
        return primary_move, policy

    monkeypatch.setattr(play_script, "_get_ai_move_with_policy", _fake_get_ai_move_with_policy)

    asyncio.run(
        play_script.play_game_with_data(
            model=None,
            browser=fake_browser,
            save_data=False,
            device="cpu",
        )
    )

    assert len(fake_browser.executed_moves) >= 2
    assert fake_browser.executed_moves[0][:2] == primary_move[:2]
    assert fake_browser.executed_moves[1][:2] != primary_move[:2]
