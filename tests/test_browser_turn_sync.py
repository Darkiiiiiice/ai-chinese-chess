import asyncio

from browser.automate import XiangqiBrowser
from game.engine import GameState


class _FakeWaitBrowser(XiangqiBrowser):
    def __init__(self):
        super().__init__()
        self.read_calls = 0

    async def read_board(self):
        self.read_calls += 1
        return {(0, 0): "r"}

    async def is_game_over(self):
        return False

    async def is_my_turn(self):
        return None


class _FakeTimeoutRecoveryBrowser(XiangqiBrowser):
    async def read_board(self):
        return {(0, 0): "r"}

    async def is_game_over(self):
        return False

    async def is_my_turn(self):
        return None

    async def detect_our_turn_from_hints(self, board_state=None, max_pieces=6):
        return True


class _FakeTimeoutLimitedProbeBrowser(XiangqiBrowser):
    def __init__(self):
        super().__init__()
        self.probe_args = []

    async def read_board(self):
        return {(0, 0): "r"}

    async def is_game_over(self):
        return False

    async def is_my_turn(self):
        return None

    async def detect_our_turn_from_hints(self, board_state=None, max_pieces=6):
        self.probe_args.append(max_pieces)
        # Simulate partial sampling miss; only exhaustive probe can recover.
        return max_pieces is None


def test_sync_after_opponent_move_updates_turn_and_board():
    game_state = GameState([["" for _ in range(9)] for _ in range(10)])
    game_state.current_player = -1  # Opponent to move
    board_state = {(3, 4): "p"}

    XiangqiBrowser.sync_after_opponent_move(game_state, board_state)

    assert game_state.current_player == 1
    assert game_state.board[4][3] == "p"


def test_sync_after_opponent_move_can_force_our_turn():
    game_state = GameState([["" for _ in range(9)] for _ in range(10)])
    game_state.current_player = 1
    board_state = {(2, 3): "P"}

    XiangqiBrowser.sync_after_opponent_move(game_state, board_state, our_color=1)

    assert game_state.current_player == 1
    assert game_state.board[3][2] == "P"


def test_wait_for_opponent_move_uses_provided_baseline_board():
    browser = _FakeWaitBrowser()

    detected = asyncio.run(browser.wait_for_opponent_move(timeout=1500, baseline_board={}))

    # With provided baseline, wait function should only read for new + confirm boards.
    assert detected is True
    assert browser.read_calls == 2


def test_wait_for_opponent_move_timeout_can_recover_with_hint_probe():
    browser = _FakeTimeoutRecoveryBrowser()

    detected = asyncio.run(
        browser.wait_for_opponent_move(timeout=50, baseline_board={(0, 0): "r"})
    )

    assert detected is True


def test_wait_for_opponent_move_timeout_retries_with_exhaustive_hint_probe():
    browser = _FakeTimeoutLimitedProbeBrowser()

    detected = asyncio.run(
        browser.wait_for_opponent_move(timeout=50, baseline_board={(0, 0): "r"})
    )

    assert detected is True
    assert browser.probe_args
    assert browser.probe_args[-1] is None
