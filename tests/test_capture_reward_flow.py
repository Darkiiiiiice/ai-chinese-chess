import math

from game.engine import GameState
from game.reward import compute_draw_penalty_by_player, compute_speed_bonus_by_player
from scripts.play import OnlineGameData


def _empty_board():
    return [["" for _ in range(9)] for _ in range(10)]


def test_get_capture_reward_is_opposite_between_sides():
    board = _empty_board()
    board[9][4] = "k"  # Red king
    board[0][4] = "K"  # Black king
    board[5][4] = "p"  # Block flying-general line
    board[6][0] = "p"  # Red pawn
    board[5][0] = "P"  # Black pawn

    state = GameState(board)
    assert state.do_move((0, 6, 0, 5)) is True

    assert math.isclose(state.get_capture_reward(1), 1 / 9, rel_tol=1e-9)
    assert math.isclose(state.get_capture_reward(-1), -1 / 9, rel_tol=1e-9)


def test_game_state_copy_preserves_capture_tracking():
    board = _empty_board()
    board[9][4] = "k"
    board[0][4] = "K"
    board[5][4] = "p"
    board[6][0] = "p"
    board[5][0] = "P"

    state = GameState(board)
    assert state.do_move((0, 6, 0, 5)) is True

    copied = state.copy()
    assert copied.captured_by == state.captured_by


def test_online_game_data_set_values_uses_player_specific_capture_reward():
    data = OnlineGameData()
    data.samples = [
        {"player": -1},  # black sample
        {"player": 1},   # red sample
    ]

    data.set_values(
        result=-1,  # black wins
        capture_rewards_by_player={-1: 1.0, 1: -1.0},
    )

    # Winner gets positive capture bonus.
    assert data.samples[0]["value"] > 1.0
    # Loser gets additional penalty.
    assert data.samples[1]["value"] < -1.0


def test_online_game_data_set_values_applies_repeat_penalty():
    data = OnlineGameData()
    data.samples = [
        {"player": 1},   # red sample
        {"player": -1},  # black sample
    ]

    data.set_values(
        result=0,
        capture_rewards_by_player={1: 0.0, -1: 0.0},
        repeat_penalty_by_player={1: 0.6, -1: 0.0},
    )

    assert data.samples[0]["value"] == -0.6
    assert data.samples[1]["value"] == 0.0


def test_compute_speed_bonus_by_player_rewards_faster_win():
    fast_bonus = compute_speed_bonus_by_player(
        result=1,
        total_moves=20,
        max_moves=200,
        max_bonus=0.3,
    )
    slow_bonus = compute_speed_bonus_by_player(
        result=1,
        total_moves=180,
        max_moves=200,
        max_bonus=0.3,
    )

    assert fast_bonus[1] > slow_bonus[1] > 0.0
    assert fast_bonus[-1] == 0.0
    assert slow_bonus[-1] == 0.0


def test_online_game_data_set_values_applies_speed_bonus_for_winner():
    data = OnlineGameData()
    data.samples = [
        {"player": 1},
        {"player": -1},
    ]

    data.set_values(
        result=1,
        capture_rewards_by_player={1: 0.0, -1: 0.0},
        speed_bonus_by_player={1: 0.25, -1: 0.0},
    )

    assert data.samples[0]["value"] == 1.25
    assert data.samples[1]["value"] == -1.0


def test_compute_draw_penalty_by_player_only_penalizes_draws():
    draw_penalty = compute_draw_penalty_by_player(result=0, penalty=0.2)
    win_penalty = compute_draw_penalty_by_player(result=1, penalty=0.2)

    assert draw_penalty == {1: 0.2, -1: 0.2}
    assert win_penalty == {1: 0.0, -1: 0.0}


def test_online_game_data_set_values_applies_draw_penalty_on_draw():
    data = OnlineGameData()
    data.samples = [
        {"player": 1},
        {"player": -1},
    ]

    data.set_values(
        result=0,
        capture_rewards_by_player={1: 0.0, -1: 0.0},
        draw_penalty_by_player={1: 0.2, -1: 0.2},
    )

    assert data.samples[0]["value"] == -0.2
    assert data.samples[1]["value"] == -0.2


def test_online_game_data_set_values_applies_step_capture_reward():
    data = OnlineGameData()
    data.samples = [
        {"player": 1, "step_capture_reward": 1 / 9},
        {"player": -1, "step_capture_reward": 0.0},
    ]

    data.set_values(
        result=0,
        capture_rewards_by_player={1: 0.0, -1: 0.0},
    )

    assert math.isclose(data.samples[0]["value"], 1 / 9, rel_tol=1e-9)
    assert data.samples[1]["value"] == 0.0


def test_online_game_data_set_values_accumulates_signed_step_rewards_from_event_timeline():
    data = OnlineGameData()
    data.step_reward_events = [
        {1: 1 / 9, -1: -(1 / 9)},
        {1: -(4 / 9), -1: 4 / 9},
    ]
    data.samples = [
        {"player": 1, "event_index": 0},
        {"player": -1, "event_index": 1},
    ]

    data.set_values(
        result=0,
        capture_rewards_by_player={1: 0.0, -1: 0.0},
    )

    assert math.isclose(data.samples[0]["value"], -3 / 9, rel_tol=1e-9)
    assert math.isclose(data.samples[1]["value"], 4 / 9, rel_tol=1e-9)


def test_undo_move_restores_board_and_capture_tracking_for_capture():
    board = _empty_board()
    board[9][4] = "k"
    board[0][4] = "K"
    board[5][4] = "p"
    board[6][0] = "p"
    board[5][0] = "P"

    state = GameState(board)
    assert state.do_move((0, 6, 0, 5)) is True
    assert state.captured_by[1].get("P") == 1
    assert state.current_player == -1

    undone = state.undo_move()

    assert undone == (0, 6, 0, 5)
    assert state.get_piece(0, 6) == "p"
    assert state.get_piece(0, 5) == "P"
    assert state.current_player == 1
    assert state.captured_by[1].get("P", 0) == 0
