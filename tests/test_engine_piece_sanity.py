from game.engine import GameState


def _empty_board():
    return [["" for _ in range(9)] for _ in range(10)]


def test_horse_leg_blocking_rule():
    board = _empty_board()
    board[4][4] = "h"
    board[5][4] = "p"  # Block downward horse-leg
    state = GameState(board)
    state.current_player = 1

    moves = state.get_piece_moves(4, 4)

    assert (4, 4, 5, 6) not in moves
    assert (4, 4, 3, 6) not in moves
    assert (4, 4, 6, 5) in moves


def test_elephant_cannot_cross_river_and_obeys_eye_block():
    board = _empty_board()
    board[5][4] = "e"
    board[4][3] = "p"  # Block left-up elephant eye
    state = GameState(board)
    state.current_player = 1

    moves = state.get_piece_moves(4, 5)

    assert (4, 5, 2, 3) not in moves  # Cannot cross river.
    assert (4, 5, 2, 7) in moves
    assert (4, 5, 6, 7) in moves


def test_advisor_stays_inside_palace():
    board = _empty_board()
    board[7][3] = "a"
    state = GameState(board)
    state.current_player = 1

    moves = state.get_piece_moves(3, 7)

    assert (3, 7, 4, 8) in moves
    assert (3, 7, 2, 6) not in moves


def test_chariot_stops_at_first_blocking_piece():
    board = _empty_board()
    board[4][4] = "r"
    board[6][4] = "p"  # Own blocker
    board[8][4] = "R"  # Enemy behind blocker, should be unreachable
    state = GameState(board)
    state.current_player = 1

    moves = state.get_piece_moves(4, 4)

    assert (4, 4, 4, 5) in moves
    assert (4, 4, 4, 7) not in moves
    assert (4, 4, 4, 8) not in moves


def test_is_draw_uses_configurable_move_limit():
    state = GameState(draw_move_limit=3)

    state.move_history = [(0, 0, 0, 1), (0, 1, 0, 0)]
    assert state.is_draw() is False

    state.move_history.append((1, 0, 2, 2))
    assert state.is_draw() is True
