from game.engine import GameState


def _empty_board():
    return [["" for _ in range(9)] for _ in range(10)]


def test_cannon_cannot_move_to_empty_square_beyond_screen_piece():
    board = _empty_board()
    board[4][4] = "c"  # Red cannon at (4, 4)
    board[6][4] = "p"  # Screen piece at (4, 6)
    state = GameState(board)
    state.current_player = 1

    moves = state.get_piece_moves(4, 4)

    # Empty squares behind first blocking piece must be illegal.
    assert (4, 4, 4, 7) not in moves
    assert (4, 4, 4, 8) not in moves
    assert (4, 4, 4, 9) not in moves


def test_cannon_can_capture_enemy_after_own_screen_piece():
    board = _empty_board()
    board[4][4] = "c"  # Red cannon at (4, 4)
    board[6][4] = "p"  # Own screen piece at (4, 6)
    board[8][4] = "R"  # Enemy capture target at (4, 8)
    state = GameState(board)
    state.current_player = 1

    moves = state.get_piece_moves(4, 4)

    assert (4, 4, 4, 8) in moves
