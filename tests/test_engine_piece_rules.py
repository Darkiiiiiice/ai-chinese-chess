from game.engine import GameState


def _empty_board():
    return [["" for _ in range(9)] for _ in range(10)]


def test_red_pawn_cannot_move_sideways_before_crossing_river():
    board = _empty_board()
    board[5][4] = "p"  # Red pawn at y=5, not crossed yet
    board[5][3] = "P"
    board[5][5] = "P"
    state = GameState(board)
    state.current_player = 1

    moves = state.get_piece_moves(4, 5)

    assert (4, 5, 3, 5) not in moves
    assert (4, 5, 5, 5) not in moves


def test_red_pawn_can_move_sideways_after_crossing_river_to_empty():
    board = _empty_board()
    board[4][4] = "p"  # Red pawn at y=4, crossed river
    state = GameState(board)
    state.current_player = 1

    moves = state.get_piece_moves(4, 4)

    assert (4, 4, 3, 4) in moves
    assert (4, 4, 5, 4) in moves


def test_black_pawn_cannot_move_sideways_before_crossing_river():
    board = _empty_board()
    board[4][4] = "P"  # Black pawn at y=4, not crossed yet
    board[4][3] = "p"
    board[4][5] = "p"
    state = GameState(board)
    state.current_player = -1

    moves = state.get_piece_moves(4, 4)

    assert (4, 4, 3, 4) not in moves
    assert (4, 4, 5, 4) not in moves


def test_black_pawn_can_move_sideways_after_crossing_river_to_empty():
    board = _empty_board()
    board[5][4] = "P"  # Black pawn at y=5, crossed river
    state = GameState(board)
    state.current_player = -1

    moves = state.get_piece_moves(4, 5)

    assert (4, 5, 3, 5) in moves
    assert (4, 5, 5, 5) in moves


def test_king_can_fly_capture_enemy_king_when_facing():
    board = _empty_board()
    board[9][4] = "k"
    board[0][4] = "K"
    state = GameState(board)
    state.current_player = 1

    moves = state.get_piece_moves(4, 9)

    assert (4, 9, 4, 0) in moves


def test_do_move_allows_move_that_checks_opponent():
    board = _empty_board()
    board[9][4] = "k"
    board[0][4] = "K"
    board[2][4] = "r"
    state = GameState(board)
    state.current_player = 1

    assert state.do_move((4, 2, 4, 1)) is True


def test_do_move_rejects_move_that_leaves_own_king_in_check():
    board = _empty_board()
    board[9][4] = "k"
    board[0][4] = "R"  # Black rook attacking file
    board[5][4] = "r"  # Red rook currently blocking the file
    state = GameState(board)
    state.current_player = 1

    assert state.do_move((4, 5, 3, 5)) is False


def test_get_all_valid_moves_excludes_moves_that_leave_own_king_in_check():
    board = _empty_board()
    board[9][4] = "k"
    board[0][4] = "R"  # Black rook attacking file
    board[5][4] = "r"  # Red rook currently blocking the file
    state = GameState(board)
    state.current_player = 1

    moves = state.get_all_valid_moves()

    assert (4, 5, 3, 5) not in moves


def test_get_all_valid_moves_does_not_copy_state_per_candidate(monkeypatch):
    state = GameState()
    original_board = [row[:] for row in state.board]
    original_player = state.current_player
    copy_calls = {"count": 0}
    original_copy = GameState.copy

    def _counting_copy(self):
        copy_calls["count"] += 1
        return original_copy(self)

    monkeypatch.setattr(GameState, "copy", _counting_copy)

    moves = state.get_all_valid_moves()

    assert moves
    assert copy_calls["count"] == 0
    assert state.board == original_board
    assert state.current_player == original_player
