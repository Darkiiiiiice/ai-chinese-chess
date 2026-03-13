"""Debug script to analyze is_game_over() issue"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from game.engine import GameState
from game.pieces import INITIAL_BOARD


def test_initial_state():
    """Test with initial board state"""
    print("=" * 50)
    print("Testing with INITIAL_BOARD")
    print("=" * 50)

    game = GameState()

    # Print board
    print("\nBoard state:")
    print(game)

    # Check king positions
    print("\n--- King positions ---")
    for y in range(10):
        for x in range(9):
            piece = game.board[y][x]
            if piece and piece.lower() == 'k':
                color = "Red" if piece.islower() else "Black"
                print(f"  {color} King at ({x}, {y}): '{piece}'")

    # Check current player
    print(f"\nCurrent player: {game.current_player} ({'Red' if game.current_player == 1 else 'Black'})")

    # Check if in check
    print("\n--- Check status ---")
    red_in_check = game.is_in_check(1)
    black_in_check = game.is_in_check(-1)
    print(f"Red in check: {red_in_check}")
    print(f"Black in check: {black_in_check}")

    # Check valid moves
    moves = game.get_all_valid_moves()
    print(f"\nValid moves count: {len(moves)}")

    # Check game over
    print(f"\nis_game_over: {game.is_game_over()}")
    print(f"is_checkmate: {game.is_checkmate()}")
    print(f"is_draw: {game.is_draw()}")
    result = game.get_game_result()
    print(f"get_game_result: {result}")

    return game


def test_is_in_check_detailed(player: int):
    """Detailed analysis of is_in_check()"""
    print(f"\n{'=' * 50}")
    print(f"Detailed is_in_check({player}) analysis")
    print("=" * 50)

    game = GameState()

    # Find king
    king = "k" if player == 1 else "K"
    king_pos = None
    for y in range(10):
        for x in range(9):
            if game.board[y][x] == king:
                king_pos = (x, y)
                break
        if king_pos:
            break

    print(f"Looking for king: '{king}'")
    print(f"King position: {king_pos}")

    if not king_pos:
        print("ERROR: King not found!")
        return

    enemy_color = -player
    print(f"\nEnemy color: {enemy_color}")

    attacking_pieces = []

    for y in range(10):
        for x in range(9):
            piece = game.board[y][x]
            if piece and game.get_piece_color(piece) == enemy_color:
                moves = game._get_piece_moves_for_color(x, y, enemy_color)
                targets = [(m[2], m[3]) for m in moves]
                if king_pos in targets:
                    attacking_pieces.append((x, y, piece, len(moves)))

    if attacking_pieces:
        print(f"\nPieces that can attack king:")
        for x, y, piece, num_moves in attacking_pieces:
            print(f"  '{piece}' at ({x}, {y}) with {num_moves} possible moves")
    else:
        print("\nNo pieces can attack king")

    print(f"\nis_in_check({player}) = {game.is_in_check(player)}")


def test_checkmate_detailed():
    """Detailed analysis of is_checkmate()"""
    print(f"\n{'=' * 50}")
    print("Detailed is_checkmate() analysis")
    print("=" * 50)

    game = GameState()

    print(f"Current player: {game.current_player}")
    print(f"is_in_check(current_player): {game.is_in_check(game.current_player)}")

    if not game.is_in_check(game.current_player):
        print("Not in check, so not checkmate")
        return

    moves = game.get_all_valid_moves()
    print(f"\nChecking {len(moves)} moves to see if any can escape check...")

    escape_moves = []
    for move in moves:
        # Try move
        piece = game.get_piece(move[0], move[1])
        captured = game.get_piece(move[2], move[3])
        game.set_piece(move[2], move[3], piece)
        game.set_piece(move[0], move[1], "")

        in_check = game.is_in_check(game.current_player)

        # Undo
        game.set_piece(move[0], move[1], piece)
        game.set_piece(move[2], move[3], captured)

        if not in_check:
            escape_moves.append(move)

    if escape_moves:
        print(f"Found {len(escape_moves)} moves that can escape check:")
        for m in escape_moves[:5]:  # Show first 5
            print(f"  {m}")
        if len(escape_moves) > 5:
            print(f"  ... and {len(escape_moves) - 5} more")
    else:
        print("No moves can escape check - CHECKMATE!")


if __name__ == "__main__":
    test_initial_state()
    test_is_in_check_detailed(1)  # Red
    test_is_in_check_detailed(-1)  # Black
    test_checkmate_detailed()
