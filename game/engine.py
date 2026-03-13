"""Chinese Chess Game Engine - Board, moves, and game rules"""

import numpy as np
from typing import List, Tuple, Optional
from game.pieces import Color, INITIAL_BOARD, PIECE_VALUES


class GameState:
    """Full game state representation"""

    # Board dimensions
    BOARD_WIDTH = 9
    BOARD_HEIGHT = 10

    # Palace positions (for king and advisor)
    RED_PALACE = [(x, y) for x in [3, 4, 5] for y in [7, 8, 9]]
    BLACK_PALACE = [(x, y) for x in [3, 4, 5] for y in [0, 1, 2]]

    def __init__(self, board: List[List[str]] = None, draw_move_limit: int = 200):
        if board is None:
            self.board = [row[:] for row in INITIAL_BOARD]
        else:
            self.board = [row[:] for row in board]
        self.current_player = Color.RED.value  # 1 for red, -1 for black
        self.draw_move_limit = draw_move_limit
        self.move_history: List[Tuple[int, int, int, int]] = []
        self.captured_history: List[str] = []
        self.last_move: Optional[Tuple[int, int, int, int]] = None

        # Track captured pieces for reward calculation
        # Format: {1: {'r': 0, 'h': 0, ...}, -1: {'R': 0, 'H': 0, ...}}
        self.captured_by = {1: {}, -1: {}}  # 红方吃掉的棋子, 黑方吃掉的棋子

    def copy(self) -> "GameState":
        """Create a deep copy of the game state"""
        new_state = GameState(self.board, draw_move_limit=self.draw_move_limit)
        new_state.current_player = self.current_player
        new_state.draw_move_limit = self.draw_move_limit
        new_state.move_history = self.move_history[:]
        new_state.captured_history = self.captured_history[:]
        new_state.last_move = self.last_move
        new_state.captured_by = {
            1: dict(self.captured_by.get(1, {})),
            -1: dict(self.captured_by.get(-1, {})),
        }
        return new_state

    def get_piece(self, x: int, y: int) -> str:
        """Get piece at position (x, y)"""
        if 0 <= x < self.BOARD_WIDTH and 0 <= y < self.BOARD_HEIGHT:
            return self.board[y][x]
        return ""

    def set_piece(self, x: int, y: int, piece: str):
        """Set piece at position (x, y)"""
        self.board[y][x] = piece

    def is_empty(self, x: int, y: int) -> bool:
        """Check if position is empty"""
        return self.get_piece(x, y) == ""

    def is_enemy(self, x: int, y: int) -> bool:
        """Check if position contains enemy piece"""
        piece = self.get_piece(x, y)
        if not piece:
            return False
        return self.get_piece_color(piece) == -self.current_player

    def is_our_piece(self, x: int, y: int) -> bool:
        """Check if position contains our piece"""
        piece = self.get_piece(x, y)
        if not piece:
            return False
        return self.get_piece_color(piece) == self.current_player

    @staticmethod
    def get_piece_color(piece: str) -> int:
        """Get piece color: 1 for red, -1 for black, 0 for empty"""
        if not piece:
            return 0
        if piece.islower():
            return 1  # Red
        return -1  # Black

    @staticmethod
    def is_red_piece(piece: str) -> bool:
        """Check if piece is red"""
        return piece.islower()

    @staticmethod
    def is_black_piece(piece: str) -> bool:
        """Check if piece is black"""
        return piece.isupper()

    # ==================== Move Generation ====================

    def get_all_valid_moves(self) -> List[Tuple[int, int, int, int]]:
        """Get all valid moves for current player"""
        moves = []
        for y in range(self.BOARD_HEIGHT):
            for x in range(self.BOARD_WIDTH):
                if self.is_our_piece(x, y):
                    piece_moves = self.get_piece_moves(x, y)
                    for move in piece_moves:
                        if self.do_move(move):
                            moves.append(move)
                            self.undo_move()
        return moves

    def get_piece_moves(self, x: int, y: int) -> List[Tuple[int, int, int, int]]:
        """Get all valid moves for piece at (x, y)"""
        piece = self.get_piece(x, y)
        if not piece:
            return []

        piece_type = piece.lower()

        if piece_type == "k":  # King/General
            return self._get_king_moves(x, y)
        elif piece_type == "a":  # Advisor
            return self._get_advisor_moves(x, y)
        elif piece_type == "e":  # Elephant
            return self._get_elephant_moves(x, y)
        elif piece_type == "h":  # Horse
            return self._get_horse_moves(x, y)
        elif piece_type == "r":  # Chariot
            return self._get_chariot_moves(x, y)
        elif piece_type == "c":  # Cannon
            return self._get_cannon_moves(x, y)
        elif piece_type == "p":  # Pawn
            return self._get_pawn_moves(x, y)

        return []

    def _get_king_moves(self, x: int, y: int) -> List[Tuple[int, int, int, int]]:
        """Get valid moves for King/General"""
        moves = []
        is_red = self.is_red_piece(self.get_piece(x, y))

        # Define palace based on color
        palace = self.RED_PALACE if is_red else self.BLACK_PALACE

        # Horizontal and vertical moves within palace
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in palace:
                if self.is_empty(nx, ny) or self.is_enemy(nx, ny):
                    moves.append((x, y, nx, ny))

        # Flying general: kings on same file with no pieces between.
        enemy_king = "K" if is_red else "k"
        step = -1 if is_red else 1
        ny = y + step
        while 0 <= ny < self.BOARD_HEIGHT:
            piece = self.get_piece(x, ny)
            if piece:
                if piece == enemy_king:
                    moves.append((x, y, x, ny))
                break
            ny += step

        return moves

    def _get_advisor_moves(self, x: int, y: int) -> List[Tuple[int, int, int, int]]:
        """Get valid moves for Advisor"""
        moves = []
        is_red = self.is_red_piece(self.get_piece(x, y))

        palace = self.RED_PALACE if is_red else self.BLACK_PALACE

        # Diagonal moves within palace
        for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in palace:
                if self.is_empty(nx, ny) or self.is_enemy(nx, ny):
                    moves.append((x, y, nx, ny))

        return moves

    def _get_elephant_moves(self, x: int, y: int) -> List[Tuple[int, int, int, int]]:
        """Get valid moves for Elephant"""
        moves = []
        is_red = self.is_red_piece(self.get_piece(x, y))

        # Red can only move down (y > 4), Black can only move up (y < 5)
        min_y = 5 if is_red else 0
        max_y = 9 if is_red else 4

        for dx, dy in [(2, 2), (2, -2), (-2, 2), (-2, -2)]:
            nx, ny = x + dx, y + dy
            # Check bounds
            if not (0 <= nx < self.BOARD_WIDTH and min_y <= ny <= max_y):
                continue
            # Check blocking piece (the "eye")
            bx, by = x + dx // 2, y + dy // 2
            if self.is_empty(bx, by) and (
                self.is_empty(nx, ny) or self.is_enemy(nx, ny)
            ):
                moves.append((x, y, nx, ny))

        return moves

    def _get_horse_moves(self, x: int, y: int) -> List[Tuple[int, int, int, int]]:
        """Get valid moves for Horse"""
        moves = []

        # Horse moves: 2 in one direction, 1 perpendicular
        offsets = [
            (1, 2, 0, 1),
            (-1, 2, 0, 1),
            (1, -2, 0, -1),
            (-1, -2, 0, -1),
            (2, 1, 1, 0),
            (2, -1, 1, 0),
            (-2, 1, -1, 0),
            (-2, -1, -1, 0),
        ]

        for dx, dy, bx, by in offsets:
            nx, ny = x + dx, y + dy
            bx, by = x + bx, y + by

            if 0 <= nx < self.BOARD_WIDTH and 0 <= ny < self.BOARD_HEIGHT:
                # Check blocking piece
                if self.is_empty(bx, by) and (
                    self.is_empty(nx, ny) or self.is_enemy(nx, ny)
                ):
                    moves.append((x, y, nx, ny))

        return moves

    def _get_chariot_moves(self, x: int, y: int) -> List[Tuple[int, int, int, int]]:
        """Get valid moves for Chariot (Rook)"""
        moves = []

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.BOARD_WIDTH and 0 <= ny < self.BOARD_HEIGHT:
                if self.is_empty(nx, ny):
                    moves.append((x, y, nx, ny))
                elif self.is_enemy(nx, ny):
                    moves.append((x, y, nx, ny))
                    break
                else:
                    break
                nx, ny = nx + dx, ny + dy

        return moves

    def _get_cannon_moves(self, x: int, y: int) -> List[Tuple[int, int, int, int]]:
        """Get valid moves for Cannon"""
        moves = []

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            screen_found = False

            while 0 <= nx < self.BOARD_WIDTH and 0 <= ny < self.BOARD_HEIGHT:
                if not screen_found:
                    if self.is_empty(nx, ny):
                        moves.append((x, y, nx, ny))
                    else:
                        # First piece in this direction becomes the cannon screen,
                        # regardless of color.
                        screen_found = True
                else:
                    if self.is_enemy(nx, ny):
                        moves.append((x, y, nx, ny))
                        break
                    elif not self.is_empty(nx, ny):
                        break
                nx, ny = nx + dx, ny + dy

        return moves

    def _get_pawn_moves(self, x: int, y: int) -> List[Tuple[int, int, int, int]]:
        """Get valid moves for Pawn"""
        moves = []
        is_red = self.is_red_piece(self.get_piece(x, y))

        # Forward move
        forward = -1 if is_red else 1
        ny = y + forward
        if 0 <= ny < self.BOARD_HEIGHT:
            if self.is_empty(x, ny) or self.is_enemy(x, ny):
                moves.append((x, y, x, ny))

        # Side moves (after crossing the river)
        crossed_river = (is_red and y <= 4) or (not is_red and y >= 5)
        if crossed_river:
            for dx in [-1, 1]:
                nx = x + dx
                if 0 <= nx < self.BOARD_WIDTH:
                    if self.is_empty(nx, y) or self.is_enemy(nx, y):
                        moves.append((x, y, nx, y))

        return moves

    # ==================== Move Execution ====================

    def do_move(self, move: Tuple[int, int, int, int]) -> bool:
        """Execute a move. Returns True if move is valid."""
        x1, y1, x2, y2 = move

        if not self.is_our_piece(x1, y1):
            return False

        valid_moves = self.get_piece_moves(x1, y1)
        if move not in valid_moves:
            return False

        # Execute move
        piece = self.get_piece(x1, y1)
        captured = self.get_piece(x2, y2)

        # Record capture for reward calculation
        if captured:
            mover = self.current_player  # 1 = red, -1 = black
            if captured not in self.captured_by[mover]:
                self.captured_by[mover][captured] = 0
            self.captured_by[mover][captured] += 1

        self.set_piece(x2, y2, piece)
        self.set_piece(x1, y1, "")

        # Check if move leaves king in check (illegal)
        if self.is_in_check(self.current_player):
            # Undo move
            self.set_piece(x1, y1, piece)
            self.set_piece(x2, y2, captured)
            if captured:
                self.captured_by[self.current_player][captured] -= 1
                if self.captured_by[self.current_player][captured] <= 0:
                    del self.captured_by[self.current_player][captured]
            return False

        self.move_history.append(move)
        self.captured_history.append(captured)
        self.last_move = move
        self.current_player = -self.current_player
        return True

    def undo_move(self) -> Optional[Tuple[int, int, int, int]]:
        """Undo last move. Returns the undone move."""
        if not self.move_history:
            return None

        x1, y1, x2, y2 = self.move_history.pop()
        captured = self.captured_history.pop() if self.captured_history else ""
        piece = self.get_piece(x2, y2)
        self.set_piece(x1, y1, piece)
        self.set_piece(x2, y2, captured)

        mover = -self.current_player
        if captured:
            captures = self.captured_by.get(mover, {})
            if captures.get(captured, 0) > 0:
                captures[captured] -= 1
                if captures[captured] <= 0:
                    del captures[captured]

        self.current_player = mover
        self.last_move = self.move_history[-1] if self.move_history else None
        return (x1, y1, x2, y2)

    # ==================== Game Status ====================

    def is_in_check(self, player: int) -> bool:
        """Check if player is in check"""
        # Find king position
        king = "k" if player == 1 else "K"
        king_pos = None

        for y in range(self.BOARD_HEIGHT):
            for x in range(self.BOARD_WIDTH):
                if self.board[y][x] == king:
                    king_pos = (x, y)
                    break
            if king_pos:
                break

        if not king_pos:
            return True  # King captured (shouldn't happen)

        # Check if any enemy piece can attack king's position
        # We need to check from the perspective of the enemy (-player)
        enemy_color = -player

        for y in range(self.BOARD_HEIGHT):
            for x in range(self.BOARD_WIDTH):
                piece = self.board[y][x]
                if piece and self.get_piece_color(piece) == enemy_color:
                    # Get moves for this enemy piece
                    moves = self._get_piece_moves_for_color(x, y, enemy_color)
                    if king_pos in [(m[2], m[3]) for m in moves]:
                        return True

        return False

    def _get_piece_moves_for_color(self, x: int, y: int, color: int) -> List[Tuple[int, int, int, int]]:
        """Get moves for a piece assuming a specific color"""
        piece = self.board[y][x]
        if not piece:
            return []

        # Temporarily set current_player to match the color
        original_player = self.current_player
        self.current_player = color
        moves = self.get_piece_moves(x, y)
        self.current_player = original_player
        return moves

    def is_checkmate(self) -> bool:
        """Check if current player is checkmated"""
        if not self.is_in_check(self.current_player):
            return False

        # Check if any move can escape check
        moves = self.get_all_valid_moves()
        for move in moves:
            # Try move
            piece = self.get_piece(move[0], move[1])
            captured = self.get_piece(move[2], move[3])
            self.set_piece(move[2], move[3], piece)
            self.set_piece(move[0], move[1], "")

            in_check = self.is_in_check(self.current_player)

            # Undo
            self.set_piece(move[0], move[1], piece)
            self.set_piece(move[2], move[3], captured)

            if not in_check:
                return False

        return True

    def is_draw(self) -> bool:
        """Check if game is a draw"""
        # Simplified: too many moves without capture or check
        if len(self.move_history) >= self.draw_move_limit:
            return True
        return False

    def get_game_result(self) -> Optional[int]:
        """Get game result: 1 for red win, -1 for black win, 0 for draw, None for ongoing"""
        if self.is_checkmate():
            return -self.current_player  # Winner is the other player
        if self.is_draw():
            return 0
        return None

    def is_game_over(self) -> bool:
        """Check if game is over"""
        return self.get_game_result() is not None

    def get_capture_reward(self, player: int, verbose: bool = False) -> float:
        """
        Calculate capture reward for a player

        Args:
            player: 1 for red, -1 for black
            verbose: Whether to print detailed log

        Returns:
            Capture reward (positive = captured more valuable pieces)
        """
        player_name = "红方" if player == 1 else "黑方"

        # 我们吃掉的子 (正向奖励)
        our_captures = self.captured_by.get(player, {})
        our_score = sum(PIECE_VALUES.get(piece, 0) * count
                        for piece, count in our_captures.items())

        # 被对手吃掉的子 (负向惩罚)
        opponent = -player
        opponent_captures = self.captured_by.get(opponent, {})
        opponent_score = sum(PIECE_VALUES.get(piece, 0) * count
                             for piece, count in opponent_captures.items())

        # 归一化: 除以最大可能值 (一个车的价值9分)
        # 这样奖励范围大约在 [-2, 2] 之间
        max_piece_value = 9.0  # 車的价值

        reward = (our_score - opponent_score) / max_piece_value

        if verbose:
            print(f"  [吃子奖励] {player_name}:")
            print(f"    我方吃子: {our_captures} (得分: {our_score:.1f})")
            print(f"    被吃子: {opponent_captures} (失分: {opponent_score:.1f})")
            print(f"    净得分: {our_score - opponent_score:.1f}, 归一化奖励: {reward:.3f}")

        return reward

    def get_combined_reward(self, player: int, game_result: int,
                            capture_weight: float = 0.1) -> float:
        """
        Get combined reward including game result and capture bonus

        Args:
            player: Current player (1=red, -1=black)
            game_result: Game result (1=red wins, -1=black wins, 0=draw)
            capture_weight: Weight for capture reward (default 0.1)

        Returns:
            Combined reward value
        """
        # 游戏结果奖励 (主要)
        result_reward = game_result * player

        # 吃子奖励 (辅助)
        capture_reward = self.get_capture_reward(player)

        return result_reward + capture_weight * capture_reward

    # ==================== Board Representation ====================

    def to_numpy(self) -> np.ndarray:
        """Convert board to numpy array for neural network"""
        # 10x9 board, encode pieces as channels
        # Channel encoding:
        # 0-6: Red pieces (k, a, e, h, r, c, p)
        # 7-13: Black pieces (K, A, E, H, R, C, P)
        # 14: Current player indicator

        piece_to_channel = {
            "k": 0,
            "a": 1,
            "e": 2,
            "h": 3,
            "r": 4,
            "c": 5,
            "p": 6,
            "K": 7,
            "A": 8,
            "E": 9,
            "H": 10,
            "R": 11,
            "C": 12,
            "P": 13,
        }

        board = np.zeros((15, self.BOARD_HEIGHT, self.BOARD_WIDTH), dtype=np.float32)

        for y in range(self.BOARD_HEIGHT):
            for x in range(self.BOARD_WIDTH):
                piece = self.board[y][x]
                if piece and piece in piece_to_channel:
                    board[piece_to_channel[piece], y, x] = 1

        # Current player channel
        if self.current_player == 1:
            board[14, :, :] = 1

        return board

    def __str__(self) -> str:
        """String representation of board"""
        lines = []
        lines.append("  0 1 2 3 4 5 6 7 8")
        lines.append("  " + "-" * 17)

        for y in range(self.BOARD_HEIGHT):
            line = str(y) + "|"
            for x in range(self.BOARD_WIDTH):
                piece = self.board[y][x]
                line += piece if piece else "."
                line += " "
            lines.append(line)

        lines.append("  " + "-" * 17)

        # Add move info
        if self.last_move:
            x1, y1, x2, y2 = self.last_move
            lines.append(f"Last move: ({x1},{y1}) -> ({x2},{y2})")

        player = "Red" if self.current_player == 1 else "Black"
        lines.append(f"Current player: {player}")

        return "\n".join(lines)
