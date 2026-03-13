"""Chinese Chess Pieces Definition"""

from enum import Enum


class PieceType(Enum):
    """Piece types in Chinese Chess"""
    # Red pieces
    RED_KING = 'k'      # 帥
    RED_ADVISOR = 'a'   # 仕
    RED_ELEPHANT = 'e'  # 相
    RED_HORSE = 'h'     # 馬
    RED_CHARIOT = 'r'   # 車
    RED_CANNON = 'c'    # 炮
    RED_PAWN = 'p'      # 兵
    
    # Black pieces
    BLACK_KING = 'K'    # 將
    BLACK_ADVISOR = 'A' # 士
    BLACK_ELEPHANT = 'E' # 象
    BLACK_HORSE = 'H'   # 馬
    BLACK_CHARIOT = 'R' # 車
    BLACK_CANNON = 'C'  # 炮
    BLACK_PAWN = 'P'    # 卒


class Color(Enum):
    """Player color"""
    RED = 1
    BLACK = -1


# Piece Unicode characters
PIECE_CHARS = {
    'k': '帥', 'a': '仕', 'e': '相', 'h': '馬', 'r': '車', 'c': '炮', 'p': '兵',
    'K': '將', 'A': '士', 'E': '象', 'H': '馬', 'R': '車', 'C': '炮', 'P': '卒',
}

# Initial board setup
INITIAL_BOARD = [
    ['R', 'H', 'E', 'A', 'K', 'A', 'E', 'H', 'R'],  # 0: Black chariot row
    ['',  '',  '',  '',  '',  '',  '',  '',  ''],    # 1
    ['',  'C',  '',  '',  '',  '',  '',  'C',  ''],  # 2: Black cannons
    ['P',  '',  'P',  '',  'P',  '',  'P',  '', 'P'],# 3: Black pawns
    ['',  '',  '',  '',  '',  '',  '',  '',  ''],    # 4
    ['',  '',  '',  '',  '',  '',  '',  '',  ''],    # 5
    ['p',  '',  'p',  '',  'p',  '',  'p',  '', 'p'],# 6: Red pawns
    ['',  'c',  '',  '',  '',  '',  '',  'c',  ''],  # 7: Red cannons
    ['',  '',  '',  '',  '',  '',  '',  '',  ''],    # 8
    ['r', 'h', 'e', 'a', 'k', 'a', 'e', 'h', 'r'],  # 9: Red chariot row
]


def get_piece_color(piece: str) -> int:
    """Get piece color: 1 for red, -1 for black, 0 for empty"""
    if not piece:
        return 0
    if piece.islower():
        return Color.RED.value
    return Color.BLACK.value


def is_red_piece(piece: str) -> bool:
    """Check if piece is red"""
    return piece.islower()


def is_black_piece(piece: str) -> bool:
    """Check if piece is black"""
    return piece.isupper()


# Piece values for capture reward calculation (中国象棋传统子力价值)
# 車=9, 馬=4, 炮=4.5, 相=2, 仕=2, 兵=1 (未过河) / 2 (过河)
PIECE_VALUES = {
    'k': 1000,  # 帅/将 - 极高价值 (实际上被将死就输了)
    'K': 1000,
    'r': 9,     # 車
    'R': 9,
    'h': 4,     # 馬
    'H': 4,
    'c': 4.5,   # 炮
    'C': 4.5,
    'e': 2,     # 相/象
    'E': 2,
    'a': 2,     # 仕/士
    'A': 2,
    'p': 1,     # 兵/卒
    'P': 1,
}


def get_piece_value(piece: str) -> float:
    """Get piece value for capture reward calculation"""
    return PIECE_VALUES.get(piece, 0)
