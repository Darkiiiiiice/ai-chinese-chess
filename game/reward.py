"""Reward-related helpers."""

from typing import Dict, Iterable, List, Tuple

from game.pieces import get_piece_value


Move = Tuple[int, int, int, int]


def is_reverse_move(current: Move, previous: Move) -> bool:
    """Return True when `current` exactly reverses `previous`."""
    return current[0] == previous[2] and current[1] == previous[3] and \
        current[2] == previous[0] and current[3] == previous[1]


def compute_repeat_penalty_by_player(
    move_entries: Iterable[dict],
    threshold: int = 10,
    penalty_unit: float = 0.2,
) -> Dict[int, float]:
    """
    Compute repetition penalties by player for back-and-forth moves.

    A streak increments when a player makes the reverse move of their previous move.
    Penalty starts once streak >= threshold and increases with streak length.
    """
    penalties: Dict[int, float] = {1: 0.0, -1: 0.0}
    last_move: Dict[int, Move] = {}
    streak: Dict[int, int] = {1: 0, -1: 0}

    for entry in move_entries:
        player = entry.get("player")
        move = entry.get("move")
        if player not in (1, -1):
            continue
        if not (isinstance(move, tuple) and len(move) == 4):
            continue

        prev = last_move.get(player)
        if prev is not None and is_reverse_move(move, prev):
            streak[player] += 1
        else:
            streak[player] = 0

        last_move[player] = move

        if streak[player] >= threshold:
            extra = streak[player] - threshold + 1
            penalties[player] += penalty_unit * extra

    return penalties


def compute_speed_bonus_by_player(
    result: int,
    total_moves: int,
    max_moves: int,
    max_bonus: float = 0.3,
) -> Dict[int, float]:
    """
    Reward faster wins with a larger positive bonus for the winner only.

    The bonus decays linearly from `max_bonus` towards 0 as `total_moves`
    approaches `max_moves`. Draws or invalid limits produce no bonus.
    """
    bonuses: Dict[int, float] = {1: 0.0, -1: 0.0}

    if result not in (1, -1):
        return bonuses
    if max_moves <= 0 or max_bonus <= 0:
        return bonuses

    clamped_moves = min(max(total_moves, 0), max_moves)
    remaining_ratio = 1.0 - (clamped_moves / max_moves)
    bonuses[result] = max_bonus * remaining_ratio
    return bonuses


def compute_draw_penalty_by_player(
    result: int,
    penalty: float = 0.1,
) -> Dict[int, float]:
    """Apply a symmetric penalty to both sides when the game ends in a draw."""
    if result != 0 or penalty <= 0:
        return {1: 0.0, -1: 0.0}
    return {1: penalty, -1: penalty}


def compute_step_capture_reward(captured_piece: str) -> float:
    """Reward an immediate capture using normalized piece value."""
    if not captured_piece:
        return 0.0
    return get_piece_value(captured_piece) / 9.0


def compute_signed_step_reward_by_player(
    captured_piece: str,
    mover: int,
) -> Dict[int, float]:
    """Return a zero-sum immediate reward keyed by player."""
    reward = compute_step_capture_reward(captured_piece)
    if mover not in (1, -1) or reward == 0.0:
        return {1: 0.0, -1: 0.0}
    return {
        mover: reward,
        -mover: -reward,
    }


def accumulate_step_reward_events(
    step_reward_events: Iterable[Dict[int, float]],
) -> List[Dict[int, float]]:
    """Build suffix sums of signed step rewards for each player."""
    events = list(step_reward_events)
    cumulative: List[Dict[int, float]] = [{1: 0.0, -1: 0.0} for _ in events]
    running = {1: 0.0, -1: 0.0}

    for idx in range(len(events) - 1, -1, -1):
        event = events[idx] or {}
        running = {
            1: running[1] + float(event.get(1, 0.0)),
            -1: running[-1] + float(event.get(-1, 0.0)),
        }
        cumulative[idx] = dict(running)

    return cumulative
