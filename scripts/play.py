"""Play script - Automated gameplay in browser using trained model"""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.model import AlphaZero, create_model
from browser.automate import XiangqiBrowser
from game.engine import GameState
from game.reward import (
    accumulate_step_reward_events,
    compute_draw_penalty_by_player,
    compute_repeat_penalty_by_player,
    compute_signed_step_reward_by_player,
    compute_step_capture_reward,
    compute_speed_bonus_by_player,
)


class OnlineGameData:
    """Helper class to manage online game data in format compatible with self_play.py"""

    def __init__(self):
        self.samples = []
        self.step_reward_events = []

        # Move encoder (same as self_play.py)
        self._init_move_encoder()

    def _init_move_encoder(self):
        """Initialize move encoder"""
        self.move_to_idx = {}
        self.idx_to_move = {}

        idx = 0
        for x1 in range(9):
            for y1 in range(10):
                for x2 in range(9):
                    for y2 in range(10):
                        if (x1, y1) != (x2, y2):
                            move = (x1, y1, x2, y2)
                            self.move_to_idx[move] = idx
                            self.idx_to_move[idx] = move
                            idx += 1

        self.num_moves = idx

    def encode_move(self, move):
        """Encode move to index"""
        return self.move_to_idx.get(move, 0)

    def add_sample(
        self,
        board: np.ndarray,
        policy: np.ndarray,
        move: tuple,
        player: int,
        game_id: str | None = None,
        step_capture_reward: float = 0.0,
        event_index: int | None = None,
    ):
        """Add a sample to the dataset"""
        self.samples.append({
            'board': board,
            'policy': policy,
            'move': move,
            'player': player,
            'game_id': game_id,
            'step_capture_reward': step_capture_reward,
            'event_index': event_index,
        })

    def add_step_reward_event(self, step_reward_by_player: dict[int, float]) -> int:
        """Append a signed immediate reward event and return its index."""
        self.step_reward_events.append({
            1: float(step_reward_by_player.get(1, 0.0)),
            -1: float(step_reward_by_player.get(-1, 0.0)),
        })
        return len(self.step_reward_events) - 1

    def set_values(
        self,
        result: int,
        capture_rewards_by_player: dict[int, float] | None = None,
        repeat_penalty_by_player: dict[int, float] | None = None,
        speed_bonus_by_player: dict[int, float] | None = None,
        draw_penalty_by_player: dict[int, float] | None = None,
    ):
        """Set the value (reward) for all samples based on game result

        Args:
            result: Game result (1=red wins, -1=black wins, 0=draw)
            capture_rewards_by_player: Capture rewards by player side.
            repeat_penalty_by_player: Repetition penalties by player side.
            speed_bonus_by_player: Faster-win bonus by player side.
            draw_penalty_by_player: Draw penalty by player side.
        """
        rewards = capture_rewards_by_player or {1: 0.0, -1: 0.0}
        penalties = repeat_penalty_by_player or {1: 0.0, -1: 0.0}
        speed_bonuses = speed_bonus_by_player or {1: 0.0, -1: 0.0}
        draw_penalties = draw_penalty_by_player or {1: 0.0, -1: 0.0}
        cumulative_step_rewards = accumulate_step_reward_events(self.step_reward_events)

        for entry in self.samples:
            player = entry['player']

            # 游戏结果奖励 (主要)
            result_reward = result * player

            # 吃子奖励 (辅助) - 从当前样本玩家视角
            cap_reward = rewards.get(player, 0.0)

            # 组合奖励
            if result_reward > 0:
                # 赢棋: 吃子加成
                entry['value'] = result_reward + cap_reward * 0.3
            elif result_reward < 0:
                # 输棋: 被吃惩罚更重
                entry['value'] = result_reward + cap_reward * 0.5
            else:
                # 和棋: 吃子奖励权重降低
                entry['value'] = cap_reward * 0.5

            event_index = entry.get('event_index')
            if (
                isinstance(event_index, int)
                and 0 <= event_index < len(cumulative_step_rewards)
            ):
                entry['value'] += cumulative_step_rewards[event_index].get(player, 0.0)
            else:
                entry['value'] += entry.get('step_capture_reward', 0.0)
            entry['value'] += speed_bonuses.get(player, 0.0)
            entry['value'] -= draw_penalties.get(player, 0.0)
            entry['value'] -= penalties.get(player, 0.0)

            # 归一化到 [-1, 1] 范围，匹配 tanh 输出
            entry['value'] = max(-1.0, min(1.0, entry['value']))

    def save(self, filepath: str):
        """Save data to PyTorch format (compatible with train.py)"""
        if not self.samples:
            return

        boards = []
        policies = []
        values = []
        game_ids = []

        fallback_game_id = Path(filepath).stem
        for entry in self.samples:
            boards.append(torch.from_numpy(entry['board']))
            policies.append(torch.from_numpy(entry['policy']))
            values.append(torch.tensor(entry.get('value', 0), dtype=torch.float32))
            game_ids.append(entry.get('game_id') or fallback_game_id)

        dataset = {
            'boards': torch.stack(boards),
            'policies': torch.stack(policies),
            'values': torch.stack(values),
            'game_ids': game_ids,
        }

        torch.save(dataset, filepath)
        print(f"已保存 {len(self.samples)} 个样本到 {filepath}")


def _get_piece_at_destination(game_state, move: tuple[int, int, int, int]) -> str:
    """Best-effort lookup of the piece currently sitting on a move destination."""
    _, _, x2, y2 = move
    if hasattr(game_state, "get_piece"):
        return game_state.get_piece(x2, y2)
    board = getattr(game_state, "board", None)
    if board is not None:
        return board[y2][x2]
    return ""


def _infer_captured_piece_from_board_transition(
    before_board: dict[tuple[int, int], str],
    after_board: dict[tuple[int, int], str],
) -> str:
    """Infer which piece disappeared between two board states."""
    before_counts: dict[str, int] = {}
    after_counts: dict[str, int] = {}

    for piece in before_board.values():
        before_counts[piece] = before_counts.get(piece, 0) + 1
    for piece in after_board.values():
        after_counts[piece] = after_counts.get(piece, 0) + 1

    for piece, count in before_counts.items():
        if after_counts.get(piece, 0) < count:
            return piece
    return ""


async def play_game_with_data(
    model: AlphaZero,
    browser: XiangqiBrowser,
    save_data: bool = True,
    data_dir: str = "data",
    device: str = "cpu",
    batch_size: int = 16,
    wait_timeout_ms: int = 45000,
    speed_bonus_max: float = 0.3,
    draw_penalty: float = 0.1,
) -> dict:
    """
    Play one game against AI opponent with full data collection

    Args:
        model: Trained AlphaZero model
        browser: Browser automation instance
        save_data: Whether to save game data
        data_dir: Directory to save game data
        device: Device for model inference
        batch_size: Batch size for parallel MCTS inference
        wait_timeout_ms: Timeout when waiting for opponent move (milliseconds)
        speed_bonus_max: Maximum faster-win reward bonus for the winner
        draw_penalty: Symmetric penalty applied to both sides on draws

    Returns:
        Game result and statistics
    """
    print("\n" + "=" * 50)
    print("开始新游戏")
    print("=" * 50)

    # Reset browser game data
    browser.game_data = []
    browser.current_move_idx = 0

    # Initialize data collector
    game_data = OnlineGameData()

    # Initialize game state with empty board
    game_state = GameState([["" for _ in range(9)] for _ in range(10)])
    my_color = browser.player_color
    game_id = f"online_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    # Read initial board from browser
    board_state = await browser.read_board()

    # Debug: print what was read
    print(f"从浏览器读取到 {len(board_state)} 个棋子")
    kings_found = [(x, y, p) for (x, y), p in board_state.items() if p.lower() == 'k']
    print(f"找到的将/帅: {kings_found}")

    # Sync board with browser - use full sync
    XiangqiBrowser.sync_game_state_from_board(game_state, board_state)

    # Determine who goes first - Red always goes first in Xiangqi
    # game_state.current_player is 1 (Red) after initialization
    my_side, opponent_side = XiangqiBrowser.describe_sides(my_color)
    print(f"我方执: {my_side}")
    print(f"对手执: {opponent_side}")
    print(f"当前回合: {'红方' if game_state.current_player == 1 else '黑方'}")

    max_moves = 200
    move_count = 0
    forced_result = None
    forced_my_outcome = "unknown"

    while not game_state.is_game_over() and move_count < max_moves:
        move_count += 1

        # Check if game is over in browser
        if await browser.is_game_over():
            print("浏览器检测到游戏结束")
            break

        current_player = game_state.current_player
        is_our_turn = browser.is_our_turn(current_player)

        print(f"\n--- 第 {move_count} 步 ---")
        print(f"当前玩家: {'红方' if current_player == 1 else '黑方'}")
        print(f"是否我方回合: {is_our_turn}")

        if is_our_turn:
            print(f"\n我方落子 {move_count} (玩家={current_player})")

            # Get board state before move
            board_np = game_state.to_numpy()

            # Get move and policy from model (with timing)
            import time
            start_time = time.time()
            move, policy = await _get_ai_move_with_policy(
                game_state, model, browser.num_simulations, device, browser.batch_size
            )
            elapsed_time = time.time() - start_time
            print(f"思考耗时: {elapsed_time:.2f} 秒")

            if move:
                ranked_moves = _rank_legal_moves_by_policy(game_state, policy)
                move_candidates = _build_move_candidates(move, ranked_moves, max_candidates=4)

                executed_move = None
                latest_board = await browser.read_board()
                for idx, candidate in enumerate(move_candidates, start=1):
                    x1, y1, x2, y2 = candidate
                    prefix = "我方" if idx == 1 else f"我方候选{idx}"
                    print(f"{prefix}: ({x1},{y1}) -> ({x2},{y2})")

                    if (x1, y1) not in latest_board:
                        print(f"浏览器棋盘无源棋子 ({x1},{y1})，跳过该候选")
                        if idx < len(move_candidates):
                            latest_board = await browser.read_board()
                        continue

                    move_ok = await browser.execute_move(x1, y1, x2, y2)
                    if move_ok:
                        executed_move = candidate
                        break

                    if idx < len(move_candidates):
                        print("浏览器执行落子失败，尝试下一候选落子")
                        latest_board = await browser.read_board()

                if executed_move is None:
                    print("候选落子全部失败，判定我方负")
                    board_state = await browser.read_board()
                    XiangqiBrowser.sync_game_state_from_board(game_state, board_state)
                    forced_result = -browser.player_color
                    forced_my_outcome = "loss"
                    break

                # Record sample AFTER browser confirms move
                captured_piece = _get_piece_at_destination(game_state, executed_move)
                step_capture_reward = compute_step_capture_reward(
                    captured_piece
                )
                event_index = game_data.add_step_reward_event(
                    compute_signed_step_reward_by_player(captured_piece, current_player)
                )
                game_data.add_sample(
                    board=board_np,
                    policy=policy,
                    move=executed_move,
                    player=current_player,
                    game_id=game_id,
                    step_capture_reward=step_capture_reward,
                    event_index=event_index,
                )

                # Update local game state
                if not game_state.do_move(executed_move):
                    # Local engine can be stale vs scraped board; keep turn progression sane.
                    game_state.current_player = -browser.player_color
            else:
                print("未找到有效落子!")
                break

            if game_state.is_game_over() or await browser.is_game_over():
                print(f"\n游戏结束: {game_state.get_game_result()}")
                break

            # Wait for opponent
            import time
            wait_start = time.time()
            baseline_board = XiangqiBrowser.game_state_to_board_dict(game_state)
            detected = await browser.wait_for_opponent_move(
                timeout=wait_timeout_ms,
                baseline_board=baseline_board,
            )
            wait_time = time.time() - wait_start
            print(f"对手思考耗时: {wait_time:.2f} 秒")
            if not detected:
                recovery_board = await browser.read_board()
                XiangqiBrowser.sync_game_state_from_board(game_state, recovery_board)
                recovered_to_our_turn = await browser.detect_our_turn_from_hints(
                    board_state=recovery_board,
                    max_pieces=None,
                )
                if recovered_to_our_turn:
                    game_state.current_player = browser.player_color
                    print("等待超时兜底: 检测到我方可走，恢复我方回合")
                else:
                    game_state.current_player = -browser.player_color
                    print("等待超时兜底: 未检测到我方可走，继续等待对手")
                continue
            print("\n轮到你落子了")

            # Sync board with browser
            board_state = await browser.read_board()
            print("\n当前棋盘:")
            opponent_captured_piece = _infer_captured_piece_from_board_transition(
                baseline_board,
                board_state,
            )
            game_data.add_step_reward_event(
                compute_signed_step_reward_by_player(
                    opponent_captured_piece,
                    -browser.player_color,
                )
            )
            XiangqiBrowser.sync_after_opponent_move(game_state, board_state, browser.player_color)

        else:
            # Wait for opponent move
            print(f"\n等待对手(网页AI)落子 {move_count}")
            import time
            wait_start = time.time()
            baseline_board = XiangqiBrowser.game_state_to_board_dict(game_state)
            detected = await browser.wait_for_opponent_move(
                timeout=wait_timeout_ms,
                baseline_board=baseline_board,
            )
            wait_time = time.time() - wait_start
            print(f"对手思考耗时: {wait_time:.2f} 秒")
            if not detected:
                recovery_board = await browser.read_board()
                XiangqiBrowser.sync_game_state_from_board(game_state, recovery_board)
                recovered_to_our_turn = await browser.detect_our_turn_from_hints(
                    board_state=recovery_board,
                    max_pieces=None,
                )
                if recovered_to_our_turn:
                    game_state.current_player = browser.player_color
                    print("等待超时兜底: 检测到我方可走，恢复我方回合")
                else:
                    game_state.current_player = -browser.player_color
                    print("等待超时兜底: 未检测到我方可走，继续等待对手")
                continue

            # Sync board with browser
            board_state = await browser.read_board()
            opponent_captured_piece = _infer_captured_piece_from_board_transition(
                baseline_board,
                board_state,
            )
            game_data.add_step_reward_event(
                compute_signed_step_reward_by_player(
                    opponent_captured_piece,
                    -browser.player_color,
                )
            )
            XiangqiBrowser.sync_after_opponent_move(game_state, board_state, browser.player_color)

        await asyncio.sleep(0.5)

    # Get game result
    result = game_state.get_game_result()
    result_text = await browser.get_game_result_text()
    my_outcome = await browser.get_my_game_outcome()

    if forced_result is not None:
        result = forced_result
        my_outcome = forced_my_outcome

    # Override result from browser if available
    if forced_result is not None:
        pass
    elif result_text == 'red_wins':
        result = 1
    elif result_text == 'black_wins':
        result = -1
    elif result_text == 'draw':
        result = 0
    elif result is None:
        # Guard rail: treat unresolved terminal status as draw for training labels.
        result = 0

    # Set values for all samples (include capture reward)
    if result is not None:
        repeat_penalties = compute_repeat_penalty_by_player(
            game_data.samples,
            threshold=10,
            penalty_unit=0.2,
        )
        capture_rewards = {
            1: game_state.get_capture_reward(1),
            -1: game_state.get_capture_reward(-1),
        }
        speed_bonuses = compute_speed_bonus_by_player(
            result=result,
            total_moves=move_count,
            max_moves=max_moves,
            max_bonus=speed_bonus_max,
        )
        draw_penalties = compute_draw_penalty_by_player(
            result=result,
            penalty=draw_penalty,
        )
        game_data.set_values(
            result,
            capture_rewards,
            repeat_penalty_by_player=repeat_penalties,
            speed_bonus_by_player=speed_bonuses,
            draw_penalty_by_player=draw_penalties,
        )

    # Print result
    result_text_cn = "红方胜" if result == 1 else ("黑方胜" if result == -1 else "和棋")
    print(f"\n[游戏结束] 结果: {result_text_cn}")

    if my_outcome == "win":
        print("我方结果: 胜")
    elif my_outcome == "loss":
        print("我方结果: 负")
    elif my_outcome == "draw":
        print("我方结果: 和")

    print(f"总步数: {move_count}")

    # Print reward summary
    print("[奖励统计]")
    print(f"  红方吃子: {game_state.captured_by.get(1, {})}")
    print(f"  黑方吃子: {game_state.captured_by.get(-1, {})}")
    red_samples = [e for e in game_data.samples if e['player'] == 1]
    black_samples = [e for e in game_data.samples if e['player'] == -1]
    if red_samples:
        avg_red = sum(e['value'] for e in red_samples) / len(red_samples)
        print(f"  红方样本平均奖励: {avg_red:.3f} ({len(red_samples)}个样本)")
        for i, s in enumerate(red_samples[:5]):  # 只打印前5个
            move = s.get('move', (0,0,0,0))
            print(f"    [{i+1}] 回合{s.get('move_idx', 0):3d} | 落子{move} | 奖励:{s['value']:+.3f}")
        if len(red_samples) > 5:
            print(f"    ... 还有 {len(red_samples)-5} 个样本")
    if black_samples:
        avg_black = sum(e['value'] for e in black_samples) / len(black_samples)
        print(f"  黑方样本平均奖励: {avg_black:.3f} ({len(black_samples)}个样本)")
        for i, s in enumerate(black_samples[:5]):  # 只打印前5个
            move = s.get('move', (0,0,0,0))
            print(f"    [{i+1}] 回合{s.get('move_idx', 0):3d} | 落子{move} | 奖励:{s['value']:+.3f}")
        if len(black_samples) > 5:
            print(f"    ... 还有 {len(black_samples)-5} 个样本")

    print(f"收集了 {len(game_data.samples)} 个训练样本")

    # Save data in PyTorch format
    if save_data and game_data.samples:
        os.makedirs(data_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as PyTorch format for training
        pt_file = os.path.join(data_dir, f"online_{timestamp}.pt")
        game_data.save(pt_file)

        # Also save game record as JSON for analysis
        json_file = os.path.join(data_dir, f"game_{timestamp}.json")
        import json

        with open(json_file, "w") as f:
            json.dump({
                "result": result,
                "num_moves": move_count,
                "player_color": browser.player_color,
                "game_id": game_id,
                "timestamp": timestamp,
            }, f, indent=2)

        print(f"游戏记录已保存到 {json_file}")

    return {
        'result': result,
        'my_outcome': my_outcome,
        'moves': browser.game_data,
        'num_moves': move_count,
        'samples': len(game_data.samples)
    }


async def _get_ai_move_with_policy(
    game_state: GameState,
    model: AlphaZero,
    num_simulations: int,
    device: str,
    batch_size: int = 16
):
    """Get AI move and policy from model"""
    if model is None:
        # Random move for testing
        import random
        moves = game_state.get_all_valid_moves()
        if moves:
            move = random.choice(moves)
            # Create uniform policy
            policy = np.zeros(8010, dtype=np.float32)
            for m in moves:
                idx = _encode_move(m)
                policy[idx] = 1.0 / len(moves)
            return move, policy
        return None, None

    # Use MCTS with batch inference
    from ai.mcts import MCTSPlayer

    mcts = MCTSPlayer(
        model=model,
        num_simulations=num_simulations,
        temperature=1.0,  # Use temperature for exploration
        device=device,
        batch_size=batch_size
    )

    if hasattr(mcts, "get_move_and_policy"):
        move, policy = mcts.get_move_and_policy(
            game_state,
            temperature=1.0,
            policy_temperature=1.0,
        )
    else:
        move = mcts.get_move(game_state)
        policy = mcts.get_policy(game_state, temperature=1.0)

    return move, policy


def _rank_legal_moves_by_policy(
    game_state: GameState, policy: np.ndarray | None
) -> list[tuple[int, int, int, int]]:
    """Return legal moves sorted by policy score descending."""
    legal_moves = game_state.get_all_valid_moves()
    if not legal_moves:
        return []

    if policy is None:
        return legal_moves

    policy_len = len(policy)
    scored = []
    for candidate in legal_moves:
        idx = _encode_move(candidate)
        score = float(policy[idx]) if idx < policy_len else 0.0
        scored.append((score, candidate))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in scored]


def _build_move_candidates(
    primary_move: tuple[int, int, int, int],
    ranked_moves: list[tuple[int, int, int, int]],
    max_candidates: int = 4,
) -> list[tuple[int, int, int, int]]:
    """Build candidate list preferring different source pieces first."""
    candidates = [primary_move]
    seen_moves = {primary_move}
    seen_sources = {primary_move[:2]}

    # First pass: diversify source pieces.
    for move in ranked_moves:
        if move in seen_moves or move[:2] in seen_sources:
            continue
        candidates.append(move)
        seen_moves.add(move)
        seen_sources.add(move[:2])
        if len(candidates) >= max_candidates:
            return candidates

    # Second pass: fill remaining slots regardless of source.
    for move in ranked_moves:
        if move in seen_moves:
            continue
        candidates.append(move)
        seen_moves.add(move)
        if len(candidates) >= max_candidates:
            break

    return candidates


def _encode_move(move: tuple) -> int:
    """Encode move to index"""
    x1, y1, x2, y2 = move
    idx = 0
    for xx1 in range(9):
        for yy1 in range(10):
            for xx2 in range(9):
                for yy2 in range(10):
                    if (xx1, yy1) != (xx2, yy2):
                        if (xx1, yy1, xx2, yy2) == move:
                            return idx
                        idx += 1
    return 0


async def run_automated_play(
    model_path: str = None,
    num_games: int = 1,
    difficulty: int = 1,
    player_color: int = 1,  # 1=red, -1=black, 0=random
    red_first: bool = True,
    num_simulations: int = 400,
    batch_size: int = 16,
    headless: bool = False,
    save_data: bool = True,
    data_dir: str = "data",
    device: str = "cpu",
    restart_after_game: bool = True,
    wait_timeout_ms: int = 45000,
    speed_bonus_max: float = 0.3,
    draw_penalty: float = 0.1,
):
    """
    Run automated play against browser AI

    Args:
        model_path: Path to trained model
        num_games: Number of games to play
        difficulty: Difficulty level (0=easy, 1=medium, 2=hard)
        player_color: 1 for red, -1 for black, 0 for random
        red_first: True if player goes first (red)
        num_simulations: MCTS simulations per move
        headless: Run browser in headless mode
        save_data: Save game data
        data_dir: Directory to save data
        device: Device for model
        restart_after_game: Restart game after each game
        wait_timeout_ms: Timeout when waiting for opponent move (milliseconds)
        speed_bonus_max: Maximum faster-win reward bonus for the winner
        draw_penalty: Symmetric penalty applied to both sides on draws
    """
    print("=" * 50)
    print("象棋自动化对弈")
    print("=" * 50)

    # Load model
    if model_path and os.path.exists(model_path):
        print(f"从 {model_path} 加载模型")
        model = create_model({"device": device})
        model.load(model_path)
        model.set_training(False)
    else:
        print("未找到模型，使用随机落子")
        model = None

    # Statistics
    stats = {"wins": 0, "losses": 0, "draws": 0}
    total_samples = 0

    browser = None
    try:
        for game_idx in range(num_games):
            print(f"\n{'=' * 50}")
            print(f"第 {game_idx + 1}/{num_games} 局")
            print(f"{'=' * 50}")

            try:
                if browser is None:
                    browser = XiangqiBrowser(
                        headless=headless,
                        model=model,
                        player_color=player_color,
                        difficulty=difficulty,
                        num_simulations=num_simulations,
                        batch_size=batch_size,
                    )

                    await browser.initialize()
                    await browser.navigate_to_game()
                    await browser.setup_game(
                        difficulty=difficulty, player_color=player_color, red_first=red_first
                    )
                elif game_idx > 0 and restart_after_game:
                    await browser.restart_game(
                        difficulty=difficulty,
                        player_color=player_color,
                        red_first=red_first,
                    )

                # Play game with data collection
                result = await play_game_with_data(
                    model,
                    browser,
                    save_data,
                    data_dir,
                    device,
                    batch_size,
                    wait_timeout_ms,
                    speed_bonus_max,
                    draw_penalty,
                )

                # Update stats
                actual_player_color = browser.player_color
                if result["result"] == actual_player_color:
                    stats["wins"] += 1
                elif result["result"] == -actual_player_color:
                    stats["losses"] += 1
                else:
                    stats["draws"] += 1

                total_samples += result.get("samples", 0)

                # Print stats
                print("\n--- 统计 ---")
                print(f"胜: {stats['wins']}")
                print(f"负: {stats['losses']}")
                print(f"和: {stats['draws']}")
                print(f"总样本数: {total_samples}")

                # Restart or exit
                if game_idx < num_games - 1 and restart_after_game:
                    print("\n重新开始游戏...")
                    await asyncio.sleep(2)
                elif game_idx < num_games - 1:
                    break

            except Exception as e:
                print(f"\n游戏出错: {e}")
                import traceback
                traceback.print_exc()

                if browser is not None:
                    try:
                        await browser.close()
                    finally:
                        browser = None

                # Try to continue
                if game_idx < num_games - 1:
                    await asyncio.sleep(2)
    finally:
        if browser is not None:
            await browser.close()

    # Final statistics
    print("\n" + "=" * 50)
    print("最终统计")
    print("=" * 50)
    print(f"对局数: {num_games}")
    print(f"胜: {stats['wins']}")
    print(f"负: {stats['losses']}")
    print(f"和: {stats['draws']}")
    print(f"总训练样本: {total_samples}")

    if num_games > 0:
        win_rate = stats["wins"] / num_games * 100
        print(f"胜率: {win_rate:.1f}%")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Automated Xiangqi Play")

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="models/model_best.pt",
        help="Path to trained model",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for model (cpu/cuda)"
    )

    # Game settings
    parser.add_argument("--games", type=int, default=1, help="Number of games to play")
    parser.add_argument(
        "--difficulty",
        type=int,
        default=1,
        help="Difficulty (0=easy, 1=medium, 2=hard)",
    )
    parser.add_argument(
        "--color",
        type=int,
        default=1,
        choices=[1, -1, 0],
        help="Our side color (1=red, -1=black, 0=random)",
    )
    parser.add_argument(
        "--red-first", action="store_true", default=True, help="Red moves first"
    )
    parser.add_argument(
        "--black-first",
        action="store_false",
        dest="red_first",
        help="Black moves first",
    )

    # MCTS settings
    parser.add_argument(
        "--simulations",
        type=int,
        default=400,
        help="Number of MCTS simulations per move",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for parallel MCTS inference (higher = better GPU utilization)",
    )
    parser.add_argument(
        "--wait-timeout-ms",
        type=int,
        default=45000,
        help="Timeout when waiting for opponent move in milliseconds",
    )
    parser.add_argument(
        "--speed-bonus-max",
        type=float,
        default=0.3,
        help="Maximum faster-win reward bonus for the winner",
    )
    parser.add_argument(
        "--draw-penalty",
        type=float,
        default=0.1,
        help="Symmetric penalty applied to both sides on draws",
    )

    # Browser settings
    parser.add_argument(
        "--headless", action="store_true", help="Run browser in headless mode"
    )
    parser.add_argument(
        "--visible", action="store_false", dest="headless", help="Show browser window"
    )

    # Data settings
    parser.add_argument("--no-save", action="store_true", help="Do not save game data")
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Directory to save game data"
    )

    args = parser.parse_args()

    # Determine player color (use --color parameter, not --red-first)
    player_color = args.color

    await run_automated_play(
        model_path=args.model if os.path.exists(args.model) else None,
        num_games=args.games,
        difficulty=args.difficulty,
        player_color=player_color,
        red_first=args.red_first,
        num_simulations=args.simulations,
        batch_size=args.batch_size,
        headless=args.headless,
        save_data=not args.no_save,
        data_dir=args.data_dir,
        device=args.device,
        wait_timeout_ms=args.wait_timeout_ms,
        speed_bonus_max=args.speed_bonus_max,
        draw_penalty=args.draw_penalty,
    )


if __name__ == "__main__":
    asyncio.run(main())
