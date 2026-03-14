"""Self-play to generate training data"""

import multiprocessing as mp
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.mcts import MCTSPlayer
from ai.model import AlphaZero, create_model
from game.engine import GameState
from game.reward import (
    accumulate_step_reward_events,
    compute_draw_penalty_by_player,
    compute_repeat_penalty_by_player,
    compute_signed_step_reward_by_player,
    compute_step_capture_reward,
    compute_speed_bonus_by_player,
)
from utils.log import wprint


# Worker ID for logging (set by each worker process)
_worker_id = None


def set_worker_id(worker_id: int):
    """Set the worker ID for the current process"""
    global _worker_id
    _worker_id = f"Worker-{worker_id}"


def log(message: str):
    """Log with worker prefix"""
    wprint(message, _worker_id)


def build_repetition_key(board: List[List[str]], current_player: int) -> str:
    """Build repetition key including side-to-move."""
    board_str = "".join(["".join(row) for row in board])
    return f"{board_str}|{current_player}"


def split_games_across_workers(num_games: int, num_workers: int) -> List[int]:
    """Split games as evenly as possible across workers."""
    if num_games <= 0 or num_workers <= 0:
        return []

    worker_count = min(num_games, num_workers)
    base = num_games // worker_count
    remainder = num_games % worker_count

    chunks = []
    for idx in range(worker_count):
        chunks.append(base + (1 if idx < remainder else 0))
    return chunks


def _get_piece_at_destination(game_state, move: Tuple[int, int, int, int]) -> str:
    """Best-effort lookup of the piece currently sitting on a move destination."""
    _, _, x2, y2 = move
    if hasattr(game_state, "get_piece"):
        return game_state.get_piece(x2, y2)
    board = getattr(game_state, "board", None)
    if board is not None:
        return board[y2][x2]
    return ""


def save_dataset(data: List[Dict], save_dir: str, suffix: str):
    """Save self-play data to disk."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"selfplay_{suffix}_{timestamp}.pt")

    boards = []
    policies = []
    values = []
    game_ids = []

    for entry in data:
        boards.append(torch.from_numpy(entry["board"]))
        policies.append(torch.from_numpy(entry["policy"]))
        values.append(torch.tensor(entry.get("value", 0), dtype=torch.float32))
        game_ids.append(entry.get("game_id"))

    if boards:
        dataset = {
            "boards": torch.stack(boards),
            "policies": torch.stack(policies),
            "values": torch.stack(values),
            "game_ids": game_ids,
        }
        torch.save(dataset, filename)
        log(f"已保存 {len(data)} 个局面到 {filename}")


def _run_selfplay_worker(worker_args: Dict) -> Dict:
    """Worker entrypoint for parallel self-play."""
    # Set worker ID for logging
    worker_id = worker_args["worker_id"]
    set_worker_id(worker_id)

    model_path = worker_args["model_path"]
    device = worker_args["device"]

    if model_path and os.path.exists(model_path):
        model = create_model({"device": device})
        log(f"[模型] 从 {model_path} 加载模型")
        model.load(model_path)
        log(f"[模型] 加载成功，设备: {device}")
    else:
        if model_path:
            log(f"[模型] 路径 {model_path} 不存在，创建新模型")
        else:
            log("[模型] 未指定模型路径，创建新模型")
        model = create_model({"device": device})
        log(f"[模型] 新模型创建成功，设备: {device}")

    model.set_training(False)

    sp = SelfPlay(
        model=model,
        num_simulations=worker_args["num_simulations"],
        temperature=worker_args["temperature"],
        max_moves=worker_args["max_moves"],
        repetition_draw_count=worker_args["repetition_draw_count"],
        resign_threshold=worker_args["resign_threshold"],
        min_resign_moves=worker_args["min_resign_moves"],
        speed_bonus_max=worker_args["speed_bonus_max"],
        draw_penalty=worker_args["draw_penalty"],
        batch_size=worker_args["batch_size"],
        device=device,
    )

    num_games = worker_args["num_games"]
    all_data = []
    results = {"red": 0, "black": 0, "draw": 0}

    for game_idx in range(num_games):
        log(f"\n=== 第 {game_idx + 1}/{num_games} 局 ===")
        game_id = f"selfplay_worker{worker_id}_game{game_idx}"
        result, move_data = sp.play_game(
            temperature=worker_args["temperature"],
            game_id=game_id,
        )
        if result == 1:
            results["red"] += 1
        elif result == -1:
            results["black"] += 1
        else:
            results["draw"] += 1
        all_data.extend(move_data)

    return {"results": results, "data": all_data}


def _run_selfplay_workers(num_workers: int, worker_args: List[Dict]) -> List[Dict]:
    """Execute self-play workers in parallel."""
    if num_workers <= 1:
        return [_run_selfplay_worker(args) for args in worker_args]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        return pool.map(_run_selfplay_worker, worker_args)


class SelfPlay:
    """Self-play for generating training data"""

    def __init__(
        self,
        model: AlphaZero,
        num_simulations: int = 400,
        temperature: float = 1.0,
        max_moves: int = 300,
        repetition_draw_count: int = 6,
        resign_threshold: Optional[float] = -0.95,
        min_resign_moves: int = 30,
        speed_bonus_max: float = 0.3,
        draw_penalty: float = 0.1,
        batch_size: int = 16,
        dirichlet_alpha: float = 0.03,
        epsilon: float = 0.25,
        device: str = "cpu",
    ):
        self.model = model
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.max_moves = max_moves
        self.repetition_draw_count = repetition_draw_count
        self.resign_threshold = resign_threshold
        self.min_resign_moves = max(0, min_resign_moves)
        self.speed_bonus_max = speed_bonus_max
        self.draw_penalty = draw_penalty
        self.batch_size = batch_size
        self.dirichlet_alpha = dirichlet_alpha
        self.epsilon = epsilon
        self.device = device

        # Move encoder
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

    def encode_move(self, move: Tuple[int, int, int, int]) -> int:
        return self.move_to_idx.get(move, 0)

    def decode_move(self, idx: int) -> Tuple[int, int, int, int]:
        return self.idx_to_move.get(idx, (0, 0, 0, 0))

    def play_game(
        self,
        temperature: float = None,
        record_moves: bool = True,
        game_id: str | None = None,
    ) -> Tuple[int, List[Dict]]:
        """
        Play one self-play game

        Args:
            temperature: Temperature for policy (0 = deterministic)
            record_moves: Whether to record move data

        Returns:
            (game_result, move_data)
        """
        if temperature is None:
            temperature = self.temperature

        # Initialize game
        game_state = GameState(draw_move_limit=self.max_moves)

        # Players for both sides
        players = {
            1: MCTSPlayer(  # Red player
                model=self.model,
                num_simulations=self.num_simulations,
                temperature=temperature,
                dirichlet_alpha=self.dirichlet_alpha,
                epsilon=self.epsilon,
                device=self.device,
                batch_size=self.batch_size,
            ),
            -1: MCTSPlayer(  # Black player
                model=self.model,
                num_simulations=self.num_simulations,
                temperature=temperature,
                dirichlet_alpha=self.dirichlet_alpha,
                epsilon=self.epsilon,
                device=self.device,
                batch_size=self.batch_size,
            ),
        }

        move_data = []
        move_count = 0
        max_moves = self.max_moves
        resigned_player = None

        # Track positions for threefold repetition
        position_history = {}

        while not game_state.is_game_over() and move_count < max_moves:
            current_player = game_state.current_player
            step_no = move_count + 1

            log(f"\n--- 第 {step_no} 步 ---")
            log(f"当前玩家: {'红方' if current_player == 1 else '黑方'}")

            # Optional resignation based on value head confidence.
            if (
                self.resign_threshold is not None
                and move_count >= self.min_resign_moves
            ):
                _, predicted_value = self.model.predict(game_state.to_numpy())
                if predicted_value <= self.resign_threshold:
                    resigned_player = current_player
                    resigned_text = "红方" if current_player == 1 else "黑方"
                    log(
                        f"  {resigned_text}认输: 价值 {predicted_value:.3f} "
                        f"<= 阈值 {self.resign_threshold:.3f}"
                    )
                    break

            # Get current player
            mcts_player = players[current_player]

            # Prefer a single-search API when available to avoid doubling MCTS work.
            if hasattr(mcts_player, "get_move_and_policy"):
                move, policy = mcts_player.get_move_and_policy(
                    game_state,
                    temperature=temperature,
                    policy_temperature=temperature,
                )
            else:
                move = mcts_player.get_move(game_state, temperature=temperature)
                policy = mcts_player.get_policy(game_state, temperature=temperature)

            if move is None:
                # No valid moves
                log("未找到有效落子，结束本局")
                break

            log(f"选择落子: ({move[0]},{move[1]}) -> ({move[2]},{move[3]})")

            board = game_state.to_numpy()
            captured_piece = _get_piece_at_destination(game_state, move)
            step_capture_reward = compute_step_capture_reward(captured_piece)
            step_reward_by_player = compute_signed_step_reward_by_player(
                captured_piece,
                current_player,
            )

            # Execute move
            if not game_state.do_move(move):
                log(f"非法落子: ({move[0]},{move[1]}) -> ({move[2]},{move[3]})，结束本局")
                break

            # Record data (from perspective of current player)
            if record_moves:
                move_entry = {
                    "board": board,
                    "policy": policy,
                    "move": move,
                    "player": current_player,
                    "game_id": game_id,
                    "move_idx": move_count,
                    "step_capture_reward": step_capture_reward,
                    "step_reward_by_player": step_reward_by_player,
                }
                move_data.append(move_entry)

            move_count += 1

            # Check for repetition
            position_key = build_repetition_key(
                game_state.board, game_state.current_player
            )
            if position_key in position_history:
                position_history[position_key] += 1
                repeat_count = position_history[position_key]
                if repeat_count >= self.repetition_draw_count:
                    # Draw by repetition
                    log(f"  [重复检测] 局面第 {repeat_count} 次重复，判定和棋")
                    break
                elif repeat_count >= 3:
                    log(f"  [重复检测] 局面第 {repeat_count} 次出现 (阈值: {self.repetition_draw_count})")
            else:
                position_history[position_key] = 1

            # Reset players periodically to clear MCTS trees
            if move_count % 10 == 0:
                players[1].reset()
                players[-1].reset()

        # Get final result (1 = red wins, -1 = black wins, 0 = draw)
        if resigned_player is not None:
            result = -resigned_player
        else:
            result = game_state.get_game_result()
            if result is None:
                # Self-play can terminate by repetition/guard rails without a winner.
                result = 0

        repeat_penalties = compute_repeat_penalty_by_player(
            move_data,
            threshold=10,
            penalty_unit=0.2,
        )
        speed_bonuses = compute_speed_bonus_by_player(
            result=result,
            total_moves=move_count,
            max_moves=self.max_moves,
            max_bonus=self.speed_bonus_max,
        )
        draw_penalties = compute_draw_penalty_by_player(
            result=result,
            penalty=self.draw_penalty,
        )

        # Print game result
        result_text = "红方胜" if result == 1 else ("黑方胜" if result == -1 else "和棋")
        log(f"\n[游戏结束] 结果: {result_text}")

        # Add result to all move data (from winner's perspective)
        if result is not None:
            cumulative_step_rewards = accumulate_step_reward_events(
                [entry.get("step_reward_by_player", {1: 0.0, -1: 0.0}) for entry in move_data]
            )
            for event_index, entry in enumerate(move_data):
                # 游戏结果奖励 (主要)
                result_reward = result * entry["player"]

                # 吃子奖励 (辅助)
                capture_reward = game_state.get_capture_reward(entry["player"])

                # 组合奖励: 结果占主导，
                # 如果赢了，吃子奖励加成
                # 如果输了，吃子奖励是负的（被吃比吃更重要）
                # 如果和棋，吃子奖励影响较小
                if result_reward > 0:
                    entry["value"] = result_reward + capture_reward * 0.3
                elif result_reward < 0:
                    entry["value"] = result_reward + capture_reward * 0.5
                else:
                    entry["value"] = capture_reward * 0.5  # 和棋时吃子奖励权重降低

                entry["value"] += cumulative_step_rewards[event_index].get(entry["player"], 0.0)
                entry["value"] += speed_bonuses.get(entry["player"], 0.0)
                entry["value"] -= draw_penalties.get(entry["player"], 0.0)
                entry["value"] -= repeat_penalties.get(entry["player"], 0.0)

                # 归一化到 [-1, 1] 范围，匹配 tanh 输出
                entry["value"] = max(-1.0, min(1.0, entry["value"]))

            # Print reward summary
            log("[奖励统计]")
            log(f"  红方吃子: {game_state.captured_by.get(1, {})}")
            log(f"  黑方吃子: {game_state.captured_by.get(-1, {})}")
            red_samples = [e for e in move_data if e["player"] == 1]
            black_samples = [e for e in move_data if e["player"] == -1]
            if red_samples:
                avg_red = sum(e["value"] for e in red_samples) / len(red_samples)
                log(f"  红方样本平均奖励: {avg_red:.3f} ({len(red_samples)}个样本)")
                for i, s in enumerate(red_samples[:5]):  # 只打印前5个
                    log(f"    [{i+1}] 回合{s['move_idx']:3d} | 落子{s['move']} | 奖励:{s['value']:+.3f}")
                if len(red_samples) > 5:
                    log(f"    ... 还有 {len(red_samples)-5} 个样本")
            if black_samples:
                avg_black = sum(e["value"] for e in black_samples) / len(black_samples)
                log(f"  黑方样本平均奖励: {avg_black:.3f} ({len(black_samples)}个样本)")
                for i, s in enumerate(black_samples[:5]):  # 只打印前5个
                    log(f"    [{i+1}] 回合{s['move_idx']:3d} | 落子{s['move']} | 奖励:{s['value']:+.3f}")
                if len(black_samples) > 5:
                    log(f"    ... 还有 {len(black_samples)-5} 个样本")

        return result, move_data

    def _board_to_string(self, board: List[List[str]]) -> str:
        """Convert board to string for repetition detection"""
        return "".join(["".join(row) for row in board])

    def generate_dataset(
        self, num_games: int = 100, temperature: float = 1.0, save_dir: str = "data"
    ) -> List[Dict]:
        """
        Generate dataset through self-play

        Args:
            num_games: Number of games to play
            temperature: Temperature for moves (higher = more exploration)
            save_dir: Directory to save data

        Returns:
            List of all game data
        """
        os.makedirs(save_dir, exist_ok=True)

        all_data = []
        results = {"red": 0, "black": 0, "draw": 0}

        for game_idx in range(num_games):
            log(f"\n=== 第 {game_idx + 1}/{num_games} 局 ===")

            # Play game
            game_id = f"selfplay_game_{game_idx}"
            result, move_data = self.play_game(
                temperature=temperature,
                game_id=game_id,
            )

            # Record result
            if result == 1:
                results["red"] += 1
            elif result == -1:
                results["black"] += 1
            else:
                results["draw"] += 1

            log(f"对局结果: {result} ({results})")

            # Add to dataset
            all_data.extend(move_data)

            log(f"总局面数: {len(all_data)}")

            # Save periodically
            if (game_idx + 1) % 10 == 0:
                self._save_data(all_data, save_dir, game_idx + 1)

        # Final save
        self._save_data(all_data, save_dir, "final")

        log("\n=== 最终结果 ===")
        log(f"红方胜: {results['red']}")
        log(f"黑方胜: {results['black']}")
        log(f"和棋: {results['draw']}")
        log(f"总局面数: {len(all_data)}")

        return all_data

    def _save_data(self, data: List[Dict], save_dir: str, suffix: str):
        """Save data to file"""
        save_dataset(data, save_dir, suffix)


def run_selfplay(
    model_path: str = None,
    num_games: int = 100,
    num_simulations: int = 400,
    temperature: float = 1.0,
    max_moves: int = 300,
    repetition_draw_count: int = 6,
    resign_threshold: Optional[float] = -0.95,
    min_resign_moves: int = 30,
    speed_bonus_max: float = 0.3,
    draw_penalty: float = 0.1,
    batch_size: int = 16,
    num_workers: int = 1,
    device: str = "cpu",
):
    """Run self-play to generate training data"""
    log("=" * 50)
    log("自我对弈数据生成")
    log("=" * 50)

    if num_workers <= 1 or num_games <= 1:
        # Load model
        if model_path and os.path.exists(model_path):
            log(f"[模型] 从 {model_path} 加载模型")
            model = create_model({"device": device})
            model.load(model_path)
            log(f"[模型] 加载成功，设备: {device}")
        else:
            if model_path:
                log(f"[模型] 路径 {model_path} 不存在，创建新模型")
            else:
                log("[模型] 未指定模型路径，创建新模型")
            model = create_model({"device": device})
            log(f"[模型] 新模型创建成功，设备: {device}")

        model.set_training(False)

        sp = SelfPlay(
            model=model,
            num_simulations=num_simulations,
            temperature=temperature,
            max_moves=max_moves,
            repetition_draw_count=repetition_draw_count,
            resign_threshold=resign_threshold,
            min_resign_moves=min_resign_moves,
            speed_bonus_max=speed_bonus_max,
            draw_penalty=draw_penalty,
            batch_size=batch_size,
            device=device,
        )
        return sp.generate_dataset(num_games=num_games, temperature=temperature)

    log(f"[并行] 启动 {num_workers} 个 worker")
    chunks = split_games_across_workers(num_games, num_workers)
    worker_args = []
    for worker_id, chunk_games in enumerate(chunks, start=1):
        worker_args.append(
            {
                "worker_id": worker_id,
                "model_path": model_path,
                "num_games": chunk_games,
                "num_simulations": num_simulations,
                "temperature": temperature,
                "max_moves": max_moves,
                "repetition_draw_count": repetition_draw_count,
                "resign_threshold": resign_threshold,
                "min_resign_moves": min_resign_moves,
                "speed_bonus_max": speed_bonus_max,
                "draw_penalty": draw_penalty,
                "batch_size": batch_size,
                "device": device,
            }
        )

    worker_results = _run_selfplay_workers(len(worker_args), worker_args)
    results = {"red": 0, "black": 0, "draw": 0}
    all_data: List[Dict] = []

    for item in worker_results:
        worker_stats = item.get("results", {})
        results["red"] += worker_stats.get("red", 0)
        results["black"] += worker_stats.get("black", 0)
        results["draw"] += worker_stats.get("draw", 0)
        all_data.extend(item.get("data", []))

    log("\n=== 最终结果 ===")
    log(f"红方胜: {results['red']}")
    log(f"黑方胜: {results['black']}")
    log(f"和棋: {results['draw']}")
    log(f"总局面数: {len(all_data)}")
    save_dataset(all_data, "data", "final")
    return all_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Model path")
    parser.add_argument("--games", type=int, default=100, help="Number of games")
    parser.add_argument("--simulations", type=int, default=400, help="MCTS simulations")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument("--max-moves", type=int, default=300, help="Max moves per self-play game")
    parser.add_argument(
        "--repetition-draw-count",
        type=int,
        default=6,
        help="Repetition occurrences before declaring draw",
    )
    parser.add_argument(
        "--resign-threshold",
        type=float,
        default=-0.95,
        help="Resign when predicted value <= threshold (set < -1 to effectively disable)",
    )
    parser.add_argument(
        "--min-resign-moves",
        type=int,
        default=30,
        help="Do not resign before this many half-moves",
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for MCTS neural network inference",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel self-play workers",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device")

    args = parser.parse_args()

    run_selfplay(
        model_path=args.model,
        num_games=args.games,
        num_simulations=args.simulations,
        temperature=args.temperature,
        max_moves=args.max_moves,
        repetition_draw_count=args.repetition_draw_count,
        resign_threshold=args.resign_threshold,
        min_resign_moves=args.min_resign_moves,
        speed_bonus_max=args.speed_bonus_max,
        draw_penalty=args.draw_penalty,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
