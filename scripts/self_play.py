"""Self-play to generate training data"""

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
    compute_draw_penalty_by_player,
    compute_repeat_penalty_by_player,
    compute_speed_bonus_by_player,
)


def build_repetition_key(board: List[List[str]], current_player: int) -> str:
    """Build repetition key including side-to-move."""
    board_str = "".join(["".join(row) for row in board])
    return f"{board_str}|{current_player}"


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
        self, temperature: float = None, record_moves: bool = True
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
            ),
            -1: MCTSPlayer(  # Black player
                model=self.model,
                num_simulations=self.num_simulations,
                temperature=temperature,
                dirichlet_alpha=self.dirichlet_alpha,
                epsilon=self.epsilon,
                device=self.device,
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

            print(f"\n--- 第 {step_no} 步 ---")
            print(f"当前玩家: {'红方' if current_player == 1 else '黑方'}")

            # Optional resignation based on value head confidence.
            if (
                self.resign_threshold is not None
                and move_count >= self.min_resign_moves
            ):
                _, predicted_value = self.model.predict(game_state.to_numpy())
                if predicted_value <= self.resign_threshold:
                    resigned_player = current_player
                    resigned_text = "红方" if current_player == 1 else "黑方"
                    print(
                        f"  {resigned_text}认输: 价值 {predicted_value:.3f} "
                        f"<= 阈值 {self.resign_threshold:.3f}"
                    )
                    break

            # Get current player
            mcts_player = players[current_player]

            # Get move from MCTS
            move = mcts_player.get_move(game_state, temperature=temperature)

            if move is None:
                # No valid moves
                print("未找到有效落子，结束本局")
                break

            print(f"选择落子: ({move[0]},{move[1]}) -> ({move[2]},{move[3]})")

            # Get policy before move
            policy = mcts_player.get_policy(game_state, temperature=0.0)

            board = game_state.to_numpy()

            # Execute move
            if not game_state.do_move(move):
                print(f"非法落子: ({move[0]},{move[1]}) -> ({move[2]},{move[3]})，结束本局")
                break

            # Record data (from perspective of current player)
            if record_moves:
                move_entry = {
                    "board": board,
                    "policy": policy,
                    "move": move,
                    "player": current_player,
                    "move_idx": move_count,
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
                    print(f"  [重复检测] 局面第 {repeat_count} 次重复，判定和棋")
                    break
                elif repeat_count >= 3:
                    print(f"  [重复检测] 局面第 {repeat_count} 次出现 (阈值: {self.repetition_draw_count})")
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
        print(f"\n[游戏结束] 结果: {result_text}")

        # Add result to all move data (from winner's perspective)
        if result is not None:
            for entry in move_data:
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

                entry["value"] += speed_bonuses.get(entry["player"], 0.0)
                entry["value"] -= draw_penalties.get(entry["player"], 0.0)
                entry["value"] -= repeat_penalties.get(entry["player"], 0.0)

            # Print reward summary
            print("[奖励统计]")
            print(f"  红方吃子: {game_state.captured_by.get(1, {})}")
            print(f"  黑方吃子: {game_state.captured_by.get(-1, {})}")
            red_samples = [e for e in move_data if e["player"] == 1]
            black_samples = [e for e in move_data if e["player"] == -1]
            if red_samples:
                avg_red = sum(e["value"] for e in red_samples) / len(red_samples)
                print(f"  红方样本平均奖励: {avg_red:.3f} ({len(red_samples)}个样本)")
                for i, s in enumerate(red_samples[:5]):  # 只打印前5个
                    print(f"    [{i+1}] 回合{s['move_idx']:3d} | 落子{s['move']} | 奖励:{s['value']:+.3f}")
                if len(red_samples) > 5:
                    print(f"    ... 还有 {len(red_samples)-5} 个样本")
            if black_samples:
                avg_black = sum(e["value"] for e in black_samples) / len(black_samples)
                print(f"  黑方样本平均奖励: {avg_black:.3f} ({len(black_samples)}个样本)")
                for i, s in enumerate(black_samples[:5]):  # 只打印前5个
                    print(f"    [{i+1}] 回合{s['move_idx']:3d} | 落子{s['move']} | 奖励:{s['value']:+.3f}")
                if len(black_samples) > 5:
                    print(f"    ... 还有 {len(black_samples)-5} 个样本")

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
            print(f"\n=== 第 {game_idx + 1}/{num_games} 局 ===")

            # Play game
            result, move_data = self.play_game(temperature=temperature)

            # Record result
            if result == 1:
                results["red"] += 1
            elif result == -1:
                results["black"] += 1
            else:
                results["draw"] += 1

            print(f"对局结果: {result} ({results})")

            # Add to dataset
            all_data.extend(move_data)

            print(f"总局面数: {len(all_data)}")

            # Save periodically
            if (game_idx + 1) % 10 == 0:
                self._save_data(all_data, save_dir, game_idx + 1)

        # Final save
        self._save_data(all_data, save_dir, "final")

        print("\n=== 最终结果 ===")
        print(f"红方胜: {results['red']}")
        print(f"黑方胜: {results['black']}")
        print(f"和棋: {results['draw']}")
        print(f"总局面数: {len(all_data)}")

        return all_data

    def _save_data(self, data: List[Dict], save_dir: str, suffix: str):
        """Save data to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"selfplay_{suffix}_{timestamp}.pt")

        # Convert to tensors for efficient storage
        boards = []
        policies = []
        values = []

        for entry in data:
            boards.append(torch.from_numpy(entry["board"]))
            policies.append(torch.from_numpy(entry["policy"]))
            values.append(torch.tensor(entry.get("value", 0), dtype=torch.float32))

        if boards:
            dataset = {
                "boards": torch.stack(boards),
                "policies": torch.stack(policies),
                "values": torch.stack(values),
            }
            torch.save(dataset, filename)
            print(f"已保存 {len(data)} 个局面到 {filename}")


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
    device: str = "cpu",
):
    """Run self-play to generate training data"""
    print("=" * 50)
    print("自我对弈数据生成")
    print("=" * 50)

    # Load model
    if model_path and os.path.exists(model_path):
        print(f"[模型] 从 {model_path} 加载模型")
        model = create_model({"device": device})
        model.load(model_path)
        print(f"[模型] 加载成功，设备: {device}")
    else:
        if model_path:
            print(f"[模型] 路径 {model_path} 不存在，创建新模型")
        else:
            print("[模型] 未指定模型路径，创建新模型")
        model = create_model({"device": device})
        print(f"[模型] 新模型创建成功，设备: {device}")

    model.set_training(False)

    # Create self-play
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
        device=device,
    )

    # Generate data
    data = sp.generate_dataset(num_games=num_games, temperature=temperature)

    return data


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
        device=args.device,
    )
