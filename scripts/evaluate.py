"""Evaluation script - Evaluate trained model performance"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.mcts import MCTSPlayer
from ai.model import AlphaZero, create_model
from game.engine import GameState


def evaluate_model(
    model_path: str,
    num_games: int = 100,
    num_simulations: int = 400,
    batch_size: int = 16,
    device: str = "cpu"
):
    """Evaluate model against random moves"""
    print("=" * 50)
    print("模型评估")
    print("=" * 50)
    print(f"模型路径: {model_path}")
    print(f"评估局数: {num_games}")
    print(f"MCTS 模拟: {num_simulations}")
    print(f"批量大小: {batch_size}")
    print(f"设备: {device}")

    # Load model
    model = create_model({"device": device})
    if os.path.exists(model_path):
        print(f"从 {model_path} 加载模型")
        model.load(model_path)
    else:
        print("模型未找到!")
        return

    model.set_training(False)
    print(f"模型已切换至评估模式")

    # Statistics
    results = {"wins": 0, "losses": 0, "draws": 0}
    total_moves = 0
    start_time = time.time()
    print(f"评估开始")

    for i in range(num_games):
        print(f"评估局数: {i + 1}/{num_games}")
        game_start = time.time()
        game = GameState()

        while not game.is_game_over():
            if game.current_player == 1:
                # Model's turn
                mcts = MCTSPlayer(
                    model,
                    num_simulations=num_simulations,
                    temperature=0.0,
                    device=device,
                    batch_size=batch_size
                )
                move = mcts.get_move(game)
            else:
                # Random move
                import random
                moves = game.get_all_valid_moves()
                move = random.choice(moves) if moves else None

            if move:
                game.do_move(move)
                total_moves += 1

        result = game.get_game_result()
        game_time = time.time() - game_start

        if result == 1:
            results["wins"] += 1
        elif result == -1:
            results["losses"] += 1
        else:
            results["draws"] += 1

        if (i + 1) % 10 == 0:
            print(f"已完成 {i + 1}/{num_games} 局 | 胜: {results['wins']} | 负: {results['losses']} | 和: {results['draws']} | 上局耗时: {game_time:.1f}s")

    total_time = time.time() - start_time

    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)
    print(f"总对局: {num_games}")
    print(f"胜: {results['wins']} ({results['wins'] / num_games * 100:.1f}%)")
    print(f"负: {results['losses']} ({results['losses'] / num_games * 100:.1f}%)")
    print(f"和: {results['draws']} ({results['draws'] / num_games * 100:.1f}%)")
    print(f"平均步数: {total_moves / num_games:.1f}")
    print(f"总耗时: {total_time:.1f}s ({total_time / num_games:.1f}s/局)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="评估训练好的模型")
    parser.add_argument("--model", type=str, default="models/model_best.pt", help="模型路径")
    parser.add_argument("--games", type=int, default=100, help="评估局数")
    parser.add_argument("--simulations", type=int, default=400, help="MCTS 模拟次数")
    parser.add_argument("--batch-size", type=int, default=16, help="批量推理大小")
    parser.add_argument("--device", type=str, default="cpu", help="计算设备 (cpu/cuda)")

    args = parser.parse_args()

    evaluate_model(
        args.model,
        args.games,
        args.simulations,
        args.batch_size,
        args.device
    )
