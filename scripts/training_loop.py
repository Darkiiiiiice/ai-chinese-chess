#!/usr/bin/env python3
"""Training loop - Full AlphaZero cycle: selfplay + train + evaluate + online play"""

import os
import sys
import time
import torch
import asyncio
from pathlib import Path
from datetime import datetime
import argparse
import glob

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.model import AlphaZero, create_model
from scripts.self_play import SelfPlay
from scripts.train import train as train_model
from browser.automate import XiangqiBrowser
from game.engine import GameState
from ai.mcts import MCTSPlayer


class TrainingLoop:
    """Main training loop for AlphaZero"""

    def __init__(
        self,
        model_path: str = None,
        data_dir: str = "data",
        model_dir: str = "models",
        log_dir: str = "logs",
        device: str = "cpu",
        # Self-play config
        sp_games: int = 100,
        sp_simulations: int = 400,
        sp_temperature: float = 1.0,
        sp_batch_size: int = 16,
        speed_bonus_max: float = 0.3,
        draw_penalty: float = 0.1,
        # Training config
        train_epochs: int = 10,
        train_batch_size: int = 256,
        train_lr: float = 0.001,
        # Evaluation config
        eval_games: int = 10,
        eval_simulations: int = 400,
        # Online play config
        online_games: int = 10,
        online_difficulty: int = 1,
        online_headless: bool = True,
        # Misc
        iterations: int = 10,
        save_interval: int = 1,
        mode: str = "selfplay",  # "selfplay", "online", "mixed"
    ):
        self.model_path = model_path
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.device = device

        # Self-play config
        self.sp_games = sp_games
        self.sp_simulations = sp_simulations
        self.sp_temperature = sp_temperature
        self.sp_batch_size = sp_batch_size
        self.speed_bonus_max = speed_bonus_max
        self.draw_penalty = draw_penalty

        # Training config
        self.train_epochs = train_epochs
        self.train_batch_size = train_batch_size
        self.train_lr = train_lr

        # Evaluation config
        self.eval_games = eval_games
        self.eval_simulations = eval_simulations

        # Online play config
        self.online_games = online_games
        self.online_difficulty = online_difficulty
        self.online_headless = online_headless

        # Iteration config
        self.iterations = iterations
        self.save_interval = save_interval
        self.mode = mode

        # Create directories
        for d in [data_dir, model_dir, log_dir]:
            os.makedirs(d, exist_ok=True)

        # Model
        self.model = None

        # Training history
        self.history = []

    def load_or_create_model(self):
        """Load existing model or create new one"""
        # Try to load best model first, then latest
        model_paths = [
            os.path.join(self.model_dir, "model_best.pt"),
            os.path.join(self.model_dir, "model_latest.pt"),
            self.model_path
        ]

        for path in model_paths:
            if path and os.path.exists(path):
                print(f"从 {path} 加载模型")
                self.model = create_model({"device": self.device})
                self.model.load(path)
                self.model.set_training(False)
                return

        print("创建新模型")
        self.model = create_model({"device": self.device})
        self.model.set_training(False)

    def selfplay(self) -> dict:
        """Generate training data through self-play"""
        print("\n" + "=" * 50)
        print("自我对弈: 生成训练数据")
        print("=" * 50)

        sp = SelfPlay(
            model=self.model,
            num_simulations=self.sp_simulations,
            temperature=self.sp_temperature,
            batch_size=self.sp_batch_size,
            speed_bonus_max=self.speed_bonus_max,
            draw_penalty=self.draw_penalty,
            device=self.device,
        )

        data = sp.generate_dataset(
            num_games=self.sp_games,
            temperature=self.sp_temperature,
            save_dir=self.data_dir,
        )

        return {"samples": len(data)}

    async def online_play(self) -> dict:
        """Generate training data through online play"""
        print("\n" + "=" * 50)
        print("在线对弈: 与网页 AI 对战")
        print("=" * 50)

        from scripts.play import play_game_with_data

        stats = {"wins": 0, "losses": 0, "draws": 0}
        total_samples = 0

        for game_idx in range(self.online_games):
            print(f"\n--- 在线对局 {game_idx + 1}/{self.online_games} ---")

            # Alternate colors
            player_color = 1 if game_idx % 2 == 0 else -1

            browser = XiangqiBrowser(
                headless=self.online_headless,
                model=self.model,
                player_color=player_color,
                difficulty=self.online_difficulty,
                num_simulations=self.sp_simulations,
            )

            try:
                await browser.initialize()
                await browser.navigate_to_game()
                await browser.setup_game(
                    player_color=player_color,
                    difficulty=self.online_difficulty
                )

                result = await play_game_with_data(
                    self.model, browser,
                    save_data=True,
                    data_dir=self.data_dir,
                    device=self.device,
                    speed_bonus_max=self.speed_bonus_max,
                    draw_penalty=self.draw_penalty,
                )

                if result["result"] == player_color:
                    stats["wins"] += 1
                elif result["result"] == -player_color:
                    stats["losses"] += 1
                else:
                    stats["draws"] += 1

                total_samples += result.get("samples", 0)

            except Exception as e:
                print(f"在线对局出错: {e}")
                import traceback
                traceback.print_exc()
            finally:
                await browser.close()

            await asyncio.sleep(2)

        win_rate = stats["wins"] / self.online_games * 100 if self.online_games > 0 else 0
        print(f"\n在线对弈结果: {stats['wins']}胜/{stats['losses']}负/{stats['draws']}和 ({win_rate:.1f}%)")
        print(f"收集了 {total_samples} 个训练样本")

        return {
            "wins": stats["wins"],
            "losses": stats["losses"],
            "draws": stats["draws"],
            "samples": total_samples
        }

    def train(self) -> dict:
        """Train the model"""
        print("\n" + "=" * 50)
        print("训练模型")
        print("=" * 50)

        # Count available data
        data_files = glob.glob(os.path.join(self.data_dir, "*.pt"))
        print(f"找到 {len(data_files)} 个数据文件")

        if not data_files:
            print("未找到训练数据!")
            return {"status": "no_data"}

        # Save model path for checkpoint
        checkpoint_path = os.path.join(self.model_dir, "model_latest.pt")

        train_model(
            data_dir=self.data_dir,
            model_path=checkpoint_path if os.path.exists(checkpoint_path) else None,
            num_epochs=self.train_epochs,
            batch_size=self.train_batch_size,
            learning_rate=self.train_lr,
            device=self.device,
            save_dir=self.model_dir,
            checkpoint_interval=self.save_interval,
        )

        # Reload best model
        best_path = os.path.join(self.model_dir, "model_best.pt")
        if os.path.exists(best_path):
            self.model.load(best_path)
            self.model.set_training(False)

        return {"status": "trained"}

    def evaluate(self) -> dict:
        """Evaluate the model"""
        print("\n" + "=" * 50)
        print("评估模型")
        print("=" * 50)

        results = {"wins": 0, "losses": 0, "draws": 0}

        for i in range(self.eval_games):
            game = GameState()

            while not game.is_game_over():
                mcts = MCTSPlayer(
                    self.model,
                    num_simulations=self.eval_simulations,
                    temperature=0.0,
                    device=self.device,
                )
                move = mcts.get_move(game)

                if not move:
                    break

                game.do_move(move)

            result = game.get_game_result()
            if result == 1:
                results["wins"] += 1
            elif result == -1:
                results["losses"] += 1
            else:
                results["draws"] += 1

        win_rate = results["wins"] / self.eval_games * 100 if self.eval_games > 0 else 0
        print(
            f"评估结果: {results['wins']}胜/{results['losses']}负/{results['draws']}和 ({win_rate:.1f}%)"
        )

        return results

    def run_sync(self):
        """Run the training loop (synchronous part)"""
        print("=" * 50)
        print("AlphaZero 训练循环")
        print(f"模式: {self.mode}")
        print("=" * 50)

        # Load or create model
        self.load_or_create_model()

        start_time = time.time()

        for iteration in range(1, self.iterations + 1):
            print(f"\n{'=' * 50}")
            print(f"第 {iteration}/{self.iterations} 轮迭代")
            print(f"{'=' * 50}")

            iter_start = time.time()

            # Generate data based on mode
            if self.mode == "selfplay":
                sp_result = self.selfplay()
            elif self.mode == "online":
                # Run async online play
                sp_result = asyncio.run(self.online_play())
            else:  # mixed
                # Mix of self-play and online play
                self.selfplay()
                sp_result = asyncio.run(self.online_play())

            # Train
            train_result = self.train()

            # Evaluate (every other iteration)
            if iteration % 2 == 0:
                eval_result = self.evaluate()

            iter_time = time.time() - iter_start

            # Record history
            self.history.append(
                {
                    "iteration": iteration,
                    "time": iter_time,
                    "samples": sp_result.get("samples", 0),
                    "status": train_result.get("status"),
                }
            )

            print(f"\n第 {iteration} 轮完成，耗时 {iter_time:.1f} 秒")

            # Save history
            self._save_history()

        total_time = time.time() - start_time

        print("\n" + "=" * 50)
        print("训练完成!")
        print("=" * 50)
        print(f"总耗时: {total_time:.1f} 秒 ({total_time / 60:.1f} 分钟)")
        print(f"最佳模型: {os.path.join(self.model_dir, 'model_best.pt')}")

    def run(self):
        """Run the training loop"""
        # Handle async parts
        if self.mode == "online":
            # Need to run online play in async context
            asyncio.run(self._run_async())
        elif self.mode == "mixed":
            asyncio.run(self._run_async())
        else:
            self.run_sync()

    async def _run_async(self):
        """Async run for online modes"""
        print("=" * 50)
        print("AlphaZero 训练循环")
        print(f"模式: {self.mode}")
        print("=" * 50)

        # Load or create model
        self.load_or_create_model()

        start_time = time.time()

        for iteration in range(1, self.iterations + 1):
            print(f"\n{'=' * 50}")
            print(f"第 {iteration}/{self.iterations} 轮迭代")
            print(f"{'=' * 50}")

            iter_start = time.time()

            # Generate data based on mode
            if self.mode == "online":
                sp_result = await self.online_play()
            else:  # mixed
                self.selfplay()
                sp_result = await self.online_play()

            # Train
            train_result = self.train()

            # Evaluate (every other iteration)
            if iteration % 2 == 0:
                eval_result = self.evaluate()

            iter_time = time.time() - iter_start

            # Record history
            self.history.append(
                {
                    "iteration": iteration,
                    "time": iter_time,
                    "samples": sp_result.get("samples", 0),
                    "status": train_result.get("status"),
                }
            )

            print(f"\n第 {iteration} 轮完成，耗时 {iter_time:.1f} 秒")

            # Save history
            self._save_history()

        total_time = time.time() - start_time

        print("\n" + "=" * 50)
        print("训练完成!")
        print("=" * 50)
        print(f"总耗时: {total_time:.1f} 秒 ({total_time / 60:.1f} 分钟)")
        print(f"最佳模型: {os.path.join(self.model_dir, 'model_best.pt')}")

    def _save_history(self):
        """Save training history"""
        import json

        history_file = os.path.join(self.log_dir, "training_history.json")
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)


def train(
    data_dir: str = "data",
    model_path: str = None,
    num_epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    device: str = "cpu",
    save_dir: str = "models",
    checkpoint_interval: int = 1,
):
    """Wrapper for training function"""
    from scripts.train import train as train_func

    train_func(
        data_dir=data_dir,
        model_path=model_path,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        save_dir=save_dir,
        checkpoint_interval=checkpoint_interval,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaZero Training Loop")

    # Paths
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--device", type=str, default="cpu")

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        default="selfplay",
        choices=["selfplay", "online", "mixed"],
        help="Training mode: selfplay, online, or mixed"
    )

    # Self-play
    parser.add_argument("--sp-games", type=int, default=100)
    parser.add_argument("--sp-simulations", type=int, default=400)
    parser.add_argument("--sp-temperature", type=float, default=1.0)
    parser.add_argument("--sp-batch-size", type=int, default=16)
    parser.add_argument("--speed-bonus-max", type=float, default=0.3)
    parser.add_argument("--draw-penalty", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)

    # Evaluation
    parser.add_argument("--eval-games", type=int, default=10)
    parser.add_argument("--eval-simulations", type=int, default=400)

    # Online play
    parser.add_argument("--online-games", type=int, default=10)
    parser.add_argument("--online-difficulty", type=int, default=1)
    parser.add_argument("--online-visible", action="store_true", help="Show browser window")

    # Loop
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=1)

    args = parser.parse_args()

    loop = TrainingLoop(
        model_path=args.model,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        device=args.device,
        sp_games=args.sp_games,
        sp_simulations=args.sp_simulations,
        sp_temperature=args.sp_temperature,
        sp_batch_size=args.sp_batch_size,
        speed_bonus_max=args.speed_bonus_max,
        draw_penalty=args.draw_penalty,
        train_epochs=args.epochs,
        train_batch_size=args.batch,
        train_lr=args.lr,
        eval_games=args.eval_games,
        eval_simulations=args.eval_simulations,
        online_games=args.online_games,
        online_difficulty=args.online_difficulty,
        online_headless=not args.online_visible,
        iterations=args.iterations,
        save_interval=args.save_interval,
        mode=args.mode,
    )

    loop.run()
