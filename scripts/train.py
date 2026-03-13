"""Training script for AlphaZero model"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict
import glob

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.model import AlphaZero, create_model


class AlphaZeroDataset(Dataset):
    """Dataset for AlphaZero training"""

    def __init__(self, data_dir: str = "data"):
        self.samples = self._load_data(data_dir)

    def _load_data(self, data_dir: str) -> List[Dict]:
        """Load all training data"""
        samples = []

        # Find all supported training data files.
        files = []
        for pattern in ("selfplay_*.pt", "online_*.pt"):
            files.extend(glob.glob(os.path.join(data_dir, pattern)))
        files = sorted(set(files))

        if not files:
            print(f"在 {data_dir} 中未找到数据文件")
            return samples

        print(f"[数据加载] 找到 {len(files)} 个数据文件:")
        print("-" * 60)

        total_samples = 0
        for f in files:
            try:
                data = torch.load(f)
                file_samples = len(data["boards"])
                total_samples += file_samples

                # 获取文件大小
                file_size = os.path.getsize(f) / 1024  # KB

                # 判断文件类型
                filename = os.path.basename(f)
                file_type = "自我对弈" if filename.startswith("selfplay") else "在线对弈"

                print(f"  [{file_type}] {filename}")
                print(f"      样本数: {file_samples}, 文件大小: {file_size:.1f} KB")

                for i in range(len(data["boards"])):
                    samples.append(
                        {
                            "board": data["boards"][i],
                            "policy": data["policies"][i],
                            "value": data["values"][i],
                        }
                    )
            except Exception as e:
                print(f"  [错误] 加载 {os.path.basename(f)} 出错: {e}")

        print("-" * 60)
        print(f"[数据加载] 总计: {total_samples} 个样本")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class Trainer:
    """AlphaZero Trainer"""

    def __init__(
        self,
        model: AlphaZero,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        weight_decay: float = 0.0001,
        l2_reg: float = 0.0001,
        device: str = "cpu",
    ):
        self.model = model
        self.device = device

        # Move model to device
        self.model.policy_net.to(device)
        self.model.value_net.to(device)

        # Optimizers
        self.policy_optimizer = optim.Adam(
            model.policy_net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.value_optimizer = optim.Adam(
            model.value_net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Loss functions
        self.policy_loss_fn = nn.KLDivLoss(reduction="batchmean")
        self.value_loss_fn = nn.MSELoss()

        self.batch_size = batch_size
        self.l2_reg = l2_reg

        # Training state
        self.epoch = 0
        self.step = 0

    def train_step(self, boards, policies, values):
        """Single training step"""
        boards = boards.to(self.device)
        policies = policies.to(self.device).float()
        values = values.to(self.device).unsqueeze(1)

        # KLDivLoss expects a probability target by default.
        policy_sums = policies.sum(dim=1, keepdim=True).clamp_min(1e-8)
        policies = policies / policy_sums

        # Forward pass - use separate networks
        policy_log = self.model.policy_net(boards)
        value = self.model.value_net(boards)

        # Policy loss (KL divergence)
        policy_loss = self.policy_loss_fn(policy_log, policies)

        # Value loss (MSE)
        value_loss = self.value_loss_fn(value, values)

        # Total loss
        loss = policy_loss + value_loss

        # Backward pass
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.policy_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.value_net.parameters(), 1.0)

        self.policy_optimizer.step()
        self.value_optimizer.step()

        self.step += 1

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }

    def train_epoch(self, dataloader: DataLoader) -> Dict:
        """Train one epoch"""
        self.model.set_training(True)

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0

        for batch in dataloader:
            stats = self.train_step(batch["board"], batch["policy"], batch["value"])

            total_loss += stats["loss"]
            total_policy_loss += stats["policy_loss"]
            total_value_loss += stats["value_loss"]
            num_batches += 1

            if self.step % 100 == 0:
                print(
                    f"步数 {self.step}: 损失={stats['loss']:.4f}, "
                    f"策略={stats['policy_loss']:.4f}, "
                    f"价值={stats['value_loss']:.4f}"
                )

        self.epoch += 1

        return {
            "loss": total_loss / num_batches,
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
        }

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )

        checkpoint = {
            "epoch": self.epoch,
            "step": self.step,
            "policy_state_dict": self.model.policy_net.state_dict(),
            "value_state_dict": self.model.value_net.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
        }

        torch.save(checkpoint, path)
        print(f"检查点已保存到 {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.model.value_net.load_state_dict(checkpoint["value_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])

        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]

        print(f"从 {path} 加载检查点")


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
    """
    Train AlphaZero model

    Args:
        data_dir: Directory containing training data
        model_path: Path to existing model (optional)
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        save_dir: Directory to save models
        checkpoint_interval: Save every N epochs
    """
    print("=" * 50)
    print("AlphaZero 训练")
    print("=" * 50)

    os.makedirs(save_dir, exist_ok=True)

    # Create or load model
    if model_path and os.path.exists(model_path):
        print(f"从 {model_path} 加载模型")
        model = create_model({"device": device})
        model.load(model_path)
    else:
        print("创建新模型")
        model = create_model({"device": device})

    # Create trainer
    trainer = Trainer(
        model=model, learning_rate=learning_rate, batch_size=batch_size, device=device
    )

    # Load data
    dataset = AlphaZeroDataset(data_dir)

    if len(dataset) == 0:
        print("未找到训练数据!")
        print("请先运行自我对弈生成数据:")
        print("  python scripts/self_play.py")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print(f"使用 {len(dataset)} 个样本训练")
    print(f"批次大小: {batch_size}, 训练轮数: {num_epochs}")

    # Training loop
    best_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"\n=== 第 {epoch + 1}/{num_epochs} 轮 ===")

        start_time = time.time()

        # Train
        stats = trainer.train_epoch(dataloader)

        epoch_time = time.time() - start_time

        print(f"第 {epoch + 1} 轮完成，耗时 {epoch_time:.1f} 秒")
        print(
            f"损失: {stats['loss']:.4f}, "
            f"策略: {stats['policy_loss']:.4f}, "
            f"价值: {stats['value_loss']:.4f}"
        )

        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            model_path = os.path.join(save_dir, f"model_epoch{epoch + 1}.pt")
            trainer.save_checkpoint(model_path)

            # Keep best model
            if stats["loss"] < best_loss:
                best_loss = stats["loss"]
                best_path = os.path.join(save_dir, "model_best.pt")
                trainer.save_checkpoint(best_path)

    # Save final model
    final_path = os.path.join(save_dir, "model_latest.pt")
    trainer.save_checkpoint(final_path)

    print("\n训练完成!")
    print(f"最佳模型已保存到 {os.path.join(save_dir, 'model_best.pt')}")
    print(f"最新模型已保存到 {final_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data", help="Data directory")
    parser.add_argument("--model", type=str, default=None, help="Model path to resume")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--save", type=str, default="models", help="Save directory")
    parser.add_argument("--interval", type=int, default=1, help="Checkpoint interval")

    args = parser.parse_args()

    train(
        data_dir=args.data,
        model_path=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        device=args.device,
        save_dir=args.save,
        checkpoint_interval=args.interval,
    )
