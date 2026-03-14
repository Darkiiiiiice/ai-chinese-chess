"""Training script for AlphaZero model"""

import os
import sys
import time
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.model import AlphaZero, create_model
from utils.log import get_short_worker_id, wprint


class AlphaZeroDataset(Dataset):
    """Dataset for AlphaZero training"""

    def __init__(self, data_dir: str = "data"):
        self.samples = self._load_data(data_dir)

    def _load_data(self, data_dir: str) -> List[Dict]:
        """Load all training data"""
        samples = []

        # Find all supported training data files.
        files = []
        for pattern in ("selfplay_*.pt", "online_*.pt", "merged_*.pt"):
            files.extend(glob.glob(os.path.join(data_dir, pattern)))
        files = sorted(set(files))

        if not files:
            wprint(f"在 {data_dir} 中未找到数据文件")
            return samples

        wprint(f"[数据加载] 找到 {len(files)} 个数据文件:")
        wprint("-" * 60)

        total_samples = 0
        for f in files:
            try:
                data = torch.load(f)
                file_samples = len(data["boards"])
                file_game_ids = data.get("game_ids")
                if file_game_ids is None:
                    file_game_ids = [None] * file_samples
                total_samples += file_samples

                # 获取文件大小
                file_size = os.path.getsize(f) / 1024  # KB

                # 判断文件类型
                filename = os.path.basename(f)
                file_type = "自我对弈" if filename.startswith("selfplay") else "在线对弈"

                wprint(f"  [{file_type}] {filename}")
                wprint(f"      样本数: {file_samples}, 文件大小: {file_size:.1f} KB")

                for i in range(len(data["boards"])):
                    samples.append(
                        {
                            "board": data["boards"][i],
                            "policy": data["policies"][i],
                            "value": data["values"][i],
                            "game_id": file_game_ids[i] if i < len(file_game_ids) else None,
                            "source_id": filename,
                        }
                    )
            except Exception as e:
                wprint(f"  [错误] 加载 {os.path.basename(f)} 出错: {e}")

        wprint("-" * 60)
        wprint(f"[数据加载] 总计: {total_samples} 个样本")
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
        lr_scheduler_patience: int = 3,
        lr_scheduler_factor: float = 0.5,
    ):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate

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

        # Learning rate schedulers
        self.policy_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.policy_optimizer, mode="min", factor=lr_scheduler_factor,
            patience=lr_scheduler_patience
        )
        self.value_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.value_optimizer, mode="min", factor=lr_scheduler_factor,
            patience=lr_scheduler_patience
        )

        # Loss functions
        self.policy_loss_fn = nn.KLDivLoss(reduction="batchmean")
        self.value_loss_fn = nn.SmoothL1Loss()  # Huber Loss，对异常值更鲁棒

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
                wprint(
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

    def eval_epoch(self, dataloader: DataLoader) -> Dict:
        """Evaluate on validation set"""
        self.model.set_training(False)

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                boards = batch["board"].to(self.device)
                policies = batch["policy"].to(self.device).float()
                values = batch["value"].to(self.device).unsqueeze(1)

                # Normalize policies
                policy_sums = policies.sum(dim=1, keepdim=True).clamp_min(1e-8)
                policies = policies / policy_sums

                # Forward pass
                policy_log = self.model.policy_net(boards)
                value = self.model.value_net(boards)

                # Losses
                policy_loss = self.policy_loss_fn(policy_log, policies)
                value_loss = self.value_loss_fn(value, values)
                loss = policy_loss + value_loss

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
        }

    def update_schedulers(self, policy_loss: float, value_loss: float):
        """Update learning rate schedulers based on losses"""
        self.policy_scheduler.step(policy_loss)
        self.value_scheduler.step(value_loss)

    def get_current_lr(self) -> Tuple[float, float]:
        """Get current learning rates"""
        policy_lr = self.policy_optimizer.param_groups[0]["lr"]
        value_lr = self.value_optimizer.param_groups[0]["lr"]
        return policy_lr, value_lr

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
            "policy_scheduler": self.policy_scheduler.state_dict(),
            "value_scheduler": self.value_scheduler.state_dict(),
        }

        torch.save(checkpoint, path)
        wprint(f"检查点已保存到 {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.model.value_net.load_state_dict(checkpoint["value_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])

        # Load scheduler states if available
        if "policy_scheduler" in checkpoint:
            self.policy_scheduler.load_state_dict(checkpoint["policy_scheduler"])
        if "value_scheduler" in checkpoint:
            self.value_scheduler.load_state_dict(checkpoint["value_scheduler"])

        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]

        wprint(f"从 {path} 加载检查点")


def print_value_statistics(dataset: AlphaZeroDataset):
    """Print value label distribution statistics"""
    values = torch.tensor([s["value"] for s in dataset.samples], dtype=torch.float32)

    wprint("\n[价值标签分布统计]")
    wprint(f"  范围: [{values.min():.4f}, {values.max():.4f}]")
    wprint(f"  均值: {values.mean():.4f}")
    wprint(f"  标准差: {values.std():.4f}")
    wprint(f"  中位数: {values.median():.4f}")

    # 分布统计
    in_range = ((values >= -1) & (values <= 1)).sum().item()
    wprint(f"  在 [-1, 1] 范围内: {in_range}/{len(values)} ({100*in_range/len(values):.1f}%)")

    # 分位数
    percentiles = [10, 25, 50, 75, 90]
    parts = []
    for p in percentiles:
        val = torch.quantile(values, p / 100).item()
        parts.append(f"P{p}={val:.3f}")
    wprint(f"  分位数: {'  '.join(parts)}")


def split_indices_by_source(
    samples: List[Dict], val_split: float, seed: int = 42
) -> Tuple[List[int], List[int], bool]:
    """Split samples by game_id first, then source_id, to reduce train/val leakage."""
    if val_split <= 0 or len(samples) < 2:
        return list(range(len(samples))), [], False

    target_val_size = int(len(samples) * val_split)
    if target_val_size <= 0:
        return list(range(len(samples))), [], False

    grouped_indices: Dict[str, List[int]] = {}
    for idx, sample in enumerate(samples):
        group_id = sample.get("game_id") or sample.get("source_id")
        if group_id is None:
            return list(range(len(samples))), [], False
        grouped_indices.setdefault(group_id, []).append(idx)

    source_ids = list(grouped_indices)
    if len(source_ids) < 2:
        return list(range(len(samples))), [], False

    rng = random.Random(seed)
    rng.shuffle(source_ids)

    val_indices: List[int] = []
    for source_id in source_ids:
        if len(val_indices) >= target_val_size:
            break
        val_indices.extend(grouped_indices[source_id])

    if not val_indices or len(val_indices) >= len(samples):
        return list(range(len(samples))), [], False

    val_index_set = set(val_indices)
    train_indices = [idx for idx in range(len(samples)) if idx not in val_index_set]
    if not train_indices:
        return list(range(len(samples))), [], False

    return train_indices, val_indices, True


class EarlyStopper:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = None
        self.bad_epochs = 0

    def step(self, metric: float) -> bool:
        if self.patience is None or self.patience <= 0:
            return False

        if self.best_metric is None or metric < self.best_metric - self.min_delta:
            self.best_metric = metric
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def train(
    data_dir: str = "data",
    model_path: str = None,
    num_epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    device: str = "cpu",
    save_dir: str = "models",
    checkpoint_interval: int = 1,
    val_split: float = 0.1,
    lr_patience: int = 3,
    lr_factor: float = 0.5,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 1e-4,
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
        val_split: Validation set ratio (default 0.1)
        lr_patience: Learning rate scheduler patience
        lr_factor: Learning rate reduction factor
        early_stopping_patience: Epoch patience for early stopping
        early_stopping_min_delta: Minimum validation loss improvement to reset patience
    """
    wprint("=" * 50)
    wprint("AlphaZero 训练")
    wprint("=" * 50)

    os.makedirs(save_dir, exist_ok=True)

    # Create or load model
    if model_path and os.path.exists(model_path):
        wprint(f"从 {model_path} 加载模型")
        model = create_model({"device": device})
        model.load(model_path)
    else:
        wprint("创建新模型")
        model = create_model({"device": device})

    # Create trainer
    trainer = Trainer(
        model=model,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device,
        lr_scheduler_patience=lr_patience,
        lr_scheduler_factor=lr_factor,
    )

    # Load data
    dataset = AlphaZeroDataset(data_dir)

    if len(dataset) == 0:
        wprint("未找到训练数据!")
        wprint("请先运行自我对弈生成数据:")
        wprint("  python scripts/self_play.py")
        return

    # Print value distribution statistics
    print_value_statistics(dataset)

    # Split into train and validation sets
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_indices = list(range(len(dataset)))
    val_indices: List[int] = []
    grouped_split_used = False

    if val_size > 0:
        train_indices, val_indices, grouped_split_used = split_indices_by_source(
            dataset.samples, val_split, seed=42
        )

        if grouped_split_used:
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
            train_size = len(train_indices)
            val_size = len(val_indices)
            game_groups = {
                sample["game_id"] for sample in dataset.samples if sample.get("game_id")
            }
            split_label = "按对局分组" if len(game_groups) >= 2 else "按来源分组"
            wprint(
                f"\n数据集分割: 训练集 {train_size} 样本, 验证集 {val_size} 样本 "
                f"({split_label})"
            )
        else:
            wprint("\n[数据集分割] 无法按来源分组切分，回退到随机样本切分")
            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            train_indices = train_dataset.indices
            val_indices = val_dataset.indices

        # 诊断：打印训练集和验证集的价值分布
        wprint("\n[数据集分割诊断]")

        train_values = torch.tensor(
            [dataset.samples[i]["value"] for i in train_indices], dtype=torch.float32
        )
        val_values = torch.tensor(
            [dataset.samples[i]["value"] for i in val_indices], dtype=torch.float32
        )

        wprint(
            f"  训练集价值: 范围[{train_values.min():.3f}, {train_values.max():.3f}], "
            f"均值={train_values.mean():.3f}, 唯一值={len(train_values.unique())}"
        )
        wprint(
            f"  验证集价值: 范围[{val_values.min():.3f}, {val_values.max():.3f}], "
            f"均值={val_values.mean():.3f}, 唯一值={len(val_values.unique())}"
        )

        # 检查验证集是否有足够的价值变化
        if len(val_values.unique()) < 3:
            wprint("  ⚠️ 警告: 验证集价值标签过于单一，可能导致评估不准确!")
            wprint(
                f"     验证集价值分布: -1={(val_values == -1).sum().item()}, "
                f"0={((val_values > -1) & (val_values < 1)).sum().item()}, "
                f"1={(val_values == 1).sum().item()}"
            )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = None
        wprint(f"\n使用 {len(dataset)} 个样本训练 (无验证集)")

    wprint(f"批次大小: {batch_size}, 训练轮数: {num_epochs}")
    wprint(f"初始学习率: {learning_rate}, LR衰减因子: {lr_factor}, LR耐心: {lr_patience}")

    # Training loop
    best_val_loss = float("inf")
    early_stopper = EarlyStopper(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
    )

    for epoch in range(num_epochs):
        wprint(f"\n=== 第 {epoch + 1}/{num_epochs} 轮 ===")

        # Print current learning rate
        policy_lr, value_lr = trainer.get_current_lr()
        wprint(f"学习率: 策略={policy_lr:.6f}, 价值={value_lr:.6f}")

        start_time = time.time()

        # Train
        train_stats = trainer.train_epoch(train_loader)

        epoch_time = time.time() - start_time

        wprint(f"第 {epoch + 1} 轮完成，耗时 {epoch_time:.1f} 秒")
        wprint(
            f"[训练] 损失: {train_stats['loss']:.4f}, "
            f"策略: {train_stats['policy_loss']:.4f}, "
            f"价值: {train_stats['value_loss']:.4f}"
        )

        # Validate
        if val_loader is not None:
            val_stats = trainer.eval_epoch(val_loader)
            wprint(
                f"[验证] 损失: {val_stats['loss']:.4f}, "
                f"策略: {val_stats['policy_loss']:.4f}, "
                f"价值: {val_stats['value_loss']:.4f}"
            )

            # Update learning rate schedulers based on validation loss
            trainer.update_schedulers(val_stats["policy_loss"], val_stats["value_loss"])

            # Save best model based on validation loss
            if val_stats["loss"] < best_val_loss:
                best_val_loss = val_stats["loss"]
                best_path = os.path.join(save_dir, "model_best.pt")
                trainer.save_checkpoint(best_path)
        else:
            # Update schedulers based on training loss if no validation set
            trainer.update_schedulers(train_stats["policy_loss"], train_stats["value_loss"])

        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            model_path = os.path.join(save_dir, f"model_epoch{epoch + 1}.pt")
            trainer.save_checkpoint(model_path)

        if val_loader is not None and early_stopper.step(val_stats["loss"]):
            wprint(
                f"早停触发: 验证损失连续 {early_stopper.patience} 轮未改善 "
                f"(min_delta={early_stopping_min_delta})"
            )
            break

    # Save final model
    final_path = os.path.join(save_dir, "model_latest.pt")
    trainer.save_checkpoint(final_path)

    wprint("\n训练完成!")
    if val_loader is not None:
        wprint(f"最佳验证损失: {best_val_loss:.4f}")
        wprint(f"最佳模型已保存到 {os.path.join(save_dir, 'model_best.pt')}")
    wprint(f"最新模型已保存到 {final_path}")


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
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--lr-patience", type=int, default=3, help="LR scheduler patience")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="LR reduction factor")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience on validation loss",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation loss improvement for early stopping",
    )

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
        val_split=args.val_split,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )
