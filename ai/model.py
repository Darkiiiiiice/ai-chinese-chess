"""AlphaZero Neural Network - Policy and Value Networks"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block for deep networks"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class PolicyHead(nn.Module):
    """Policy network head - outputs move probabilities"""

    def __init__(self, in_channels: int, num_moves: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 32, 1, padding=0)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(32 * 10 * 9, num_moves)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ValueHead(nn.Module):
    """Value network head - outputs game outcome"""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 32, 1, padding=0)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(32 * 10 * 9, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))


class AlphaZeroNet(nn.Module):
    """AlphaZero Network with separate policy and value heads"""

    def __init__(
        self,
        num_moves: int = 8010,  # 9*10*9*10 - 90 = 8010 possible moves
        num_channels: int = 128,
        num_res_blocks: int = 10,
    ):
        super().__init__()

        # Initial convolution
        self.input_conv = ConvBlock(15, num_channels)  # 15 input channels

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_head = PolicyHead(num_channels, num_moves)

        # Value head
        self.value_head = ValueHead(num_channels)

        # Move encoding
        self.num_moves = num_moves

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch, 15, 10, 9)

        Returns:
            policy: Move probabilities (batch, num_moves)
            value: Game value estimate (batch, 1)
        """
        # Input convolution
        x = self.input_conv(x)

        # Residual blocks
        x = self.res_blocks(x)

        # Heads
        policy = F.log_softmax(self.policy_head(x), dim=1)
        value = self.value_head(x)

        return policy, value

    def predict(self, board: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predict policy and value from board state

        Args:
            board: Board state as numpy array (15, 10, 9)

        Returns:
            policy: Move probabilities as numpy array
            value: Game value as float
        """
        self.eval()

        with torch.no_grad():
            x = torch.from_numpy(board).unsqueeze(0).float()
            policy_log, value = self.forward(x)

            policy = np.exp(policy_log.cpu().numpy()[0])
            value = float(value.cpu().numpy()[0])

        return policy, value


class PolicyNet(nn.Module):
    """Separate Policy Network for AlphaZero"""

    def __init__(
        self, num_moves: int = 8010, num_channels: int = 128, num_res_blocks: int = 10
    ):
        super().__init__()

        self.input_conv = ConvBlock(15, num_channels)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        self.policy_head = PolicyHead(num_channels, num_moves)
        self.num_moves = num_moves

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        x = self.res_blocks(x)
        return F.log_softmax(self.policy_head(x), dim=1)

    def predict(self, board: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(board).unsqueeze(0).float()
            policy = np.exp(self.forward(x).cpu().numpy()[0])
        return policy


class ValueNet(nn.Module):
    """Separate Value Network for AlphaZero"""

    def __init__(self, num_channels: int = 128, num_res_blocks: int = 10):
        super().__init__()

        self.input_conv = ConvBlock(15, num_channels)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        self.value_head = ValueHead(num_channels)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        x = self.res_blocks(x)
        return self.value_head(x)

    def predict(self, board: np.ndarray) -> float:
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(board).unsqueeze(0).float()
            value = float(self.forward(x).cpu().numpy()[0])
        return value


class AlphaZero:
    """AlphaZero wrapper with separate policy and value networks"""

    def __init__(
        self,
        num_moves: int = 8010,
        num_channels: int = 128,
        num_res_blocks: int = 10,
        device: str = "cpu",
    ):
        self.device = device

        # Create separate networks
        self.policy_net = PolicyNet(num_moves, num_channels, num_res_blocks).to(device)
        self.value_net = ValueNet(num_channels, num_res_blocks).to(device)

        # Training mode
        self.training_mode = True

    def set_training(self, training: bool):
        """Set training or evaluation mode"""
        self.training_mode = training
        if training:
            self.policy_net.train()
            self.value_net.train()
        else:
            self.policy_net.eval()
            self.value_net.eval()

    def predict(self, board: np.ndarray) -> Tuple[np.ndarray, float]:
        """Predict policy and value for a single board"""
        board_tensor = torch.from_numpy(board).unsqueeze(0).float().to(self.device)

        if self.training_mode:
            self.policy_net.train()
            self.value_net.train()
        else:
            self.policy_net.eval()
            self.value_net.eval()

        with torch.no_grad():
            policy = np.exp(self.policy_net(board_tensor).cpu().numpy()[0])
            value = float(self.value_net(board_tensor).cpu().numpy().item())

        return policy, value

    def predict_batch(self, boards: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch prediction for multiple boards

        Args:
            boards: Board states as numpy array (N, 15, 10, 9)

        Returns:
            policies: Move probabilities (N, num_moves)
            values: Game values (N,)
        """
        self.policy_net.eval()
        self.value_net.eval()

        board_tensor = torch.from_numpy(boards).float().to(self.device)

        with torch.no_grad():
            policies = np.exp(self.policy_net(board_tensor).cpu().numpy())
            values = self.value_net(board_tensor).cpu().numpy().flatten()

        return policies, values

    def save(self, path: str):
        """Save model checkpoint"""
        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "value_state_dict": self.value_net.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.value_net.load_state_dict(checkpoint["value_state_dict"])


def create_model(config: dict = None) -> AlphaZero:
    """Create AlphaZero model from config"""
    if config is None:
        config = {}

    num_moves = config.get("num_moves", 8010)
    num_channels = config.get("num_channels", 128)
    num_res_blocks = config.get("num_res_blocks", 10)
    device = config.get("device", "cpu")

    return AlphaZero(num_moves, num_channels, num_res_blocks, device)
