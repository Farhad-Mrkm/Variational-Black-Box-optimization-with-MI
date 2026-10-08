from __future__ import annotations

import torch
from torch import nn


class ActionNet(nn.Module):
    """Three-affine-layer action network with a bounded tanh output."""

    def __init__(self, seed_dim: int, action_dim: int, lower: torch.Tensor, upper: torch.Tensor, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.register_buffer("lower", lower.detach().clone())
        self.register_buffer("upper", upper.detach().clone())

    def forward(self, seeds: torch.Tensor) -> torch.Tensor:
        z = self.net(seeds)
        return self.lower + 0.5 * (z + 1.0) * (self.upper - self.lower)


class HelperCritic(nn.Module):
    """Pairwise LSTM + MLP critic for the DV mutual-information bound."""

    def __init__(self, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.lstm_x = nn.LSTM(action_dim, hidden_dim, num_layers=1, batch_first=True)
        self.lstm_y = nn.LSTM(1, hidden_dim, num_layers=1, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_x(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm_x(x.unsqueeze(1))
        return h[-1]

    def encode_y(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        _, (h, _) = self.lstm_y(y.unsqueeze(1))
        return h[-1]

    def score_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        hx = self.encode_x(x)
        hy = self.encode_y(y)
        b = x.shape[0]
        hx_pair = hx.unsqueeze(1).expand(b, b, -1)
        hy_pair = hy.unsqueeze(0).expand(b, b, -1)
        return self.mlp(torch.cat([hx_pair, hy_pair], dim=-1)).squeeze(-1)

    forward = score_matrix


class SurrogateNet(nn.Module):
    """Small MLP used for the exploitation term and oracle-free inner loops."""

    def __init__(self, action_dim: int, hidden_dim: int = 128, hidden_layers: int = 2):
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")
        layers: list[nn.Module] = [nn.Linear(action_dim, hidden_dim), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
