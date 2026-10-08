from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .common import standardize_targets, unstandardize_targets
from .gp import ExactGPBaseline


class DKLFeatureExtractor(nn.Module):
    """3-layer Tanh feature extractor, width 128 in the TMLR baseline."""

    def __init__(
        self,
        input_dim: int,
        hidden_width: int = 128,
        feature_dim: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DKLBaseline(ExactGPBaseline):
    """
    Deep Kernel Learning baseline.

    TMLR configuration:
      - 3-layer MLP feature extractor
      - Tanh activations
      - width 128
      - Matérn-5/2 GP kernel
    """

    def __init__(
        self,
        input_dim: int,
        hidden_width: int = 128,
        feature_dim: int = 128,
        fit_steps: int = 300,
        learning_rate: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            fit_steps=fit_steps,
            learning_rate=learning_rate,
            **kwargs,
        )
        self.feature_extractor = DKLFeatureExtractor(
            input_dim=input_dim,
            hidden_width=hidden_width,
            feature_dim=feature_dim,
        ).to(device=self.device, dtype=self.dtype)
        # The GP kernel operates in feature space, so the ARD lengthscale has
        # one parameter per learned feature rather than per raw input dimension.
        self.raw_log_lengthscale = nn.Parameter(
            torch.zeros(feature_dim, dtype=self.dtype, device=self.device)
        )

    def kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        z1 = self.feature_extractor(x1)
        z2 = self.feature_extractor(x2)
        return super().kernel(z1, z2)

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        focal_points: Optional[torch.Tensor] = None,
    ) -> None:
        del focal_points
        X = X.to(device=self.device, dtype=self.dtype)
        y = y.to(device=self.device, dtype=self.dtype).reshape(-1)
        y_std, y_mean, y_scale = standardize_targets(y)
        self.set_target_stats(y_mean.detach(), y_scale.detach())
        self._X = X.detach()
        self._y = y_std.detach()

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.train()
        for _ in range(self.fit_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = self._nll(self._X, self._y)
            loss.backward()
            optimizer.step()


DeepKernelLearningBaseline = DKLBaseline
