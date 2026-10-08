from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .common import (
    SurrogateBaseline,
    relu_nngp_covariance,
    safe_cholesky,
    standardize_targets,
    unstandardize_targets,
)


class IBNNBaseline(SurrogateBaseline, nn.Module):
    """
    Infinite-width Bayesian Neural Network / NNGP baseline.

    TMLR configuration:
      - NNGP kernel
      - depth 3
      - ReLU
      - weight variance 10.0
      - bias variance 1.6

    This is not a finite-width BNN with dropout. It is an exact GP using the
    corresponding NNGP covariance.
    """

    def __init__(
        self,
        input_dim: int,
        depth: int = 3,
        weight_variance: float = 10.0,
        bias_variance: float = 1.6,
        fit_steps: int = 300,
        learning_rate: float = 0.05,
        noise_init: float = 0.05,
        jitter: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        SurrogateBaseline.__init__(self)
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.depth = depth
        self.weight_variance = weight_variance
        self.bias_variance = bias_variance
        self.fit_steps = fit_steps
        self.learning_rate = learning_rate
        self.jitter = jitter
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        self.raw_log_noise = nn.Parameter(
            torch.tensor(noise_init, dtype=dtype, device=self.device).log()
        )
        self.mean_offset = nn.Parameter(
            torch.zeros((), dtype=dtype, device=self.device)
        )
        self.register_buffer("_X", torch.empty(0, input_dim, dtype=dtype, device=self.device))
        self.register_buffer("_y", torch.empty(0, dtype=dtype, device=self.device))

    @property
    def noise(self) -> torch.Tensor:
        return self.raw_log_noise.exp().clamp_min(1e-6)

    def kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return relu_nngp_covariance(
            x1,
            x2,
            depth=self.depth,
            weight_variance=self.weight_variance,
            bias_variance=self.bias_variance,
        )

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
            K = self.kernel(self._X, self._X)
            K = K + (self.noise.square() + self.jitter) * torch.eye(
                len(self._X), dtype=self.dtype, device=self.device
            )
            L = safe_cholesky(K, jitter=self.jitter)
            alpha = torch.cholesky_solve(
                (self._y - self.mean_offset)[:, None], L
            )[:, 0]
            centered_y = self._y - self.mean_offset
            loss = (
                0.5 * centered_y.dot(alpha)
                + L.diagonal().log().sum()
                + 0.5 * len(self._X) * torch.log(
                    torch.tensor(2.0 * torch.pi, dtype=self.dtype, device=self.device)
                )
            )
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                self.raw_log_noise.clamp_(-8.0, 1.0)

    def posterior_moments(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X = X.to(device=self.device, dtype=self.dtype)
        self.eval()

        K = self.kernel(self._X, self._X)
        K = K + (self.noise.square() + self.jitter) * torch.eye(
            len(self._X), dtype=self.dtype, device=self.device
        )
        L = safe_cholesky(K, jitter=self.jitter)
        Kxs = self.kernel(self._X, X)
        alpha = torch.cholesky_solve(
            (self._y - self.mean_offset)[:, None], L
        )[:, 0]
        mean_std = self.mean_offset + Kxs.T @ alpha
        v = torch.linalg.solve_triangular(L, Kxs, upper=False)
        Kss = self.kernel(X, X)
        covariance_std = Kss - v.T @ v
        covariance_std = 0.5 * (covariance_std + covariance_std.T)
        covariance_std = covariance_std + self.jitter * torch.eye(
            len(X), dtype=self.dtype, device=self.device
        )

        mean = unstandardize_targets(mean_std, self.y_mean, self.y_scale)
        covariance = covariance_std * self.y_scale.square()
        return mean, covariance
