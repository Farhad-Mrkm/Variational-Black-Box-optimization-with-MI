from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Optional, Tuple

import torch
from torch import nn


Tensor = torch.Tensor


def standardize_targets(y: Tensor, eps: float = 1e-8) -> Tuple[Tensor, Tensor, Tensor]:
    y = y.reshape(-1)
    mean = y.mean()
    scale = y.std(unbiased=False).clamp_min(eps)
    return (y - mean) / scale, mean, scale


def unstandardize_targets(y_std: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
    return y_std * scale + mean


def safe_cholesky(matrix: Tensor, jitter: float = 1e-6, max_tries: int = 8) -> Tensor:
    eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    current = jitter
    for _ in range(max_tries):
        chol, info = torch.linalg.cholesky_ex(matrix + current * eye)
        if torch.all(info == 0):
            return chol
        current *= 10.0
    raise RuntimeError("Cholesky factorization failed after increasing jitter.")


def matern52(x1: Tensor, x2: Tensor, lengthscale: Tensor, outputscale: Tensor) -> Tensor:
    ls = torch.as_tensor(lengthscale, dtype=x1.dtype, device=x1.device).reshape(1, 1, -1)
    diff = (x1[:, None, :] - x2[None, :, :]) / ls
    r2 = diff.square().sum(dim=-1).clamp_min(0.0)
    # Smooth the square-root at r=0 to avoid infinite derivatives on the diagonal.
    r = torch.sqrt(r2 + 1e-12)
    sr5 = math.sqrt(5.0) * r
    return outputscale * (1.0 + sr5 + (5.0 / 3.0) * r2) * torch.exp(-sr5)


def _relu_covariance_map(k: Tensor, q1: Tensor, q2: Tensor, weight_variance: float, bias_variance: float):
    denom = (q1[:, None].clamp_min(1e-12) * q2[None, :].clamp_min(1e-12)).sqrt()
    cos_theta = (k / denom).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta)
    relu_cov = denom * (
        torch.sin(theta) + (math.pi - theta) * torch.cos(theta)
    ) / (2.0 * math.pi)
    return weight_variance * relu_cov + bias_variance


def relu_nngp_covariance(
    x1: Tensor,
    x2: Tensor,
    depth: int = 3,
    weight_variance: float = 10.0,
    bias_variance: float = 1.6,
) -> Tensor:
    """NNGP covariance of a depth-`depth` fully connected ReLU network."""
    d = x1.shape[-1]
    k11 = weight_variance * (x1 @ x1.T) / max(d, 1) + bias_variance
    k22 = weight_variance * (x2 @ x2.T) / max(d, 1) + bias_variance
    k12 = weight_variance * (x1 @ x2.T) / max(d, 1) + bias_variance

    for _ in range(depth):
        k12 = _relu_covariance_map(k12, torch.diagonal(k11), torch.diagonal(k22), weight_variance, bias_variance)
        k11 = _relu_covariance_map(k11, torch.diagonal(k11), torch.diagonal(k11), weight_variance, bias_variance)
        k22 = _relu_covariance_map(k22, torch.diagonal(k22), torch.diagonal(k22), weight_variance, bias_variance)
    return k12


class MLP(nn.Module):
    """Three-layer MLP: input -> hidden -> hidden -> output."""
    def __init__(
        self,
        input_dim: int,
        hidden_width: int = 128,
        output_dim: int = 1,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_width)
        self.hidden = nn.Linear(hidden_width, hidden_width)
        self.output = nn.Linear(hidden_width, output_dim)
        activation = activation.lower()
        if activation not in {"tanh", "relu"}:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        act = torch.tanh if self.activation == "tanh" else torch.relu
        z = act(self.input(x))
        z = act(self.hidden(z))
        return self.output(z)


def module_to_flat(module: nn.Module) -> Tensor:
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()])


def functional_mlp(
    theta: Tensor,
    x: Tensor,
    input_dim: int,
    hidden_width: int,
    output_dim: int = 1,
    activation: str = "tanh",
) -> Tensor:
    n_w1 = hidden_width * input_dim
    n_b1 = hidden_width
    n_w2 = hidden_width * hidden_width
    n_b2 = hidden_width
    n_w3 = output_dim * hidden_width
    n_b3 = output_dim
    i = 0
    w1 = theta[i:i+n_w1].view(hidden_width, input_dim); i += n_w1
    b1 = theta[i:i+n_b1].view(hidden_width); i += n_b1
    w2 = theta[i:i+n_w2].view(hidden_width, hidden_width); i += n_w2
    b2 = theta[i:i+n_b2].view(hidden_width); i += n_b2
    w3 = theta[i:i+n_w3].view(output_dim, hidden_width); i += n_w3
    b3 = theta[i:i+n_b3].view(output_dim)
    act = torch.tanh if activation.lower() == "tanh" else torch.relu
    h = act(x @ w1.T + b1)
    h = act(h @ w2.T + b2)
    return h @ w3.T + b3


class SurrogateBaseline(ABC):
    """Interface expected by the common BO/qEI runner."""
    def __init__(self) -> None:
        self.y_mean: Optional[Tensor] = None
        self.y_scale: Optional[Tensor] = None

    @abstractmethod
    def fit(self, X: Tensor, y: Tensor, focal_points: Optional[Tensor] = None) -> None:
        ...

    @abstractmethod
    def posterior_moments(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        ...

    def posterior_samples(self, X: Tensor, num_samples: int = 128) -> Tensor:
        mean, covariance = self.posterior_moments(X)
        chol = safe_cholesky(covariance, jitter=1e-6)
        z = torch.randn(
            num_samples, X.shape[0], dtype=mean.dtype, device=mean.device
        )
        return mean.unsqueeze(0) + z @ chol.T

    def set_target_stats(self, mean: Tensor, scale: Tensor) -> None:
        self.y_mean = mean
        self.y_scale = scale
