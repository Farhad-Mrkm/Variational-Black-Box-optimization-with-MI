from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .common import SurrogateBaseline, MLP, standardize_targets, unstandardize_targets


class LLABaseline(SurrogateBaseline):
    """
    Linearized Laplace Approximation baseline.

    The TMLR appendix specifies:
      - 3-layer MLP
      - Tanh
      - width 128

    For faithful full-network LLA, this class uses `laplace-torch`, matching
    the implementation family used by Li et al. (2024):
      subset_of_weights='all'
      hessian_structure='kron'

    Install:
      pip install laplace-torch
    """

    def __init__(
        self,
        input_dim: int,
        hidden_width: int = 128,
        fit_steps: int = 1000,
        learning_rate: float = 1e-1,
        weight_decay: float = 1e-3,
        prior_precision_steps: int = 50,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_width = hidden_width
        self.fit_steps = fit_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.prior_precision_steps = prior_precision_steps
        self.dtype = dtype
        self.device = device or torch.device("cpu")

        self.model = MLP(
            input_dim=input_dim,
            hidden_width=hidden_width,
            output_dim=1,
            activation="tanh",
        ).to(self.device, dtype=self.dtype)
        self.la = None
        self.y_mean = None
        self.y_scale = None

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        focal_points: Optional[torch.Tensor] = None,
    ) -> None:
        del focal_points
        try:
            from laplace import Laplace
        except ImportError as exc:
            raise RuntimeError(
                "LLABaseline requires laplace-torch. Install with `pip install laplace-torch`."
            ) from exc

        X = X.to(device=self.device, dtype=self.dtype)
        y = y.to(device=self.device, dtype=self.dtype).reshape(-1)
        y_std, y_mean, y_scale = standardize_targets(y)
        self.y_mean = y_mean.detach()
        self.y_scale = y_scale.detach()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.fit_steps)
        )

        self.model.train()
        for _ in range(self.fit_steps):
            optimizer.zero_grad(set_to_none=True)
            pred = self.model(X)[:, 0]
            loss = 0.5 * (pred - y_std).square().mean()
            loss.backward()
            optimizer.step()
            schedule.step()

        self.model.eval()
        self.la = Laplace(
            self.model,
            likelihood="regression",
            subset_of_weights="all",
            hessian_structure="kron",
            enable_backprop=True,
        )
        self.la.fit(X, y_std)
        try:
            self.la.optimize_prior_precision(
                method="marglik",
                pred_type="glm",
                n_steps=self.prior_precision_steps,
            )
        except TypeError:
            # Compatibility with older laplace-torch versions.
            self.la.optimize_prior_precision(
                method="marglik",
                n_steps=self.prior_precision_steps,
            )

    def posterior_moments(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.la is None:
            raise RuntimeError("LLA model must be fit before prediction.")
        X = X.to(device=self.device, dtype=self.dtype)
        mean_std, var_std = self.la(X, pred_type="glm")
        mean_std = mean_std.reshape(-1)
        var_std = var_std.reshape(-1).clamp_min(1e-10)
        mean = unstandardize_targets(mean_std, self.y_mean, self.y_scale)
        covariance = torch.diag(var_std * self.y_scale.square())
        return mean, covariance

    def posterior_samples(self, X: torch.Tensor, num_samples: int = 128) -> torch.Tensor:
        mean, covariance = self.posterior_moments(X)
        std = covariance.diagonal().clamp_min(1e-10).sqrt()
        z = torch.randn(num_samples, len(X), dtype=X.dtype, device=X.device)
        return mean.unsqueeze(0) + z * std.unsqueeze(0)
