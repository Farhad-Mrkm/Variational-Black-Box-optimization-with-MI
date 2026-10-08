from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .common import SurrogateBaseline, matern52, safe_cholesky, standardize_targets, unstandardize_targets


class FocalSparseGPBaseline(SurrogateBaseline, nn.Module):
    """
    Focalized sparse GP baseline.

    TMLR configuration:
      - Matérn-5/2 kernel
      - m=50 inducing variables for synthetic benchmarks
      - m=200 inducing variables for real-world benchmarks

    The focalized variational objective follows the weighting pattern of the
    official FocalBO implementation: observations receive larger likelihood
    weights according to their kernel similarity to a local test distribution.
    The BO loop supplies the current incumbent as the local focal point.

    Reference implementation:
      https://github.com/yunyuewei/FocalBO
    """

    def __init__(
        self,
        input_dim: int,
        num_inducing: int = 50,
        fit_steps: int = 300,
        learning_rate: float = 0.01,
        noise_init: float = 0.05,
        jitter: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        SurrogateBaseline.__init__(self)
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.num_inducing = num_inducing
        self.fit_steps = fit_steps
        self.learning_rate = learning_rate
        self.noise_init = noise_init
        self.jitter = jitter
        self.dtype = dtype
        self.device = device or torch.device("cpu")

        self.raw_log_lengthscale = nn.Parameter(torch.zeros(input_dim, dtype=dtype, device=self.device))
        self.raw_log_outputscale = nn.Parameter(torch.tensor(0.0, dtype=dtype, device=self.device))
        self.raw_log_noise = nn.Parameter(torch.tensor(noise_init, dtype=dtype, device=self.device).log())
        self.mean_offset = nn.Parameter(
            torch.zeros((), dtype=dtype, device=self.device)
        )

        # Lazy initialization because the design size is task dependent.
        self.Z = nn.Parameter(
            torch.empty(num_inducing, input_dim, dtype=dtype, device=self.device)
        )
        self.variational_mean = nn.Parameter(
            torch.zeros(num_inducing, dtype=dtype, device=self.device)
        )
        self.raw_variational_tril = nn.Parameter(
            torch.eye(num_inducing, dtype=dtype, device=self.device)
        )
        self._initialized_inducing = False

        self.register_buffer("_X", torch.empty(0, input_dim, dtype=dtype, device=self.device))
        self.register_buffer("_y", torch.empty(0, dtype=dtype, device=self.device))

    @property
    def lengthscale(self) -> torch.Tensor:
        return self.raw_log_lengthscale.exp().clamp_min(1e-4)

    @property
    def outputscale(self) -> torch.Tensor:
        return self.raw_log_outputscale.exp().clamp_min(1e-8)

    @property
    def noise(self) -> torch.Tensor:
        return self.raw_log_noise.exp().clamp_min(1e-6)

    def kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return matern52(x1, x2, self.lengthscale, self.outputscale)

    def _ensure_inducing(self, X: torch.Tensor) -> None:
        if self._initialized_inducing:
            return
        m = self.num_inducing
        if len(X) == 0:
            raise ValueError("At least one observation is required.")
        reps = (m + len(X) - 1) // len(X)
        Z = X.repeat((reps, 1))[:m].detach().clone()
        if len(X) < m:
            Z = Z + 1e-3 * torch.randn_like(Z)
        with torch.no_grad():
            self.Z.copy_(Z)
        self._initialized_inducing = True

    def _variational_covariance(self) -> torch.Tensor:
        raw = torch.tril(self.raw_variational_tril)
        diag = torch.diagonal(raw, 0)
        diag_pos = torch.nn.functional.softplus(diag) + 1e-4
        raw = raw - torch.diag(diag) + torch.diag(diag_pos)
        return raw @ raw.T

    def _focal_weights(self, X: torch.Tensor, focal_points: Optional[torch.Tensor]) -> torch.Tensor:
        if focal_points is None:
            focal_points = X[torch.argmax(self._y)].view(1, -1)
        focal_points = focal_points.to(device=X.device, dtype=X.dtype)
        K_xt = self.kernel(X, focal_points)
        weights = K_xt.max(dim=1).values
        self_cov = self.kernel(X[:1], X[:1]).diag().clamp_min(1e-8)
        weights = weights / self_cov
        return weights.clamp_min(1e-8)

    def _elbo(self, X: torch.Tensor, y: torch.Tensor, focal_points: Optional[torch.Tensor]) -> torch.Tensor:
        Z = self.Z
        Kmm = self.kernel(Z, Z)
        Kmm = Kmm + self.jitter * torch.eye(len(Z), dtype=X.dtype, device=X.device)
        Lmm = safe_cholesky(Kmm, jitter=self.jitter)

        Knm = self.kernel(X, Z)
        A = torch.cholesky_solve(Knm.T, Lmm).T  # N x M, K_nm K_mm^-1
        mean = self.mean_offset + A @ self.variational_mean

        S = self._variational_covariance()
        # Diagonal correction of K_nm K_mm^-1 (S-Kmm) K_mm^-1 K_mn.
        var_correction = (A * (A @ (S - Kmm))).sum(dim=1)
        kxx = self.kernel(X, X).diag()
        latent_var = (kxx + var_correction).clamp_min(1e-8)

        sigma2 = self.noise.square()
        log_prob = -0.5 * (
            torch.log(2.0 * torch.pi * sigma2)
            + ((y - mean).square() + latent_var) / sigma2
        )

        weight = self._focal_weights(X, focal_points)
        weighted_ll = (weight * log_prob).sum() / (weight.sum() + 1e-7)

        Kinv_m = torch.cholesky_solve(self.variational_mean[:, None], Lmm)[:, 0]
        trace_term = torch.trace(torch.cholesky_solve(S, Lmm))
        quad_term = self.variational_mean.dot(Kinv_m)
        logdet_p = 2.0 * Lmm.diagonal().log().sum()
        Ls = safe_cholesky(S, jitter=self.jitter)
        logdet_q = 2.0 * Ls.diagonal().log().sum()
        m = len(Z)
        kl = 0.5 * (
            trace_term
            + quad_term
            - m
            + logdet_p
            - logdet_q
        ) / max(len(X), 1)

        return weighted_ll - kl

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        focal_points: Optional[torch.Tensor] = None,
    ) -> None:
        X = X.to(device=self.device, dtype=self.dtype)
        y = y.to(device=self.device, dtype=self.dtype).reshape(-1)
        y_std, y_mean, y_scale = standardize_targets(y)
        self.set_target_stats(y_mean.detach(), y_scale.detach())
        self._X = X.detach()
        self._y = y_std.detach()
        self._ensure_inducing(self._X)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.train()
        for _ in range(self.fit_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = -self._elbo(self._X, self._y, focal_points)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                self.raw_log_lengthscale.clamp_(-5.0, 5.0)
                self.raw_log_outputscale.clamp_(-5.0, 5.0)
                self.raw_log_noise.clamp_(-8.0, 1.0)

    def posterior_moments(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X = X.to(device=self.device, dtype=self.dtype)
        Z = self.Z
        Kmm = self.kernel(Z, Z)
        Kmm = Kmm + self.jitter * torch.eye(len(Z), dtype=self.dtype, device=self.device)
        Lmm = safe_cholesky(Kmm, jitter=self.jitter)
        Kxm = self.kernel(X, Z)
        A = torch.cholesky_solve(Kxm.T, Lmm).T

        mean_std = self.mean_offset + A @ self.variational_mean
        S = self._variational_covariance()
        correction = A @ (S - Kmm) @ A.T
        Kxx = self.kernel(X, X)
        covariance_std = Kxx + correction
        covariance_std = 0.5 * (covariance_std + covariance_std.T)
        covariance_std = covariance_std + self.jitter * torch.eye(
            len(X), dtype=self.dtype, device=self.device
        )

        mean = unstandardize_targets(mean_std, self.y_mean, self.y_scale)
        covariance = covariance_std * self.y_scale.square()
        return mean, covariance
