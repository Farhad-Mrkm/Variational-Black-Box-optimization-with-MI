from __future__ import annotations

from typing import Optional, Tuple

import math
import torch
from torch import nn

from .common import (
    SurrogateBaseline,
    MLP,
    functional_mlp,
    module_to_flat,
    standardize_targets,
    unstandardize_targets,
)


class HMCBaseline(SurrogateBaseline):
    """
    Full-batch Hamiltonian Monte Carlo BNN baseline.

    TMLR configuration:
      - 3-layer MLP
      - Tanh
      - width 128
      - HMC posterior sampling
      - FLOP table: 50 posterior samples and 20 leapfrog steps

    The remaining sampler hyperparameters are intentionally configurable because
    the TMLR appendix delegates them to Li et al. (2024).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_width: int = 128,
        samples: int = 50,
        warmup_steps: int = 100,
        leapfrog_steps: int = 20,
        step_size: float = 5e-4,
        target_acceptance: float = 0.65,
        prior_std: float = 1.0,
        noise_std: float = 0.1,
        map_steps: int = 500,
        map_learning_rate: float = 1e-2,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        seed: int = 0,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_width = hidden_width
        self.samples = samples
        self.warmup_steps = warmup_steps
        self.leapfrog_steps = leapfrog_steps
        self.step_size = step_size
        self.target_acceptance = target_acceptance
        self.prior_std = prior_std
        self.noise_std = noise_std
        self.map_steps = map_steps
        self.map_learning_rate = map_learning_rate
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        self.seed = seed

        self.theta: Optional[torch.Tensor] = None
        self.posterior_theta: Optional[torch.Tensor] = None
        self.y_mean: Optional[torch.Tensor] = None
        self.y_scale: Optional[torch.Tensor] = None

        # Construct a deterministic MAP network to obtain an initial state.
        self.map_model = MLP(
            input_dim=input_dim,
            hidden_width=hidden_width,
            output_dim=1,
            activation="tanh",
        ).to(self.device, dtype=self.dtype)

    def _log_posterior(self, theta: torch.Tensor, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pred = functional_mlp(
            theta, X, self.input_dim, self.hidden_width, output_dim=1, activation="tanh"
        )[:, 0]
        sigma2 = self.noise_std ** 2
        log_lik = -0.5 * (((y - pred) ** 2) / sigma2 + math.log(2.0 * math.pi * sigma2)).sum()
        log_prior = -0.5 * (theta / self.prior_std).square().sum()
        return log_lik + log_prior

    def _potential_and_grad(
        self,
        theta: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        theta_req = theta.detach().clone().requires_grad_(True)
        logp = self._log_posterior(theta_req, X, y)
        grad = torch.autograd.grad(logp, theta_req)[0]
        return -logp.detach(), -grad.detach()

    def _leapfrog(
        self,
        theta: torch.Tensor,
        momentum: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
        step_size: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p = momentum.clone()
        q = theta.clone()
        _, grad = self._potential_and_grad(q, X, y)
        p -= 0.5 * step_size * grad

        for j in range(self.leapfrog_steps):
            q = q + step_size * p
            _, grad = self._potential_and_grad(q, X, y)
            if j != self.leapfrog_steps - 1:
                p -= step_size * grad

        p -= 0.5 * step_size * grad
        return q, p

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        focal_points: Optional[torch.Tensor] = None,
    ) -> None:
        del focal_points
        torch.manual_seed(self.seed)
        X = X.to(device=self.device, dtype=self.dtype)
        y = y.to(device=self.device, dtype=self.dtype).reshape(-1)
        y_std, y_mean, y_scale = standardize_targets(y)
        self.y_mean = y_mean.detach()
        self.y_scale = y_scale.detach()

        # MAP warm start.
        map_opt = torch.optim.Adam(self.map_model.parameters(), lr=self.map_learning_rate)
        for _ in range(self.map_steps):
            map_opt.zero_grad(set_to_none=True)
            pred = self.map_model(X)[:, 0]
            loss = 0.5 * ((pred - y_std) / self.noise_std).square().mean()
            loss = loss + 0.5 * sum((p / self.prior_std).square().mean() for p in self.map_model.parameters())
            loss.backward()
            map_opt.step()

        theta = module_to_flat(self.map_model).to(self.device, self.dtype)

        current_eps = float(self.step_size)
        samples = []
        total_warmup = max(0, self.warmup_steps)
        total_steps = total_warmup + self.samples

        for it in range(total_steps):
            momentum = torch.randn_like(theta)
            current_U, _ = self._potential_and_grad(theta, X, y_std)
            current_K = 0.5 * momentum.square().sum()

            proposed_theta, proposed_momentum = self._leapfrog(
                theta, momentum, X, y_std, current_eps
            )
            proposed_U, _ = self._potential_and_grad(proposed_theta, X, y_std)
            proposed_K = 0.5 * proposed_momentum.square().sum()

            log_accept = (current_U + current_K - proposed_U - proposed_K).clamp(max=0.0)
            accept_prob = torch.exp(log_accept)
            accepted = torch.rand((), device=self.device) < accept_prob
            if accepted:
                theta = proposed_theta.detach()

            if it < total_warmup:
                # Multiplicative adaptation toward the target acceptance.
                direction = float(accept_prob.detach().cpu()) - self.target_acceptance
                current_eps *= math.exp(0.05 * direction)
                current_eps = min(max(current_eps, 1e-6), 5e-2)
            else:
                samples.append(theta.detach().clone())

        self.theta = theta.detach()
        self.posterior_theta = torch.stack(samples, dim=0)

    def posterior_moments(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        samples = self.posterior_samples(X, num_samples=min(self.samples, len(self.posterior_theta)))
        mean = samples.mean(dim=0)
        covariance = torch.cov(samples.T) if samples.shape[0] > 1 else torch.diag(
            torch.ones(len(X), dtype=self.dtype, device=self.device) * 1e-6
        )
        mean = unstandardize_targets(mean, self.y_mean, self.y_scale)
        covariance = covariance * self.y_scale.square()
        return mean, covariance

    def posterior_samples(self, X: torch.Tensor, num_samples: int = 128) -> torch.Tensor:
        if self.posterior_theta is None:
            raise RuntimeError("HMC model must be fit before prediction.")
        X = X.to(device=self.device, dtype=self.dtype)
        count = min(num_samples, self.posterior_theta.shape[0])
        theta_samples = self.posterior_theta[:count]
        preds = []
        for theta in theta_samples:
            pred = functional_mlp(
                theta, X, self.input_dim, self.hidden_width, output_dim=1, activation="tanh"
            )[:, 0]
            preds.append(pred)
        samples_std = torch.stack(preds, dim=0)
        return unstandardize_targets(samples_std, self.y_mean, self.y_scale)
