from __future__ import annotations

from typing import Optional, Tuple
import math

import torch

from .common import SurrogateBaseline, MLP, functional_mlp, module_to_flat, standardize_targets, unstandardize_targets


class SGHMCBaseline(SurrogateBaseline):
    """
    Stochastic Gradient Hamiltonian Monte Carlo BNN baseline.

    TMLR configuration:
      - 3-layer MLP
      - Tanh
      - width 128
      - adaptive step size
      - heteroskedastic likelihood produced by the network

    The sampler is implemented directly rather than replacing SGHMC by SVI.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_width: int = 128,
        samples: int = 50,
        burnin_steps: int = 1000,
        total_steps: int = 2000,
        batch_size: int = 32,
        step_size: float = 1e-3,
        friction: float = 0.05,
        prior_std: float = 1.0,
        map_steps: int = 300,
        map_learning_rate: float = 1e-2,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        seed: int = 0,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_width = hidden_width
        self.samples = samples
        self.burnin_steps = burnin_steps
        self.total_steps = max(total_steps, burnin_steps + samples)
        self.batch_size = batch_size
        self.step_size = step_size
        self.friction = friction
        self.prior_std = prior_std
        self.map_steps = map_steps
        self.map_learning_rate = map_learning_rate
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        self.seed = seed

        self.model_output_dim = 2
        self.map_model = MLP(
            input_dim=input_dim,
            hidden_width=hidden_width,
            output_dim=2,
            activation="tanh",
        ).to(self.device, dtype=self.dtype)
        self.posterior_theta: Optional[torch.Tensor] = None
        self.y_mean: Optional[torch.Tensor] = None
        self.y_scale: Optional[torch.Tensor] = None

    def _log_posterior_batch(
        self,
        theta: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
        full_n: int,
    ) -> torch.Tensor:
        out = functional_mlp(
            theta, X, self.input_dim, self.hidden_width, output_dim=2, activation="tanh"
        )
        mu = out[:, 0]
        log_sigma = out[:, 1].clamp(-8.0, 4.0)
        sigma = torch.nn.functional.softplus(log_sigma) + 1e-3

        log_lik = -0.5 * (
            ((y - mu) / sigma).square()
            + 2.0 * sigma.log()
            + math.log(2.0 * math.pi)
        ).sum()
        scale = full_n / max(len(X), 1)
        log_prior = -0.5 * (theta / self.prior_std).square().sum()
        return scale * log_lik + log_prior

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

        # MAP warm-start.
        map_opt = torch.optim.Adam(self.map_model.parameters(), lr=self.map_learning_rate)
        for _ in range(self.map_steps):
            map_opt.zero_grad(set_to_none=True)
            out = self.map_model(X)
            mu = out[:, 0]
            sigma = torch.nn.functional.softplus(out[:, 1]) + 1e-3
            nll = 0.5 * (((y_std - mu) / sigma).square() + 2 * sigma.log()).mean()
            prior = 0.5 * sum((p / self.prior_std).square().mean() for p in self.map_model.parameters())
            (nll + prior).backward()
            map_opt.step()

        theta = module_to_flat(self.map_model).to(self.device, self.dtype)
        momentum = torch.zeros_like(theta)
        grad_sq_ema = torch.ones_like(theta)
        samples = []

        data_n = len(X)
        for step in range(self.total_steps):
            idx = torch.randint(
                low=0,
                high=data_n,
                size=(min(self.batch_size, data_n),),
                device=self.device,
            )
            Xb = X[idx]
            yb = y_std[idx]

            theta_req = theta.detach().clone().requires_grad_(True)
            logp = self._log_posterior_batch(theta_req, Xb, yb, full_n=data_n)
            grad = torch.autograd.grad(logp, theta_req)[0].detach()

            # Adaptive diagonal preconditioning from the stochastic gradient.
            grad_sq_ema.mul_(0.99).addcmul_(grad, grad, value=0.01)
            precond = (grad_sq_ema.sqrt() + 1e-6).reciprocal()
            eps_t = self.step_size * precond

            noise = torch.randn_like(theta)
            momentum = (
                (1.0 - self.friction) * momentum
                + eps_t * grad
                + torch.sqrt(
                    torch.clamp(2.0 * self.friction * eps_t, min=1e-12)
                ) * noise
            )
            theta = theta + momentum

            if step >= self.burnin_steps:
                samples.append(theta.detach().clone())
                if len(samples) >= self.samples:
                    break

        self.posterior_theta = torch.stack(samples, dim=0)

    def posterior_samples(self, X: torch.Tensor, num_samples: int = 128) -> torch.Tensor:
        if self.posterior_theta is None:
            raise RuntimeError("SGHMC model must be fit before prediction.")
        X = X.to(device=self.device, dtype=self.dtype)
        count = min(num_samples, self.posterior_theta.shape[0])
        preds = []
        for theta in self.posterior_theta[:count]:
            out = functional_mlp(
                theta, X, self.input_dim, self.hidden_width, output_dim=2, activation="tanh"
            )
            preds.append(out[:, 0])
        samples_std = torch.stack(preds, dim=0)
        return unstandardize_targets(samples_std, self.y_mean, self.y_scale)

    def posterior_moments(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        samples = self.posterior_samples(X, num_samples=min(self.samples, len(self.posterior_theta)))
        mean = samples.mean(dim=0)
        if samples.shape[0] > 1:
            covariance = torch.cov(samples.T)
        else:
            covariance = torch.diag(torch.ones(len(X), dtype=self.dtype, device=self.device) * 1e-6)
        return mean, covariance
