from __future__ import annotations

import torch

from ..baselines.common import SurrogateBaseline


def mc_qei(
    model: SurrogateBaseline,
    X: torch.Tensor,
    best_f: torch.Tensor,
    num_samples: int = 64,
    seed: int = 12345,
) -> torch.Tensor:
    """
    Monte-Carlo qExpected Improvement.

    The same random seed is reset inside a forked RNG context at every
    acquisition evaluation. This keeps the objective deterministic during
    gradient-based optimization while avoiding changes to the caller's RNG.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        samples = model.posterior_samples(X, num_samples=num_samples)

    batch_max = samples.max(dim=-1).values
    improvement = (batch_max - best_f).clamp_min(0.0)
    return improvement.mean()


def _candidate_init(bounds: torch.Tensor, num_restarts: int, q: int) -> torch.Tensor:
    low, high = bounds[0], bounds[1]
    return low + torch.rand(
        num_restarts, q, bounds.shape[1], device=bounds.device, dtype=bounds.dtype
    ) * (high - low)


def optimize_qei(
    model: SurrogateBaseline,
    bounds: torch.Tensor,
    best_f: torch.Tensor,
    q: int = 1,
    num_restarts: int = 10,
    steps: int = 50,
    lr: float = 0.05,
    mc_samples: int = 64,
    seed: int = 12345,
) -> torch.Tensor:
    """Optimize MC-qEI with gradient-based local refinement."""
    params = list(getattr(model, "parameters", lambda: [])())
    device = params[0].device if params else bounds.device
    bounds = bounds.to(device=device)
    best_f = best_f.to(dtype=bounds.dtype, device=device).reshape(())

    starts = _candidate_init(bounds, num_restarts=num_restarts, q=q)
    best_value = torch.tensor(float("-inf"), dtype=bounds.dtype, device=device)
    best_x = starts[0].clone()

    for r in range(num_restarts):
        x = starts[r].clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=lr)

        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            x_eval = torch.max(torch.min(x, bounds[1]), bounds[0])
            acq = mc_qei(
                model, x_eval, best_f,
                num_samples=mc_samples, seed=seed
            )
            (-acq).backward()
            optimizer.step()

        x_final = torch.max(torch.min(x.detach(), bounds[1]), bounds[0])
        value = mc_qei(
            model, x_final, best_f,
            num_samples=mc_samples, seed=seed
        ).detach()
        if value > best_value:
            best_value = value
            best_x = x_final

    return best_x
