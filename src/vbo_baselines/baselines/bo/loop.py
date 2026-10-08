from dataclasses import dataclass, asdict
import time
from typing import Callable, Dict, Iterable, Mapping, Optional

import pandas as pd
import torch

from ..baselines.common import SurrogateBaseline
from .acquisition import optimize_qei


Tensor = torch.Tensor
Objective = Callable[[Tensor], Tensor]
ModelFactory = Callable[[int, int], SurrogateBaseline]


@dataclass
class BOConfig:
    n_init: int = 10
    total_evals: int = 500
    q: int = 1
    qei_restarts: int = 10
    qei_steps: int = 50
    qei_mc_samples: int = 64
    seed: int = 0


def sobol_initial_design(dim: int, n: int, bounds: Tensor, seed: int) -> Tensor:
    engine = torch.quasirandom.SobolEngine(
        dimension=dim,
        scramble=True,
        seed=int(seed),
    )
    unit = engine.draw(n)
    return bounds[0] + unit * (bounds[1] - bounds[0])


def _evaluate_objective(objective: Objective, X: Tensor) -> Tensor:
    y = objective(X)
    return y.to(dtype=X.dtype, device=X.device).reshape(-1)


def run_baseline(
    method_name: str,
    model_factory: ModelFactory,
    objective: Objective,
    bounds: Tensor,
    input_dim: int,
    config: BOConfig,
) -> pd.DataFrame:
    """
    Run one baseline independently.

    Crucial experimental invariant:
      - the initial design is identical for every method for a given seed
      - thereafter each method has its own X/y history
      - no method sees another method's evaluations
    """
    torch.manual_seed(config.seed)
    bounds = bounds.to(dtype=torch.float32)
    X = sobol_initial_design(input_dim, config.n_init, bounds, config.seed)
    y = _evaluate_objective(objective, X)

    rows = []
    best_so_far = y.max().item()

    rows.extend(
        {
            "method": method_name,
            "evaluation": i + 1,
            "best_so_far": best_so_far if i == len(y) - 1 else max(best_so_far, float(y[i].item())),
            "iteration_time_sec": 0.0,
            "seed": config.seed,
        }
        for i in range(len(y))
    )

    # Replace initial best values correctly.
    running_best = float("-inf")
    for row, yi in zip(rows, y.tolist()):
        running_best = max(running_best, float(yi))
        row["best_so_far"] = running_best

    remaining = config.total_evals - len(y)
    if remaining <= 0:
        return pd.DataFrame(rows)

    n_iter = remaining // config.q
    for _ in range(n_iter):
        model = model_factory(input_dim, config.seed)

        start = time.perf_counter()
        focal = X[y.argmax()].view(1, -1)
        try:
            model.fit(X, y, focal_points=focal)
        except TypeError:
            model.fit(X, y)
        x_next = optimize_qei(
            model=model,
            bounds=bounds,
            best_f=y.max(),
            q=config.q,
            num_restarts=config.qei_restarts,
            steps=config.qei_steps,
            mc_samples=config.qei_mc_samples,
            seed=config.seed + 17,
        )
        iteration_time = time.perf_counter() - start

        y_next = _evaluate_objective(objective, x_next)
        X = torch.cat([X, x_next.detach()], dim=0)
        y = torch.cat([y, y_next.detach()], dim=0)
        best_so_far = max(best_so_far, float(y_next.max().item()))

        for j, value in enumerate(y_next.tolist(), start=1):
            rows.append(
                {
                    "method": method_name,
                    "evaluation": len(rows) + 1,
                    "best_so_far": best_so_far,
                    "iteration_time_sec": iteration_time,
                    "seed": config.seed,
                }
            )

    return pd.DataFrame(rows)


def run_suite(
    factories: Mapping[str, ModelFactory],
    objective: Objective,
    bounds: Tensor,
    input_dim: int,
    config: BOConfig,
) -> pd.DataFrame:
    """Run all methods independently under identical initial-design seeds."""
    all_results = []
    for name, factory in factories.items():
        result = run_baseline(
            method_name=name,
            model_factory=factory,
            objective=objective,
            bounds=bounds,
            input_dim=input_dim,
            config=config,
        )
        all_results.append(result)
    return pd.concat(all_results, ignore_index=True)
