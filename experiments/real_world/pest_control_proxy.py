"""Categorical Pest Control proxy benchmark.

The TMLR task is a 25-dimensional categorical optimization problem with five
choices per stage. The paper specifies one-hot handling and a categorical
projection for VBO-MI, but does not include the full simulator in the
manuscript. This file provides a small stochastic control proxy with the same
25 x 5 search-space shape. It is useful for testing categorical wrappers and
is explicitly *not* presented as the original benchmark simulator.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch

from vbo_mi.config import VBOConfig
from vbo_mi.objectives import BlackBoxObjective
from vbo_mi.runner import VBOMI


ACTION_EFFECT = torch.tensor([0.00, 0.18, 0.32, 0.46, 0.60])
ACTION_COST = torch.tensor([0.00, 0.20, 0.55, 1.00, 1.70])


def pest_control_proxy(x: torch.Tensor) -> torch.Tensor:
    """Return negative cumulative loss for 25 categorical stages."""
    categories = torch.clamp(torch.round(x), 0, 4).long()
    batch = x.shape[0]
    pest = torch.full((batch,), 0.30, dtype=x.dtype, device=x.device)
    total_loss = torch.zeros_like(pest)
    effect = ACTION_EFFECT.to(device=x.device, dtype=x.dtype)
    cost = ACTION_COST.to(device=x.device, dtype=x.dtype)

    for t in range(25):
        a = categories[:, t]
        total_loss = total_loss + 10.0 * pest + cost[a]
        growth = 1.08
        pest = torch.clamp(pest * growth - effect[a], 0.0, 1.0)

    return (-total_loss).unsqueeze(-1)


def make_objective() -> BlackBoxObjective:
    return BlackBoxObjective(
        function=pest_control_proxy,
        lower=torch.zeros(25),
        upper=torch.full((25,), 4.0),
        name="pest_control_proxy",
        optimum_is_known=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = VBOMI(make_objective(), VBOConfig(seed=args.seed)).run(
        batch_size=args.batch_size,
        warmup_iterations=args.warmup,
        main_iterations=args.iterations,
    )
    print(f"Final best proxy reward: {result.best_y:.4f}")
    print(f"Function evaluations: {int(result.evaluation_counts[-1])}")


if __name__ == "__main__":
    main()
