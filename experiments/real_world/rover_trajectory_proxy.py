"""60-D Rover Trajectory Planning proxy.

The manuscript describes a 60-dimensional task consisting of 30 two-dimensional
waypoints. This adapter reproduces that search-space structure and the main
optimization ingredients (path length, obstacle collisions, and deviation)
without claiming to reproduce the original simulator exactly.
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


START = torch.tensor([0.05, 0.10])
GOAL = torch.tensor([0.95, 0.90])
OBSTACLES = torch.tensor([
    [0.35, 0.35, 0.12],
    [0.55, 0.55, 0.10],
    [0.65, 0.30, 0.09],
    [0.30, 0.70, 0.08],
])


def rover_trajectory_proxy(x: torch.Tensor) -> torch.Tensor:
    waypoints = x.view(-1, 30, 2)
    start = START.to(device=x.device, dtype=x.dtype).expand(x.shape[0], -1).unsqueeze(1)
    goal = GOAL.to(device=x.device, dtype=x.dtype).expand(x.shape[0], -1).unsqueeze(1)
    pts = torch.cat([start, waypoints, goal], dim=1)

    step = pts[:, 1:] - pts[:, :-1]
    length = torch.linalg.vector_norm(step, dim=-1).sum(dim=1)

    # Smoothness penalty discourages highly oscillatory trajectories.
    curvature = torch.linalg.vector_norm(pts[:, 2:] - 2.0 * pts[:, 1:-1] + pts[:, :-2], dim=-1).sum(dim=1)

    obstacles = OBSTACLES.to(device=x.device, dtype=x.dtype)
    centers = obstacles[:, :2]
    radii = obstacles[:, 2]
    distances = torch.linalg.vector_norm(waypoints.unsqueeze(2) - centers.view(1, 1, -1, 2), dim=-1)
    collision = torch.relu(radii.view(1, 1, -1) - distances).sum(dim=(1, 2))

    # Encourage a near-direct route while penalizing collisions.
    path_deviation = torch.abs(waypoints[:, :, 0] - waypoints[:, :, 1]).mean(dim=1)
    cost = length + 0.25 * curvature + 30.0 * collision + 0.10 * path_deviation
    return (-cost).unsqueeze(-1)


def make_objective() -> BlackBoxObjective:
    return BlackBoxObjective(
        function=rover_trajectory_proxy,
        lower=torch.zeros(60),
        upper=torch.ones(60),
        name="rover_trajectory_proxy",
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
