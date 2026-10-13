"""Lunar Lander experiment adapter.


"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

try:
    import gymnasium as gym
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Lunar Lander requires Gymnasium Box2D. Install with: "
        "pip install -e '.[realworld]'"
    ) from exc

from vbo_mi.config import VBOConfig
from vbo_mi.objectives import BlackBoxObjective
from vbo_mi.runner import VBOMI


def controller_action(theta: np.ndarray, obs: np.ndarray) -> int:
    """A compact 12-parameter tanh controller for the 8D lander state."""
    # 12 parameters are split into per-action linear biases/weights.
    w = theta[:8]
    b = theta[8:]
    score_main = np.tanh(np.dot(w, obs) + b[0])
    score_left = np.tanh(np.dot(w * theta[0:8], obs) + b[1])
    score_right = np.tanh(np.dot(w * theta[4:12], obs[:8]) + b[2]) if len(theta) >= 12 else 0.0
    scores = np.array([0.0, score_left, score_main, score_right], dtype=np.float64)
    return int(np.argmax(scores))


def evaluate_controller(theta: torch.Tensor, episodes: int = 10, seed: int = 0) -> torch.Tensor:
    """Evaluate each parameter vector over a small set of deterministic episode seeds."""
    values = []
    theta_np = theta.detach().cpu().numpy()
    base_seed = int(seed)
    for row in theta_np:
        returns = []
        for ep in range(episodes):
            env = gym.make("LunarLander-v3", continuous=False)
            obs, _ = env.reset(seed=base_seed + ep)
            total = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                action = controller_action(row, np.asarray(obs, dtype=np.float64))
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
            env.close()
            returns.append(total)
        values.append(np.mean(returns))
    return torch.tensor(values, dtype=theta.dtype, device=theta.device).unsqueeze(-1)


def make_objective(episodes: int, seed: int) -> BlackBoxObjective:
    return BlackBoxObjective(
        function=lambda x: evaluate_controller(x, episodes=episodes, seed=seed),
        lower=torch.full((12,), -2.0),
        upper=torch.full((12,), 2.0),
        name="lunar_lander",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per oracle evaluation.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    objective = make_objective(args.episodes, args.seed)
    cfg = VBOConfig(seed=args.seed)
    result = VBOMI(objective, cfg).run(
        batch_size=args.batch_size,
        warmup_iterations=args.warmup,
        main_iterations=args.iterations,
    )
    print(f"Final best reward: {result.best_y:.3f}")
    print(f"Function evaluations: {int(result.evaluation_counts[-1])}")


if __name__ == "__main__":
    main()
