from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vbo_baselines.baselines.gp import ExactGPBaseline
from vbo_baselines.baselines.dkl import DKLBaseline
from vbo_baselines.baselines.ibnn import IBNNBaseline
from vbo_baselines.baselines.focal_sparse_gp import FocalSparseGPBaseline
from vbo_baselines.baselines.hmc import HMCBaseline
from vbo_baselines.baselines.sghmc import SGHMCBaseline
from vbo_baselines.bo.acquisition import optimize_qei
from vbo_baselines.objectives import branin_normalized


def main() -> None:
    torch.set_default_dtype(torch.float64)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    X = torch.rand(8, 2)
    y = branin_normalized(X)

    models = {
        "GP": ExactGPBaseline(2, fit_steps=5, learning_rate=0.03),
        "DKL": DKLBaseline(2, hidden_width=8, feature_dim=4, fit_steps=5, learning_rate=0.01),
        "IBNN": IBNNBaseline(2, fit_steps=5, learning_rate=0.03),
        "Sparse": FocalSparseGPBaseline(2, num_inducing=6, fit_steps=5, learning_rate=0.01),
        "HMC": HMCBaseline(
            2, hidden_width=8, samples=2, warmup_steps=2, leapfrog_steps=2,
            step_size=1e-4, map_steps=2, seed=0
        ),
        "SGHMC": SGHMCBaseline(
            2, hidden_width=8, samples=2, burnin_steps=2, total_steps=4,
            batch_size=4, map_steps=2, step_size=1e-4, seed=0
        ),
    }

    for name, model in models.items():
        print(f"Running {name} ...")
        focal = X[y.argmax()].view(1, -1)
        model.fit(X, y, focal_points=focal)
        mean, cov = model.posterior_moments(X[:3])
        assert mean.shape == (3,), (name, mean.shape)
        assert cov.shape == (3, 3), (name, cov.shape)
        assert torch.isfinite(mean).all()
        assert torch.isfinite(cov).all()

        # One small qEI optimization run.
        x_next = optimize_qei(
            model=model,
            bounds=bounds,
            best_f=y.max(),
            q=1,
            num_restarts=2,
            steps=2,
            mc_samples=4,
        )
        assert x_next.shape == (1, 2), (name, x_next.shape)
        print(f"  OK: next point={x_next.detach().cpu().numpy().round(4)}")

    try:
        from vbo_baselines.baselines.lla import LLABaseline
        lla = LLABaseline(2, hidden_width=8, fit_steps=2, prior_precision_steps=1)
        lla.fit(X, y)
        mean, cov = lla.posterior_moments(X[:2])
        assert mean.shape == (2,)
        assert cov.shape == (2, 2)
        print("LLA OK")
    except RuntimeError as exc:
        print(f"LLA skipped: {exc}")

    print("Smoke test completed.")


if __name__ == "__main__":
    main()
