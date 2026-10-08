from __future__ import annotations

from typing import Any, Dict

from .baselines import (
    DKLBaseline,
    ExactGPBaseline,
    FocalSparseGPBaseline,
    HMCBaseline,
    IBNNBaseline,
    LLABaseline,
    SGHMCBaseline,
)


def make_baseline(name: str, input_dim: int, seed: int = 0, **overrides):
    """Construct one of the seven TMLR baselines."""
    key = name.strip().lower().replace("-", "").replace("_", "")
    common = dict(overrides)

    if key == "gp":
        return ExactGPBaseline(input_dim=input_dim, **common)
    if key == "dkl":
        return DKLBaseline(input_dim=input_dim, **common)
    if key in {"ibnn", "infinitewidthbnn", "nngp"}:
        return IBNNBaseline(input_dim=input_dim, **common)
    if key in {"focalsparse", "focalsparsegp", "sparse"}:
        return FocalSparseGPBaseline(input_dim=input_dim, **common)
    if key == "hmc":
        return HMCBaseline(input_dim=input_dim, seed=seed, **common)
    if key == "sghmc":
        return SGHMCBaseline(input_dim=input_dim, seed=seed, **common)
    if key in {"lla", "laplace"}:
        return LLABaseline(input_dim=input_dim, **common)

    raise ValueError(
        f"Unknown baseline {name!r}. "
        "Choose GP, DKL, IBNN, HMC, SGHMC, Sparse, or LLA."
    )
