from __future__ import annotations

import contextlib
import random
from typing import Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def frozen(module: torch.nn.Module):
    """Freeze parameters while preserving gradients with respect to inputs."""
    old = [p.requires_grad for p in module.parameters()]
    try:
        for p in module.parameters():
            p.requires_grad_(False)
        yield module
    finally:
        for p, value in zip(module.parameters(), old):
            p.requires_grad_(value)


def finite_tensor(x: torch.Tensor, name: str) -> None:
    if not torch.isfinite(x).all():
        raise FloatingPointError(f"{name} contains NaN/Inf")


def running_best(values: Iterable[float]) -> list[float]:
    best = -float("inf")
    out: list[float] = []
    for value in values:
        best = max(best, float(value))
        out.append(best)
    return out
