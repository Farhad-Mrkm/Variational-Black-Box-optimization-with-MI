from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class TaskConfig:
    """Optimization-domain settings."""

    input_dim: int
    lower: torch.Tensor
    upper: torch.Tensor
    batch_size: int = 60
    main_iterations: int = 70
    warmup_iterations: int = 30
    noise_std: float = 0.0
    name: str = "task"
    max_buffer_batches: Optional[int] = 100

    def validate(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.lower.numel() != self.input_dim or self.upper.numel() != self.input_dim:
            raise ValueError("lower/upper must have length input_dim")
        if torch.any(self.upper <= self.lower):
            raise ValueError("upper bounds must be strictly greater than lower bounds")
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2 for pairwise DV estimation")
        if self.main_iterations < 0 or self.warmup_iterations < 0:
            raise ValueError("iteration counts must be non-negative")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")


@dataclass
class VBOConfig:
    """VBO-MI network and optimization hyperparameters.

    Values marked as manuscript-derived are based on Appendix C.1.1 of the
    supplied TMLR manuscript. The manuscript does not specify every runtime
    detail (e.g. seed dimension or a single default beta), so those remain
    explicitly configurable here.
    """

    seed_dim: int = 16
    action_hidden_dim: int = 128
    helper_hidden_dim: int = 64
    surrogate_hidden_dim: int = 128
    helper_mlp_depth: int = 2
    surrogate_hidden_layers: int = 2

    action_lr: float = 2e-3
    helper_lr: float = 2e-3
    surrogate_lr: float = 1e-3

    ka: int = 5
    kb: int = 10
    ks: int = 10
    surrogate_pretrain_steps: int = 200

    beta: float = 100.0
    seed: int = 0
    device: Optional[str] = None
    dtype: torch.dtype = torch.float32

    # The supplied implementation caps history at 100 batches; the manuscript
    # states that the replay buffer is capped for the complexity discussion but
    # does not give a separate numeric cap in Appendix C. This is configurable.
    max_buffer_batches: Optional[int] = 100

    normalize_surrogate_targets: bool = False

    def validate(self) -> None:
        if self.seed_dim <= 0:
            raise ValueError("seed_dim must be positive")
        if min(self.action_hidden_dim, self.helper_hidden_dim, self.surrogate_hidden_dim) <= 0:
            raise ValueError("hidden dimensions must be positive")
        if min(self.action_lr, self.helper_lr, self.surrogate_lr) <= 0:
            raise ValueError("learning rates must be positive")
        if min(self.ka, self.kb, self.ks, self.surrogate_pretrain_steps) < 0:
            raise ValueError("update counts must be non-negative")
        if self.beta < 0:
            raise ValueError("beta must be non-negative")
