from __future__ import annotations

import math
import torch


def logmeanexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(mean(exp(x)))."""
    n = x.numel()
    if n == 0:
        raise ValueError("Cannot compute logmeanexp over an empty tensor")
    return torch.logsumexp(x.reshape(-1), dim=0) - math.log(n)


def dv_mi_bound(scores: torch.Tensor) -> torch.Tensor:
    """Equation (13)/(32): sample-average DV lower bound.

    The joint expectation is represented by the diagonal. The product of
    marginals is represented by the full B x B pairing, matching Equation 32
    of the supplied manuscript. Stable log-sum-exp is used for the exponential
    term as described in Remark 4.2.
    """
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must have shape [B, B]")
    joint = scores.diag().mean()
    marginal_logmeanexp = logmeanexp(scores)
    return joint - marginal_logmeanexp


def helper_loss(scores: torch.Tensor) -> torch.Tensor:
    """Negative DV lower bound for helper-network training."""
    return -dv_mi_bound(scores)


def action_loss(scores: torch.Tensor, exploitation: torch.Tensor, beta: float) -> torch.Tensor:
    """Equation (32), returned as a minimization loss."""
    if beta < 0:
        raise ValueError("beta must be non-negative")
    return -(exploitation + math.sqrt(beta) * dv_mi_bound(scores))
