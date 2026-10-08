import math
import torch


def ackley(X: torch.Tensor) -> torch.Tensor:
    """Ackley, conventionally minimized; this implementation returns -f for BO maximization."""
    a, b, c = 20.0, 0.2, 2.0 * math.pi
    d = X.shape[-1]
    x = X
    term1 = -a * torch.exp(-b * torch.sqrt((x.square().mean(dim=-1)).clamp_min(1e-12)))
    term2 = -torch.exp(torch.cos(c * x).mean(dim=-1))
    f = term1 + term2 + a + math.e
    return -f


def levy(X: torch.Tensor) -> torch.Tensor:
    w = 1.0 + (X - 1.0) / 4.0
    term1 = torch.sin(math.pi * w[..., 0]).square()
    term2 = ((w[..., 1:] - 1.0).square() * (1.0 + 10.0 * torch.sin(math.pi * w[..., 1:] + 1.0).square())).sum(dim=-1)
    term3 = (w[..., -1] - 1.0).square() * (1.0 + torch.sin(2.0 * math.pi * w[..., -1]).square())
    return -(term1 + term2 + term3)


def griewank(X: torch.Tensor) -> torch.Tensor:
    d = X.shape[-1]
    idx = torch.arange(1, d + 1, dtype=X.dtype, device=X.device)
    value = (X.square().sum(dim=-1) / 4000.0) - torch.prod(torch.cos(X / torch.sqrt(idx)), dim=-1) + 1.0
    return -value


def branin_normalized(X: torch.Tensor) -> torch.Tensor:
    x1 = X[..., 0] * 15.0 - 5.0
    x2 = X[..., 1] * 15.0
    a = 1.0
    b = 5.1 / (4.0 * math.pi**2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)
    term1 = a * (x2 - b * x1.square() + c * x1 - r).square()
    term2 = s * (1.0 - t) * torch.cos(x1)
    return term1 + term2 + s
