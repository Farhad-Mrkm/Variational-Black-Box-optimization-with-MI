from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class BlackBoxObjective:
    """A callable black-box objective used by VBO-MI.

    The runner calls this objective with detached tensors. The objective should
    therefore never rely on autograd and should return one scalar reward per row.
    """

    function: Callable[[torch.Tensor], torch.Tensor]
    lower: torch.Tensor
    upper: torch.Tensor
    name: str = "objective"
    noise_std: float = 0.0
    batch_size: Optional[int] = None
    warmup_iterations: Optional[int] = None
    main_iterations: Optional[int] = None
    optimum_is_known: bool = False
    optimum_value: Optional[float] = None

    @property
    def input_dim(self) -> int:
        return int(self.lower.numel())

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach()
        with torch.no_grad():
            y = self.function(x)
            if y.ndim == 1:
                y = y.unsqueeze(-1)
            if y.ndim != 2 or y.shape[1] != 1:
                raise ValueError("Objective must return shape [B, 1] or [B]")
            if self.noise_std > 0:
                y = y + torch.randn_like(y) * self.noise_std
        return y


def ackley_negative(x: torch.Tensor) -> torch.Tensor:
    a, b, c = 20.0, 0.2, 2.0 * torch.pi
    term1 = -a * torch.exp(-b * torch.sqrt(torch.mean(x * x, dim=1)))
    term2 = -torch.exp(torch.mean(torch.cos(c * x), dim=1))
    value = term1 + term2 + a + torch.e
    return -value.unsqueeze(-1)


def levy_negative(x: torch.Tensor) -> torch.Tensor:
    w = 1.0 + (x - 1.0) / 4.0
    value = torch.sin(torch.pi * w[:, 0]) ** 2
    value = value + torch.sum((w[:, :-1] - 1.0) ** 2 * (1.0 + 10.0 * torch.sin(torch.pi * w[:, :-1] + 1.0) ** 2), dim=1)
    value = value + (w[:, -1] - 1.0) ** 2 * (1.0 + torch.sin(2.0 * torch.pi * w[:, -1]) ** 2)
    return -value.unsqueeze(-1)


def griewank_negative(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[1]
    idx = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
    value = torch.sum(x * x, dim=1) / 4000.0 - torch.prod(torch.cos(x / torch.sqrt(idx)), dim=1) + 1.0
    return -value.unsqueeze(-1)


def rastrigin_negative(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[1]
    value = 10.0 * d + torch.sum(x * x - 10.0 * torch.cos(2.0 * torch.pi * x), dim=1)
    return -value.unsqueeze(-1)


def hartmann6_negative(x: torch.Tensor) -> torch.Tensor:
    # Standard 6-D Hartmann convention; returned as negative for maximization.
    alpha = torch.tensor([1.0, 1.2, 3.0, 3.2], dtype=x.dtype, device=x.device)
    A = torch.tensor(
        [[10, 3, 17, 3.5, 1.7, 8],
         [0.05, 10, 17, 0.1, 8, 14],
         [3, 3.5, 1.7, 10, 17, 8],
         [17, 8, 0.05, 10, 0.1, 14]],
        dtype=x.dtype, device=x.device,
    )
    P = 1e-4 * torch.tensor(
        [[3689, 1170, 2673, 5569, 1446, 5681],
         [4699, 4387, 7470,  556, 8825, 8732],
         [1091, 8732, 5547, 5743, 8828, 8481],
         [381,  5743, 8828,  446,  40,   5743]],
        dtype=x.dtype, device=x.device,
    )
    inner = torch.sum(A.unsqueeze(0) * (x.unsqueeze(1) - P.unsqueeze(0)) ** 2, dim=-1)
    value = -torch.sum(alpha.unsqueeze(0) * torch.exp(-inner), dim=1)
    return -value.unsqueeze(-1)


def paper_synthetic_objective(name: str, device: str | torch.device = "cpu") -> BlackBoxObjective:
    name = name.lower()
    if name == "ackley":
        d, lo, hi, fn = 10, -32.768, 32.768, ackley_negative
    elif name == "levy":
        d, lo, hi, fn = 20, -10.0, 10.0, levy_negative
    elif name == "griewank":
        d, lo, hi, fn = 20, -600.0, 600.0, griewank_negative
    elif name == "rastrigin":
        d, lo, hi, fn = 20, -5.12, 5.12, rastrigin_negative
    elif name == "hartmann6":
        d, lo, hi, fn = 6, 0.0, 1.0, hartmann6_negative
    else:
        raise KeyError(f"Unknown synthetic benchmark: {name}")
    dtype = torch.float32
    return BlackBoxObjective(
        function=fn,
        lower=torch.full((d,), lo, device=device, dtype=dtype),
        upper=torch.full((d,), hi, device=device, dtype=dtype),
        name=name,
        optimum_is_known=(name != "hartmann6"),
        optimum_value=(0.0 if name != "hartmann6" else None),
    )
