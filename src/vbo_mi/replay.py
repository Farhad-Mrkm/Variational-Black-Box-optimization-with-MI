from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ReplayBuffer:
    max_batches: Optional[int] = 100

    def __post_init__(self) -> None:
        self._x: list[torch.Tensor] = []
        self._y: list[torch.Tensor] = []

    def __len__(self) -> int:
        return sum(t.shape[0] for t in self._x)

    @property
    def num_batches(self) -> int:
        return len(self._x)

    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.ndim != 2 or y.ndim != 2 or y.shape[1] != 1 or x.shape[0] != y.shape[0]:
            raise ValueError("Replay entries must be x:[B,D] and y:[B,1]")
        self._x.append(x.detach().cpu())
        self._y.append(y.detach().cpu())
        if self.max_batches is not None and len(self._x) > self.max_batches:
            excess = len(self._x) - self.max_batches
            del self._x[:excess]
            del self._y[:excess]

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._x:
            raise RuntimeError("Replay buffer is empty")
        return torch.cat(self._x, dim=0), torch.cat(self._y, dim=0)

    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.tensors()
        n = x.shape[0]
        indices = torch.randint(0, n, (batch_size,))
        return x[indices].to(device=device, dtype=dtype), y[indices].to(device=device, dtype=dtype)
