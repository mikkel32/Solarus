from __future__ import annotations

from typing import Any

from .. import Optimizer


class LRScheduler:
    optimizer: Optimizer

    def __init__(self, optimizer: Optimizer, *args: Any, **kwargs: Any) -> None: ...

    def state_dict(self) -> Any: ...

    def load_state_dict(self, state_dict: Any) -> None: ...

    def get_last_lr(self) -> list[float]: ...

    def step(self, *args: Any, **kwargs: Any) -> None: ...


class ReduceLROnPlateau(LRScheduler):
    ...


class CosineAnnealingLR(LRScheduler):
    ...


class CosineAnnealingWarmRestarts(LRScheduler):
    ...


class OneCycleLR(LRScheduler):
    ...


class LinearLR(LRScheduler):
    ...


class PolynomialLR(LRScheduler):
    ...


__all__ = [
    "LRScheduler",
    "ReduceLROnPlateau",
    "CosineAnnealingLR",
    "CosineAnnealingWarmRestarts",
    "OneCycleLR",
    "LinearLR",
    "PolynomialLR",
]
