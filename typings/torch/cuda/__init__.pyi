from __future__ import annotations

from typing import Any

from typing import Any

from .. import device
from . import amp as amp_module


def is_available() -> bool: ...

def current_device() -> int: ...

def device_count() -> int: ...

def manual_seed_all(seed: int) -> None: ...

def synchronize(device: device | int | None = ...) -> None: ...

def Stream() -> Any: ...

GradScaler = amp_module.GradScaler
autocast = amp_module.autocast

__all__ = [
    "is_available",
    "current_device",
    "device_count",
    "manual_seed_all",
    "synchronize",
    "Stream",
    "GradScaler",
    "autocast",
]
