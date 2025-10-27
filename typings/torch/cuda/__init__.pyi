from __future__ import annotations

from typing import Any

from .. import device


def is_available() -> bool: ...

def current_device() -> int: ...

def device_count() -> int: ...

def manual_seed_all(seed: int) -> None: ...

def synchronize(device: device | int | None = ...) -> None: ...

def Stream() -> Any: ...


from .amp import GradScaler, autocast
