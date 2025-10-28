from __future__ import annotations

from typing import Any


class c_nvmlDevice_t:
    ...


NVML_VALUE_NOT_AVAILABLE: int


def nvmlInit() -> None: ...

def nvmlShutdown() -> None: ...

def nvmlDeviceGetHandleByIndex(index: int) -> c_nvmlDevice_t: ...

def nvmlDeviceGetMemoryInfo(handle: c_nvmlDevice_t) -> Any: ...

def nvmlDeviceGetFanSpeed(handle: c_nvmlDevice_t) -> int: ...

def nvmlDeviceGetFanSpeed_v2(handle: c_nvmlDevice_t, fan: int) -> Any: ...

