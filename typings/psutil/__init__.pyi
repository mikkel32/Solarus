from __future__ import annotations

from typing import Any, Iterable


def cpu_count(logical: bool | None = ...) -> int | None: ...


def cpu_percent(interval: float | None = ..., percpu: bool = ...) -> float | list[float]: ...


def virtual_memory() -> Any: ...


def process_iter(attrs: Iterable[str] | None = ...) -> Iterable[Any]: ...
