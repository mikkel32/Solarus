from __future__ import annotations

from typing import Any, Iterable


def find_namespace_packages(*, include: Iterable[str] | None = ..., exclude: Iterable[str] | None = ...) -> list[str]: ...


def setup(**kwargs: Any) -> None: ...


__all__ = ["find_namespace_packages", "setup"]

