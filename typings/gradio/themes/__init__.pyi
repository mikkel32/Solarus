from __future__ import annotations

from typing import Any


class Theme:
    ...


class Soft(Theme):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


__all__ = ["Theme", "Soft"]

