from __future__ import annotations

from typing import Any

class cudnn:
    @staticmethod
    def version() -> int | None: ...
    deterministic: bool
    benchmark: bool


class mps:
    @staticmethod
    def is_available() -> bool: ...
    @staticmethod
    def is_built() -> bool: ...


def __getattr__(name: str) -> Any: ...
