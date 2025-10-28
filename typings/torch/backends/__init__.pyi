from __future__ import annotations

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

__all__ = ["cudnn", "mps"]
