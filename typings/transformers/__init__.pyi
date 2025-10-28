from __future__ import annotations

from typing import Any


from types import ModuleType


from . import optimization as optimization_module


class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args: Any, **kwargs: Any) -> AutoTokenizer: ...

    def __len__(self) -> int: ...


optimization: ModuleType = optimization_module


__all__ = ["AutoTokenizer", "optimization"]
