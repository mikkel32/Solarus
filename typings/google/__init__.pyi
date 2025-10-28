from __future__ import annotations

from types import ModuleType

from . import colab as colab_module

colab: ModuleType = colab_module

__all__ = ["colab"]
