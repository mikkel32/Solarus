from __future__ import annotations

from typing import Any, Iterable

from .. import Tensor


def clip_grad_norm_(
    parameters: Iterable[Tensor | Any],
    max_norm: float,
    norm_type: float | int = ...,
) -> float:
    ...
