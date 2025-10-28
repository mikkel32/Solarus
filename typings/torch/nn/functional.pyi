from __future__ import annotations

from typing import Any

from .. import Tensor


def one_hot(tensor: Tensor, num_classes: int | None = ..., *, dtype: Any | None = ...) -> Tensor: ...

def log_softmax(tensor: Tensor, dim: int = ..., *, dtype: Any | None = ...) -> Tensor: ...

def softmax(tensor: Tensor, dim: int = ..., *, dtype: Any | None = ...) -> Tensor: ...

def cosine_similarity(x1: Tensor, x2: Tensor, dim: int = ..., eps: float = ...) -> Tensor: ...

def mse_loss(input: Tensor, target: Tensor, reduction: str = ...) -> Tensor: ...

def kl_div(
    input: Tensor,
    target: Tensor,
    reduction: str = ..., 
    *,
    log_target: bool = ...,
) -> Tensor: ...


__all__ = [
    "one_hot",
    "log_softmax",
    "softmax",
    "cosine_similarity",
    "mse_loss",
    "kl_div",
]
