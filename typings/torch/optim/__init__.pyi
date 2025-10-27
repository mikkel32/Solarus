from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from .. import Tensor


class Optimizer:
    defaults: Mapping[str, Any]
    param_groups: Sequence[Mapping[str, Any]]
    def __init__(self, params: Iterable[Any], defaults: Mapping[str, Any] | None = ...) -> None: ...
    def step(self, closure: Any | None = ...) -> Any: ...
    def zero_grad(self, set_to_none: bool | None = ...) -> None: ...
    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None: ...


class SGD(Optimizer):
    ...


class Adam(Optimizer):
    ...


class AdamW(Optimizer):
    ...


class RMSprop(Optimizer):
    ...


class Adadelta(Optimizer):
    ...


def lr_scheduler() -> Any: ...
