from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


class Optimizer:
    defaults: Mapping[str, Any]
    param_groups: Sequence[Mapping[str, Any]]
    micro_batch_size: int | None
    grad_accumulation_steps: int | None
    ema_active: bool
    ema_model: Any | None
    swa_active: bool
    swa_model: Any | None
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


from . import lr_scheduler as lr_scheduler
