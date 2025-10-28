from __future__ import annotations

from typing import Any, overload
from collections.abc import Iterable, Iterator, Mapping, Sequence

from . import functional as functional_module

from .. import Tensor

functional = functional_module

class Module:
    training: bool
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
    def to(self, *args: Any, **kwargs: Any) -> Module: ...
    def eval(self) -> Module: ...
    def train(self, mode: bool = ...) -> Module: ...
    def parameters(self, *args: Any, **kwargs: Any) -> Iterable[Any]: ...
    def named_parameters(self, *args: Any, **kwargs: Any) -> Iterable[tuple[str, Any]]: ...
    def state_dict(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]: ...
    def load_state_dict(self, *args: Any, **kwargs: Any) -> Any: ...
    def buffers(self, *args: Any, **kwargs: Any) -> Iterable[Any]: ...
    def named_buffers(self, *args: Any, **kwargs: Any) -> Iterable[tuple[str, Any]]: ...
    def register_buffer(self, name: str, tensor: Any | None, persistent: bool = ...) -> None: ...
    def cpu(self) -> Module: ...
    def cuda(self, *args: Any, **kwargs: Any) -> Module: ...
    def zero_grad(self, *args: Any, **kwargs: Any) -> None: ...
    def update_parameters(self, *args: Any, **kwargs: Any) -> None: ...
    def __getattr__(self, name: str) -> Any: ...


class ModuleList(Module, Sequence[Module]):
    def __init__(self, modules: Iterable[Module] | None = ...) -> None: ...
    def append(self, module: Module) -> None: ...
    def extend(self, modules: Iterable[Module]) -> None: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, index: int) -> Module: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[Module]: ...
    def __iter__(self) -> Iterator[Module]: ...


class ModuleDict(Module, Mapping[str, Module]):
    ...


class Sequential(Module):
    def __init__(self, *modules: Module) -> None: ...


class DataParallel(Module):
    module: Module
    device_ids: Sequence[int] | None
    output_device: int | None

    def __init__(self, module: Module, *args: Any, **kwargs: Any) -> None: ...


class Linear(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    weight: Tensor
    bias: Tensor | None


class Conv1d(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    weight: Tensor
    bias: Tensor | None


class Dropout(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class LayerNorm(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class Parameter(Tensor):
    def __new__(cls, data: Tensor | Any, requires_grad: bool = ...) -> Parameter: ...


class Embedding(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class MultiheadAttention(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    def forward(
        self,
        query: Tensor,
        key: Tensor | None = ...,
        value: Tensor | None = ...,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor | None]: ...


class LSTM(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    bidirectional: bool


class GRU(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class BatchNorm1d(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class CrossEntropyLoss(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class BCEWithLogitsLoss(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class GELU(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class ReLU(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class SiLU(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class Tanh(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class Softmax(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


__all__ = [
    "Module",
    "ModuleList",
    "ModuleDict",
    "Sequential",
    "DataParallel",
    "Linear",
    "Conv1d",
    "Dropout",
    "LayerNorm",
    "Parameter",
    "Embedding",
    "MultiheadAttention",
    "LSTM",
    "GRU",
    "BatchNorm1d",
    "CrossEntropyLoss",
    "BCEWithLogitsLoss",
    "GELU",
    "ReLU",
    "SiLU",
    "Tanh",
    "Softmax",
    "functional",
]
