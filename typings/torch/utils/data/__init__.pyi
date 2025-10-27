from __future__ import annotations

from typing import Any, Callable, Generic, Iterable, Iterator, Sequence, TypeVar

T_co = TypeVar("T_co", covariant=True)


class Dataset(Generic[T_co]):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> T_co: ...


class DataLoader(Iterable[T_co], Generic[T_co]):
    dataset: Dataset[T_co]
    batch_size: int | None
    drop_last: bool
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None
    def __init__(
        self,
        dataset: Dataset[T_co],
        batch_size: int | None = ...,
        shuffle: bool = ...,
        sampler: Any | None = ...,
        batch_sampler: Any | None = ...,
        num_workers: int = ...,
        collate_fn: Callable[[Sequence[T_co]], Any] | None = ...,
        pin_memory: bool = ...,
        drop_last: bool = ...,
        timeout: float = ...,
        worker_init_fn: Callable[[int], None] | None = ...,
        multiprocessing_context: Any | None = ...,
        generator: Any | None = ...,
        prefetch_factor: int | None = ...,
        persistent_workers: bool = ...,
        pin_memory_device: str | None = ...,
    ) -> None: ...
    def __iter__(self) -> Iterator[T_co]: ...
    def __len__(self) -> int: ...


def default_collate(batch: Sequence[T_co]) -> Any: ...
