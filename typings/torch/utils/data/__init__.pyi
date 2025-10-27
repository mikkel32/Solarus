from __future__ import annotations

from typing import Any, Generic, Iterable, Iterator, Sequence, TypeVar

T_co = TypeVar("T_co", covariant=True)


class Dataset(Generic[T_co]):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> T_co: ...


class DataLoader(Iterable[T_co], Generic[T_co]):
    dataset: Dataset[T_co]
    def __iter__(self) -> Iterator[T_co]: ...
    def __len__(self) -> int: ...


def default_collate(batch: Sequence[T_co]) -> Any: ...
