from __future__ import annotations

from typing import Any, Sequence


class SentenceTransformer:
    def __init__(self, model_name_or_path: str, *args: Any, **kwargs: Any) -> None: ...

    def encode(
        self,
        sentences: str | Sequence[str],
        *args: Any,
        **kwargs: Any,
    ) -> Sequence[float] | Sequence[Sequence[float]]: ...


__all__ = ["SentenceTransformer"]

