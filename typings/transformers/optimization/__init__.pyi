from __future__ import annotations

from typing import Any, Iterable, Mapping


class Adafactor:
    def __init__(self, params: Iterable[Any], *args: Any, **kwargs: Any) -> None: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None: ...


__all__ = ["Adafactor"]
