from __future__ import annotations

from typing import Any, Callable, Sequence

from . import themes


class Blocks:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __enter__(self) -> Blocks: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: Any | None) -> None: ...
    def queue(self, *args: Any, **kwargs: Any) -> Blocks: ...
    def launch(self, *args: Any, **kwargs: Any) -> Any: ...


class Column:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __enter__(self) -> Column: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: Any | None) -> None: ...


class Row:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __enter__(self) -> Row: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: Any | None) -> None: ...


class Markdown:
    def __init__(self, value: Any = ..., *args: Any, **kwargs: Any) -> None: ...
    def update(self, value: Any = ..., *args: Any, **kwargs: Any) -> Markdown: ...


class State:
    value: Any

    def __init__(self, value: Any = ...) -> None: ...


class Textbox:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def submit(
        self,
        fn: Callable[..., Any],
        inputs: Sequence[Any] | Any,
        outputs: Sequence[Any] | Any,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...


class Button:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def click(
        self,
        fn: Callable[..., Any],
        inputs: Sequence[Any] | Any,
        outputs: Sequence[Any] | Any,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...


class Chatbot:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class Dataframe:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


__all__ = [
    "Blocks",
    "Column",
    "Row",
    "Markdown",
    "State",
    "Textbox",
    "Button",
    "Chatbot",
    "Dataframe",
    "themes",
]

