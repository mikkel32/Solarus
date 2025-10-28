from __future__ import annotations

from typing import Any, Mapping, Sequence

class _DriveModule:
    def mount(self, mountpoint: str, force_remount: bool | None = ..., *, timeout_ms: int | None = ...) -> None: ...
    def flush_and_unmount(self) -> None: ...


class _FilesModule:
    def upload(
        self,
        *,
        destination_directory: str | None = ...,
        accept: Sequence[str] | None = ...,
        multiple_files: bool = ...,
        use_native_file_dialog: bool = ...,
    ) -> Mapping[str, Any]: ...
    def view(self, path: str) -> None: ...
    def download(self, path: str) -> None: ...


class _AuthModule:
    def authorize(self, force_reauth: bool = ...) -> None: ...


drive: _DriveModule
files: _FilesModule
auth: _AuthModule

__all__ = ["drive", "files", "auth"]
