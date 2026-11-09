"""Bounded, exclusive file sinks used by Contract-4 media delivery.

The sink is deliberately small and provider-neutral.  It owns exactly one
pre-created partial and never replaces a destination or accumulates a response
body in memory.  The provider worker writes and flushes it; the coordinator
may only remove it after the worker has stopped.
"""

from __future__ import annotations

import hashlib
import ctypes
import errno
import os
import secrets
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CHUNK_BYTES = 1024 * 1024


class DownloadError(IOError):
    """Finite failure for bounded download/sink operations."""


@dataclass(frozen=True)
class DownloadDigest:
    sha256: str
    bytes_written: int



def _flush_parent(path: Path) -> None:
    """Flush a directory after a no-replace publication where supported."""
    # On POSIX a directory descriptor gives us the durability boundary.  The
    # Windows path boundary has a stronger native implementation; use it when
    # the synthetic studio-core package is loaded, while keeping this module
    # independently importable for unit tests.
    if os.name == "nt":
        import sys

        for name, module in tuple(sys.modules.items()):
            if name.endswith(".path_boundary") and hasattr(module, "_fsync_parent"):
                module._fsync_parent(path)  # type: ignore[attr-defined]
                return
        return
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        os.fsync(fd)
    except OSError as exc:
        raise DownloadError("parent directory flush failed") from exc
    finally:
        if fd is not None:
            os.close(fd)


def _windows_create_exclusive(path: Path):
    """Create a partial with DELETE access and delete-sharing enabled."""
    if os.name != "nt":
        raise RuntimeError("Windows partial creation requested on POSIX")
    import msvcrt
    import ctypes.wintypes as wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL

    generic_write = 0x40000000
    delete = 0x00010000
    share_read = 0x1
    share_write = 0x2
    share_delete = 0x4
    create_new = 1
    file_attribute_normal = 0x80
    invalid_handle = ctypes.c_void_p(-1).value
    native = create_file(
        str(path),
        generic_write | delete,
        share_read | share_write | share_delete,
        None,
        create_new,
        file_attribute_normal,
        None,
    )
    raw = int(native)
    if raw in (-1, invalid_handle):
        raise OSError(ctypes.get_last_error() or errno.EIO, "partial could not be created exclusively")
    try:
        cleanup_native = _windows_duplicate_native(native)
    except Exception:
        try:
            _windows_dispose_native(native)
        finally:
            close_handle(native)
        raise
    fd = None
    wrapped = None
    try:
        fd = msvcrt.open_osfhandle(raw, os.O_WRONLY | getattr(os, "O_BINARY", 0))
        wrapped = os.fdopen(fd, "wb", buffering=0)
        _windows_close_native(cleanup_native)
        cleanup_native = None
        return wrapped
    except Exception:
        # ``open_osfhandle`` does not transfer ownership when it fails.  Mark
        # the exact object for deletion before closing the raw handle; merely
        # closing it leaves an empty .part behind on Windows.
        try:
            _windows_dispose_native(cleanup_native if cleanup_native is not None else native)
        finally:
            if cleanup_native is not None:
                try:
                    _windows_close_native(cleanup_native)
                except Exception:
                    pass
            if wrapped is not None:
                try:
                    wrapped.close()
                except Exception:
                    pass
            elif fd is None:
                close_handle(native)
            else:
                try:
                    os.close(fd)
                except OSError:
                    pass
        raise


def _windows_duplicate_native(native: Any) -> Any:
    """Duplicate one raw handle so wrapper-construction failure stays disposable."""
    if os.name != "nt":
        raise RuntimeError("Windows handle duplication requested on POSIX")
    import ctypes.wintypes as wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = wintypes.HANDLE
    duplicate_handle = kernel32.DuplicateHandle
    duplicate_handle.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    duplicate_handle.restype = wintypes.BOOL
    process = get_current_process()
    duplicate = wintypes.HANDLE()
    if not duplicate_handle(process, native, process, ctypes.byref(duplicate), 0, False, 2):
        raise OSError(ctypes.get_last_error() or errno.EIO, "native handle could not be duplicated")
    return duplicate


def _windows_close_native(native: Any) -> None:
    """Close one duplicated raw Windows handle exactly once."""
    if os.name != "nt":
        raise RuntimeError("Windows handle close requested on POSIX")
    import ctypes.wintypes as wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    if native and not close_handle(native):
        raise OSError(ctypes.get_last_error() or errno.EIO, "native handle close failed")


def _windows_dispose_native(native: Any) -> None:
    """Mark one raw Windows file handle for deletion, never by pathname."""
    if os.name != "nt":
        raise RuntimeError("Windows file disposition requested on POSIX")
    import ctypes.wintypes as wintypes

    class IOStatusBlock(ctypes.Structure):
        _fields_ = [("Status", ctypes.c_long), ("Information", ctypes.c_size_t)]

    class FileDispositionInformation(ctypes.Structure):
        _fields_ = [("DeleteFile", wintypes.BOOLEAN)]

    ntdll = ctypes.WinDLL("ntdll", use_last_error=False)
    set_information = ntdll.NtSetInformationFile
    set_information.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(IOStatusBlock),
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    set_information.restype = ctypes.c_long
    information = FileDispositionInformation(1)
    io_status = IOStatusBlock()
    status = ctypes.c_ulong(
        set_information(
            native,
            ctypes.byref(io_status),
            ctypes.byref(information),
            ctypes.sizeof(information),
            13,  # FileDispositionInformation
        )
    ).value
    if status in (0, 0xC0000056, 0xC0000123):  # STATUS_SUCCESS / DELETE_PENDING / FILE_DELETED
        return
    if status in (0xC0000022, 0xC0000043, 0xC0000121):
        raise PermissionError(errno.EACCES, "owned partial cannot be deleted")
    if status in (0xC0000034, 0xC000003A):
        raise FileNotFoundError(errno.ENOENT, "owned partial is unavailable")
    raise OSError(errno.EIO, "native owned-partial disposition failed")


def _windows_dispose_exact(handle: Any) -> None:
    """Mark the object behind ``handle`` for deletion, never by pathname."""
    if os.name != "nt":
        raise RuntimeError("Windows file disposition requested on POSIX")
    import msvcrt

    native = msvcrt.get_osfhandle(handle.fileno())
    if native == -1:
        raise OSError(errno.EBADF, "partial handle is closed")
    _windows_dispose_native(native)


_RENAME_NOREPLACE = 1
_RENAME_EXCHANGE = 2


def _posix_renameat2(parent_fd: int, source_name: str, target_name: str, flags: int) -> None:
    """Perform one same-parent atomic rename operation or fail closed."""
    if os.name == "nt":
        raise RuntimeError("POSIX renameat2 requested on Windows")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError) as exc:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable") from exc
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        int(parent_fd),
        os.fsencode(source_name),
        int(parent_fd),
        os.fsencode(target_name),
        int(flags),
    )
    if result == 0:
        return
    error = ctypes.get_errno() or errno.EIO
    if error == errno.EEXIST and flags == _RENAME_NOREPLACE:
        raise FileExistsError(error, "rename target already exists", target_name)
    if error in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
        raise OSError(errno.ENOSYS, "requested renameat2 operation is unavailable")
    raise OSError(error, "same-parent rename failed", source_name)


def _posix_child_stat(parent_fd: int, name: str) -> os.stat_result:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(name, flags, dir_fd=parent_fd)
    try:
        return os.fstat(fd)
    finally:
        os.close(fd)


def _posix_same_regular_object(observed: os.stat_result, expected: os.stat_result) -> bool:
    return (
        stat.S_ISREG(observed.st_mode)
        and stat.S_ISREG(expected.st_mode)
        and int(observed.st_dev) == int(expected.st_dev)
        and int(observed.st_ino) == int(expected.st_ino)
        and int(observed.st_size) == int(expected.st_size)
    )


def _posix_unique_absent_name(parent_fd: int, source_name: str) -> str:
    for _attempt in range(32):
        candidate = f".exclusive-sink-tombstone-{secrets.token_hex(16)}.tmp"
        if candidate == source_name:
            continue
        try:
            _posix_child_stat(parent_fd, candidate)
        except FileNotFoundError:
            return candidate
    raise DownloadError("owned partial tombstone name collision")


def _posix_move_placeholder(parent_fd: int, source_name: str, expected: os.stat_result) -> bool:
    """Move only a verified placeholder, restoring any foreign rebind."""
    destination = _posix_unique_absent_name(parent_fd, source_name)
    try:
        _posix_renameat2(parent_fd, source_name, destination, _RENAME_NOREPLACE)
    except FileNotFoundError:
        return False
    moved: os.stat_result | None = None
    try:
        moved = _posix_child_stat(parent_fd, destination)
        if _posix_same_regular_object(moved, expected):
            try:
                _posix_child_stat(parent_fd, source_name)
            except FileNotFoundError:
                return True
            return False
    except (FileNotFoundError, OSError):
        pass

    # The source may have been rebound in the final window.  Restore whatever
    # was moved, without replacing a new occupant.  If restoration is not
    # provable, preserve both names and report failure.
    try:
        _posix_renameat2(parent_fd, destination, source_name, _RENAME_NOREPLACE)
    except (FileExistsError, FileNotFoundError, OSError):
        return False
    return False


def _posix_dispose_exact(handle: Any, path: Path) -> None:
    """Logically remove one exact object through a tombstone exchange."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    parent_fd = os.open(path.parent, flags)
    tombstone_name: str | None = None
    tombstone_fd: int | None = None
    try:
        expected = os.fstat(handle.fileno())
        if not stat.S_ISREG(expected.st_mode):
            raise DownloadError("owned partial is not a regular file")
        tombstone_name = _posix_unique_absent_name(parent_fd, path.name)
        tombstone_fd = os.open(
            tombstone_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_fd,
        )
        placeholder = os.fstat(tombstone_fd)
        _posix_renameat2(parent_fd, path.name, tombstone_name, _RENAME_EXCHANGE)
        displaced = _posix_child_stat(parent_fd, tombstone_name)
        if not _posix_same_regular_object(displaced, expected):
            current_placeholder = _posix_child_stat(parent_fd, path.name)
            if _posix_same_regular_object(current_placeholder, placeholder):
                _posix_renameat2(parent_fd, path.name, tombstone_name, _RENAME_EXCHANGE)
            raise DownloadError("owned partial pathname was replaced")
        if not _posix_move_placeholder(parent_fd, path.name, placeholder):
            raise DownloadError("owned partial placeholder could not be removed safely")
        os.fsync(parent_fd)
    except DownloadError:
        raise
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise DownloadError("owned partial cleanup failed closed") from exc
    finally:
        if tombstone_fd is not None:
            os.close(tombstone_fd)
        os.close(parent_fd)


def _dispose_exact(handle: Any, path: Path) -> None:
    if os.name == "nt":
        _windows_dispose_exact(handle)
    else:
        _posix_dispose_exact(handle, path)


class ExclusiveSink:
    """A lock-protected ``open|abort_requested|closed|aborted`` sink."""

    __slots__ = (
        "_path",
        "_parent",
        "_boundary",
        "_parent_handle",
        "_stage_name",
        "_identity",
        "_native_created",
        "_file",
        "_lock",
        "_state",
        "_worker_done",
        "_bytes",
        "_abort_disposed",
    )

    def __init__(
        self,
        partial_path: os.PathLike[str] | str,
        *,
        boundary: Any | None = None,
        parent_handle: Any | None = None,
        created_handle: Any | None = None,
        created_identity: Any | None = None,
    ) -> None:
        path = Path(partial_path)
        if not path.is_absolute() or path.name in ("", ".", ".."):
            raise DownloadError("partial path is invalid")
        if parent_handle is None:
            if path.parent.exists() and path.parent.is_symlink():
                raise DownloadError("partial parent is a reparse path")
            if not path.parent.is_dir():
                raise DownloadError("partial parent is unavailable")
        elif boundary is None and created_handle is None:
            raise DownloadError("pinned parent requires a path boundary")
        self._path = path
        self._parent = path.parent
        self._boundary = boundary
        self._parent_handle = parent_handle
        self._stage_name = path.name
        self._identity = created_identity
        self._native_created = None
        self._lock = threading.RLock()
        self._state = "open"
        self._worker_done = False
        self._bytes = 0
        self._abort_disposed = False
        self._file = None
        try:
            if parent_handle is not None:
                if created_handle is None:
                    creator = getattr(boundary, "create_exclusive_child", None)
                    if not callable(creator):
                        creator = getattr(boundary, "_create_exclusive_child", None)
                    if not callable(creator):
                        raise DownloadError("pinned child creation is unavailable")
                    created_handle, created_identity = creator(
                        parent_handle,
                        path.parent,
                        path.name,
                    )
                    self._identity = created_identity
                self._file = self._adopt_created_handle(created_handle)
                if os.name != "nt" and not hasattr(created_handle, "write"):
                    self._native_created = created_handle
            else:
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
                if os.name == "nt":
                    self._file = _windows_create_exclusive(path)
                else:
                    fd = os.open(path, flags, 0o600)
                    self._file = os.fdopen(fd, "wb", buffering=0)
        except FileExistsError:
            self._close_parent_handle()
            raise
        except DownloadError:
            cleanup_error = self._cleanup_failed_created_handle(created_handle)
            self._close_parent_handle()
            if cleanup_error is not None:
                raise DownloadError("partial cleanup failed") from cleanup_error
            raise
        except OSError as exc:
            cleanup_error = self._cleanup_failed_created_handle(created_handle)
            self._close_parent_handle()
            if cleanup_error is not None:
                raise DownloadError("partial cleanup failed") from cleanup_error
            raise DownloadError("partial could not be created exclusively") from exc
        except Exception as exc:
            cleanup_error = self._cleanup_failed_created_handle(created_handle)
            self._close_parent_handle()
            if cleanup_error is not None:
                raise DownloadError("partial cleanup failed") from cleanup_error
            raise DownloadError("partial could not be adopted") from exc
        if os.name != "nt" and parent_handle is None:
            try:
                os.fchmod(self._file.fileno(), stat.S_IRUSR | stat.S_IWUSR)
            except OSError as exc:
                try:
                    _dispose_exact(self._file, path)
                except Exception:
                    pass
                try:
                    self._file.close()
                except Exception:
                    pass
                self._file = None
                raise DownloadError("partial permissions could not be secured") from exc

    def _adopt_created_handle(self, handle: Any) -> Any:
        """Wrap a boundary-created native handle without replacing its object."""
        if hasattr(handle, "write") and hasattr(handle, "fileno"):
            return handle
        if os.name == "nt":
            import msvcrt

            try:
                raw = getattr(handle.raw, "value", handle.raw)
                if raw is None:
                    raise OSError(errno.EBADF, "created handle is unavailable")
                cleanup_native = _windows_duplicate_native(raw)
            except Exception:
                raise
            fd = None
            wrapped = None
            try:
                fd = msvcrt.open_osfhandle(int(raw), os.O_WRONLY | getattr(os, "O_BINARY", 0))
            except Exception:
                try:
                    _windows_close_native(cleanup_native)
                except Exception as exc:
                    raise DownloadError("partial cleanup failed") from exc
                raise
            try:
                # The CRT fd now owns the original raw handle.  The duplicate
                # remains the exact cleanup authority until wrapping succeeds.
                if hasattr(handle, "_closed"):
                    handle._closed = True
                wrapped = os.fdopen(fd, "wb", buffering=0)
                _windows_close_native(cleanup_native)
                cleanup_native = None
                return wrapped
            except Exception:
                cleanup_error: Exception | None = None
                try:
                    _windows_dispose_native(cleanup_native)
                except Exception as exc:
                    cleanup_error = exc
                finally:
                    try:
                        _windows_close_native(cleanup_native)
                    except Exception as exc:
                        if cleanup_error is None:
                            cleanup_error = exc
                    if wrapped is not None:
                        try:
                            wrapped.close()
                        except Exception as exc:
                            if cleanup_error is None:
                                cleanup_error = exc
                    elif fd is not None:
                        try:
                            os.close(fd)
                        except OSError as exc:
                            if cleanup_error is None:
                                cleanup_error = exc
                if cleanup_error is not None:
                    raise DownloadError("partial cleanup failed") from cleanup_error
                raise
        fd = os.dup(int(handle.raw))
        try:
            return os.fdopen(fd, "wb", buffering=0)
        except Exception:
            os.close(fd)
            raise

    def _cleanup_failed_created_handle(self, handle: Any | None) -> Exception | None:
        if handle is None or self._parent_handle is None or self._identity is None:
            return None
        cleanup_error: Exception | None = None
        if not bool(getattr(handle, "closed", False)):
            disposer = getattr(self._boundary, "dispose_exclusive_child", None)
            if not callable(disposer):
                disposer = getattr(self._boundary, "_dispose_exclusive_child", None)
            if callable(disposer):
                try:
                    disposer(
                        self._parent_handle,
                        self._parent,
                        self._stage_name,
                        handle,
                        self._identity,
                        max_bytes=1 << 30,
                    )
                except Exception as exc:
                    cleanup_error = exc
            elif os.name == "nt":
                try:
                    _windows_dispose_native(handle.raw)
                except Exception as exc:
                    cleanup_error = exc
            else:
                cleanup_error = DownloadError("pinned child cleanup is unavailable")
        close = getattr(handle, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        return cleanup_error

    def _close_parent_handle(self) -> None:
        handle = self._parent_handle
        if handle is None:
            return
        self._parent_handle = None
        close = getattr(handle, "close", None)
        if callable(close):
            close()

    def _close_native_created(self) -> None:
        handle = self._native_created
        if handle is None:
            return
        self._native_created = None
        handle.close()

    def _flush_file(self, handle: Any) -> None:
        flush = getattr(self._boundary, "flush_private_file", None)
        if not callable(flush):
            flush = getattr(self._boundary, "_flush_private_file", None)
        if callable(flush):
            flush(handle.fileno())
        else:
            os.fsync(handle.fileno())

    def _flush_parent_handle(self) -> None:
        if self._parent_handle is None:
            _flush_parent(self._parent)
            return
        flush = getattr(self._boundary, "fsync_directory_handle", None)
        if not callable(flush):
            flush = getattr(self._boundary, "_fsync_directory_handle", None)
        if callable(flush):
            flush(self._parent_handle)
            return
        # Explicit test seams may provide a pinned-looking handle without a
        # directory flush primitive; retain the historical seam rather than
        # silently changing its contract.
        _flush_parent(self._parent)

    def _dispose_owned(self, handle: Any) -> bool:
        if self._parent_handle is not None and self._boundary is not None and self._identity is not None:
            disposer = getattr(self._boundary, "dispose_exclusive_child", None)
            if not callable(disposer):
                disposer = getattr(self._boundary, "_dispose_exclusive_child", None)
            if not callable(disposer):
                raise DownloadError("pinned child cleanup is unavailable")
            return bool(
                disposer(
                    self._parent_handle,
                    self._parent,
                    self._stage_name,
                    handle,
                    self._identity,
                    expected_sha256=None,
                    max_bytes=1 << 30,
                )
            )
        _dispose_exact(handle, self._path)
        return True

    @property
    def partial_path(self) -> Path:
        """Internal path for the owning coordinator; never provider input."""
        return self._path

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def bytes_written(self) -> int:
        with self._lock:
            return self._bytes

    def _ensure_open(self) -> Any:
        if self._state != "open" or self._file is None:
            raise DownloadError("sink is not writable")
        return self._file

    def write(self, chunk: bytes | bytearray | memoryview) -> None:
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise DownloadError("download chunk is not bytes")
        view = memoryview(chunk)
        with self._lock:
            handle = self._ensure_open()
            remaining = view
            written_total = 0
            try:
                while remaining:
                    written = handle.write(remaining)
                    if written is None:
                        written = len(remaining)
                    if not isinstance(written, int) or written <= 0:
                        raise OSError("short partial write")
                    written_total += written
                    remaining = remaining[written:]
                self._bytes += written_total
            except Exception as exc:
                raise DownloadError("partial write failed") from exc

    def flush_and_close(self) -> None:
        with self._lock:
            handle = self._ensure_open()
            try:
                handle.flush()
                self._flush_file(handle)
                # Keep both the exact file object and the pinned parent alive
                # until the directory durability boundary has completed.
                self._flush_parent_handle()
                handle.close()
                self._file = None
                self._close_native_created()
                self._close_parent_handle()
            except Exception as exc:
                # In particular, a parent flush failure must leave the exact
                # file handle available for abort_owned().
                self._state = "abort_requested"
                if isinstance(exc, DownloadError):
                    raise
                raise DownloadError("partial flush failed") from exc
            self._state = "closed"

    def request_abort(self) -> None:
        with self._lock:
            if self._state in {"closed", "aborted"}:
                return
            self._state = "abort_requested"

    def worker_finished(self) -> None:
        with self._lock:
            self._worker_done = True

    def abort_owned(self) -> None:
        """Delete only this sink after its worker has definitively stopped."""
        with self._lock:
            if not self._worker_done:
                raise DownloadError("sink worker has not finished")
            if self._state in {"closed", "aborted"}:
                return
            if self._state == "open":
                self._state = "abort_requested"
            handle = self._file
            if handle is None:
                raise DownloadError("owned partial handle is unavailable")
            try:
                if not self._abort_disposed:
                    if not self._dispose_owned(handle):
                        raise DownloadError("owned partial identity changed")
                    self._abort_disposed = True
                # Keep the exact file handle alive through the parent flush.
                # A failed parent flush can then be retried without losing the
                # only handle that proves ownership of the disposed object.
                self._flush_parent_handle()
                handle.close()
                self._file = None
                self._close_native_created()
                self._close_parent_handle()
            except Exception as exc:
                self._state = "abort_requested"
                if not self._abort_disposed:
                    try:
                        handle.close()
                    finally:
                        self._file = None
                        self._close_native_created()
                        self._close_parent_handle()
                if isinstance(exc, DownloadError):
                    raise
                raise DownloadError("owned partial cleanup failed") from exc
            self._state = "aborted"

    def digest(self) -> DownloadDigest:
        with self._lock:
            if self._state != "closed":
                raise DownloadError("digest requested before close")
            digest = hashlib.sha256()
            try:
                with self._path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
                        digest.update(chunk)
            except OSError as exc:
                raise DownloadError("closed partial could not be hashed") from exc
            return DownloadDigest(digest.hexdigest(), self._bytes)



def stream_to_sink(url: str, *, client: Any, sink: ExclusiveSink, min_bytes: int = 65536, max_bytes: int = 1 << 30) -> DownloadDigest:
    """Small adapter for providers whose client exposes ``stream_url``."""
    if hasattr(client, "stream_url"):
        return client.stream_url(url, sink=sink, min_bytes=min_bytes, max_bytes=max_bytes)
    raise DownloadError("provider client has no bounded stream operation")


__all__ = ["DownloadDigest", "DownloadError", "ExclusiveSink", "stream_to_sink"]
