from __future__ import annotations


def prepared_cache_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    directory = state_root(Path(base) / "hermes") / "prepared"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def prepared_cache_key(
    source: Path,
    in_frame: int,
    out_frame: int,
    fps: float,
    source_sha256: str | None = None,
) -> str:
    if source_sha256 is None:
        stat = source.stat()
        source_identity = f"{stat.st_mtime_ns}|{stat.st_size}"
    else:
        source_identity = source_sha256
    material = json.dumps(
        {
            "version": "prepare-byte-snapshot-v2",
            "source_path": os.path.normcase(os.path.realpath(source)),
            "source_sha256": source_identity,
            "in_frame": in_frame,
            "out_frame": out_frame,
            "fps": format(float(fps), ".12g"),
            "encoder_profile": "libx264-crf16-fast-yuv420p-aac",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def evict_prepared_cache(directory: Path, protected: set[Path] | None = None, now: float | None = None) -> None:
    protected_paths = {path.resolve() for path in (protected or set())}
    stamp = time.time() if now is None else now
    maintenance_key = hashlib.sha256(b"prepare-cache-maintenance-v1").hexdigest()
    maintenance_lock = PrepareKeyLock(directory, maintenance_key)
    maintenance_lock.acquire_blocking()

    def candidate_lock(candidate: Path) -> tuple[bool, PrepareKeyLock | None]:
        if not re.fullmatch(r"[0-9a-f]{64}", candidate.stem):
            return True, None
        lock = PrepareKeyLock(directory, candidate.stem, wait_seconds=0)
        try:
            lock.acquire_blocking()
        except RecipeError:
            return False, None
        return True, lock

    try:
        entries: list[tuple[Path, os.stat_result]] = []
        for candidate in directory.glob("*.mp4"):
            try:
                candidate_stat = candidate.stat()
            except OSError:
                continue
            if candidate.is_file():
                entries.append((candidate, candidate_stat))

        for candidate, candidate_stat in entries:
            if candidate.resolve() in protected_paths:
                continue
            acquired, lock = candidate_lock(candidate)
            if not acquired:
                continue
            try:
                try:
                    current_stat = candidate.stat()
                except OSError:
                    continue
                if stamp - current_stat.st_mtime > PREPARED_CACHE_MAX_AGE_SECONDS:
                    try:
                        candidate.unlink()
                    except OSError:
                        pass
            finally:
                if lock is not None:
                    lock.release_blocking()

        remaining: list[tuple[Path, os.stat_result]] = []
        for candidate in directory.glob("*.mp4"):
            try:
                candidate_stat = candidate.stat()
            except OSError:
                continue
            if candidate.is_file():
                remaining.append((candidate, candidate_stat))
        total = sum(candidate_stat.st_size for _, candidate_stat in remaining)
        for candidate, _candidate_stat in sorted(
            remaining,
            key=lambda item: (item[1].st_atime_ns, item[1].st_mtime_ns),
        ):
            if total <= PREPARED_CACHE_MAX_BYTES or candidate.resolve() in protected_paths:
                continue
            acquired, lock = candidate_lock(candidate)
            if not acquired:
                continue
            try:
                try:
                    current_stat = candidate.stat()
                except OSError:
                    continue
                if total <= PREPARED_CACHE_MAX_BYTES:
                    break
                if stamp - current_stat.st_mtime < PREPARED_CACHE_MIN_RESIDENCY_SECONDS:
                    continue
                try:
                    candidate.unlink()
                    total -= current_stat.st_size
                except OSError:
                    continue
            finally:
                if lock is not None:
                    lock.release_blocking()
    finally:
        maintenance_lock.release_blocking()


class PrepareKeyLock:
    def __init__(self, cache_dir: Path, cache_key: str, wait_seconds: float = 190.0):
        if not re.fullmatch(r"[0-9a-f]{64}", cache_key):
            raise RecipeError("Prepared cache identity is invalid.")
        self.path: Path | None = None
        self._mutex_name: str | None = None
        if os.name == "nt":
            lexical_domain = os.path.normcase(os.path.abspath(os.path.normpath(str(cache_dir))))
            identity = hashlib.sha256(f"{lexical_domain}|{cache_key}".encode("utf-8")).hexdigest()
            self._mutex_name = f"Local\\HermesFlux3.{identity}"
        else:
            lock_dir = cache_dir / "locks"
            lock_dir.mkdir(parents=True, exist_ok=True)
            self.path = lock_dir / f"{cache_key}.lock"
        self.wait_seconds = max(0.0, float(wait_seconds))
        self._handle: Any | None = None

    def _open(self) -> Any:
        if os.name == "nt":
            assert self._mutex_name is not None
            handle = _CREATE_MUTEX(None, False, self._mutex_name)
            if not handle:
                error = ctypes.get_last_error()
                raise OSError(error, "Could not create the media publication mutex.")
            return handle

        assert self.path is not None
        if os.path.lexists(self.path):
            before = os.lstat(self.path)
            attributes = int(getattr(before, "st_file_attributes", 0))
            reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
            if stat.S_ISLNK(before.st_mode) or attributes & reparse_flag:
                raise RecipeError("Prepared cache lock is not a regular local file.")
        handle = self.path.open("a+b", buffering=0)
        after = os.fstat(handle.fileno())
        current = os.lstat(self.path)
        if not stat.S_ISREG(after.st_mode) or (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino):
            handle.close()
            raise RecipeError("Prepared cache lock identity changed while opening.")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        return handle

    def _try_lock(self) -> bool:
        assert self._handle is not None
        if os.name == "nt":
            result = int(_WAIT_FOR_SINGLE_OBJECT(self._handle, 0))
            if result in (_WAIT_OBJECT_0, _WAIT_ABANDONED):
                return True
            if result == _WAIT_TIMEOUT:
                return False
            if result == _WAIT_FAILED:
                error = ctypes.get_last_error()
                raise OSError(error, "Could not wait for the media publication mutex.")
            raise OSError(f"Unexpected media publication mutex wait result: {result}.")

        self._handle.seek(0)
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                return False
            raise

    def _close_unowned_handle(self, handle: Any) -> None:
        if os.name == "nt":
            if not _CLOSE_HANDLE(handle):
                error = ctypes.get_last_error()
                raise OSError(error, "Could not close the media publication mutex.")
        else:
            handle.close()

    async def acquire(self) -> "PrepareKeyLock":
        if self._handle is not None:
            return self
        self._handle = self._open()
        deadline = time.monotonic() + self.wait_seconds
        try:
            while not self._try_lock():
                if time.monotonic() >= deadline:
                    raise RecipeError("Media preparation is already in progress.")
                await asyncio.sleep(0.025)
        except BaseException:
            handle = self._handle
            self._handle = None
            if handle is not None:
                self._close_unowned_handle(handle)
            raise
        return self

    def acquire_blocking(self) -> "PrepareKeyLock":
        if self._handle is not None:
            return self
        self._handle = self._open()
        deadline = time.monotonic() + self.wait_seconds
        try:
            while not self._try_lock():
                if time.monotonic() >= deadline:
                    raise RecipeError("Media publication is already in progress.")
                time.sleep(0.025)
        except BaseException:
            handle = self._handle
            self._handle = None
            if handle is not None:
                self._close_unowned_handle(handle)
            raise
        return self

    def release_blocking(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        if os.name == "nt":
            try:
                if not _RELEASE_MUTEX(handle):
                    error = ctypes.get_last_error()
                    raise OSError(error, "Could not release the media publication mutex.")
            finally:
                _CLOSE_HANDLE(handle)
            return
        try:
            handle.seek(0)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    async def release(self) -> None:
        self.release_blocking()


class StampPublicationLocks:
    def __init__(self, media: Path, sidecar: Path, wait_seconds: float = 190.0):
        base = prepared_cache_dir().parent / "stamp-publication"
        resources = sorted(
            {
                os.path.normcase(os.path.realpath(media)),
                os.path.normcase(os.path.realpath(sidecar)),
            }
        )
        self._locks = [
            PrepareKeyLock(
                base,
                hashlib.sha256(resource.encode("utf-8")).hexdigest(),
                wait_seconds=wait_seconds,
            )
            for resource in resources
        ]
        self._acquired: list[PrepareKeyLock] = []

    def __enter__(self) -> "StampPublicationLocks":
        try:
            for lock in self._locks:
                lock.acquire_blocking()
                self._acquired.append(lock)
        except BaseException:
            for lock in reversed(self._acquired):
                lock.release_blocking()
            self._acquired.clear()
            raise
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        for lock in reversed(self._acquired):
            lock.release_blocking()
        self._acquired.clear()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _cleanup_stage_orphans(directory: Path, prefix: str, now: float | None = None) -> None:
    stamp = time.time() if now is None else now
    for candidate in directory.glob(f"{prefix}*"):
        try:
            age = stamp - candidate.stat().st_mtime
        except OSError:
            continue
        if age > MEDIA_STAGE_MAX_AGE_SECONDS:
            _safe_unlink(candidate)


def _copy_hashed_media(source: Path, stage_name: str, prefix: str) -> tuple[Path, str]:
    stage_dir = prepared_cache_dir() / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_stage_orphans(stage_dir, prefix)
    fd, raw_path = tempfile.mkstemp(
        prefix=prefix,
        suffix=source.suffix or ".mp4",
        dir=stage_dir,
    )
    staged = Path(raw_path)
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as target, source.open("rb") as source_handle:
            for block in iter(lambda: source_handle.read(1024 * 1024), b""):
                digest.update(block)
                target.write(block)
            target.flush()
            os.fsync(target.fileno())
        return staged, digest.hexdigest()
    except BaseException:
        _safe_unlink(staged)
        raise


def _stage_verified_media(source: Path, expected_hash: str) -> Path:
    """Copy and hash once so the provider uploads stable, verified bytes."""
    staged, actual_hash = _copy_hashed_media(source, "submit", ".submit-")
    if actual_hash != expected_hash:
        _safe_unlink(staged)
        raise RecipeError("The prepared continuation changed after preparation; prepare the source again.")
    return staged


def _snapshot_derivation_source(source: Path) -> tuple[Path, str]:
    """Return private stable bytes and the hash FFmpeg/ffprobe will consume."""
    return _copy_hashed_media(source, "derive", ".derive-")
