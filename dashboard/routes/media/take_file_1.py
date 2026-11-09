from __future__ import annotations

# Lane w24-rest (VID-030/VID-035): a Contract-4 row never carries published_path; its file is the one its private
# operation record published (_shoot_take_ledger, the resolver the player and poster use). has_file read
# bool(published_path) and so said False for every V4 take - the two Hermes-made H3 takes of the artist's window. A
# verified path is kept per job with its size and mtime: a later listing re-checks only those (one stat), never re-hashes.
_V4_TAKE_FILES: dict[str, tuple[str, int, int]] = {}


def _media_take_has_file(record: Any) -> bool:
    if getattr(record, "published_path", None):
        return True
    if getattr(record, "row_contract", None) != 4 or getattr(record, "state", None) != "ready":
        return False
    job_id = record.studio_job_id
    known = _V4_TAKE_FILES.get(job_id)
    if known is not None:
        try:
            st = os.stat(known[0])
            if (st.st_size, st.st_mtime_ns) == known[1:]:
                return True
        except OSError:
            pass
        _V4_TAKE_FILES.pop(job_id, None)
    try:
        ops = _private_ops()
        path = ops.saved_path_for_receipt(ops.load_for_job(record))
        st = os.stat(path)
    except Exception:  # noqa: BLE001 - no verified file: none
        return False
    if len(_V4_TAKE_FILES) >= 1024:
        _V4_TAKE_FILES.pop(next(iter(_V4_TAKE_FILES)))
    _V4_TAKE_FILES[job_id] = (str(path), st.st_size, st.st_mtime_ns)
    return True
