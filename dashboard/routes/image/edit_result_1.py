from __future__ import annotations


# Lane w29-fix2 (P1 IMG-221; the owner's 2912 x 2912 Sky restyle, 3/3: the result GET never answered within the desktop's
# 60 s and every repeat started another stitch): one build per variant at a time (a second GET, Regenerate's repeat and
# the panel's Retry wait for the same one), the built answer kept on disk next to the edit record (served at once after a
# restart), and a GET that waits at most _EDIT_RESULT_WAIT_SECONDS: past that it answers 202 result_building (retryable)
# and the build goes on. Reference: Lightroom / Photoshop render a heavy result once in the background and show it when
# ready; a viewer never blocks the app.
_EDIT_RESULT_BUILDS: dict[str, "asyncio.Task[Any]"] = {}
_EDIT_RESULT_WAIT_SECONDS = 40.0


def _edit_result_remember(cache_key: str, body: dict[str, Any]) -> None:
    _EDIT_RESULT_CACHE[cache_key] = body
    while len(_EDIT_RESULT_CACHE) > _EDIT_RESULT_CACHE_MAX:
        _EDIT_RESULT_CACHE.popitem(last=False)


async def _edit_result_shared(cache_key: str, build: Any, store: Any) -> Any:
    """IMG-221: the answer kept in memory or on disk, else one build per variant; the GET waits for it at most
    _EDIT_RESULT_WAIT_SECONDS (then 202 result_building, retryable)."""
    cached = _EDIT_RESULT_CACHE.get(cache_key)
    if cached is not None:
        _EDIT_RESULT_CACHE.move_to_end(cache_key)
        return cached
    kept = await asyncio.to_thread(store.result, cache_key)
    if kept is not None:
        _edit_result_remember(cache_key, kept)
        return kept
    task = _EDIT_RESULT_BUILDS.get(cache_key)
    if task is None or task.done() and task.exception() is not None:
        async def run() -> Any:
            body = await build()
            if isinstance(body, dict) and body.get("schema") == "studio.image-edit-result/1":
                _edit_result_remember(cache_key, body)
                await asyncio.to_thread(store.put_result, cache_key, body)
            return body
        task = _EDIT_RESULT_BUILDS[cache_key] = asyncio.ensure_future(run())
        task.add_done_callback(lambda t, k=cache_key: _EDIT_RESULT_BUILDS.pop(k, None) if _EDIT_RESULT_BUILDS.get(k) is t else None)
    try:
        return await asyncio.wait_for(asyncio.shield(task), _EDIT_RESULT_WAIT_SECONDS)
    except asyncio.TimeoutError:
        return JSONResponse(status_code=202, content={"error": {"code": "result_building", "message": "The result is still being put together", "retryable": True}})
