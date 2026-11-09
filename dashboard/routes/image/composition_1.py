from __future__ import annotations


def _image_film_profile_key(home: Path | None) -> Any | None:
    composition_root = importlib.import_module(
        f"{_STUDIO_CORE_PACKAGE_NAME}.image_core.composition_root"
    )
    try:
        return composition_root._profile_anchor_for_home(home)
    except Exception:
        raise composition_root.ImageFilmCompositionError("image_film_unavailable") from None


def _drop_image_film_root(root: Any, composition_root: Any) -> None:
    global _IMAGE_FILM_ROOT, _IMAGE_FILM_ROOT_PROFILE
    if _IMAGE_FILM_ROOT is root:
        _IMAGE_FILM_ROOT = None
        _IMAGE_FILM_ROOT_PROFILE = None
    try:
        root.close()
    except Exception as exc:
        if isinstance(exc, composition_root.ImageFilmCompositionError):
            raise
        raise composition_root.ImageFilmCompositionError("image_film_unavailable") from None


def _image_film_root() -> Any:
    """Lazily return the one trusted Image/Film root for the active profile."""
    global _IMAGE_FILM_ROOT, _IMAGE_FILM_ROOT_PROFILE
    with _IMAGE_FILM_ROOT_LOCK:
        composition_root = importlib.import_module(
            f"{_STUDIO_CORE_PACKAGE_NAME}.image_core.composition_root"
        )
        composition_root._bind_composition_lock(_IMAGE_FILM_ROOT_LOCK)
        cached = _IMAGE_FILM_ROOT
        candidate: Any | None = None
        cleaned: set[int] = set()

        def discard(root: Any | None, *, preserve_control: bool) -> None:
            global _IMAGE_FILM_ROOT, _IMAGE_FILM_ROOT_PROFILE
            if root is None or id(root) in cleaned:
                return
            cleaned.add(id(root))
            if _IMAGE_FILM_ROOT is root:
                _IMAGE_FILM_ROOT = None
                _IMAGE_FILM_ROOT_PROFILE = None
            try:
                root.close()
            except BaseException as cleanup_error:
                if preserve_control:
                    return
                if isinstance(cleanup_error, Exception):
                    raise composition_root.ImageFilmCompositionError(
                        "image_film_unavailable"
                    ) from None
                raise

        try:
            # Every resolver/anchor observation is inside this finite boundary.
            requested = composition_root._stable_active_profile_snapshot(_active_hermes_home)
            if requested is None:
                discard(cached, preserve_control=False)
                raise composition_root.ImageFilmCompositionError("image_film_unavailable")
            requested_path = requested.path
            requested_anchor = requested.anchor
            if requested_path is not None and requested_anchor is None:
                if cached is not None:
                    if Path(requested_path).exists():
                        discard(cached, preserve_control=False)
                        raise composition_root.ImageFilmCompositionError("image_film_unavailable")
                    discard(cached, preserve_control=False)
                    cached = None
                if Path(requested_path).exists():
                    raise composition_root.ImageFilmCompositionError("image_film_unavailable")
            if cached is not None:
                requested_matches_cached = (
                    requested_path == getattr(cached._profile_anchor, "path", None)
                    and composition_root._same_profile_anchor(cached._profile_anchor, requested_anchor)
                )
                if not requested_matches_cached:
                    discard(cached, preserve_control=False)
                    cached = None
                else:
                    cached._require_open()
                    final = composition_root._stable_active_profile_snapshot(_active_hermes_home)
                    if (
                        final is None
                        or final.path != requested_path
                        or not composition_root._same_profile_anchor(cached._profile_anchor, final.anchor)
                    ):
                        discard(cached, preserve_control=False)
                        raise composition_root.ImageFilmCompositionError("image_film_unavailable")
                    _IMAGE_FILM_ROOT_PROFILE = final.anchor
                    return cached

            candidate = composition_root.ImageFilmCompositionRoot.open()
            if (
                requested_path is None
                or requested_path != getattr(candidate._profile_anchor, "path", None)
                or (
                    requested_anchor is not None
                    and not composition_root._same_profile_anchor(requested_anchor, candidate._profile_anchor)
                )
            ):
                discard(candidate, preserve_control=False)
                candidate = None
                raise composition_root.ImageFilmCompositionError("image_film_unavailable")
            final = composition_root._stable_active_profile_snapshot(_active_hermes_home)
            if (
                final is None
                or final.path != requested_path
                or not composition_root._same_profile_anchor(candidate._profile_anchor, final.anchor)
            ):
                discard(candidate, preserve_control=False)
                candidate = None
                raise composition_root.ImageFilmCompositionError("image_film_unavailable")
            _IMAGE_FILM_ROOT = candidate
            _IMAGE_FILM_ROOT_PROFILE = final.anchor
            return candidate
        except composition_root.ImageFilmCompositionError:
            discard(candidate, preserve_control=False)
            discard(cached, preserve_control=False)
            raise
        except Exception as exc:
            discard(candidate, preserve_control=False)
            discard(cached, preserve_control=False)
            raise composition_root._map_exception(
                exc,
                default="image_film_unavailable",
            ) from None
        except BaseException:
            discard(candidate, preserve_control=True)
            discard(cached, preserve_control=True)
            raise


def _initialized_image_film_root() -> Any:
    """Return the already initialized profile root without bootstrapping it."""
    composition_root = importlib.import_module(
        f"{_STUDIO_CORE_PACKAGE_NAME}.image_core.composition_root"
    )
    with _IMAGE_FILM_ROOT_LOCK:
        composition_root._bind_composition_lock(_IMAGE_FILM_ROOT_LOCK)
        root = _IMAGE_FILM_ROOT
        if root is None:
            raise composition_root.ImageFilmCompositionError("image_film_unavailable")
        try:
            root._require_open()
        except composition_root.ImageFilmCompositionError:
            raise
        except Exception as exc:
            raise composition_root._map_exception(
                exc,
                default="image_film_unavailable",
            ) from None
        return root
