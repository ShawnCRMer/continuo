from __future__ import annotations


def _active_hermes_home() -> Path | None:
    from hermes_constants import get_hermes_home

    value = get_hermes_home()
    return None if value is None else Path(value)
