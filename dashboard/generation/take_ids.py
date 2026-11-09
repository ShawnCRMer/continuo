"""The ledger's Shoot take id, the one definition every reader validates against.

A take id is ``studio:`` + a UUID4 in either spelling the ledger mints today:
V4 submits (``routes/generation/managed_submit_1.py``, ``gmi_submit_1.py``)
write the 32-hex form, Contract-3 rows (``job_store.create_or_get_job``) the
36-character dashed form. Both already exist in artists' ledgers, so the media
poster, the media marks and the timeline resolver accept both and nothing else.

Pure (``re`` only) and loaded by path, never as a package member: the timeline
core and the media routes each exec this one file (the generation package's
loaded module tree is pinned by tests/backend/test_route_loader.py).
"""
from __future__ import annotations

import re

STUDIO_JOB_ID_BODY = r"studio:(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
STUDIO_JOB_ID_PATTERN = "^" + STUDIO_JOB_ID_BODY + "$"
STUDIO_JOB_ID_RE = re.compile(STUDIO_JOB_ID_PATTERN)

__all__ = ["STUDIO_JOB_ID_BODY", "STUDIO_JOB_ID_PATTERN", "STUDIO_JOB_ID_RE"]
