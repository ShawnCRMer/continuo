"""Backend facade for the model-agnostic Film Suite desktop plugin.

This is a direct UI-to-service adapter.  It does not invoke Hermes chat, an
agent loop, or a plugin LLM.  It delegates the managed authentication, local
media upload, validation-compatible request construction, polling, and secure
video download to Hermes' existing FLUX 3 service adapter.
"""

from __future__ import annotations

import base64
import _thread
import asyncio
import collections
import errno
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from types import MappingProxyType
from urllib.parse import unquote, urlparse

if os.name == "nt":
    import ctypes
    from ctypes import wintypes
else:
    import fcntl

# Stateless migration is also used by top-level, unregistered host loaders.
import importlib.util as _migration_import
_migration_spec = _migration_import.spec_from_file_location("film_suite_legacy_migration", Path(__file__).with_name("generation") / "legacy_plugin_migration.py")
_migration_module = _migration_import.module_from_spec(_migration_spec)
_migration_spec.loader.exec_module(_migration_module)
state_root = _migration_module.state_root

from fastapi import APIRouter, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.responses import JSONResponse, Response


# Keep the dashboard API as the public namespace and import/reload boundary.
# The area files contain the declarations; no route logic is defined here once
# extraction is complete. Use source filenames for tracing moved functions.
_routes_spec = importlib.util.spec_from_file_location(
    "film_suite_routes_composition", Path(__file__).with_name("routes") / "__init__.py"
)
if _routes_spec is None or _routes_spec.loader is None:
    raise RuntimeError("The local route composition package is unavailable.")
_routes_composition = importlib.util.module_from_spec(_routes_spec)
_routes_spec.loader.exec_module(_routes_composition)
_load_route_area = _routes_composition.load_area

_load_route_area("shared/bootstrap_1.py", globals())

_load_route_area("image/composition_1.py", globals())

_load_route_area("shared/bridges_1.py", globals())

_load_route_area("shared/validation_1.py", globals())

_load_route_area("projects/models_1.py", globals())

_load_route_area("image/producer_models_1.py", globals())

_load_route_area("projects/models_2.py", globals())

_load_route_area("shared/routers_1.py", globals())

_load_route_area("shared/errors_1.py", globals())

_load_route_area("shared/profile_1.py", globals())

_load_route_area("shared/events_1.py", globals())

_load_route_area("music/bridge_1.py", globals())

_load_route_area("image/bridge_1.py", globals())

_load_route_area("music/evidence_1.py", globals())

_load_route_area("projects/helpers_1.py", globals())

_load_route_area("generation/bootstrap_1.py", globals())

_load_route_area("generation/models_1.py", globals())

_load_route_area("media/models_1.py", globals())

_load_route_area("music/models_1.py", globals())

_load_route_area("shared/constants_1.py", globals())

_load_route_area("shared/paths_1.py", globals())

_load_route_area("music/cache_1.py", globals())

_load_route_area("shared/paths_2.py", globals())

_load_route_area("media/preparation_1.py", globals())

_load_route_area("generation/lineage_1.py", globals())

_load_route_area("media/probe_1.py", globals())

_load_route_area("media/routes_1.py", globals())

_load_route_area("generation/recipe_1.py", globals())

_load_route_area("shared/responses_1.py", globals())

_load_route_area("generation/recipe_routes_1.py", globals())

_load_route_area("music/analysis_1.py", globals())

_load_route_area("music/analysis_jobs_1.py", globals())

_load_route_area("generation/health_scope_1.py", globals())

_load_route_area("projects/reference_1.py", globals())

_load_route_area("music/interpretation_1.py", globals())

_load_route_area("generation/managed_media_1.py", globals())

_load_route_area("generation/h3_sound_1.py", globals())

_load_route_area("generation/gmi_submit_1.py", globals())

_load_route_area("generation/managed_submit_1.py", globals())

_load_route_area("generation/lifecycle_1.py", globals())

_load_route_area("generation/comfy_catalog_1.py", globals())

_load_route_area("image/source_catalog_1.py", globals())

_load_route_area("generation/model_catalog_bootstrap_1.py", globals())

_load_route_area("image/preview_bootstrap_1.py", globals())

_load_route_area("generation/model_catalog_1.py", globals())

_load_route_area("projects/routes_1.py", globals())

_load_route_area("image/producer_1.py", globals())

_load_route_area("image/preview_1.py", globals())

_load_route_area("image/mask_1.py", globals())

_load_route_area("image/edit_result_1.py", globals())  # w29-fix2 IMG-221: one shared result build per variant
_load_route_area("image/edit_1.py", globals())
_load_route_area("image/workflow_1.py", globals())
_load_route_area("image/import_1.py", globals())
_load_route_area("image/model_files_1.py", globals())
_load_route_area("image/loras_1.py", globals())
_load_route_area("image/edits_1.py", globals())
_load_route_area("image/upscale_1.py", globals())
_load_route_area("image/export_1.py", globals())

_load_route_area("generation/jobs_1.py", globals())
_load_route_area("generation/h3_lineage_1.py", globals())
_load_route_area("generation/h3_finish_1.py", globals())
_load_route_area("generation/video_upscale_1.py", globals())  # lane w23-vsr: RelayVSR video upscale

_load_route_area("media/feed_1.py", globals())

_load_route_area("projects/film_image_1.py", globals())

_load_route_area("projects/state_1.py", globals())

_load_route_area("projects/timeline_1.py", globals())

_load_route_area("projects/timeline_export_1.py", globals())

_load_route_area("projects/timeline_footage_1.py", globals())

_load_route_area("projects/timeline_audio_1.py", globals())

_load_route_area("projects/timeline_peaks_1.py", globals())

_load_route_area("projects/script_1.py", globals())

_load_route_area("projects/timeline_duck_1.py", globals())
# Deliver: check before render + render queue (lane w2-cutdel).
_load_route_area("projects/timeline_deliver_1.py", globals())

_load_route_area("media/sound_1.py", globals())

_load_route_area("shared/frame_1.py", globals())

_load_route_area("media/control_1.py", globals())
# Audio page, Create: speak one line (lane fs-voice).
_load_route_area("voice/voice_1.py", globals())
_load_route_area("generation/h3_controls_1.py", globals())  # H3 inspector controls + Generate x N (lane w2-h3ctl)
_load_route_area("image/box_1.py", globals())
_load_route_area("audio/stems_1.py", globals())  # Audio page stems (lane w3-audio)
_load_route_area("audio/takes_1.py", globals())  # Audio page takes: recording, words, re-cut, history (lane w4-audio)
_load_route_area("projects/review_1.py", globals())  # Take A/B review: comparisons, notes, decisions (lane w5-review)
_load_route_area("projects/timeline_qc_1.py", globals())  # File QC of the written movie in Deliver (lane w5-review)
_load_route_area("generation/job_authors_1.py", globals())  # who sent a job / wrote the shot: hermes | artist (lane w5-backend)
_load_route_area("generation/model_sources_1.py", globals())  # GET /models/missing: model files a workflow needs here (lane w5-backend)
_load_route_area("music/song_map_1.py", globals())  # song map: one clock, measured tracks (lane w6-song)
_load_route_area("music/range_1.py", globals())  # song IN-OUT stored per project (lane p12-aud)
_load_route_area("scene/scene_1.py", globals())  # scene analysis: cuts, speakers, sync, motion -> studio.scene-map/1 (lane w6-scene)
_load_route_area("generation/video_steps_1.py", globals())  # studio.video-step/1 manifests + support (lane w9-chain)
_load_route_area("projects/chain_1.py", globals())  # shot chain studio.shot-chain/1: guide, marks, pictures, takes (lane w9-chain)
_load_route_area("generation/job_causes_1.py", globals())  # why a Video job failed; refused submits as jobs (lane w11-backend)
_load_route_area("projects/loose_1.py", globals())  # Loose-work approved folders, page-only writer (lane p5-prj)
_load_route_area("generation/jobs_dismiss_1.py", globals())  # POST /jobs/dismiss: dismiss failed jobs (lane p5c-frm)
_load_route_area("projects/settings_1.py", globals())  # one output folder per project, PUT /projects/settings/output (lane p5c-frm)
_load_route_area("projects/recent_cover_1.py", globals())  # GET /projects/recent: item_count + cover; project create/open announce media.changed (lane p5e-frm)
_load_route_area("projects/naming_1.py", globals())  # one naming template per project + project rename (lane p6-name)
_load_route_area("projects/references_1.py", globals())  # Use as reference…: a name on media = one reference (lane p13-med)
_load_route_area("generation/job_names_1.py", globals())  # /jobs/list workflow_label = "<runs on> · <workflow>" from the catalogs (lane w14-shell)
_load_route_area("music/generate_1.py", globals())  # local music generation: SA3, YuE2, Foundation-1 (lane w16-mus)
_load_route_area("projects/use_in_cut_1.py", globals())  # the Audio page's Use in Cut as one route (lane w19-g7hermes, CUT-071)
_load_route_area("planning/plan_1.py", globals())  # planning layer: shot record + elements with versions (lane w20-plan)
_load_route_area("planning/previz_1.py", globals())  # Previz stage 1: one set per scene, shot cameras, block proposal, guide clip (lane w20-plan2)
_load_route_area("projects/sample_1.py", globals())  # Sample film: a copy of the shipped sample project (lane w21-pkg, BCK-212)
_load_route_area("projects/subtitles_1.py", globals())  # the cut's subtitle track ST1: read, edit, transcribe, export (lane w23-subs, CUT-173)

router.include_router(image_catalog_router)
router.include_router(model_catalog_router)
router.include_router(project_router)

# Style browser (package 6): an injected, independently importable unit.
_styles_spec = importlib.util.spec_from_file_location(
    "film_suite_style_routes", Path(__file__).with_name("routes") / "styles" / "browser.py"
)
if _styles_spec is None or _styles_spec.loader is None:
    raise RuntimeError("The style routes are unavailable.")
_styles_module = importlib.util.module_from_spec(_styles_spec)
_styles_spec.loader.exec_module(_styles_module)
router.include_router(_styles_module.create_style_router(
    hermes_home=lambda: _active_hermes_home(),
    state_root=lambda parent, *parts: state_root(parent, *parts),
    require_project=lambda project_id: _recent_directory(_recent_project_store_factory(), project_id),
))
_load_route_area("shared/dependencies_1.py", globals())  # dependency readiness on a fresh machine (lane w19-g1fresh)
_load_route_area("media/take_file_1.py", globals())  # a V4 take has its file in the media feed (lane w24-rest VID-030)
_load_route_area("generation/jobs_audio_1.py", globals())  # /jobs/list lists the Audio runs (stems, analysis, song map, subtitles) like the Jobs popover (lane w24-rest)
