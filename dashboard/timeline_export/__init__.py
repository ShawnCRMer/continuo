"""Timeline export: exact reference movie, FCPXML 1.9 for Resolve, CMX3600 EDL.

Works from a ``studio.timeline/1`` document (TIMELINE-INPUT.md section 2.3)
and a source resolver (``resolver.SourceResolver``): ``ref -> {path, frames,
rate, width, height, has_audio}``.  Integer frames and rational rates only;
seconds appear at the ffmpeg edge.

The FCPXML mapping follows FilmCraft's interchange crate (MIT/Apache-2.0,
``crates/interchange/src/fcpxml.rs:1-15``): V1 is the spine, other video
tracks are connected clips on positive lanes, audio on negative lanes, markers
on the covering spine element.  Written fresh in Python, not translated.
"""
from __future__ import annotations

from .errors import ExportError
from .model import Plan, build_plan, load_timeline
from .resolver import MappingResolver, MusicInfo, ProbeResolver, ResolvedResolver, SourceInfo, SourceResolver

__all__ = [
    "ExportError",
    "MappingResolver",
    "MusicInfo",
    "Plan",
    "ProbeResolver",
    "ResolvedResolver",
    "SourceInfo",
    "SourceResolver",
    "build_plan",
    "load_timeline",
]
