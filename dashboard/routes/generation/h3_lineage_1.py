from __future__ import annotations


# H3 take lineage (lane fs-obvb): one take's sidecar facts, which continuations it offers and its
# parent/child takes in the same scope.  Read-only: the take facts are written when a take is
# published (its .mctx header, read by generation/h3_mctx.py); nothing is uploaded or generated.
_H3_LINEAGE_LIMIT = 200
_H3_LINEAGE_RELATIONS = {"extends": "continues", "prepends": "leads into", "bridges": "bridges"}


class H3TakeLineageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    owner_scope: str = Field(pattern=_OWNER_SCOPE_PATTERN, min_length=18, max_length=47)
    studio_job_id: str = Field(pattern=r"^studio:[0-9a-f]{32}$")

    @property
    def project_id(self) -> str | None:
        suffix = self.owner_scope[len("film_owner:"):]
        return None if suffix == "scratch" else suffix


def _h3_lineage_entry(record: Any, facts: Mapping[str, Any]) -> dict[str, Any]:
    mctx = facts.get("mctx") or {}
    prompt = None
    try:
        prompt = (record.request.get("args") or {}).get("prompt")
    except Exception:  # noqa: BLE001 - a row with an unreadable request still lists
        prompt = None
    return {"studio_job_id": record.studio_job_id, "target_id": record.target_id, "label": _job_label(prompt),
            "relation": mctx.get("relation"), "self_id": mctx.get("self_id")}


@router.post("/generation/take/lineage")
async def h3_take_lineage(request: H3TakeLineageRequest) -> dict[str, Any]:
    """Which continuations a take offers (with the plain reason when none) and its lineage."""
    try:
        records = _comfy_store().list_recent(project_id=request.project_id, limit=_H3_LINEAGE_LIMIT)
    except Exception:  # noqa: BLE001 - the ledger is unreadable: say so, never guess
        return _v4_error("private_state_unavailable")
    takes = [record for record in records if _h3_is_take(record) and getattr(record, "state", None) == "ready"]
    by_id = {record.studio_job_id: record for record in takes}
    record = by_id.get(request.studio_job_id)
    if record is None:
        return {"error": "That take is not a finished take of this scope", "reason_code": "invalid_request"}
    root = _h3_facts_root()
    facts = _h3m_module.read_take_facts(root, record.studio_job_id) or {}
    mctx = facts.get("mctx") if facts.get("mctx_state") == "paired" else None
    answer: dict[str, Any] = {"studio_job_id": record.studio_job_id, "continuable": mctx is not None,
                              "offers": [], "reason": None, "mctx": mctx, "parents": [], "children": [],
                              "continuation": facts.get("continuation")}
    if mctx is None:
        answer["reason"] = _h3x_module.NOT_CONTINUABLE
        return answer
    answer["offers"] = ["extend", "prepend", "bridge"]
    by_self: dict[str, Any] = {}
    sidecars: dict[str, Mapping[str, Any]] = {}
    for other in takes:
        other_facts = _h3m_module.read_take_facts(root, other.studio_job_id) or {}
        other_mctx = other_facts.get("mctx") if other_facts.get("mctx_state") == "paired" else None
        if other_mctx:
            by_self[other_mctx["self_id"]] = other
            sidecars[other.studio_job_id] = other_facts
    for parent_id in mctx.get("parents") or []:
        parent = by_self.get(parent_id)
        if parent is not None:
            answer["parents"].append(_h3_lineage_entry(parent, sidecars[parent.studio_job_id]))
        else:
            answer["parents"].append({"studio_job_id": None, "self_id": parent_id, "relation": None, "label": None})
    for other in takes:
        other_mctx = (sidecars.get(other.studio_job_id) or {}).get("mctx") or {}
        if mctx["self_id"] in (other_mctx.get("parents") or []):
            answer["children"].append(_h3_lineage_entry(other, sidecars[other.studio_job_id]))
    answer["relation_label"] = _H3_LINEAGE_RELATIONS.get(mctx.get("relation"))
    return answer


@router.post("/generation/take/freeze-check")
async def h3_take_freeze_check(request: H3TakeLineageRequest) -> dict[str, Any]:
    """A finished H3 take's freeze verdict (lane w1-h3): act and hold_end, stored when it was published.

    Read-only: the verdict is measured once by the poll that publishes the take
    (fs-cam2 RESULT §3.5).  A frozen take stays a normal take; it is only marked.
    """
    try:
        records = _comfy_store().list_recent(project_id=request.project_id, limit=_H3_LINEAGE_LIMIT)
    except Exception:  # noqa: BLE001 - the ledger is unreadable: say so, never guess
        return _v4_error("private_state_unavailable")
    record = next((row for row in records if row.studio_job_id == request.studio_job_id and _h3_is_take(row)
                   and getattr(row, "state", None) == "ready"), None)
    if record is None:
        return {"error": "That take is not a finished take of this scope", "reason_code": "invalid_request"}
    facts = _h3m_module.read_take_facts(_h3_facts_root(), record.studio_job_id) or {}
    check = facts.get("freeze_check")
    if not isinstance(check, Mapping):
        return {"studio_job_id": record.studio_job_id, "verdict": "unmeasured", "frozen": None, "act": None,
                "hold_end": None, "measured": False}
    return {"studio_job_id": record.studio_job_id, "measured": check.get("verdict") in ("frozen", "moving"),
            **{key: check.get(key) for key in ("verdict", "frozen", "act", "hold_end", "act_below", "reference", "advice")}}
