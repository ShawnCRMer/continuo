from __future__ import annotations

import copy
import datetime as _datetime
import hashlib
import sys
from typing import Any, Callable

from .image_contract import (
    ContractError,
    build_generated_artifact_ref,
    canonical_bytes,
    validate_image_state,
    validate_operation,
)


class ImageAttachError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        self.reason_code = code
        super().__init__(code)


class ImageAttachService:
    """Offline bridge from one published producer job to one standalone Image revision."""

    def __init__(self, job_store: Any, state_store: Any, *, clock: Callable[[], str] | None = None) -> None:
        if job_store is None or state_store is None:
            raise ImageAttachError("attach-owner-required")
        self.job_store = job_store
        self.state_store = state_store
        owner_module = sys.modules.get(type(state_store).__module__)
        owner_error_type = getattr(owner_module, "ImageStateError", None)
        self._owner_error_type = owner_error_type if type(owner_error_type) is type else None
        self._clock = clock or self._utc_second

    @staticmethod
    def _utc_second() -> str:
        return _datetime.datetime.now(_datetime.timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _target(job: dict[str, Any]) -> list[str | None]:
        basis = job["context_basis"]
        return [basis["context_id"], basis["work_id"], basis["revision_id"]]

    @staticmethod
    def _derived_ids(job_id: str, target: list[str | None]) -> tuple[str, str]:
        suffix = hashlib.sha256(canonical_bytes({"job_id": job_id, "target": target})).hexdigest()[:32]
        return "imgrev_" + suffix, "imgop_" + suffix

    @staticmethod
    def _material_ids(content: dict[str, Any], selected: str) -> list[str]:
        result = {selected}
        for field in ("references", "alternatives"):
            for item in content[field]:
                material_id = item.get("material_id") if type(item) is dict else None
                if type(material_id) is str:
                    result.add(material_id)
        return sorted(result)

    def _validate_policy(
        self, job: dict[str, Any], operation: dict[str, Any], target: list[str | None]
    ) -> None:
        policy = job["attach_policy"]
        policy_keys = ("mode", "expected_revision", "owner_token")
        if any(operation[key] != policy[key] for key in policy_keys):
            raise ImageAttachError("attach-policy-mismatch")
        if operation["mode"] == "specified":
            requested = [operation["context_id"], operation["work_id"], operation["revision_id"]]
            frozen = [policy["context_id"], policy["work_id"], policy["revision_id"]]
            if requested != target or frozen != target:
                raise ImageAttachError("attach-target-mismatch")
        if operation["expected_revision"] <= 0:
            raise ImageAttachError("attach-positive-revision-required")

    def _validate_authority(
        self, job: dict[str, Any], operation: dict[str, Any]
    ) -> dict[str, Any]:
        basis = job["context_basis"]
        claim = {
            "schema": "studio.image-owner-claim/1",
            "context_id": basis["context_id"],
            "work_id": basis["work_id"],
            "basis_kind": basis["basis_kind"],
            "revision_id": basis["revision_id"],
            "draft_generation": basis["draft_generation"],
            "basis_content_sha256": basis["basis_content_sha256"],
            "expected_revision": operation["expected_revision"],
        }
        try:
            return self.state_store.verify_current_owner(claim, operation["owner_token"])
        except Exception as exc:
            refusal_codes = {
                "owner-token-invalid", "owner-claim-shape", "owner-claim-schema",
                "owner-claim-invalid", "owner-claim-stale", "owner-basis-invalid",
                "basis-kind-invalid", "revision-required", "draft-revision-alternative",
                "draft-generation-invalid", "draft-generation-alternative",
                "draft-generation-required", "basis-content-sha256-invalid",
                "expected-revision-invalid", "work-missing-or-ambiguous",
                "revision-missing-or-ambiguous", "revision-work-mismatch",
            }
            if (
                self._owner_error_type is not None
                and isinstance(exc, self._owner_error_type)
                and getattr(exc, "code", None) in refusal_codes
            ):
                raise ImageAttachError("attach-owner-refused") from exc
            raise

    def _record_refusal(
        self, job: dict[str, Any], operation: dict[str, Any], error: ImageAttachError
    ) -> None:
        self.job_store.quarantine(
            job["image_producer_job_id"],
            "attach-authority-failure",
            evidence={
                "reason_code": error.code,
                "mode": operation.get("mode"),
                "expected_revision": operation.get("expected_revision"),
            },
        )

    @staticmethod
    def _basis_content(state: dict[str, Any], basis: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        works = [work for work in state["state"]["works"] if work["work_id"] == basis["work_id"]]
        if len(works) != 1:
            raise ImageAttachError("attach-work-missing-or-ambiguous")
        work = works[0]
        if basis["basis_kind"] == "saved_revision":
            matches = [item for item in work["saved_revisions"] if item["revision_id"] == basis["revision_id"]]
            if len(matches) != 1 or matches[0]["work_id"] != basis["work_id"]:
                raise ImageAttachError("attach-revision-missing-or-ambiguous")
            return work, copy.deepcopy(matches[0]["content"])
        draft = work["draft"]
        if draft["draft_generation"] != basis["draft_generation"]:
            raise ImageAttachError("attach-draft-stale")
        return work, {key: copy.deepcopy(value) for key, value in draft.items() if key not in {"based_on_revision_id", "draft_generation"}}

    def _find_replay(
        self,
        state: dict[str, Any],
        job: dict[str, Any],
        artifact: dict[str, Any],
        target: list[str | None],
        revision_id: str,
    ) -> dict[str, Any] | None:
        matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for work in state["state"]["works"]:
            for revision in work["saved_revisions"]:
                for alternative in revision["content"]["alternatives"]:
                    if type(alternative) is dict and alternative.get("origin") == "generated" and alternative.get("producer_job_id") == job["image_producer_job_id"]:
                        matches.append((work, revision))
        if not matches:
            return None
        if len(matches) != 1:
            raise ImageAttachError("attach-marker-ambiguous")
        work, revision = matches[0]
        generated = [
            item for item in revision["content"]["alternatives"]
            if type(item) is dict and item.get("origin") == "generated" and item.get("producer_job_id") == job["image_producer_job_id"]
        ]
        if (
            generated != [artifact]
            or work["work_id"] != target[1]
            or revision["revision_id"] != revision_id
            or revision["parent_revision_id"] != target[2]
            or revision["selected_artifact"] != artifact["material_id"]
        ):
            raise ImageAttachError("attach-marker-conflict")
        basis_content = copy.deepcopy(revision["content"])
        basis_content["alternatives"] = [item for item in basis_content["alternatives"] if item != artifact]
        if job["context_basis"]["basis_kind"] == "draft":
            draft = work["draft"]
            expected_saved = {
                key: copy.deepcopy(value)
                for key, value in draft.items()
                if key not in {"based_on_revision_id", "draft_generation"}
            }
            basis_matches = (
                draft["draft_generation"] == job["context_basis"]["draft_generation"]
                and basis_content == expected_saved
                and hashlib.sha256(canonical_bytes(draft)).hexdigest()
                == job["context_basis"]["basis_content_sha256"]
            )
        else:
            basis_matches = (
                hashlib.sha256(canonical_bytes(basis_content)).hexdigest()
                == job["context_basis"]["basis_content_sha256"]
            )
        if not basis_matches:
            raise ImageAttachError("attach-marker-basis-conflict")
        if revision["material_ids"] != self._material_ids(revision["content"], artifact["material_id"]):
            raise ImageAttachError("attach-marker-material-closure")
        material = {
            "material_id": artifact["material_id"], "sha256": artifact["artifact_sha256"],
            "media_type": "image", "label": "generated.png", "availability": "available",
        }
        if [item for item in state["state"]["materials"] if item["material_id"] == artifact["material_id"]] != [material]:
            raise ImageAttachError("attach-marker-material-conflict")
        projected_target = job.get("attached_target")
        projected_revision = job.get("attached_revision_id")
        if (projected_target is not None or projected_revision is not None) and (
            projected_target != target or projected_revision != revision_id
        ):
            raise ImageAttachError("attach-projection-conflict")
        return revision

    def attach(self, operation: dict[str, Any]) -> dict[str, Any]:
        try:
            operation = validate_operation(operation)
        except ContractError as exc:
            raise ImageAttachError("attach-operation-invalid") from exc
        if operation["kind"] != "attach":
            raise ImageAttachError("attach-operation-invalid")
        job = self.job_store.get_job(operation["image_producer_job_id"])
        if job["state"] != "published":
            raise ImageAttachError("attach-job-not-published")
        artifact = build_generated_artifact_ref(job)
        target = self._target(job)
        try:
            self._validate_policy(job, operation, target)
        except ImageAttachError as exc:
            self._record_refusal(job, operation, exc)
            raise
        revision_id, operation_id = self._derived_ids(job["image_producer_job_id"], target)
        try:
            state = self.state_store.load(target[0])
        except Exception as exc:
            if (
                self._owner_error_type is not None
                and isinstance(exc, self._owner_error_type)
                and getattr(exc, "code", None) == "state-object-unavailable"
            ):
                self._record_refusal(
                    job, operation, ImageAttachError("attach-owner-missing")
                )
            raise
        replay = self._find_replay(state, job, artifact, target, revision_id)
        if replay is not None:
            self.job_store.record_attached(
                job["image_producer_job_id"], target=target, revision_id=revision_id
            )
            return {
                "state": state,
                "image_producer_job_id": job["image_producer_job_id"],
                "attached_target": target,
                "attached_revision_id": revision_id,
                "artifact": artifact,
                "material_id": artifact["material_id"],
                "replayed": True,
            }
        if "attached_target" in job or "attached_revision_id" in job:
            raise ImageAttachError("attach-projection-without-marker")
        try:
            self._validate_authority(job, operation)
        except ImageAttachError as exc:
            self._record_refusal(job, operation, exc)
            raise

        if state["revision"] != operation["expected_revision"]:
            raise ImageAttachError("attach-revision-conflict")
        work, content = self._basis_content(state, job["context_basis"])
        basis_bytes = (
            canonical_bytes(work["draft"])
            if job["context_basis"]["basis_kind"] == "draft"
            else canonical_bytes(content)
        )
        if hashlib.sha256(basis_bytes).hexdigest() != job["context_basis"]["basis_content_sha256"]:
            raise ImageAttachError("attach-basis-content-mismatch")

        candidate = copy.deepcopy(state)
        candidate["revision"] = state["revision"] + 1
        candidate["updated_at"] = self._clock()
        candidate_work = next(item for item in candidate["state"]["works"] if item["work_id"] == target[1])
        content["alternatives"].append(copy.deepcopy(artifact))
        candidate_work["saved_revisions"].append({
            "revision_id": revision_id,
            "work_id": target[1],
            "parent_revision_id": target[2],
            "saved_at": candidate["updated_at"],
            "content": content,
            "selected_artifact": artifact["material_id"],
            "material_ids": self._material_ids(content, artifact["material_id"]),
        })
        materials = candidate["state"]["materials"]
        same_id = [item for item in materials if item["material_id"] == artifact["material_id"]]
        material = {
            "material_id": artifact["material_id"],
            "sha256": artifact["artifact_sha256"],
            "media_type": "image",
            "label": "generated.png",
            "availability": "available",
        }
        if same_id and same_id != [material]:
            raise ImageAttachError("attach-material-conflict")
        if not same_id:
            materials.append(material)
        candidate = validate_image_state(candidate)
        replayed = False
        try:
            committed = self.state_store.publish_next(
                target[0], operation["expected_revision"], candidate, operation_id
            )
        except Exception:
            observed = self.state_store.load(target[0])
            if self._find_replay(observed, job, artifact, target, revision_id) is None:
                raise
            committed = observed
            replayed = True
        self.job_store.record_attached(
            job["image_producer_job_id"], target=target, revision_id=revision_id
        )
        return {
            "state": committed,
            "image_producer_job_id": job["image_producer_job_id"],
            "attached_target": target,
            "attached_revision_id": revision_id,
            "artifact": artifact,
            "material_id": artifact["material_id"],
            "replayed": replayed,
        }


__all__ = ["ImageAttachError", "ImageAttachService"]
