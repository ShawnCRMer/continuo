"""Profile-owned generated Image material byte authority.

This module is intentionally not an HTTP surface.  The only read capability is
an opaque Image source tuple.  The state owner supplies the generated artifact
provenance and the producer ledger supplies the published-job authority before
this module derives a content-addressed filename and opens it relative to the
pinned materials directory.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from . import image_contract as contract


MAX_BYTES = contract.MAX_BYTES
_CONTEXT_RE = re.compile(r"^imgctx_[0-9a-f]{32}$")
_WORK_RE = re.compile(r"^imgwork_[0-9a-f]{32}$")
_REVISION_RE = re.compile(r"^imgrev_[0-9a-f]{32}$")
_MATERIAL_RE = re.compile(r"^imgmat_[0-9a-f]{32}$")


class ImageMaterialReadError(ValueError):
    """Finite fail-closed error for generated material reads."""

    def __init__(self, code: str) -> None:
        self.code = code
        self.reason_code = code
        self.reason = code
        super().__init__(code)


class ImagePublishedMaterialReadAuthority:
    """Read one published generated PNG from the composition-root-owned root."""

    def __init__(
        self,
        *,
        image_store: Any,
        job_store: Any,
        material_root: Path | str,
        boundary: Any,
        max_bytes: int = MAX_BYTES,
    ) -> None:
        if image_store is None or job_store is None or boundary is None:
            raise ImageMaterialReadError("image-material-authority-unavailable")
        if type(max_bytes) is not int or max_bytes < 1 or max_bytes > MAX_BYTES:
            raise ImageMaterialReadError("image-material-size-exceeded")
        root = Path(material_root)
        if root.name != "image-materials":
            raise ImageMaterialReadError("image-material-root-invalid")
        self.image_store = image_store
        self.job_store = job_store
        self.material_root = root
        self.materials_root = root / "materials"
        self.boundary = boundary
        self.max_bytes = max_bytes
        self._materials_identity = self._require_directory(self.materials_root, self.boundary)

    @staticmethod
    def _require_directory(path: Path, boundary: Any | None = None) -> Any:
        if boundary is None:
            raise ImageMaterialReadError("image-material-authority-unavailable")
        opener = getattr(boundary, "open_existing", None)
        if not callable(opener):
            opener = getattr(boundary, "_open_existing", None)
        if not callable(opener):
            raise ImageMaterialReadError("image-material-authority-unavailable")
        handle = None
        try:
            handle, identity = opener(path, directory=True)
            if not bool(getattr(identity, "is_directory", False)) or any(
                bool(getattr(identity, name, False))
                for name in ("is_reparse", "is_reparse_point", "reparse")
            ):
                raise ImageMaterialReadError("image-material-root-invalid")
        except ImageMaterialReadError:
            raise
        except Exception as exc:
            raise ImageMaterialReadError("image-material-root-invalid") from exc
        finally:
            if handle is not None:
                close = getattr(handle, "close", None)
                if callable(close):
                    close()
        return identity

    def _require_materials_root(self) -> None:
        identity = self._require_directory(self.materials_root, self.boundary)
        if not self._same_identity(self._materials_identity, identity):
            raise ImageMaterialReadError("image-material-changed")

    @staticmethod
    def _source_tuple(source: Any) -> tuple[str, str, str, str]:
        if type(source) is not tuple or len(source) != 4:
            raise ImageMaterialReadError("image-material-source-invalid")
        validators = (
            (_CONTEXT_RE, "context"),
            (_WORK_RE, "work"),
            (_REVISION_RE, "revision"),
            (_MATERIAL_RE, "material"),
        )
        for value, (pattern, _label) in zip(source, validators):
            if type(value) is not str or pattern.fullmatch(value) is None:
                raise ImageMaterialReadError("image-material-source-invalid")
        return source

    @staticmethod
    def _close(handle: Any) -> None:
        close = getattr(handle, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _regular(identity: Any) -> bool:
        return not bool(getattr(identity, "is_directory", False)) and not any(
            bool(getattr(identity, name, False))
            for name in ("is_reparse", "is_reparse_point", "reparse")
        )

    @staticmethod
    def _same_identity(left: Any, right: Any) -> bool:
        try:
            left_same = getattr(left, "same_object", None)
            right_same = getattr(right, "same_object", None)
            if callable(left_same) and callable(right_same):
                return bool(left_same(right)) and bool(right_same(left))
            fields = ("volume_serial", "file_id", "is_directory", "size")
            if any(not hasattr(left, field) or not hasattr(right, field) for field in fields):
                return False
            return (
                left.volume_serial == right.volume_serial
                and left.file_id == right.file_id
                and bool(left.is_directory) == bool(right.is_directory)
                and left.size == right.size
            )
        except Exception:
            return False

    def _state_provenance(
        self,
        source: tuple[str, str, str, str],
        state: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        context_id, work_id, revision_id, material_id = source
        try:
            state = contract.validate_image_state(state)
        except ImageMaterialReadError:
            raise
        except Exception as exc:
            raise ImageMaterialReadError("image-material-unavailable") from exc

        works = [work for work in state["state"]["works"] if work.get("work_id") == work_id]
        if len(works) != 1:
            raise ImageMaterialReadError("image-material-unavailable")
        revisions = [revision for revision in works[0]["saved_revisions"] if revision.get("revision_id") == revision_id]
        if len(revisions) != 1 or revisions[0].get("work_id") != work_id:
            raise ImageMaterialReadError("image-material-unavailable")
        revision = revisions[0]
        if revision.get("selected_artifact") != material_id:
            raise ImageMaterialReadError("image-material-provenance-invalid")
        if material_id not in revision.get("material_ids", []):
            raise ImageMaterialReadError("image-material-provenance-invalid")
        closure = {material_id}
        for field in ("references", "alternatives"):
            for item in revision["content"][field]:
                if type(item) is dict and type(item.get("material_id")) is str:
                    closure.add(item["material_id"])
        if revision["material_ids"] != sorted(closure):
            raise ImageMaterialReadError("image-material-provenance-invalid")
        generated = [
            item
            for item in revision["content"]["alternatives"]
            if type(item) is dict
            and item.get("origin") == "generated"
            and item.get("material_id") == material_id
        ]
        if len(generated) != 1:
            raise ImageMaterialReadError("image-material-provenance-invalid")
        try:
            artifact = contract.validate_artifact_ref(copy.deepcopy(generated[0]))
        except Exception as exc:
            raise ImageMaterialReadError("image-material-provenance-invalid") from exc
        materials = [item for item in state["state"]["materials"] if item.get("material_id") == material_id]
        if len(materials) != 1:
            raise ImageMaterialReadError("image-material-provenance-invalid")
        material = materials[0]
        if (
            material.get("sha256") != artifact["artifact_sha256"]
            or material.get("media_type") != "image"
            or material.get("availability") != "available"
        ):
            raise ImageMaterialReadError("image-material-provenance-invalid")
        return artifact, state

    def _published_artifact(self, artifact: dict[str, Any]) -> dict[str, Any]:
        try:
            job = self.job_store.get_job(artifact["producer_job_id"])
            if job.get("state") != "published":
                raise ImageMaterialReadError("image-material-unpublished")
            derived = contract.build_generated_artifact_ref(job)
        except ImageMaterialReadError:
            raise
        except Exception as exc:
            raise ImageMaterialReadError("image-material-unavailable") from exc
        if derived != artifact:
            raise ImageMaterialReadError("image-material-provenance-invalid")
        return derived

    def _read_relative(self, artifact: dict[str, Any]) -> bytes:
        self._require_materials_root()
        filename = f"{artifact['artifact_sha256']}.png"
        opener = getattr(self.boundary, "open_existing", None)
        if not callable(opener):
            opener = getattr(self.boundary, "_open_existing", None)
        child_opener = getattr(self.boundary, "open_child_existing", None)
        if not callable(child_opener):
            child_opener = getattr(self.boundary, "_open_child_existing", None)
        reader = getattr(self.boundary, "read_handle_bytes", None)
        if not callable(reader):
            reader = getattr(self.boundary, "read_handle", None)
        if not callable(opener) or not callable(child_opener) or not callable(reader):
            raise ImageMaterialReadError("image-material-authority-unavailable")
        parent_handle = None
        file_handle = None
        try:
            parent_handle, parent_identity = opener(self.materials_root, directory=True)
            if not bool(getattr(parent_identity, "is_directory", False)) or any(
                bool(getattr(parent_identity, name, False))
                for name in ("is_reparse", "is_reparse_point", "reparse")
            ):
                raise ImageMaterialReadError("image-material-root-invalid")
            if not self._same_identity(self._materials_identity, parent_identity):
                raise ImageMaterialReadError("image-material-changed")
            file_handle, file_identity = child_opener(
                parent_handle,
                self.materials_root,
                filename,
                directory=False,
            )
            if not self._regular(file_identity):
                raise ImageMaterialReadError("image-material-unavailable")
            if getattr(file_identity, "size", -1) != artifact["byte_size"]:
                raise ImageMaterialReadError("image-material-size-exceeded")
            try:
                data = reader(file_handle, max_bytes=self.max_bytes)
            except Exception as exc:
                raise ImageMaterialReadError("image-material-unavailable") from exc
            if type(data) is not bytes or len(data) != artifact["byte_size"] or len(data) > self.max_bytes:
                raise ImageMaterialReadError("image-material-size-exceeded")
            descriptor = {
                "filename": "generated.png",
                "subfolder": "",
                "type": "temp",
                "mime_type": "image/png",
                "extension": ".png",
            }
            try:
                facts = contract.validate_output(
                    data,
                    1,
                    descriptor,
                    artifact["width"],
                    artifact["height"],
                    artifact["artifact_sha256"],
                )
            except Exception as exc:
                raise ImageMaterialReadError("image-material-hash-mismatch") from exc
            if (
                facts["format"] != "png"
                or facts["mime_type"] != artifact["mime_type"]
                or facts["byte_size"] != artifact["byte_size"]
            ):
                raise ImageMaterialReadError("image-material-provenance-invalid")
        except ImageMaterialReadError:
            raise
        except FileNotFoundError as exc:
            raise ImageMaterialReadError("image-material-unavailable") from exc
        except Exception as exc:
            raise ImageMaterialReadError("image-material-changed") from exc
        finally:
            if file_handle is not None:
                self._close(file_handle)
            if parent_handle is not None:
                self._close(parent_handle)

        recheck_parent = None
        recheck_file = None
        try:
            recheck_parent, recheck_parent_identity = opener(self.materials_root, directory=True)
            if (
                not bool(getattr(recheck_parent_identity, "is_directory", False))
                or any(
                    bool(getattr(recheck_parent_identity, name, False))
                    for name in ("is_reparse", "is_reparse_point", "reparse")
                )
                or not self._same_identity(parent_identity, recheck_parent_identity)
                or not self._same_identity(self._materials_identity, recheck_parent_identity)
            ):
                raise ImageMaterialReadError("image-material-changed")
            recheck_file, recheck_identity = child_opener(
                recheck_parent,
                self.materials_root,
                filename,
                directory=False,
            )
            if (
                not self._regular(recheck_identity)
                or not self._same_identity(file_identity, recheck_identity)
                or getattr(recheck_identity, "size", -1) != artifact["byte_size"]
            ):
                raise ImageMaterialReadError("image-material-changed")
            return data
        except ImageMaterialReadError:
            raise
        except FileNotFoundError as exc:
            raise ImageMaterialReadError("image-material-changed") from exc
        except Exception as exc:
            raise ImageMaterialReadError("image-material-changed") from exc
        finally:
            if recheck_file is not None:
                self._close(recheck_file)
            if recheck_parent is not None:
                self._close(recheck_parent)

    def read_published_job(self, image_producer_job_id: str) -> tuple[bytes, dict[str, Any]]:
        """Return the bytes of one published run, before anyone has kept it.

        A finished run has to be visible before it is named - that is the
        whole point of looking first.  The authority is the ledger alone: the
        job must be published, and the artifact is derived from that job, so a
        caller cannot name a file.  Ownership of the job is the caller's check
        (the route resolves it through the owner-scoped state view first).
        """
        try:
            job = self.job_store.get_job(image_producer_job_id)
        except Exception as exc:
            raise ImageMaterialReadError("image-material-unavailable") from exc
        if job.get("state") != "published":
            raise ImageMaterialReadError("image-material-unpublished")
        try:
            artifact = contract.build_generated_artifact_ref(job)
        except Exception as exc:
            raise ImageMaterialReadError("image-material-provenance-invalid") from exc
        data = self._read_relative(artifact)
        if self._published_artifact(artifact) != artifact:
            raise ImageMaterialReadError("image-material-provenance-invalid")
        return data, copy.deepcopy(artifact)

    def read(self, source: tuple[str, str, str, str]) -> bytes:
        """Return bytes only for a validated ``(context, work, revision, material)`` tuple."""
        source = self._source_tuple(source)
        fenced_read = getattr(self.image_store, "_read_fenced", None)
        if not callable(fenced_read):
            raise ImageMaterialReadError("image-material-authority-unavailable")

        def consume(state: dict[str, Any]) -> bytes:
            artifact, _state = self._state_provenance(source, state)
            initial_artifact = self._published_artifact(artifact)
            data = self._read_relative(initial_artifact)
            final_artifact = self._published_artifact(artifact)
            if final_artifact != initial_artifact:
                raise ImageMaterialReadError("image-material-provenance-invalid")
            return data

        try:
            return fenced_read(source[0], consume, max_bytes=self.max_bytes)
        except ImageMaterialReadError:
            raise
        except Exception as exc:
            raise ImageMaterialReadError("image-material-unavailable") from exc

    read_source = read


__all__ = ["MAX_BYTES", "ImageMaterialReadError", "ImagePublishedMaterialReadAuthority"]
