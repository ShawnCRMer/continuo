"""One finite error type for every export failure."""
from __future__ import annotations

from typing import Any


class ExportError(Exception):
    """``code`` is stable and machine-readable; ``message`` is one English sentence."""

    def __init__(self, code: str, message: str, details: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details is not None:
            result["details"] = self.details
        return result
