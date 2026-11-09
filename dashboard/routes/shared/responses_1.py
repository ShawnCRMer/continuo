from __future__ import annotations


def _response(raw: str) -> dict[str, Any]:
    """Return the existing adapter's JSON response without exposing signed URLs."""
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "The provider returned an unreadable response."}
    return decoded if isinstance(decoded, dict) else {"error": "The provider returned an invalid response."}


def _error(message: str) -> dict[str, Any]:
    return {"error": message}
