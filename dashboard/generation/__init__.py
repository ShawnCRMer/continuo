"""Provider-neutral generation contracts.

This package is loaded by the top-level Hermes dashboard plugin loader under
``film_suite_generation``.  Keep imports relative so the package remains
independent of the dashboard directory and of any global ``sys.path`` entry.
"""

from .contract import (
    GENERATION_REQUEST_SCHEMA,
    HEALTH_STATES,
    NORMALIZED_STATUSES,
    PUBLIC_REASON_MESSAGES,
    REASON_CODES,
    GenerationRequest,
    ProviderCapabilities,
    ProviderReceipt,
    StaticHealth,
    normalize_status,
    public_reason,
)

__all__ = [
    "GENERATION_REQUEST_SCHEMA",
    "HEALTH_STATES",
    "NORMALIZED_STATUSES",
    "PUBLIC_REASON_MESSAGES",
    "REASON_CODES",
    "GenerationRequest",
    "ProviderCapabilities",
    "ProviderReceipt",
    "StaticHealth",
    "normalize_status",
    "public_reason",
]
