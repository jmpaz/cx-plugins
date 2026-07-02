from __future__ import annotations

from datetime import timedelta
from typing import Any

from ..shared.cache import provider_cache_root, read_json_entry, write_json_entry

WIKIPEDIA_CACHE_ROOT = provider_cache_root("CONTEXTUALIZE_WIKIPEDIA_CACHE", "wikipedia")
DOCUMENT_CACHE_ROOT = WIKIPEDIA_CACHE_ROOT / "documents"


def get_cached_document(identity: str, ttl: timedelta | None = None) -> Any | None:
    entry = read_json_entry(DOCUMENT_CACHE_ROOT, identity, ttl=ttl)
    return entry.value if entry is not None else None


def store_document(identity: str, payload: Any) -> None:
    write_json_entry(DOCUMENT_CACHE_ROOT, identity, payload)
