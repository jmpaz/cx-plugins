from __future__ import annotations

from datetime import timedelta
from typing import Any

from ..shared.cache import provider_cache_root, read_json_entry, write_json_entry

XWITTER_CACHE_ROOT = provider_cache_root("CONTEXTUALIZE_XWITTER_CACHE", "xwitter")
API_CACHE_ROOT = XWITTER_CACHE_ROOT / "api"
DEFAULT_API_TTL = timedelta(hours=6)


def get_cached_api_json(identity: str, ttl: timedelta | None = None) -> Any | None:
    entry = read_json_entry(
        API_CACHE_ROOT,
        identity,
        ttl=DEFAULT_API_TTL if ttl is None else ttl,
    )
    return entry.value if entry is not None else None


def store_api_json(identity: str, payload: Any) -> None:
    write_json_entry(API_CACHE_ROOT, identity, payload)
