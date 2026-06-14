from __future__ import annotations

from datetime import timedelta
from typing import Any

from ..shared.cache import (
    get_cached_media_bytes as _get_cached_media_bytes,
    provider_cache_root,
    read_json_entry,
    read_text_entry,
    store_media_bytes as _store_media_bytes,
    write_json_entry,
    write_text_entry,
)

TWITTER_CACHE_ROOT = provider_cache_root("CONTEXTUALIZE_TWITTER_CACHE", "twitter")
API_CACHE_ROOT = TWITTER_CACHE_ROOT / "api"
MEDIA_CACHE_ROOT = TWITTER_CACHE_ROOT / "media"
RENDER_CACHE_ROOT = TWITTER_CACHE_ROOT / "render"
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


def get_cached_media_bytes(identity: str) -> bytes | None:
    return _get_cached_media_bytes(MEDIA_CACHE_ROOT, identity)


def store_media_bytes(identity: str, content: bytes) -> None:
    _store_media_bytes(MEDIA_CACHE_ROOT, identity, content)


def get_cached_rendered(identity: str, ttl: timedelta | None = None) -> str | None:
    entry = read_text_entry(RENDER_CACHE_ROOT, identity, ttl=ttl)
    return entry.value if entry is not None else None


def store_rendered(identity: str, content: str) -> None:
    write_text_entry(RENDER_CACHE_ROOT, identity, content)
