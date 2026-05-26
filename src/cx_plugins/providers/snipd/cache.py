from __future__ import annotations

from datetime import timedelta

from ..shared.cache import (
    get_cached_media_bytes as _get_cached_media_bytes,
    provider_cache_root,
    read_text_entry,
    store_media_bytes as _store_media_bytes,
    write_text_entry,
)

SNIPD_CACHE_ROOT = provider_cache_root("CONTEXTUALIZE_SNIPD_CACHE", "snipd")
MEDIA_CACHE_ROOT = SNIPD_CACHE_ROOT / "media"
DEFAULT_TTL = timedelta(days=30)


def get_cached_transcript(
    identity: str,
    ttl: timedelta | None = None,
) -> str | None:
    entry = read_text_entry(
        SNIPD_CACHE_ROOT,
        identity,
        ext="content",
        ttl=DEFAULT_TTL if ttl is None else ttl,
    )
    if entry is None:
        return None
    return entry.value


def store_transcript(
    identity: str,
    content: str,
    source: str = "unknown",
) -> None:
    write_text_entry(
        SNIPD_CACHE_ROOT,
        identity,
        content,
        ext="content",
        identity_field="clip_id",
        extra_metadata={"source": source},
    )


def get_cached_media_bytes(identity: str) -> bytes | None:
    return _get_cached_media_bytes(MEDIA_CACHE_ROOT, identity)


def store_media_bytes(identity: str, content: bytes) -> None:
    _store_media_bytes(MEDIA_CACHE_ROOT, identity, content)
