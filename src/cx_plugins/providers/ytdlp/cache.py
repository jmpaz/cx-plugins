from __future__ import annotations

from datetime import timedelta

from ..shared.cache import (
    get_cached_media_bytes as _get_cached_media_bytes,
    provider_cache_root,
    read_text_entry,
    store_media_bytes as _store_media_bytes,
    write_text_entry,
)

YOUTUBE_CACHE_ROOT = provider_cache_root("CONTEXTUALIZE_YOUTUBE_CACHE", "youtube")
MEDIA_CACHE_ROOT = YOUTUBE_CACHE_ROOT / "media"
DEFAULT_TTL = timedelta(days=30)


def get_cached_transcript(
    video_id: str,
    ttl: timedelta | None = None,
    whisper_available: bool = False,
) -> str | None:
    entry = read_text_entry(
        YOUTUBE_CACHE_ROOT,
        video_id,
        ext="content",
        ttl=DEFAULT_TTL if ttl is None else ttl,
    )
    if entry is None:
        return None
    if whisper_available and entry.metadata.get("source") == "captions":
        return None
    return entry.value


def store_transcript(
    video_id: str,
    content: str,
    source: str = "unknown",
) -> None:
    write_text_entry(
        YOUTUBE_CACHE_ROOT,
        video_id,
        content,
        ext="content",
        identity_field="video_id",
        extra_metadata={"source": source},
    )


def get_cached_media_bytes(identity: str) -> bytes | None:
    return _get_cached_media_bytes(MEDIA_CACHE_ROOT, identity)


def store_media_bytes(identity: str, content: bytes) -> None:
    _store_media_bytes(MEDIA_CACHE_ROOT, identity, content)
