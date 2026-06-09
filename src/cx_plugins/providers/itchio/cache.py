from __future__ import annotations

from datetime import timedelta

from ..shared.cache import (
    provider_cache_root,
    read_text_entry,
    write_text_entry,
)
from ..shared.cache import (
    get_cached_media_bytes as _get_cached_media_bytes,
)
from ..shared.cache import (
    store_media_bytes as _store_media_bytes,
)

ITCHIO_CACHE_ROOT = provider_cache_root("CONTEXTUALIZE_ITCHIO_CACHE", "itchio")
HTML_CACHE_ROOT = ITCHIO_CACHE_ROOT / "html"
MEDIA_CACHE_ROOT = ITCHIO_CACHE_ROOT / "media"
DEFAULT_HTML_TTL = timedelta(days=7)
HTML_CACHE_VERSION = 2


def _html_cache_identity(url: str) -> str:
    return f"html-v{HTML_CACHE_VERSION}:{url}"


def get_cached_html(url: str, ttl: timedelta | None = DEFAULT_HTML_TTL) -> str | None:
    entry = read_text_entry(HTML_CACHE_ROOT, _html_cache_identity(url), ext="html", ttl=ttl)
    return entry.value if entry is not None else None


def store_html(url: str, content: str) -> None:
    write_text_entry(
        HTML_CACHE_ROOT,
        _html_cache_identity(url),
        content,
        ext="html",
        identity_field="url",
        extra_metadata={"source_url": url, "html_cache_version": HTML_CACHE_VERSION},
    )


def get_cached_media_bytes(identity: str) -> bytes | None:
    return _get_cached_media_bytes(MEDIA_CACHE_ROOT, identity)


def store_media_bytes(identity: str, content: bytes) -> None:
    _store_media_bytes(MEDIA_CACHE_ROOT, identity, content)
