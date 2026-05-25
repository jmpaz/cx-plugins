from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from ..shared.cache import (
    cache_paths,
    get_cached_media_bytes as _get_cached_media_bytes,
    provider_cache_root,
    read_json_entry,
    read_text_entry,
    store_media_bytes as _store_media_bytes,
    token_expired,
    write_json_entry,
    write_text_entry,
)

SOUNDCLOUD_CACHE_ROOT = provider_cache_root(
    "CONTEXTUALIZE_SOUNDCLOUD_CACHE",
    "soundcloud",
)
API_CACHE_ROOT = SOUNDCLOUD_CACHE_ROOT / "api"
RENDER_CACHE_ROOT = SOUNDCLOUD_CACHE_ROOT / "render"
MEDIA_CACHE_ROOT = SOUNDCLOUD_CACHE_ROOT / "media"
TOKEN_CACHE_ROOT = SOUNDCLOUD_CACHE_ROOT / "token"
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


def get_cached_rendered(identity: str, ttl: timedelta | None = None) -> str | None:
    entry = read_text_entry(RENDER_CACHE_ROOT, identity, ttl=ttl)
    return entry.value if entry is not None else None


def store_rendered(identity: str, content: str) -> None:
    write_text_entry(RENDER_CACHE_ROOT, identity, content)


def get_cached_media_bytes(identity: str) -> bytes | None:
    return _get_cached_media_bytes(MEDIA_CACHE_ROOT, identity)


def store_media_bytes(identity: str, content: bytes) -> None:
    _store_media_bytes(MEDIA_CACHE_ROOT, identity, content)


def get_cached_client_token(identity: str) -> str | None:
    entry = read_json_entry(TOKEN_CACHE_ROOT, identity)
    if entry is None or not isinstance(entry.value, dict):
        return None
    expires_at = entry.metadata.get("expires_at")
    if not isinstance(expires_at, str) or token_expired(expires_at):
        return None
    token = entry.value.get("access_token")
    if isinstance(token, str) and token.strip():
        return token
    return None


def _read_user_token_payload() -> tuple[dict[str, Any], dict[str, Any]] | None:
    entry = read_json_entry(TOKEN_CACHE_ROOT, "soundcloud-user:active")
    if entry is None or not isinstance(entry.value, dict):
        return None
    return entry.value, entry.metadata


def get_cached_user_token_record() -> dict[str, Any] | None:
    loaded = _read_user_token_payload()
    if loaded is None:
        return None
    payload, meta = loaded
    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token.strip():
        return None
    refresh_token = payload.get("refresh_token")
    token_type = payload.get("token_type")
    scope = payload.get("scope")
    return {
        "access_token": access_token.strip(),
        "refresh_token": refresh_token.strip()
        if isinstance(refresh_token, str) and refresh_token.strip()
        else None,
        "token_type": token_type.strip()
        if isinstance(token_type, str) and token_type.strip()
        else "Bearer",
        "scope": scope.strip() if isinstance(scope, str) else "",
        "expires_at": meta.get("expires_at"),
        "cached_at": meta.get("cached_at"),
        "identity": meta.get("identity"),
    }


def get_cached_user_access_token(min_valid_seconds: int = 60) -> str | None:
    record = get_cached_user_token_record()
    if record is None:
        return None
    expires_at = record.get("expires_at")
    if not isinstance(expires_at, str) or token_expired(
        expires_at, min_valid_seconds=min_valid_seconds
    ):
        return None
    token = record.get("access_token")
    if isinstance(token, str) and token.strip():
        return token.strip()
    return None


def clear_cached_user_token() -> None:
    for path in cache_paths(TOKEN_CACHE_ROOT, "soundcloud-user:active", "json"):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue


def store_user_token(
    *,
    access_token: str,
    refresh_token: str | None,
    expires_in_seconds: int,
    token_type: str = "Bearer",
    scope: str = "",
) -> None:
    if not access_token:
        return
    identity = "soundcloud-user:active"
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=max(60, int(expires_in_seconds)))
    write_json_entry(
        TOKEN_CACHE_ROOT,
        identity,
        {
            "access_token": access_token,
            "refresh_token": refresh_token or "",
            "token_type": token_type,
            "scope": scope,
        },
        extra_metadata={"cached_at": now.isoformat(), "expires_at": expires_at.isoformat()},
        indent=2,
        secure=True,
    )


def store_client_token(
    identity: str,
    *,
    access_token: str,
    expires_in_seconds: int,
) -> None:
    if not access_token:
        return
    now = datetime.now(timezone.utc)
    max_age = max(60, int(expires_in_seconds) - 60)
    expires_at = now + timedelta(seconds=max_age)
    write_json_entry(
        TOKEN_CACHE_ROOT,
        identity,
        {"access_token": access_token},
        extra_metadata={"cached_at": now.isoformat(), "expires_at": expires_at.isoformat()},
        indent=2,
        secure=True,
    )
