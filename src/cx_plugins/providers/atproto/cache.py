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

ATPROTO_CACHE_ROOT = provider_cache_root("CONTEXTUALIZE_ATPROTO_CACHE", "atproto")
API_CACHE_ROOT = ATPROTO_CACHE_ROOT / "api"
RENDER_CACHE_ROOT = ATPROTO_CACHE_ROOT / "render"
MEDIA_CACHE_ROOT = ATPROTO_CACHE_ROOT / "media"
IDENTITY_CACHE_ROOT = ATPROTO_CACHE_ROOT / "identity"
TOKEN_CACHE_ROOT = ATPROTO_CACHE_ROOT / "token"
DEFAULT_API_TTL = timedelta(hours=6)
DEFAULT_IDENTITY_TTL = timedelta(days=7)


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


def get_cached_identity(identity: str, ttl: timedelta | None = None) -> str | None:
    entry = read_text_entry(
        IDENTITY_CACHE_ROOT,
        identity,
        ttl=DEFAULT_IDENTITY_TTL if ttl is None else ttl,
    )
    if entry is None:
        return None
    value = entry.value.strip()
    return value or None


def store_identity(identity: str, value: str) -> None:
    if value:
        write_text_entry(IDENTITY_CACHE_ROOT, identity, value)


def get_cached_handle_did(handle: str, ttl: timedelta | None = None) -> str | None:
    return get_cached_identity(f"handle:{handle.lower().strip()}", ttl=ttl)


def store_handle_did(handle: str, did: str) -> None:
    store_identity(f"handle:{handle.lower().strip()}", did)


def get_cached_media_bytes(identity: str) -> bytes | None:
    return _get_cached_media_bytes(MEDIA_CACHE_ROOT, identity)


def store_media_bytes(identity: str, content: bytes) -> None:
    _store_media_bytes(MEDIA_CACHE_ROOT, identity, content)


def _read_oauth_session_payload() -> tuple[dict[str, Any], dict[str, Any]] | None:
    entry = read_json_entry(TOKEN_CACHE_ROOT, "atproto-oauth:active")
    if entry is None or not isinstance(entry.value, dict):
        return None
    return entry.value, entry.metadata


def get_cached_oauth_session_record() -> dict[str, Any] | None:
    loaded = _read_oauth_session_payload()
    if loaded is None:
        return None
    payload, meta = loaded
    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token.strip():
        return None
    refresh_token = payload.get("refresh_token")
    token_type = payload.get("token_type")
    scope = payload.get("scope")
    client_id = payload.get("client_id")
    auth_server = payload.get("auth_server")
    resource_server = payload.get("resource_server")
    dpop_private_key_pem = payload.get("dpop_private_key_pem")
    dpop_public_jwk = payload.get("dpop_public_jwk")
    auth_server_nonce = payload.get("auth_server_nonce")
    resource_server_nonce = payload.get("resource_server_nonce")
    subject_did = payload.get("subject_did")
    return {
        "access_token": access_token.strip(),
        "refresh_token": refresh_token.strip()
        if isinstance(refresh_token, str) and refresh_token.strip()
        else None,
        "token_type": token_type.strip()
        if isinstance(token_type, str) and token_type.strip()
        else "DPoP",
        "scope": scope.strip() if isinstance(scope, str) else "",
        "client_id": client_id.strip() if isinstance(client_id, str) else "",
        "auth_server": auth_server.strip() if isinstance(auth_server, str) else "",
        "resource_server": resource_server.strip()
        if isinstance(resource_server, str)
        else "",
        "dpop_private_key_pem": dpop_private_key_pem.strip()
        if isinstance(dpop_private_key_pem, str)
        else "",
        "dpop_public_jwk": dpop_public_jwk if isinstance(dpop_public_jwk, dict) else {},
        "auth_server_nonce": auth_server_nonce.strip()
        if isinstance(auth_server_nonce, str) and auth_server_nonce.strip()
        else None,
        "resource_server_nonce": resource_server_nonce.strip()
        if isinstance(resource_server_nonce, str) and resource_server_nonce.strip()
        else None,
        "subject_did": subject_did.strip()
        if isinstance(subject_did, str) and subject_did.strip()
        else None,
        "expires_at": meta.get("expires_at"),
        "cached_at": meta.get("cached_at"),
        "identity": meta.get("identity"),
    }


def get_cached_oauth_access_token(min_valid_seconds: int = 60) -> str | None:
    record = get_cached_oauth_session_record()
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


def clear_cached_oauth_session() -> None:
    for path in cache_paths(TOKEN_CACHE_ROOT, "atproto-oauth:active", "json"):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue


def store_oauth_session(
    *,
    access_token: str,
    refresh_token: str | None,
    expires_in_seconds: int,
    token_type: str,
    scope: str,
    client_id: str,
    auth_server: str,
    resource_server: str,
    dpop_private_key_pem: str,
    dpop_public_jwk: dict[str, str],
    auth_server_nonce: str | None = None,
    resource_server_nonce: str | None = None,
    subject_did: str | None = None,
) -> None:
    if not access_token:
        return
    identity = "atproto-oauth:active"
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
            "client_id": client_id,
            "auth_server": auth_server,
            "resource_server": resource_server,
            "dpop_private_key_pem": dpop_private_key_pem,
            "dpop_public_jwk": dpop_public_jwk,
            "auth_server_nonce": auth_server_nonce or "",
            "resource_server_nonce": resource_server_nonce or "",
            "subject_did": subject_did or "",
        },
        extra_metadata={"cached_at": now.isoformat(), "expires_at": expires_at.isoformat()},
        indent=2,
        secure=True,
    )
