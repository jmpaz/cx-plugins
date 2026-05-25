from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

CACHE_VERSION = 1


@dataclass(frozen=True)
class CacheEntry:
    value: Any
    metadata: dict[str, Any]


def provider_cache_root(env_var: str, provider: str) -> Path:
    return Path(
        os.environ.get(
            env_var,
            os.path.expanduser(f"~/.local/share/contextualize/cache/{provider}/v1"),
        )
    )


def cache_key(identity: str) -> str:
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def cache_paths(root: Path, identity: str, ext: str) -> tuple[Path, Path]:
    key = cache_key(identity)
    return root / f"{key}.{ext}", root / f"{key}.meta.json"


def keyed_path(root: Path, identity: str, ext: str) -> Path:
    return root / f"{cache_key(identity)}.{ext}"


def metadata_expired(metadata: dict[str, Any], ttl: timedelta | None) -> bool:
    if ttl is None:
        return False
    if ttl == timedelta(0):
        return True
    cached_at = metadata.get("cached_at")
    if not isinstance(cached_at, str):
        return True
    try:
        parsed = datetime.fromisoformat(cached_at.replace("Z", "+00:00"))
    except ValueError:
        return True
    return (datetime.now(timezone.utc) - parsed) > ttl


def token_expired(expires_at_iso: str, *, min_valid_seconds: int = 0) -> bool:
    try:
        expires_at = datetime.fromisoformat(expires_at_iso.replace("Z", "+00:00"))
    except ValueError:
        return True
    return expires_at <= (
        datetime.now(timezone.utc) + timedelta(seconds=min_valid_seconds)
    )


def read_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    version = payload.get("cache_version", CACHE_VERSION)
    if version != CACHE_VERSION:
        return None
    return payload


def read_text_entry(
    root: Path,
    identity: str,
    *,
    ext: str = "txt",
    ttl: timedelta | None = None,
) -> CacheEntry | None:
    content_path, meta_path = cache_paths(root, identity, ext)
    if not content_path.exists():
        return None
    metadata = read_metadata(meta_path)
    if metadata is None or metadata_expired(metadata, ttl):
        return None
    try:
        return CacheEntry(content_path.read_text(encoding="utf-8"), metadata)
    except (OSError, UnicodeDecodeError):
        return None


def write_text_entry(
    root: Path,
    identity: str,
    content: str,
    *,
    ext: str = "txt",
    identity_field: str = "identity",
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    try:
        root.mkdir(parents=True, exist_ok=True)
        content_path, meta_path = cache_paths(root, identity, ext)
        content_path.write_text(content, encoding="utf-8")
        metadata = {
            identity_field: identity,
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "size_bytes": len(content.encode("utf-8")),
            "cache_version": CACHE_VERSION,
        }
        metadata.update(extra_metadata or {})
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    except OSError:
        return


def read_json_entry(
    root: Path,
    identity: str,
    *,
    ttl: timedelta | None = None,
) -> CacheEntry | None:
    content_path, meta_path = cache_paths(root, identity, "json")
    if not content_path.exists():
        return None
    metadata = read_metadata(meta_path)
    if metadata is None or metadata_expired(metadata, ttl):
        return None
    try:
        return CacheEntry(json.loads(content_path.read_text(encoding="utf-8")), metadata)
    except (OSError, json.JSONDecodeError):
        return None


def write_json_entry(
    root: Path,
    identity: str,
    payload: Any,
    *,
    identity_field: str = "identity",
    extra_metadata: dict[str, Any] | None = None,
    indent: int | None = None,
    secure: bool = False,
) -> None:
    try:
        root.mkdir(parents=True, exist_ok=True)
        content_path, meta_path = cache_paths(root, identity, "json")
        text = json.dumps(payload, ensure_ascii=False, indent=indent)
        content_path.write_text(text, encoding="utf-8")
        metadata = {
            identity_field: identity,
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "size_bytes": len(text.encode("utf-8")),
            "cache_version": CACHE_VERSION,
        }
        metadata.update(extra_metadata or {})
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if secure:
            secure_file_permissions(content_path)
            secure_file_permissions(meta_path)
    except OSError:
        return


def read_keyed_text(root: Path, identity: str, *, ext: str = "txt") -> str | None:
    path = keyed_path(root, identity, ext)
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def write_keyed_text(
    root: Path,
    identity: str,
    content: str,
    *,
    ext: str = "txt",
) -> None:
    try:
        root.mkdir(parents=True, exist_ok=True)
        keyed_path(root, identity, ext).write_text(content, encoding="utf-8")
    except OSError:
        return


def read_mtime_text(
    root: Path,
    identity: str,
    *,
    ext: str = "txt",
    ttl: timedelta | None = None,
) -> str | None:
    if ttl == timedelta(0):
        return None
    path = root / f"{identity}.{ext}"
    if not path.exists():
        return None
    try:
        if ttl is not None:
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if (datetime.now(timezone.utc) - modified) > ttl:
                return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def write_mtime_text(
    root: Path,
    identity: str,
    content: str,
    *,
    ext: str = "txt",
) -> None:
    try:
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{identity}.{ext}").write_text(content, encoding="utf-8")
    except OSError:
        return


def get_cached_media_bytes(root: Path, identity: str) -> bytes | None:
    path = keyed_path(root, identity, "bin")
    if not path.exists():
        return None
    try:
        return path.read_bytes()
    except OSError:
        return None


def store_media_bytes(root: Path, identity: str, content: bytes) -> None:
    if not content:
        return
    try:
        root.mkdir(parents=True, exist_ok=True)
        keyed_path(root, identity, "bin").write_bytes(content)
    except OSError:
        return


def secure_file_permissions(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        return
