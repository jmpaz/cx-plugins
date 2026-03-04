from __future__ import annotations

from typing import Any

PLUGIN_API_VERSION = "1"
PLUGIN_NAME = "soundcloud"
PLUGIN_PRIORITY = 100


def _soundcloud_overrides(context: dict[str, Any]) -> dict[str, Any] | None:
    overrides = context.get("overrides")
    if not isinstance(overrides, dict):
        return None
    value = overrides.get("soundcloud")
    return value if isinstance(value, dict) else None


def can_resolve(target: str, context: dict[str, Any]) -> bool:
    from contextualize.references.soundcloud import is_soundcloud_url

    return is_soundcloud_url(target)


def resolve(target: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    from contextualize.references.soundcloud import (
        build_soundcloud_settings,
        resolve_soundcloud_url,
        soundcloud_settings_cache_key,
    )

    settings = build_soundcloud_settings(_soundcloud_overrides(context))
    settings_key = soundcloud_settings_cache_key(settings)
    documents = resolve_soundcloud_url(
        target,
        settings=settings,
        use_cache=bool(context.get("use_cache", True)),
        cache_ttl=context.get("cache_ttl"),
        refresh_cache=bool(context.get("refresh_cache", False)),
    )
    return [
        {
            "source": target,
            "label": document.label,
            "content": document.rendered,
            "metadata": {
                "trace_path": document.trace_path,
                "provider": PLUGIN_NAME,
                "source_ref": "soundcloud.com",
                "source_path": document.urn,
                "context_subpath": document.context_subpath,
                "source_created": document.source_created,
                "source_modified": document.source_modified,
                "kind": document.kind,
                "urn": document.urn,
                "scope_id": document.scope_id or target,
                "settings_key": settings_key,
            },
        }
        for document in documents
    ]


def register_auth_command(group: Any) -> None:
    from .soundcloud_auth import register_auth_command as _register_auth_command

    _register_auth_command(group)
