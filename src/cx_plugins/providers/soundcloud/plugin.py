from __future__ import annotations

from typing import Any

PLUGIN_API_VERSION = "1"
PLUGIN_NAME = "soundcloud"
PLUGIN_PRIORITY = 100


def normalize_manifest_config(
    raw_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if raw_config is None:
        return None
    if not isinstance(raw_config, dict):
        raise ValueError("soundcloud config must be a mapping")
    return dict(raw_config)


def _soundcloud_runtime_overrides(raw: dict[str, Any]) -> dict[str, Any] | None:
    result: dict[str, Any] = {}
    for key in (
        "max_items",
        "artist_tracks_filter",
        "artist_playlists_filter",
        "artist_reposts_filter",
        "include_comments",
        "comments_max_items",
        "comments_sort",
        "comments_nesting_depth",
        "include_artwork_descriptions",
        "media_mode",
    ):
        if key in raw:
            result[key] = raw[key]

    if "max-items" in raw:
        result["max_items"] = raw.get("max-items")

    artist_cfg = raw.get("artist")
    if artist_cfg is not None:
        if not isinstance(artist_cfg, dict):
            raise ValueError("soundcloud artist config must be a mapping")
        for config_key, result_key in (
            ("tracks", "artist_tracks_filter"),
            ("playlists", "artist_playlists_filter"),
            ("reposts", "artist_reposts_filter"),
        ):
            if config_key in artist_cfg:
                result[result_key] = artist_cfg.get(config_key)

    comments_cfg = raw.get("comments")
    if comments_cfg is not None:
        if not isinstance(comments_cfg, dict):
            raise ValueError("soundcloud comments config must be a mapping")
        for config_key, result_key in (
            ("enabled", "include_comments"),
            ("max-items", "comments_max_items"),
            ("sort", "comments_sort"),
            ("nesting-depth", "comments_nesting_depth"),
        ):
            if config_key in comments_cfg:
                result[result_key] = comments_cfg.get(config_key)

    media_cfg = raw.get("media")
    if media_cfg is not None:
        if not isinstance(media_cfg, dict):
            raise ValueError("soundcloud media config must be a mapping")
        for config_key, result_key in (
            ("describe-artwork", "include_artwork_descriptions"),
            ("mode", "media_mode"),
        ):
            if config_key in media_cfg:
                result[result_key] = media_cfg.get(config_key)

    return result or None


def _soundcloud_overrides(context: dict[str, Any]) -> dict[str, Any] | None:
    overrides = context.get("overrides")
    if not isinstance(overrides, dict):
        return None
    value = overrides.get("soundcloud")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("soundcloud overrides must be a mapping")
    return _soundcloud_runtime_overrides(value)


def can_resolve(target: str, context: dict[str, Any]) -> bool:
    from .soundcloud import is_soundcloud_url

    return is_soundcloud_url(target)


def classify_target(target: str, context: dict[str, Any]) -> dict[str, Any] | None:
    from .soundcloud import parse_soundcloud_target

    parsed = parse_soundcloud_target(target)
    if parsed is None:
        return None
    return {
        "provider": PLUGIN_NAME,
        "kind": parsed.kind,
        "is_external": True,
        "group_key": parsed.kind,
    }


def resolve(target: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    from .soundcloud import (
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
