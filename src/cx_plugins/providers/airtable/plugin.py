from __future__ import annotations

from typing import Any

PLUGIN_API_VERSION = "1"
PLUGIN_NAME = "airtable"
PLUGIN_PRIORITY = 100


def normalize_manifest_config(
    raw_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if raw_config is None:
        return None
    if not isinstance(raw_config, dict):
        raise ValueError("airtable config must be a mapping")
    return dict(raw_config)


def _airtable_runtime_overrides(raw: dict[str, Any]) -> dict[str, Any] | None:
    result: dict[str, Any] = {}

    for key in ("include_media", "max_rows", "timeout_seconds"):
        if key in raw:
            result[key] = raw[key]

    for config_key, result_key in (
        ("include-media", "include_media"),
        ("max-rows", "max_rows"),
        ("timeout", "timeout_seconds"),
        ("timeout-seconds", "timeout_seconds"),
    ):
        if config_key in raw:
            result[result_key] = raw.get(config_key)

    media_cfg = raw.get("media")
    if media_cfg is not None:
        if not isinstance(media_cfg, dict):
            raise ValueError("airtable media config must be a mapping")
        if "enabled" in media_cfg:
            result["include_media"] = media_cfg.get("enabled")

    return result or None


def _airtable_overrides(context: dict[str, Any]) -> dict[str, Any] | None:
    overrides = context.get("overrides")
    if not isinstance(overrides, dict):
        return None
    value = overrides.get("airtable")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("airtable overrides must be a mapping")
    return _airtable_runtime_overrides(value)


def can_resolve(target: str, context: dict[str, Any]) -> bool:
    from .airtable import is_airtable_target

    return is_airtable_target(target)


def classify_target(target: str, context: dict[str, Any]) -> dict[str, Any] | None:
    from .airtable import parse_airtable_target

    parsed = parse_airtable_target(target)
    if parsed is None:
        return None
    return {
        "provider": PLUGIN_NAME,
        "kind": "database",
        "is_external": True,
        "group_key": "database",
    }


def _failure_document(
    target: str,
    exc: BaseException,
    *,
    settings_key: str,
) -> dict[str, Any]:
    from .airtable import parse_airtable_target

    parsed = parse_airtable_target(target)
    canonical_id = parsed.canonical_id if parsed is not None else target
    label = f"airtable/{parsed.primary_id if parsed is not None else 'unresolved'}-error"
    message = (
        "Airtable public resolution failed. This provider only resolves public "
        f"Airtable shared views, forms, and interface pages; generic webpage "
        f"fallback was suppressed: {exc}"
    )
    return {
        "source": target,
        "label": label,
        "content": message,
        "prose": "",
        "metadata": {
            "trace_path": label,
            "provider": PLUGIN_NAME,
            "source_ref": "airtable.com",
            "source_path": canonical_id,
            "context_subpath": f"{label}.md",
            "kind": "database",
            "canonical_id": canonical_id,
            "settings_key": settings_key,
            "media_count": 0,
            "resolution_error": str(exc),
            "hydrate_dedupe": {
                "mode": "canonical_symlink",
                "key": f"airtable-error:{canonical_id}:{settings_key}",
                "rank": 0,
            },
        },
    }


def resolve(target: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    from .airtable import (
        airtable_settings_cache_key,
        build_airtable_settings,
        resolve_airtable,
    )

    settings = build_airtable_settings(_airtable_overrides(context))
    settings_key = airtable_settings_cache_key(settings)
    try:
        document = resolve_airtable(
            target,
            settings=settings,
            use_cache=bool(context.get("use_cache", True)),
            cache_ttl=context.get("cache_ttl"),
            refresh_cache=bool(context.get("refresh_cache", False)),
        )
    except (RuntimeError, ValueError) as exc:
        return [_failure_document(target, exc, settings_key=settings_key)]

    metadata: dict[str, Any] = {
        "trace_path": document.label,
        "provider": PLUGIN_NAME,
        "source_ref": document.source_ref,
        "source_path": document.source_path,
        "context_subpath": document.context_subpath,
        "kind": document.kind,
        "canonical_id": document.canonical_id,
        "settings_key": settings_key,
        "media_count": document.media_count,
        "hydrate_dedupe": {
            "mode": "canonical_symlink",
            "key": (
                f"airtable-{document.kind}:{document.canonical_id}:"
                f"{settings_key}:{document.source_path}"
            ),
            "rank": 0,
        },
    }
    if document.title:
        metadata["title"] = document.title

    return [
        {
            "source": target,
            "label": document.label,
            "content": document.rendered,
            "prose": document.prose,
            "metadata": metadata,
        }
    ]
