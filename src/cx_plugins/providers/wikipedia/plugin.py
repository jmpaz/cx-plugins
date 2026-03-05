from __future__ import annotations

from typing import Any

PLUGIN_API_VERSION = "1"
PLUGIN_NAME = "wikipedia"
PLUGIN_PRIORITY = 100


def normalize_manifest_config(
    raw_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if raw_config is None:
        return None
    if not isinstance(raw_config, dict):
        raise ValueError("wikipedia config must be a mapping")
    return dict(raw_config)


def _wikipedia_runtime_overrides(raw: dict[str, Any]) -> dict[str, Any] | None:
    result: dict[str, Any] = {}

    for key in (
        "default_lang",
        "include_media",
        "include_media_descriptions",
        "include_references",
        "include_external_links",
        "include_categories",
    ):
        if key in raw:
            result[key] = raw[key]

    for config_key, result_key in (
        ("default-lang", "default_lang"),
        ("include-media", "include_media"),
        ("media-descriptions", "include_media_descriptions"),
        ("include-references", "include_references"),
        ("include-external-links", "include_external_links"),
        ("include-categories", "include_categories"),
    ):
        if config_key in raw:
            result[result_key] = raw.get(config_key)

    media_cfg = raw.get("media")
    if media_cfg is not None:
        if not isinstance(media_cfg, dict):
            raise ValueError("wikipedia media config must be a mapping")
        for config_key, result_key in (
            ("enabled", "include_media"),
            ("describe", "include_media_descriptions"),
        ):
            if config_key in media_cfg:
                result[result_key] = media_cfg.get(config_key)

    return result or None


def _wikipedia_overrides(context: dict[str, Any]) -> dict[str, Any] | None:
    overrides = context.get("overrides")
    if not isinstance(overrides, dict):
        return None
    value = overrides.get("wikipedia")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("wikipedia overrides must be a mapping")
    return _wikipedia_runtime_overrides(value)


def _resolve_default_lang(context: dict[str, Any]) -> str:
    overrides = _wikipedia_overrides(context) or {}
    value = overrides.get("default_lang")
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return "en"


def can_resolve(target: str, context: dict[str, Any]) -> bool:
    from .wikipedia import is_wikipedia_target

    return is_wikipedia_target(target, default_lang=_resolve_default_lang(context))


def classify_target(target: str, context: dict[str, Any]) -> dict[str, Any] | None:
    from .wikipedia import parse_wikipedia_target

    parsed = parse_wikipedia_target(target, default_lang=_resolve_default_lang(context))
    if parsed is None:
        return None
    return {
        "provider": PLUGIN_NAME,
        "kind": "article",
        "is_external": True,
        "group_key": "article",
    }


def resolve(target: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    from .wikipedia import (
        build_wikipedia_settings,
        resolve_wikipedia_article,
        wikipedia_settings_cache_key,
    )

    settings = build_wikipedia_settings(_wikipedia_overrides(context))
    settings_key = wikipedia_settings_cache_key(settings)
    document = resolve_wikipedia_article(
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
                "trace_path": document.label,
                "provider": PLUGIN_NAME,
                "source_ref": document.source_ref,
                "source_path": document.source_path,
                "context_subpath": document.context_subpath,
                "kind": document.kind,
                "canonical_id": document.canonical_id,
                "settings_key": settings_key,
                "hydrate_dedupe": {
                    "mode": "canonical_symlink",
                    "key": (
                        f"wikipedia-article:{document.canonical_id}:{settings_key}:"
                        f"{document.source_path}"
                    ),
                    "rank": 0,
                },
            },
        }
    ]
