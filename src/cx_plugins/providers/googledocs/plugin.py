from __future__ import annotations

from typing import Any

PLUGIN_API_VERSION = "1"
PLUGIN_NAME = "googledocs"
PLUGIN_PRIORITY = 100


def normalize_manifest_config(
    raw_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if raw_config is None:
        return None
    if not isinstance(raw_config, dict):
        raise ValueError("googledocs config must be a mapping")
    return dict(raw_config)


def _googledocs_runtime_overrides(raw: dict[str, Any]) -> dict[str, Any] | None:
    result: dict[str, Any] = {}

    for key in (
        "include_media",
        "include_media_descriptions",
        "timeout_seconds",
    ):
        if key in raw:
            result[key] = raw[key]

    for config_key, result_key in (
        ("include-media", "include_media"),
        ("media-descriptions", "include_media_descriptions"),
        ("timeout", "timeout_seconds"),
        ("timeout-seconds", "timeout_seconds"),
    ):
        if config_key in raw:
            result[result_key] = raw.get(config_key)

    media_cfg = raw.get("media")
    if media_cfg is not None:
        if not isinstance(media_cfg, dict):
            raise ValueError("googledocs media config must be a mapping")
        for config_key, result_key in (
            ("enabled", "include_media"),
            ("describe", "include_media_descriptions"),
        ):
            if config_key in media_cfg:
                result[result_key] = media_cfg.get(config_key)

    return result or None


def _googledocs_overrides(context: dict[str, Any]) -> dict[str, Any] | None:
    overrides = context.get("overrides")
    if not isinstance(overrides, dict):
        return None
    value = overrides.get("googledocs")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("googledocs overrides must be a mapping")
    return _googledocs_runtime_overrides(value)


def can_resolve(target: str, context: dict[str, Any]) -> bool:
    from .googledocs import is_google_doc_target

    return is_google_doc_target(target)


def classify_target(target: str, context: dict[str, Any]) -> dict[str, Any] | None:
    from .googledocs import parse_google_doc_target

    parsed = parse_google_doc_target(target)
    if parsed is None:
        return None
    return {
        "provider": PLUGIN_NAME,
        "kind": "document",
        "is_external": True,
        "group_key": "document",
    }

def _failure_document(
    target: str,
    exc: BaseException,
    *,
    settings_key: str,
) -> dict[str, Any]:
    from .googledocs import parse_google_doc_target

    parsed = parse_google_doc_target(target)
    canonical_id = parsed.canonical_id if parsed is not None else target
    source_path = canonical_id
    label_id = canonical_id if parsed is not None else "unresolved"
    label = f"googledocs/{label_id}-error"
    message = (
        "Google Docs public export failed. This provider only resolves public "
        f"Google Docs export targets; generic webpage fallback was suppressed: {exc}"
    )
    return {
        "source": target,
        "label": label,
        "content": message,
        "metadata": {
            "trace_path": label,
            "provider": PLUGIN_NAME,
            "source_ref": "docs.google.com",
            "source_path": source_path,
            "context_subpath": f"{label}.md",
            "kind": "document",
            "canonical_id": canonical_id,
            "settings_key": settings_key,
            "media_count": 0,
            "resolution_error": str(exc),
            "hydrate_dedupe": {
                "mode": "canonical_symlink",
                "key": f"googledocs-error:{canonical_id}:{settings_key}",
                "rank": 0,
            },
        },
    }



def resolve(target: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    from .googledocs import (
        build_google_docs_settings,
        google_docs_settings_cache_key,
        resolve_google_doc,
    )

    settings = build_google_docs_settings(_googledocs_overrides(context))
    settings_key = google_docs_settings_cache_key(settings)
    try:
        document = resolve_google_doc(
            target,
            settings=settings,
            use_cache=bool(context.get("use_cache", True)),
            cache_ttl=context.get("cache_ttl"),
            refresh_cache=bool(context.get("refresh_cache", False)),
        )
    except RuntimeError as exc:
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
                f"googledocs-document:{document.canonical_id}:"
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
            "metadata": metadata,
        }
    ]
