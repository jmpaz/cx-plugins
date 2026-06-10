from __future__ import annotations

from typing import Any

PLUGIN_API_VERSION = "1"
PLUGIN_NAME = "xwitter"
PLUGIN_PRIORITY = 100

_EXPECTED_RESOLUTION_ERRORS = (RuntimeError, ValueError, OSError)


def normalize_manifest_config(
    raw_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if raw_config is None:
        return None
    if not isinstance(raw_config, dict):
        raise ValueError("xwitter config must be a mapping")
    return dict(raw_config)


def _xwitter_runtime_overrides(raw: dict[str, Any]) -> dict[str, Any] | None:
    result: dict[str, Any] = {}
    for key in (
        "include_html",
        "use_alias_fallback",
        "resolve_tco_links",
        "quote_depth",
    ):
        if key in raw:
            result[key] = raw[key]
    for config_key, result_key in (
        ("include-html", "include_html"),
        ("alias-fallback", "use_alias_fallback"),
        ("use-alias-fallback", "use_alias_fallback"),
        ("resolve-tco-links", "resolve_tco_links"),
        ("quote-depth", "quote_depth"),
    ):
        if config_key in raw:
            result[result_key] = raw.get(config_key)
    return result or None


def _xwitter_overrides(context: dict[str, Any]) -> dict[str, Any] | None:
    overrides = context.get("overrides")
    if not isinstance(overrides, dict):
        return None
    for key in ("xwitter", "x", "twitter"):
        value = overrides.get(key)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"{key} overrides must be a mapping")
        return _xwitter_runtime_overrides(value)
    return None


def can_resolve(target: str, context: dict[str, Any]) -> bool:
    from .xwitter import is_xwitter_url

    return is_xwitter_url(target)


def classify_target(target: str, context: dict[str, Any]) -> dict[str, Any] | None:
    from .xwitter import parse_xwitter_target

    parsed = parse_xwitter_target(target)
    if parsed is None:
        return None
    return {
        "provider": PLUGIN_NAME,
        "kind": parsed.kind,
        "is_external": True,
        "group_key": parsed.kind,
    }


def _failure_document(target: str, exc: BaseException) -> dict[str, Any]:
    from .xwitter import parse_xwitter_target

    parsed = parse_xwitter_target(target)
    source_path = target
    canonical_url = None
    tweet_id = None
    if parsed is not None:
        source_path = parsed.source_path
        canonical_url = parsed.canonical_url
        tweet_id = parsed.tweet_id
    return {
        "source": target,
        "label": target,
        "content": (
            "X/Twitter tweet URL was recognized but could not be resolved through "
            f"the public embed surfaces: {exc}"
        ),
        "metadata": {
            "trace_path": source_path,
            "provider": PLUGIN_NAME,
            "source_ref": "x.com",
            "source_path": source_path,
            "context_subpath": "xwitter-error.md",
            "kind": "tweet",
            "canonical_url": canonical_url,
            "tweet_id": tweet_id,
            "resolution_error": str(exc),
        },
    }


def resolve(target: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    from .xwitter import (
        build_xwitter_settings,
        resolve_xwitter_url,
        xwitter_settings_cache_key,
    )

    settings = build_xwitter_settings(_xwitter_overrides(context))
    settings_key = xwitter_settings_cache_key(settings)
    try:
        documents = resolve_xwitter_url(
            target,
            settings=settings,
            use_cache=bool(context.get("use_cache", True)),
            cache_ttl=context.get("cache_ttl"),
            refresh_cache=bool(context.get("refresh_cache", False)),
        )
    except _EXPECTED_RESOLUTION_ERRORS as exc:
        return [_failure_document(target, exc)]

    out: list[dict[str, Any]] = []
    for document in documents:
        out.append(
            {
                "source": target,
                "label": document.label,
                "content": document.rendered,
                "metadata": {
                    "trace_path": document.trace_path,
                    "provider": PLUGIN_NAME,
                    "source_ref": "x.com",
                    "source_path": document.source_path,
                    "context_subpath": document.context_subpath,
                    "source_created": document.source_created,
                    "source_modified": document.source_modified,
                    "kind": document.kind,
                    "canonical_url": document.canonical_url,
                    "tweet_id": document.tweet_id,
                    "author": document.author,
                    "settings_key": settings_key,
                    "hydrate_dedupe": {
                        "mode": "canonical_symlink",
                        "key": (
                            f"xwitter-tweet:{document.tweet_id}:"
                            f"{settings_key}:{document.source_path}"
                        ),
                        "rank": 0,
                    },
                },
            }
        )
    return out
