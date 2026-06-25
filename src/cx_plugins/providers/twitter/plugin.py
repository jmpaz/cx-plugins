from __future__ import annotations

from typing import Any

PLUGIN_API_VERSION = "1"
PLUGIN_NAME = "twitter"
PLUGIN_PRIORITY = 100

_EXPECTED_RESOLUTION_ERRORS = (RuntimeError, ValueError, OSError)


def normalize_manifest_config(
    raw_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if raw_config is None:
        return None
    if not isinstance(raw_config, dict):
        raise ValueError("twitter config must be a mapping")
    return dict(raw_config)


def _twitter_runtime_overrides(raw: dict[str, Any]) -> dict[str, Any] | None:
    result: dict[str, Any] = {}
    for key in (
        "include_html",
        "use_fx_api",
        "include_media_descriptions",
        "media_mode",
        "quote_depth",
    ):
        if key in raw:
            result[key] = raw[key]
    for config_key, result_key in (
        ("include-html", "include_html"),
        ("use-fx-api", "use_fx_api"),
        ("fx-api", "use_fx_api"),
        ("include-media-descriptions", "include_media_descriptions"),
        ("media-descriptions", "include_media_descriptions"),
        ("media-mode", "media_mode"),
        ("quote-depth", "quote_depth"),
    ):
        if config_key in raw:
            result[result_key] = raw.get(config_key)
    media = raw.get("media")
    if isinstance(media, dict):
        if "describe" in media:
            result["include_media_descriptions"] = media["describe"]
        if "mode" in media:
            result["media_mode"] = media["mode"]
    return result or None


def _twitter_overrides(context: dict[str, Any]) -> dict[str, Any] | None:
    overrides = context.get("overrides")
    if not isinstance(overrides, dict):
        return None
    value = overrides.get("twitter")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("twitter overrides must be a mapping")
    return _twitter_runtime_overrides(value)


def can_resolve(target: str, context: dict[str, Any]) -> bool:
    from .twitter import is_twitter_url

    return is_twitter_url(target)


def classify_target(target: str, context: dict[str, Any]) -> dict[str, Any] | None:
    from .twitter import parse_twitter_target

    parsed = parse_twitter_target(target)
    if parsed is None:
        return None
    return {
        "provider": PLUGIN_NAME,
        "kind": parsed.kind,
        "is_external": True,
        "group_key": parsed.kind,
    }


def _failure_document(target: str, exc: BaseException) -> dict[str, Any]:
    from .twitter import parse_twitter_target

    parsed = parse_twitter_target(target)
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
        "prose": "",
        "metadata": {
            "trace_path": source_path,
            "provider": PLUGIN_NAME,
            "source_ref": "x.com",
            "source_path": source_path,
            "context_subpath": "twitter-error.md",
            "kind": "tweet",
            "canonical_url": canonical_url,
            "tweet_id": tweet_id,
            "resolution_error": str(exc),
        },
    }


def resolve(target: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    from .twitter import (
        build_twitter_settings,
        resolve_twitter_url,
        twitter_settings_cache_key,
    )

    settings = build_twitter_settings(_twitter_overrides(context))
    settings_key = twitter_settings_cache_key(settings)
    try:
        documents = resolve_twitter_url(
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
                "prose": document.prose,
                "prose_authors": document.prose_authors,
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
                            f"twitter-tweet:{document.tweet_id}:"
                            f"{settings_key}:{document.source_path}"
                        ),
                        "rank": 0,
                    },
                },
            }
        )
    return out
