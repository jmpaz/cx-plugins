from __future__ import annotations

from typing import Any

PLUGIN_API_VERSION = "1"
PLUGIN_NAME = "forum_magnum"
PLUGIN_PRIORITY = 100


def normalize_manifest_config(
    raw_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if raw_config is None:
        return None
    if not isinstance(raw_config, dict):
        raise ValueError("forum_magnum config must be a mapping")
    return _forum_magnum_runtime_overrides(raw_config)


def _forum_magnum_runtime_overrides(raw: dict[str, Any]) -> dict[str, Any] | None:
    result: dict[str, Any] = {}

    for key in (
        "include_comments",
        "max_comments",
        "comment_batch_size",
        "comment_view",
    ):
        if key in raw:
            result[key] = raw[key]

    for config_key, result_key in (
        ("include-comments", "include_comments"),
        ("comments-enabled", "include_comments"),
        ("comments-limit", "max_comments"),
        ("max-comments", "max_comments"),
        ("comments-batch-size", "comment_batch_size"),
        ("comment-batch-size", "comment_batch_size"),
        ("comments-view", "comment_view"),
        ("comment-view", "comment_view"),
    ):
        if config_key in raw:
            result[result_key] = raw.get(config_key)

    comments = raw.get("comments")
    if isinstance(comments, bool):
        result["include_comments"] = comments
    elif isinstance(comments, dict):
        for config_key, result_key in (
            ("enabled", "include_comments"),
            ("include", "include_comments"),
            ("limit", "max_comments"),
            ("max", "max_comments"),
            ("batch-size", "comment_batch_size"),
            ("batch_size", "comment_batch_size"),
            ("view", "comment_view"),
        ):
            if config_key in comments:
                result[result_key] = comments.get(config_key)
    elif comments is not None:
        raise ValueError("forum_magnum comments config must be a mapping or boolean")

    return result or None


def _forum_magnum_overrides(context: dict[str, Any]) -> dict[str, Any] | None:
    overrides = context.get("overrides")
    if not isinstance(overrides, dict):
        return None
    value = overrides.get("forum_magnum")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("forum_magnum overrides must be a mapping")
    return _forum_magnum_runtime_overrides(value)


def can_resolve(target: str, context: dict[str, Any]) -> bool:
    from .forum_magnum import is_forum_magnum_target

    return is_forum_magnum_target(target)


def classify_target(target: str, context: dict[str, Any]) -> dict[str, Any] | None:
    from .forum_magnum import parse_forum_magnum_target

    parsed = parse_forum_magnum_target(target)
    if parsed is None:
        return None
    return {
        "provider": PLUGIN_NAME,
        "kind": "post",
        "is_external": True,
        "group_key": parsed.platform,
        "metadata": {
            "platform": parsed.platform,
            "post_id": parsed.post_id,
            "input_host": parsed.input_host,
        },
    }


def resolve(target: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    from .forum_magnum import (
        build_forum_magnum_settings,
        forum_magnum_settings_cache_key,
        resolve_forum_magnum_url,
    )

    settings = build_forum_magnum_settings(_forum_magnum_overrides(context))
    settings_key = forum_magnum_settings_cache_key(settings)
    documents = resolve_forum_magnum_url(
        target,
        settings=settings,
        use_cache=bool(context.get("use_cache", True)),
        cache_ttl=context.get("cache_ttl"),
        refresh_cache=bool(context.get("refresh_cache", False)),
    )

    out: list[dict[str, Any]] = []
    for document in documents:
        metadata = {
            "trace_path": document.trace_path,
            "provider": PLUGIN_NAME,
            "source_ref": document.source_ref,
            "source_path": document.source_path,
            "context_subpath": document.context_subpath,
            "source_created": document.source_created,
            "source_modified": document.source_modified,
            "kind": document.kind,
            "platform": document.platform,
            "canonical_id": document.canonical_id,
            "canonical_url": document.canonical_url,
            "post_id": document.post_id,
            "comment_id": document.comment_id,
            "parent_comment_id": document.parent_comment_id,
            "author": document.author,
            "score": document.score,
            "settings_key": settings_key,
            "hydrate_dedupe": {
                "mode": "canonical_symlink",
                "key": (
                    f"forum-magnum:{document.canonical_id}:"
                    f"{settings_key}:{document.source_path}"
                ),
                "rank": document.dedupe_rank,
            },
        }
        out.append(
            {
                "source": target,
                "label": document.label,
                "content": document.rendered,
                "prose": document.prose,
                "prose_authors": list(document.prose_authors),
                "metadata": metadata,
            }
        )
    return out
