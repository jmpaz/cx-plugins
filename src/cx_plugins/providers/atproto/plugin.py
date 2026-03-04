from __future__ import annotations

from typing import Any

PLUGIN_API_VERSION = "1"
PLUGIN_NAME = "atproto"
PLUGIN_PRIORITY = 100


def normalize_manifest_config(
    raw_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if raw_config is None:
        return None
    if not isinstance(raw_config, dict):
        raise ValueError("atproto config must be a mapping")
    return dict(raw_config)


def _atproto_runtime_overrides(raw: dict[str, Any]) -> dict[str, Any] | None:
    result: dict[str, Any] = {}
    for key in (
        "max_items",
        "thread_depth",
        "post_ancestors",
        "quote_depth",
        "max_replies",
        "reply_quote_depth",
        "replies_filter",
        "reposts_filter",
        "likes_filter",
        "created_after",
        "created_before",
        "include_media_descriptions",
        "include_embed_media_descriptions",
        "media_mode",
        "include_lineage",
    ):
        if key in raw:
            result[key] = raw[key]

    for config_key, result_key in (
        ("max-items", "max_items"),
        ("thread-depth", "thread_depth"),
        ("post-ancestors", "post_ancestors"),
        ("quote-depth", "quote_depth"),
        ("max-replies", "max_replies"),
        ("reply-quote-depth", "reply_quote_depth"),
        ("created-after", "created_after"),
        ("created-before", "created_before"),
        ("include-media-descriptions", "include_media_descriptions"),
        ("include-embed-media-descriptions", "include_embed_media_descriptions"),
        ("media-mode", "media_mode"),
        ("include-lineage", "include_lineage"),
    ):
        if config_key in raw:
            result[result_key] = raw.get(config_key)

    for config_key, result_key in (
        ("replies", "replies_filter"),
        ("reposts", "reposts_filter"),
        ("likes", "likes_filter"),
    ):
        if config_key in raw:
            result[result_key] = raw.get(config_key)

    return result or None


def _atproto_overrides(context: dict[str, Any]) -> dict[str, Any] | None:
    overrides = context.get("overrides")
    if not isinstance(overrides, dict):
        return None
    value = overrides.get("atproto")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("atproto overrides must be a mapping")
    return _atproto_runtime_overrides(value)


def can_resolve(target: str, context: dict[str, Any]) -> bool:
    from .atproto import is_atproto_url

    return is_atproto_url(target)


def classify_target(target: str, context: dict[str, Any]) -> dict[str, Any] | None:
    from .atproto import parse_atproto_target

    parsed = parse_atproto_target(target)
    if parsed is None:
        return None
    return {
        "provider": PLUGIN_NAME,
        "kind": parsed.kind,
        "is_external": True,
        "group_key": parsed.kind,
    }


def _parse_frontmatter(rendered: str) -> dict[str, Any] | None:
    if not rendered.startswith("---\n"):
        return None
    end = rendered.find("\n---\n", 4)
    if end == -1:
        return None
    block = rendered[4:end]
    try:
        import yaml

        parsed = yaml.safe_load(block)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _metadata_str(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _route_kind(frontmatter: dict[str, Any]) -> str | None:
    lineage_role = _metadata_str(frontmatter, "lineage_role")
    reply_root_uri = _metadata_str(frontmatter, "reply_root_uri")
    reply_to_uri = _metadata_str(frontmatter, "reply_to_uri")
    if lineage_role == "root_anchor" or reply_root_uri or reply_to_uri:
        return "replies"
    entry_type = _metadata_str(frontmatter, "entry_type") or "post"
    if entry_type == "repost":
        return "reposts"
    if entry_type == "like":
        return "likes"
    if entry_type == "post":
        return "posts"
    return None


def _source_path(uri: str, fallback: str) -> str:
    if uri.startswith("at://"):
        return uri.replace("at://", "", 1)
    return uri or fallback


def resolve(target: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    from .atproto import (
        atproto_settings_cache_key,
        build_atproto_settings,
        resolve_atproto_url,
    )

    settings = build_atproto_settings(_atproto_overrides(context))
    settings_key = atproto_settings_cache_key(settings)
    documents = resolve_atproto_url(
        target,
        settings=settings,
        use_cache=bool(context.get("use_cache", True)),
        cache_ttl=context.get("cache_ttl"),
        refresh_cache=bool(context.get("refresh_cache", False)),
    )
    route_priority = {"posts": 0, "replies": 1, "reposts": 2, "likes": 3}
    out: list[dict[str, Any]] = []
    for document in documents:
        uri = document.uri
        source_path = _source_path(uri, document.trace_path)
        metadata: dict[str, Any] = {
            "trace_path": document.trace_path,
            "provider": PLUGIN_NAME,
            "source_ref": "atproto",
            "source_path": source_path,
            "context_subpath": document.context_subpath,
            "source_created": document.source_created,
            "source_modified": document.source_modified,
            "kind": document.kind,
            "uri": document.uri,
            "settings_key": settings_key,
        }
        frontmatter = _parse_frontmatter(document.rendered)
        if isinstance(frontmatter, dict):
            route_kind = _route_kind(frontmatter)
            if route_kind in route_priority:
                metadata["hydrate_dedupe"] = {
                    "mode": "canonical_symlink",
                    "key": f"atproto-post:{target}:{settings_key}:{source_path}",
                    "rank": route_priority[route_kind],
                }
        out.append(
            {
                "source": target,
                "label": document.label,
                "content": document.rendered,
                "metadata": metadata,
            }
        )
    return out


def register_auth_command(group: Any) -> None:
    from .atproto_auth import register_auth_command as _register_auth_command

    _register_auth_command(group)
