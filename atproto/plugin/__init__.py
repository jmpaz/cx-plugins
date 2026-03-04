from __future__ import annotations

from typing import Any

PLUGIN_API_VERSION = "1"
PLUGIN_NAME = "atproto"
PLUGIN_PRIORITY = 100


def _atproto_overrides(context: dict[str, Any]) -> dict[str, Any] | None:
    overrides = context.get("overrides")
    if not isinstance(overrides, dict):
        return None
    value = overrides.get("atproto")
    return value if isinstance(value, dict) else None


def can_resolve(target: str, context: dict[str, Any]) -> bool:
    from contextualize.references.atproto import is_atproto_url

    return is_atproto_url(target)


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
    from contextualize.references.atproto import (
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
