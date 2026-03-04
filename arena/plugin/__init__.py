from __future__ import annotations

from typing import Any

PLUGIN_API_VERSION = "1"
PLUGIN_NAME = "arena"
PLUGIN_PRIORITY = 100


def _arena_overrides(context: dict[str, Any]) -> dict[str, Any] | None:
    overrides = context.get("overrides")
    if not isinstance(overrides, dict):
        return None
    value = overrides.get("arena")
    return value if isinstance(value, dict) else None


def can_resolve(target: str, context: dict[str, Any]) -> bool:
    from contextualize.references.arena import is_arena_url

    return is_arena_url(target)


def _settings_key(settings: Any) -> tuple[Any, ...]:
    recurse_users = settings.recurse_users
    recurse_key: tuple[str, ...] | None
    if recurse_users is None:
        recurse_key = None
    else:
        recurse_key = tuple(sorted(str(value) for value in recurse_users))
    return (
        settings.max_depth,
        settings.sort_order,
        settings.max_blocks_per_channel,
        settings.include_descriptions,
        settings.include_comments,
        settings.include_link_image_descriptions,
        settings.include_pdf_content,
        settings.include_media_descriptions,
        recurse_key,
    )


def resolve(target: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    from contextualize.references.arena import (
        ArenaReference,
        _fetch_block,
        build_arena_settings,
        extract_block_id,
        extract_channel_slug,
        is_arena_block_url,
        is_arena_channel_url,
        resolve_channel,
        warmup_arena_network_stack,
    )

    settings = build_arena_settings(_arena_overrides(context))
    settings_key = _settings_key(settings)
    use_cache = bool(context.get("use_cache", True))
    cache_ttl = context.get("cache_ttl")
    refresh_cache = bool(context.get("refresh_cache", False))

    out: list[dict[str, Any]] = []
    if is_arena_channel_url(target):
        slug = extract_channel_slug(target)
        if not slug:
            return out
        warmup_arena_network_stack()
        channel_meta, flat_blocks = resolve_channel(
            slug,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
            settings=settings,
        )
        for channel_path, block in flat_blocks:
            block_type = block.get("type", "")
            is_channel = block_type == "Channel" or block.get("base_type") == "Channel"
            block_id = block.get("id")
            label = ref_label = ""
            ref = ArenaReference(
                target,
                block=block,
                channel_path=channel_path,
                format="raw",
                inject=False,
                include_descriptions=settings.include_descriptions,
                include_comments=settings.include_comments,
                include_link_image_descriptions=settings.include_link_image_descriptions,
                include_pdf_content=settings.include_pdf_content,
                include_media_descriptions=settings.include_media_descriptions,
            )
            ref_label = ref.get_label()
            label = ref_label or str(block_id or "block")
            channel_parts = [part for part in channel_path.split("/") if part]
            if channel_parts and channel_parts[0] == slug:
                channel_parts = channel_parts[1:]
            channel_subdir = "/".join(channel_parts)
            context_subpath = (
                f"{slug}/{channel_subdir}/{label}.md"
                if channel_subdir
                else f"{slug}/{label}.md"
            )
            source_path = f"{slug}/{label}"
            dedupe = None
            if is_channel and block_id is not None:
                dedupe = {
                    "mode": "canonical_symlink",
                    "key": f"arena-channel:{block_id}:{settings_key}",
                    "rank": len(channel_parts),
                }
            elif block_id is not None:
                dedupe = {
                    "mode": "canonical_symlink",
                    "key": f"arena-block:{block_id}:{settings_key}",
                    "rank": len(channel_parts),
                }
            out.append(
                {
                    "source": target,
                    "label": ref_label,
                    "content": ref.read(),
                    "metadata": {
                        "trace_path": ref.trace_path,
                        "provider": PLUGIN_NAME,
                        "source_ref": "are.na",
                        "source_path": source_path,
                        "context_subpath": context_subpath,
                        "source_created": block.get("connected_at")
                        or block.get("created_at"),
                        "source_modified": block.get("updated_at"),
                        "dir_created": channel_meta.get("created_at")
                        if isinstance(channel_meta, dict)
                        else None,
                        "dir_modified": channel_meta.get("updated_at")
                        if isinstance(channel_meta, dict)
                        else None,
                        "settings_key": settings_key,
                        "hydrate_dedupe": dedupe,
                    },
                }
            )
        return out

    if is_arena_block_url(target):
        block_id = extract_block_id(target)
        if block_id is None:
            return out
        block = _fetch_block(block_id)
        ref = ArenaReference(
            target,
            block=block,
            format="raw",
            inject=False,
            include_descriptions=settings.include_descriptions,
            include_comments=settings.include_comments,
            include_link_image_descriptions=settings.include_link_image_descriptions,
            include_pdf_content=settings.include_pdf_content,
            include_media_descriptions=settings.include_media_descriptions,
        )
        out.append(
            {
                "source": target,
                "label": ref.get_label(),
                "content": ref.read(),
                "metadata": {
                    "trace_path": ref.trace_path,
                    "provider": PLUGIN_NAME,
                    "source_ref": "are.na",
                    "source_path": str(block_id),
                    "context_subpath": f"arena-block-{block_id}.md",
                    "source_created": block.get("connected_at")
                    or block.get("created_at"),
                    "source_modified": block.get("updated_at"),
                    "settings_key": settings_key,
                    "hydrate_dedupe": {
                        "mode": "canonical_symlink",
                        "key": f"arena-block:{block_id}:{settings_key}",
                        "rank": 0,
                    },
                },
            }
        )
    return out


def register_auth_command(group: Any) -> None:
    from .arena_auth import register_auth_command as _register_auth_command

    _register_auth_command(group)
