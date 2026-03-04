from __future__ import annotations

from typing import Any

PLUGIN_API_VERSION = "1"
PLUGIN_NAME = "arena"
PLUGIN_PRIORITY = 100


def normalize_manifest_config(
    raw_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if raw_config is None:
        return None
    if not isinstance(raw_config, dict):
        raise ValueError("arena config must be a mapping")
    return dict(raw_config)


def _arena_runtime_overrides(raw: dict[str, Any]) -> dict[str, Any] | None:
    result: dict[str, Any] = {}
    for key in (
        "max_depth",
        "sort_order",
        "max_blocks_per_channel",
        "connected_after",
        "connected_before",
        "created_after",
        "created_before",
        "recurse_users",
        "include_descriptions",
        "include_comments",
        "include_link_image_descriptions",
        "include_pdf_content",
        "include_media_descriptions",
    ):
        if key in raw:
            result[key] = raw[key]

    recurse_depth = raw.get("recurse-depth")
    max_depth_alias = raw.get("max-depth")
    if (
        recurse_depth is not None
        and max_depth_alias is not None
        and recurse_depth != max_depth_alias
    ):
        raise ValueError("arena recurse-depth and max-depth cannot differ")
    if recurse_depth is None:
        recurse_depth = max_depth_alias
    if recurse_depth is not None:
        result["max_depth"] = recurse_depth

    block_sort = raw.get("block-sort")
    sort_alias = raw.get("sort")
    if block_sort is not None and sort_alias is not None:
        if str(block_sort).strip().lower() != str(sort_alias).strip().lower():
            raise ValueError("arena block-sort and sort cannot differ")
    if block_sort is None:
        block_sort = sort_alias
    if block_sort is not None:
        result["sort_order"] = str(block_sort).strip().lower()

    if "max-blocks-per-channel" in raw:
        result["max_blocks_per_channel"] = raw.get("max-blocks-per-channel")

    for config_key, result_key in (
        ("connected-after", "connected_after"),
        ("connected-before", "connected_before"),
        ("created-after", "created_after"),
        ("created-before", "created_before"),
    ):
        if config_key in raw:
            result[result_key] = raw.get(config_key)

    block_cfg = raw.get("block")
    if block_cfg is not None:
        if not isinstance(block_cfg, dict):
            raise ValueError("arena block config must be a mapping")
        for config_key, result_key in (
            ("description", "include_descriptions"),
            ("comments", "include_comments"),
            ("link-image-desc", "include_link_image_descriptions"),
            ("pdf-content", "include_pdf_content"),
            ("media-desc", "include_media_descriptions"),
        ):
            if config_key in block_cfg:
                result[result_key] = block_cfg.get(config_key)

    if "recurse-users" in raw:
        recurse_users = raw.get("recurse-users")
        if isinstance(recurse_users, str):
            value = recurse_users.strip().lower()
            if value == "all":
                result["recurse_users"] = None
            elif value:
                result["recurse_users"] = {value}
        elif isinstance(recurse_users, list):
            values = {
                str(item).strip().lower() for item in recurse_users if str(item).strip()
            }
            result["recurse_users"] = values
        else:
            raise ValueError("arena recurse-users must be a string or list")

    return result or None


def _arena_overrides(context: dict[str, Any]) -> dict[str, Any] | None:
    overrides = context.get("overrides")
    if not isinstance(overrides, dict):
        return None
    value = overrides.get("arena")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("arena overrides must be a mapping")
    return _arena_runtime_overrides(value)


def can_resolve(target: str, context: dict[str, Any]) -> bool:
    from .arena import is_arena_url

    return is_arena_url(target)


def classify_target(target: str, context: dict[str, Any]) -> dict[str, Any] | None:
    from .arena import is_arena_block_url, is_arena_channel_url

    if is_arena_channel_url(target):
        return {
            "provider": PLUGIN_NAME,
            "kind": "channel",
            "is_external": True,
            "group_key": "channel",
        }
    if is_arena_block_url(target):
        return {
            "provider": PLUGIN_NAME,
            "kind": "block",
            "is_external": True,
            "group_key": "block",
        }
    return None


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
    from .arena import (
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
