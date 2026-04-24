from __future__ import annotations

from dataclasses import replace
from typing import Any

import click

PLUGIN_API_VERSION = "1"
PLUGIN_NAME = "arena"
PLUGIN_PRIORITY = 100
DEFAULT_RICH_MEDIA_CHANNEL_MAX_BLOCKS = 10
_ARENA_SORT_CHOICES = (
    "asc",
    "desc",
    "date-asc",
    "date-desc",
    "random",
    "position-asc",
    "position-desc",
)


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
        "recurse_blocks",
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
    if "recurse-blocks" in raw:
        result["recurse_blocks"] = raw.get("recurse-blocks")

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


def _option_already_registered(command: click.Command, *, name: str) -> bool:
    return any(getattr(param, "name", None) == name for param in command.params)


def _append_option(command: click.Command, option: click.Option) -> None:
    if _option_already_registered(command, name=option.name):
        return
    command.params.append(option)


def register_cli_options(command_name: str, command: click.Command) -> None:
    if command_name not in {"cat", "hydrate"}:
        return
    _append_option(
        command,
        click.Option(
            ["--arena-recurse-depth"],
            type=int,
            default=None,
            help="Maximum nested Are.na channel recursion depth. 0 disables recursion.",
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--arena-recurse-user", "--arena-recurse-users"],
            multiple=True,
            help=(
                "Only recurse into nested channels owned by this Are.na slug. "
                "Repeatable or comma-separated; use 'all' by itself to recurse into every owner's channels."
            ),
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--arena-max-blocks-per-channel"],
            type=int,
            default=None,
            help="Limit blocks fetched from each Are.na channel before nested recursion.",
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--arena-recurse-blocks"],
            default=None,
            help=(
                "Limit blocks fetched from recursed Are.na channels. "
                "Accepts a positive integer, decimal ratio like 0.05, or percent like 5%."
            ),
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--arena-block-sort"],
            type=click.Choice(_ARENA_SORT_CHOICES, case_sensitive=False),
            default=None,
            help="Order Are.na channel blocks before limiting and recursion.",
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--arena-connected-after"],
            default=None,
            help="Keep channel blocks connected at or after this timestamp or duration.",
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--arena-connected-before"],
            default=None,
            help="Keep channel blocks connected at or before this timestamp or duration.",
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--arena-created-after"],
            default=None,
            help="Keep channel blocks created at or after this timestamp or duration.",
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--arena-created-before"],
            default=None,
            help="Keep channel blocks created at or before this timestamp or duration.",
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--arena-block-descriptions/--no-arena-block-descriptions"],
            default=None,
            help="Include or omit Are.na block descriptions.",
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--arena-block-comments/--no-arena-block-comments"],
            default=None,
            help="Include or omit Are.na block comments.",
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--arena-link-image-descriptions/--no-arena-link-image-descriptions"],
            default=None,
            help="Describe images attached to Are.na link blocks.",
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--arena-pdf-content/--no-arena-pdf-content"],
            default=None,
            help="Extract or skip PDF attachment content in Are.na blocks.",
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--arena-media-descriptions/--no-arena-media-descriptions"],
            default=None,
            help="Describe or skip Are.na image, video, and audio block media.",
        ),
    )


def _collect_recurse_users(values: tuple[str, ...]) -> str | list[str] | None:
    users = [
        user.strip().lower()
        for value in values
        for user in value.split(",")
        if user.strip()
    ]
    if not users:
        return None
    if "all" in users:
        if len(users) > 1:
            raise ValueError("--arena-recurse-user all cannot be combined with slugs")
        return "all"
    return users


def _add_optional_value(
    mapping: dict[str, Any], key: str, value: Any, *, positive_int: bool = False
) -> None:
    if value is None:
        return
    if positive_int and int(value) <= 0:
        raise ValueError(f"--arena-{key} must be greater than 0")
    mapping[key] = value


def collect_cli_overrides(
    command_name: str,
    params: dict[str, Any],
) -> dict[str, Any] | None:
    if command_name not in {"cat", "hydrate"}:
        return None

    raw_mapping: dict[str, Any] = {}
    recurse_depth = params.get("arena_recurse_depth")
    if recurse_depth is not None:
        if int(recurse_depth) < 0:
            raise ValueError("--arena-recurse-depth must be zero or greater")
        raw_mapping["recurse-depth"] = int(recurse_depth)

    recurse_users = _collect_recurse_users(params.get("arena_recurse_user") or ())
    if recurse_users is not None:
        raw_mapping["recurse-users"] = recurse_users

    _add_optional_value(
        raw_mapping,
        "max-blocks-per-channel",
        params.get("arena_max_blocks_per_channel"),
        positive_int=True,
    )
    recurse_blocks = params.get("arena_recurse_blocks")
    if isinstance(recurse_blocks, str) and recurse_blocks.strip():
        raw_mapping["recurse-blocks"] = recurse_blocks.strip()

    block_sort = params.get("arena_block_sort")
    if isinstance(block_sort, str) and block_sort.strip():
        raw_mapping["block-sort"] = block_sort.strip().lower()

    for param_key, config_key in (
        ("arena_connected_after", "connected-after"),
        ("arena_connected_before", "connected-before"),
        ("arena_created_after", "created-after"),
        ("arena_created_before", "created-before"),
    ):
        value = params.get(param_key)
        if isinstance(value, str) and value.strip():
            raw_mapping[config_key] = value.strip()

    block: dict[str, Any] = {}
    for param_key, config_key in (
        ("arena_block_descriptions", "description"),
        ("arena_block_comments", "comments"),
        ("arena_link_image_descriptions", "link-image-desc"),
        ("arena_pdf_content", "pdf-content"),
        ("arena_media_descriptions", "media-desc"),
    ):
        value = params.get(param_key)
        if value is not None:
            block[config_key] = bool(value)
    if block:
        raw_mapping["block"] = block

    if not raw_mapping:
        return None
    return normalize_manifest_config(raw_mapping)


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
        (
            settings.recurse_blocks.cache_key()
            if settings.recurse_blocks is not None
            else None
        ),
        settings.include_descriptions,
        settings.include_comments,
        settings.include_link_image_descriptions,
        settings.include_pdf_content,
        settings.include_media_descriptions,
        recurse_key,
    )


def _has_explicit_rich_media_overrides(overrides: dict[str, Any] | None) -> bool:
    if not overrides:
        return False
    return bool(
        overrides.get("include_link_image_descriptions") is True
        or overrides.get("include_pdf_content") is True
        or overrides.get("include_media_descriptions") is True
    )


def _has_time_window(settings: Any) -> bool:
    return bool(
        settings.connected_after
        or settings.connected_before
        or settings.created_after
        or settings.created_before
    )


def _apply_channel_safety_defaults(
    settings: Any, overrides: dict[str, Any] | None
) -> Any:
    if (
        settings.max_blocks_per_channel is not None
        or settings.recurse_blocks is not None
        or _has_time_window(settings)
        or not _has_explicit_rich_media_overrides(overrides)
    ):
        return settings
    return replace(
        settings, max_blocks_per_channel=DEFAULT_RICH_MEDIA_CHANNEL_MAX_BLOCKS
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

    arena_overrides = _arena_overrides(context)
    settings = build_arena_settings(arena_overrides)
    settings_key = _settings_key(settings)
    use_cache = bool(context.get("use_cache", True))
    cache_ttl = context.get("cache_ttl")
    refresh_cache = bool(context.get("refresh_cache", False))

    out: list[dict[str, Any]] = []
    if is_arena_channel_url(target):
        settings = _apply_channel_safety_defaults(settings, arena_overrides)
        settings_key = _settings_key(settings)
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
        channel_block = dict(channel_meta) if isinstance(channel_meta, dict) else {}
        channel_block.setdefault("type", "Channel")
        channel_block.setdefault("base_type", "Channel")
        channel_block.setdefault("slug", slug)
        channel_ref = ArenaReference(
            target,
            block=channel_block,
            format="raw",
            inject=False,
            include_descriptions=settings.include_descriptions,
            include_comments=settings.include_comments,
            include_link_image_descriptions=settings.include_link_image_descriptions,
            include_pdf_content=settings.include_pdf_content,
            include_media_descriptions=settings.include_media_descriptions,
        )
        channel_id = channel_block.get("id")
        channel_dedupe = None
        if channel_id is not None:
            channel_dedupe = {
                "mode": "canonical_symlink",
                "key": f"arena-channel:{channel_id}:{settings_key}",
                "rank": -1,
            }
        out.append(
            {
                "source": target,
                "label": channel_ref.get_label(),
                "content": channel_ref.read(),
                "metadata": {
                    "trace_path": channel_ref.trace_path,
                    "provider": PLUGIN_NAME,
                    "source_ref": "are.na",
                    "source_path": slug,
                    "context_subpath": f"{slug}/_channel.md",
                    "source_created": channel_block.get("created_at"),
                    "source_modified": channel_block.get("updated_at"),
                    "dir_created": channel_block.get("created_at"),
                    "dir_modified": channel_block.get("updated_at"),
                    "settings_key": settings_key,
                    "hydrate_dedupe": channel_dedupe,
                },
            }
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
