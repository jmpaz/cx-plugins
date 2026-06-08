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
_LISTED_BLOCK_CACHE: dict[str, dict[str, Any]] = {}
_RESOLUTION_MODE_ALIASES = {
    "digest": "digest",
    "richtoplevel": "richTopLevel",
    "rich-top-level": "richTopLevel",
    "rich_top_level": "richTopLevel",
    "richTopLevel": "richTopLevel",
    "recursive": "recursive",
}


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
        "include_connections",
        "connections_max_items",
        "include_link_image_descriptions",
        "include_pdf_content",
        "include_media_descriptions",
        "exclude_channels",
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
    if "exclude-channels" in raw:
        result["exclude_channels"] = raw.get("exclude-channels")

    resolution_mode = (
        raw.get("resolution-mode")
        if "resolution-mode" in raw
        else raw.get("resolutionMode", raw.get("resolution_mode"))
    )
    if resolution_mode is not None:
        if not isinstance(resolution_mode, str):
            raise ValueError("arena resolution-mode must be a string")
        normalized_resolution_mode = resolution_mode.strip()
        alias_key = (
            normalized_resolution_mode
            if normalized_resolution_mode in _RESOLUTION_MODE_ALIASES
            else normalized_resolution_mode.lower()
        )
        if alias_key not in _RESOLUTION_MODE_ALIASES:
            raise ValueError(
                "arena resolution-mode must be one of: digest, richTopLevel, recursive"
            )
        result["resolution_mode"] = _RESOLUTION_MODE_ALIASES[alias_key]

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
            ("connections", "include_connections"),
            ("connections-max-items", "connections_max_items"),
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
    if command_name not in {"cat", "hydrate", "payload"}:
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
            ["--arena-block-connections/--no-arena-block-connections"],
            default=None,
            help="Include or omit Are.na block channel provenance and connected channels.",
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--arena-block-connections-max-items"],
            type=int,
            default=None,
            help="Maximum connected Are.na channels to render per block.",
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
    _append_option(
        command,
        click.Option(
            ["--arena-exclude-channel"],
            multiple=True,
            help="Skip an Are.na channel by slug, id, or URL. Repeatable or comma-separated.",
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
    if command_name not in {"cat", "hydrate", "payload"}:
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
        ("arena_block_connections", "connections"),
        ("arena_link_image_descriptions", "link-image-desc"),
        ("arena_pdf_content", "pdf-content"),
        ("arena_media_descriptions", "media-desc"),
    ):
        value = params.get(param_key)
        if value is not None:
            block[config_key] = bool(value)
    connections_max_items = params.get("arena_block_connections_max_items")
    if connections_max_items is not None:
        if int(connections_max_items) <= 0:
            raise ValueError(
                "--arena-block-connections-max-items must be greater than 0"
            )
        block["connections-max-items"] = int(connections_max_items)
    if block:
        raw_mapping["block"] = block

    exclude_channels = [
        channel.strip()
        for value in params.get("arena_exclude_channel") or ()
        for channel in value.split(",")
        if channel.strip()
    ]
    if exclude_channels:
        raw_mapping["exclude-channels"] = exclude_channels

    if not raw_mapping:
        return None
    return normalize_manifest_config(raw_mapping)


def can_resolve(target: str, context: dict[str, Any]) -> bool:
    from .arena import is_arena_url

    return is_arena_url(target)


def classify_target(target: str, context: dict[str, Any]) -> dict[str, Any] | None:
    from .arena import (
        is_arena_block_attachment_url,
        is_arena_block_url,
        is_arena_channel_url,
        is_arena_group_target,
        is_arena_user_target,
    )

    if is_arena_block_attachment_url(target):
        return {
            "provider": PLUGIN_NAME,
            "kind": "attachment",
            "is_external": True,
            "group_key": "attachment",
        }
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
    if is_arena_user_target(target):
        return {
            "provider": PLUGIN_NAME,
            "kind": "user",
            "is_external": True,
            "group_key": "user",
        }
    if is_arena_group_target(target):
        return {
            "provider": PLUGIN_NAME,
            "kind": "group",
            "is_external": True,
            "group_key": "group",
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
        settings.include_connections,
        settings.connections_max_items,
        settings.include_link_image_descriptions,
        settings.include_pdf_content,
        settings.include_media_descriptions,
        recurse_key,
        settings.exclude_channels.cache_key(),
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


def _join_context_path(prefix: str | None, path: str) -> str:
    if not prefix:
        return path
    return f"{prefix.strip('/')}/{path.lstrip('/')}"


def _owner_targets(target: str) -> list[tuple[str, str]]:
    from .arena import extract_group_slug, extract_profile_slug, extract_user_slug

    profile_slug = extract_profile_slug(target)
    if profile_slug is not None:
        return [("user", profile_slug), ("group", profile_slug)]

    user_slug = extract_user_slug(target)
    if user_slug is not None:
        return [("user", user_slug)]
    group_slug = extract_group_slug(target)
    if group_slug is not None:
        return [("group", group_slug)]
    return []


def _owner_not_found(exc: ValueError, kind: str, slug: str) -> bool:
    message = str(exc)
    return message.endswith(
        f"Are.na resource not found: /{kind}s/{slug}"
    ) or message.endswith(f"Are.na resource not found: /{kind}s/{slug}/contents")


def _fetch_owner_channels_for_target(
    owners: list[tuple[str, str]],
    *,
    use_cache: bool,
    cache_ttl: Any,
    refresh_cache: bool,
    settings: Any,
) -> tuple[str, str, list[dict[str, Any]]] | None:
    from .arena import fetch_owner_channels

    last_missing: ValueError | None = None
    for kind, slug in owners:
        try:
            return (
                kind,
                slug,
                fetch_owner_channels(
                    kind,
                    slug,
                    use_cache=use_cache,
                    cache_ttl=cache_ttl,
                    refresh_cache=refresh_cache,
                    settings=settings,
                ),
            )
        except ValueError as exc:
            if len(owners) > 1 and _owner_not_found(exc, kind, slug):
                last_missing = exc
                continue
            raise
    if last_missing is not None:
        raise last_missing
    return None


def _fetch_owner_data_for_target(
    owners: list[tuple[str, str]],
    *,
    use_cache: bool,
    cache_ttl: Any,
    refresh_cache: bool,
    settings: Any,
) -> tuple[str, str, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]] | None:
    from .arena import fetch_owner_channels, fetch_owner_profile, fetch_user_groups

    last_missing: ValueError | None = None
    for kind, slug in owners:
        try:
            profile = fetch_owner_profile(
                kind,
                slug,
                use_cache=use_cache,
                cache_ttl=cache_ttl,
                refresh_cache=refresh_cache,
            )
            user_groups = (
                fetch_user_groups(
                    slug,
                    use_cache=use_cache,
                    cache_ttl=cache_ttl,
                    refresh_cache=refresh_cache,
                )
                if kind == "user"
                else []
            )
            channels = fetch_owner_channels(
                kind,
                slug,
                use_cache=use_cache,
                cache_ttl=cache_ttl,
                refresh_cache=refresh_cache,
                settings=settings,
            )
            return kind, slug, profile, user_groups, channels
        except ValueError as exc:
            if len(owners) > 1 and _owner_not_found(exc, kind, slug):
                last_missing = exc
                continue
            raise
    if last_missing is not None:
        raise last_missing
    return None


def _owner_context_prefix(kind: str, slug: str) -> str:
    return f"{kind}s/{slug}"


def _channel_target(slug: str) -> str:
    return f"https://www.are.na/channel/{slug}"


def _block_target(block: dict[str, Any]) -> str | None:
    block_type = block.get("type", "")
    if block_type == "Channel" or block.get("base_type") == "Channel":
        slug = block.get("slug")
        return _channel_target(slug) if isinstance(slug, str) and slug else None
    block_id = block.get("id")
    if isinstance(block_id, int):
        return f"https://www.are.na/block/{block_id}"
    if isinstance(block_id, str) and block_id.isdigit():
        return f"https://www.are.na/block/{block_id}"
    return None


def _canonical_block_target(block_id: Any) -> str | None:
    if isinstance(block_id, int):
        return f"https://www.are.na/block/{block_id}"
    if isinstance(block_id, str) and block_id.isdigit():
        return f"https://www.are.na/block/{block_id}"
    return None


def _remember_listed_block(target: str, block: dict[str, Any]) -> None:
    snapshot = dict(block)
    _LISTED_BLOCK_CACHE[target] = snapshot
    canonical = _canonical_block_target(block.get("id"))
    if canonical is not None:
        _LISTED_BLOCK_CACHE[canonical] = snapshot


def _cached_block_for_target(target: str) -> dict[str, Any] | None:
    cached = _LISTED_BLOCK_CACHE.get(target)
    if cached is not None:
        return dict(cached)
    from .arena import extract_block_id

    canonical = _canonical_block_target(extract_block_id(target))
    if canonical is None:
        return None
    cached = _LISTED_BLOCK_CACHE.get(canonical)
    return dict(cached) if cached is not None else None


def _cached_block_for_attachment_target(
    target: str, parsed: dict[str, Any]
) -> dict[str, Any] | None:
    block = _cached_block_for_target(target)
    if block is None:
        return None
    attachment = block.get("attachment")
    if not isinstance(attachment, dict):
        return block

    attachment_id = parsed.get("attachment_id")
    if attachment_id:
        candidates = [
            str(value).strip()
            for value in (attachment.get("id"), attachment.get("uuid"))
            if value is not None and str(value).strip()
        ]
        if candidates and str(attachment_id) not in candidates:
            return None

    attachment_name = parsed.get("attachment_name")
    if attachment_name:
        from .arena import _attachment_filename

        if _attachment_filename(attachment) != str(attachment_name):
            return None
    return block


def _block_list_label(block: dict[str, Any], fallback: str) -> str:
    title = block.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return fallback


def _entity_summary(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    summary: dict[str, Any] = {}
    for key in ("id", "slug", "name"):
        item = value.get(key)
        if item is not None:
            summary[key] = item
    return summary or None


def _block_connection_metadata(block: dict[str, Any]) -> dict[str, Any]:
    connection = block.get("connection")
    if not isinstance(connection, dict):
        connection = {}

    connected_by = connection.get("connected_by") or block.get("connected_by")
    metadata: dict[str, Any] = {}
    connection_id = connection.get("id")
    if connection_id is not None:
        metadata["connection_id"] = connection_id
    position = connection.get("position") or block.get("position")
    if position is not None:
        metadata["position"] = position
    connected_at = connection.get("connected_at") or block.get("connected_at")
    if connected_at is not None:
        metadata["connected_at"] = connected_at
    connected_by_summary = _entity_summary(connected_by)
    if connected_by_summary is not None:
        metadata["connected_by"] = connected_by_summary
    return metadata


def _block_source_metadata(block: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    source = block.get("source")
    if isinstance(source, dict):
        source_url = source.get("url")
        source_title = source.get("title")
        if isinstance(source_url, str) and source_url.strip():
            metadata["source_url"] = source_url.strip()
        if isinstance(source_title, str) and source_title.strip():
            metadata["source_title"] = source_title.strip()
    return metadata


def _nested_depth(channel_path: str) -> int:
    parts = [part for part in channel_path.split("/") if part]
    return max(0, len(parts) - 1)


def _target_metadata_for_block(
    block: dict[str, Any],
    *,
    channel_path: str,
    source_channel: str,
    relation: str,
) -> dict[str, Any]:
    block_type = str(block.get("type") or block.get("base_type") or "block")
    metadata: dict[str, Any] = {
        "relation": relation,
        "block_id": block.get("id"),
        "block_type": block_type,
        "channel_path": channel_path,
        "source_channel": source_channel,
        "nested_depth": _nested_depth(channel_path),
    }
    slug = block.get("slug")
    if block_type == "Channel" and isinstance(slug, str) and slug.strip():
        metadata["channel_slug"] = slug.strip()
    owner = _entity_summary(block.get("owner") or block.get("user"))
    if owner is not None:
        metadata["owner"] = owner
    metadata.update(_block_connection_metadata(block))
    metadata.update(_block_source_metadata(block))
    return metadata


def _channel_summary(channel: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("id", "slug", "title", "status", "created_at", "updated_at"):
        value = channel.get(key)
        if value is not None:
            summary[key] = value
    counts = channel.get("counts")
    if isinstance(counts, dict):
        summary["counts"] = counts
    owner = _entity_summary(channel.get("owner") or channel.get("user"))
    if owner is not None:
        summary["owner"] = owner
    return summary


def _type_breakdown(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for block in blocks:
        block_type = str(block.get("type") or block.get("base_type") or "block")
        counts[block_type] = counts.get(block_type, 0) + 1
    return [
        {"type": block_type, "count": count}
        for block_type, count in sorted(counts.items(), key=lambda item: item[0].lower())
    ]


def _block_target_summary(
    block: dict[str, Any],
    *,
    channel_path: str,
    source_channel: str,
    relation: str = "contains",
) -> dict[str, Any] | None:
    target = _block_target(block)
    if target is None:
        return None
    block_type = str(block.get("type") or block.get("base_type") or "block")
    is_channel = block_type == "Channel" or block.get("base_type") == "Channel"
    item = {
        "target": target,
        "label": _block_list_label(block, target),
        "kind": "channel" if is_channel else "block",
        "metadata": _target_metadata_for_block(
            block,
            channel_path=channel_path,
            source_channel=source_channel,
            relation=relation,
        ),
    }
    if is_channel:
        item["traverse"] = False
    return item


def _positive_int(value: object, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _arena_resolution_mode(arena_overrides: dict[str, Any] | None) -> str | None:
    if not arena_overrides:
        return None
    value = arena_overrides.get("resolution_mode")
    return value if isinstance(value, str) else None


def _owner_profile_document(
    *,
    target: str,
    kind: str,
    slug: str,
    profile: dict[str, Any],
    user_groups: list[dict[str, Any]],
    settings_key: tuple[Any, ...],
) -> dict[str, Any]:
    from .arena import render_owner_profile

    profile_id = profile.get("id")
    profile_slug = profile.get("slug") if isinstance(profile.get("slug"), str) else slug
    profile_name = profile.get("name") or profile_slug
    context_prefix = _owner_context_prefix(kind, slug)
    dedupe = None
    if profile_id is not None:
        dedupe = {
            "mode": "canonical_symlink",
            "key": f"arena-{kind}:{profile_id}:{settings_key}",
            "rank": -2,
        }
    return {
        "source": target,
        "label": profile_name,
        "content": render_owner_profile(
            kind,
            profile,
            user_groups=user_groups if kind == "user" else None,
        ),
        "metadata": {
            "trace_path": f"{context_prefix}/{profile_id or profile_slug}",
            "provider": PLUGIN_NAME,
            "source_ref": "are.na",
            "source_path": context_prefix,
            "context_subpath": f"{context_prefix}/_{kind}.md",
            "source_created": profile.get("created_at"),
            "source_modified": profile.get("updated_at"),
            "dir_created": profile.get("created_at"),
            "dir_modified": profile.get("updated_at"),
            "settings_key": settings_key,
            "hydrate_dedupe": dedupe,
        },
    }


def _channel_documents(
    *,
    target: str,
    slug: str,
    settings: Any,
    settings_key: tuple[Any, ...],
    use_cache: bool,
    cache_ttl: Any,
    refresh_cache: bool,
    context_prefix: str | None = None,
) -> list[dict[str, Any]]:
    from .arena import ArenaReference, resolve_channel, warmup_arena_network_stack

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
        include_connections=settings.include_connections,
        connections_max_items=settings.connections_max_items,
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
    out: list[dict[str, Any]] = [
        {
            "source": target,
            "label": channel_ref.get_label(),
            "content": channel_ref.read(),
            "metadata": {
                "trace_path": channel_ref.trace_path,
                "provider": PLUGIN_NAME,
                "source_ref": "are.na",
                "source_path": _join_context_path(context_prefix, slug),
                "context_subpath": _join_context_path(
                    context_prefix, f"{slug}/_channel.md"
                ),
                "block_id": channel_id,
                "block_type": "Channel",
                "channel_path": _join_context_path(context_prefix, slug),
                "source_channel": target,
                "nested_depth": 0,
                "source_created": channel_block.get("created_at"),
                "source_modified": channel_block.get("updated_at"),
                "dir_created": channel_block.get("created_at"),
                "dir_modified": channel_block.get("updated_at"),
                "settings_key": settings_key,
                "hydrate_dedupe": channel_dedupe,
            },
        }
    ]
    for channel_path, block in flat_blocks:
        block_type = block.get("type", "")
        is_channel = block_type == "Channel" or block.get("base_type") == "Channel"
        block_id = block.get("id")
        ref = ArenaReference(
            target,
            block=block,
            channel_path=channel_path,
            format="raw",
            inject=False,
            include_descriptions=settings.include_descriptions,
            include_comments=settings.include_comments,
            include_connections=settings.include_connections,
            connections_max_items=settings.connections_max_items,
            include_link_image_descriptions=settings.include_link_image_descriptions,
            include_pdf_content=settings.include_pdf_content,
            include_media_descriptions=settings.include_media_descriptions,
        )
        ref_label = ref.get_label()
        label = ref_label or str(block_id or "block")
        label_path = label if "/" in label else f"{slug}/{label}"
        channel_parts = [part for part in channel_path.split("/") if part]
        if channel_parts and channel_parts[0] == slug:
            channel_parts = channel_parts[1:]
        context_subpath = f"{label_path}.md"
        source_path = label_path
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
                    "source_path": _join_context_path(context_prefix, source_path),
                    "context_subpath": _join_context_path(
                        context_prefix, context_subpath
                    ),
                    "source_created": block.get("connected_at")
                    or block.get("created_at"),
                    "source_modified": block.get("updated_at"),
                    "dir_created": channel_meta.get("created_at")
                    if isinstance(channel_meta, dict)
                    else None,
                    "dir_modified": channel_meta.get("updated_at")
                    if isinstance(channel_meta, dict)
                    else None,
                    "block_id": block_id,
                    "block_type": block_type or block.get("base_type"),
                    "channel_path": _join_context_path(context_prefix, channel_path),
                    "source_channel": target,
                    "nested_depth": len(channel_parts),
                    **_block_connection_metadata(block),
                    **_block_source_metadata(block),
                    "settings_key": settings_key,
                    "hydrate_dedupe": dedupe,
                },
            }
        )
    return out


def _arena_listing_settings(settings: Any) -> dict[str, Any]:
    return {
        "blockSort": settings.sort_order,
        "recurseDepth": settings.max_depth,
        "maxBlocksPerChannel": settings.max_blocks_per_channel,
    }


def _arena_listing_envelope(
    *,
    target: str,
    kind: str,
    targets: list[dict[str, Any]],
    settings: Any,
    summary: dict[str, Any] | None = None,
    pagination: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    capabilities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    envelope_metadata: dict[str, Any] = {
        "provider": PLUGIN_NAME,
        "kind": kind,
        "target": target,
        "settings": _arena_listing_settings(settings),
    }
    if metadata:
        envelope_metadata.update(metadata)
    return {
        "targets": targets,
        "summary": summary or {},
        "pagination": pagination
        or {"returned": len(targets), "totalCount": len(targets), "hasMore": False},
        "metadata": envelope_metadata,
        "capabilities": capabilities
        or {"resolve": True, "listTargets": True, "materialize": False},
    }


def list_targets(target: str, context: dict[str, Any]) -> dict[str, Any]:
    from .arena import (
        _fetch_block,
        build_arena_settings,
        extract_block_id,
        extract_channel_slug,
        is_arena_block_url,
        is_arena_channel_url,
        list_arena_block_targets,
        parse_arena_block_attachment_target,
        resolve_channel,
        warmup_arena_network_stack,
    )

    arena_overrides = _arena_overrides(context)
    settings = build_arena_settings(arena_overrides)
    use_cache = bool(context.get("use_cache", True))
    cache_ttl = context.get("cache_ttl")
    refresh_cache = bool(context.get("refresh_cache", False))

    if parse_arena_block_attachment_target(target) is not None:
        return _arena_listing_envelope(
            target=target,
            kind="attachment",
            targets=[],
            settings=settings,
            capabilities={"resolve": True, "listTargets": False, "materialize": True},
        )

    if is_arena_channel_url(target):
        slug = extract_channel_slug(target)
        if not slug:
            return _arena_listing_envelope(
                target=target,
                kind="channel",
                targets=[],
                settings=settings,
                capabilities={"resolve": False, "listTargets": True, "materialize": False},
            )
        settings = _apply_channel_safety_defaults(settings, arena_overrides)
        warmup_arena_network_stack()
        _channel_meta, flat_blocks = resolve_channel(
            slug,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
            settings=settings,
        )
        items: list[dict[str, Any]] = []
        for channel_path, block in flat_blocks:
            item = _block_target_summary(
                block,
                channel_path=channel_path,
                source_channel=target,
            )
            if item is None:
                continue
            child_target = item["target"]
            _remember_listed_block(child_target, block)
            items.append(item)
        channel_block = dict(_channel_meta) if isinstance(_channel_meta, dict) else {}
        channel_block.setdefault("type", "Channel")
        channel_block.setdefault("base_type", "Channel")
        channel_block.setdefault("slug", slug)
        if bool(context.get("include_containing", False)):
            items = [
                *_containing_channel_targets(
                    channel_block,
                    source_target=_channel_target(slug),
                    settings=settings,
                ),
                *items,
            ]
        blocks = [block for _channel_path, block in flat_blocks]
        total_count = (_channel_meta.get("counts") or {}).get("contents")
        collaborators = _channel_meta.get("collaborators") or _channel_meta.get("users") or []
        if not isinstance(collaborators, list):
            collaborators = []
        return _arena_listing_envelope(
            target=_channel_target(slug),
            kind="channel",
            targets=items,
            settings=settings,
            summary={
                "channel": _channel_summary(_channel_meta),
                "typeBreakdown": _type_breakdown(blocks),
                "sampledItems": items,
                "collaborators": [
                    summary
                    for value in collaborators
                    if (summary := _entity_summary(value)) is not None
                ],
            },
            pagination={
                "returned": len(items),
                "totalCount": total_count if isinstance(total_count, int) else None,
                "hasMore": bool(
                    isinstance(total_count, int)
                    and settings.max_blocks_per_channel is not None
                    and total_count > settings.max_blocks_per_channel
                ),
            },
            metadata={"channel_slug": slug},
        )
    if is_arena_block_url(target):
        block_id = extract_block_id(target)
        if block_id is None:
            return _arena_listing_envelope(
                target=target,
                kind="block",
                targets=[],
                settings=settings,
                capabilities={"resolve": False, "listTargets": True, "materialize": False},
            )
        block = _cached_block_for_target(target) or _fetch_block(block_id)
        items = list_arena_block_targets(block, source_target=target)
        if bool(context.get("include_containing", False)):
            items = [
                *items,
                *_containing_channel_targets(
                    block,
                    source_target=target,
                    settings=settings,
                ),
            ]
        return _arena_listing_envelope(
            target=target,
            kind="block",
            targets=items,
            settings=settings,
            summary={
                "block": _block_target_summary(
                    block,
                    channel_path="",
                    source_channel=target,
                    relation="self",
                )
            },
            pagination={"returned": len(items), "totalCount": len(items), "hasMore": False},
            metadata={
                "block_id": block_id,
                "block_type": block.get("type") or block.get("base_type"),
            },
            capabilities={
                "resolve": True,
                "listTargets": True,
                "materialize": any(
                    str(item.get("kind") or "").startswith("attachment:")
                    for item in items
                ),
            },
        )

    owners = _owner_targets(target)
    if not owners:
        return _arena_listing_envelope(
            target=target,
            kind="unknown",
            targets=[],
            settings=settings,
            capabilities={"resolve": False, "listTargets": False, "materialize": False},
        )
    warmup_arena_network_stack()
    owner_channels = _fetch_owner_channels_for_target(
        owners,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
        settings=settings,
    )
    if owner_channels is None:
        return _arena_listing_envelope(
            target=target,
            kind="owner",
            targets=[],
            settings=settings,
            capabilities={"resolve": False, "listTargets": True, "materialize": False},
        )
    kind, slug, channels = owner_channels
    items = [
        {
            "target": _channel_target(channel_slug),
            "label": channel.get("title") or channel_slug,
            "kind": "channel",
            "traverse": False,
            "metadata": {"owner_kind": kind, "owner_slug": slug},
        }
        for channel in channels
        if isinstance((channel_slug := channel.get("slug")), str) and channel_slug
    ]
    return _arena_listing_envelope(
        target=target,
        kind=kind,
        targets=items,
        settings=settings,
        summary={"owner": {"kind": kind, "slug": slug}, "channelCount": len(items)},
        metadata={"owner_kind": kind, "owner_slug": slug},
    )


def _progressive_channel_settings(
    settings: Any,
    arena_overrides: dict[str, Any] | None,
    *,
    default_depth: int,
    max_blocks: int | None = None,
) -> Any:
    changes: dict[str, Any] = {}
    if not arena_overrides or "max_depth" not in arena_overrides:
        changes["max_depth"] = default_depth
    if max_blocks is not None:
        changes["max_blocks_per_channel"] = max_blocks
    return replace(settings, **changes) if changes else settings


def _digest_sample_size(context: dict[str, Any]) -> int:
    return _positive_int(context.get("sample_size") or context.get("sampleSize"), 24)


def _containing_channel_targets(
    block: dict[str, Any],
    *,
    source_target: str,
    settings: Any,
) -> list[dict[str, Any]]:
    from .arena import _fetch_block_connections, _fetch_channel_connections

    block_id = block.get("id")
    if not isinstance(block_id, int):
        return []

    block_type = str(block.get("type") or block.get("base_type") or "block")
    source_context = block.get("_contextualize_channel_context")
    if not isinstance(source_context, dict):
        source_context = None

    if block_type == "Channel" or block.get("base_type") == "Channel":
        channels, cap_hit = _fetch_channel_connections(
            block_id,
            max_items=settings.connections_max_items,
            source_context=source_context,
        )
    else:
        channels, cap_hit = _fetch_block_connections(
            block_id,
            max_items=settings.connections_max_items,
            source_context=source_context,
        )

    items: list[dict[str, Any]] = []
    for channel in channels:
        slug = channel.get("slug")
        if not isinstance(slug, str) or not slug.strip():
            continue
        metadata: dict[str, Any] = {
            "relation": "contained_by",
            "source_target": source_target,
            "source_block_id": block_id,
            "source_block_type": block_type,
            "channel_id": channel.get("id"),
            "channel_slug": slug.strip(),
        }
        owner = _entity_summary(channel.get("owner") or channel.get("user"))
        if owner is not None:
            metadata["owner"] = owner
        metadata.update(_block_connection_metadata(channel))
        items.append(
            {
                "target": _channel_target(slug.strip()),
                "label": channel.get("title") or slug.strip(),
                "kind": "channel",
                "traverse": False,
                "metadata": metadata,
            }
        )
    if cap_hit:
        items.append(
            {
                "target": source_target,
                "label": "More containing channels available",
                "kind": "pagination",
                "metadata": {
                    "relation": "contained_by",
                    "truncated": True,
                    "max_items": settings.connections_max_items,
                },
            }
        )
    return items


def arena_digest(target: str, context: dict[str, Any]) -> dict[str, Any]:
    from .arena import (
        build_arena_settings,
        extract_channel_slug,
        is_arena_channel_url,
        resolve_channel,
        warmup_arena_network_stack,
    )

    if not is_arena_channel_url(target):
        raise ValueError("arena_digest requires an Are.na channel URL")

    slug = extract_channel_slug(target)
    if not slug:
        raise ValueError("arena_digest could not determine channel slug")

    arena_overrides = _arena_overrides(context)
    settings = build_arena_settings(arena_overrides)
    sample_size = _digest_sample_size(context)
    settings = _progressive_channel_settings(
        settings,
        arena_overrides,
        default_depth=0,
        max_blocks=sample_size,
    )
    settings = _apply_channel_safety_defaults(settings, arena_overrides)

    use_cache = bool(context.get("use_cache", True))
    cache_ttl = context.get("cache_ttl")
    refresh_cache = bool(context.get("refresh_cache", False))

    warmup_arena_network_stack()
    channel_meta, flat_blocks = resolve_channel(
        slug,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
        settings=settings,
    )
    blocks = [block for _channel_path, block in flat_blocks]
    total_count = (channel_meta.get("counts") or {}).get("contents")
    sampled_items = [
        item
        for channel_path, block in flat_blocks
        if (
            item := _block_target_summary(
                block,
                channel_path=channel_path,
                source_channel=target,
            )
        )
        is not None
    ]

    collaborators = channel_meta.get("collaborators") or channel_meta.get("users") or []
    if not isinstance(collaborators, list):
        collaborators = []

    return {
        "target": _channel_target(slug),
        "channel": _channel_summary(channel_meta),
        "typeBreakdown": _type_breakdown(blocks),
        "sampledItems": sampled_items,
        "collaborators": [
            summary
            for value in collaborators
            if (summary := _entity_summary(value)) is not None
        ],
        "pagination": {
            "sampleSize": len(sampled_items),
            "requestedSampleSize": sample_size,
            "totalCount": total_count if isinstance(total_count, int) else None,
            "hasMore": bool(isinstance(total_count, int) and total_count > len(sampled_items)),
        },
        "settings": {
            "blockSort": settings.sort_order,
            "recurseDepth": settings.max_depth,
            "maxBlocksPerChannel": settings.max_blocks_per_channel,
        },
    }


def _render_arena_digest_document(digest: dict[str, Any]) -> str:
    channel = digest.get("channel") if isinstance(digest.get("channel"), dict) else {}
    pagination = (
        digest.get("pagination") if isinstance(digest.get("pagination"), dict) else {}
    )
    title = channel.get("title") or channel.get("slug") or digest.get("target")
    lines = [f"# {title}", "", f"target: {digest.get('target')}"]
    if channel.get("slug"):
        lines.append(f"slug: {channel.get('slug')}")
    counts = channel.get("counts") if isinstance(channel.get("counts"), dict) else {}
    if "contents" in counts:
        lines.append(f"contents: {counts.get('contents')}")
    if pagination:
        if pagination.get("sampleSize") is not None:
            lines.append(f"sampled: {pagination.get('sampleSize')}")
        if pagination.get("hasMore") is not None:
            lines.append(f"has_more: {pagination.get('hasMore')}")
    lines.append("")

    type_breakdown = digest.get("typeBreakdown")
    if isinstance(type_breakdown, list) and type_breakdown:
        lines.append("## Type Breakdown")
        for item in type_breakdown:
            if not isinstance(item, dict):
                continue
            block_type = item.get("type")
            count = item.get("count")
            if block_type is not None and count is not None:
                lines.append(f"- {block_type}: {count}")
        lines.append("")

    sampled_items = digest.get("sampledItems")
    if isinstance(sampled_items, list) and sampled_items:
        lines.append("## Sampled Targets")
        for item in sampled_items:
            if not isinstance(item, dict):
                continue
            item_target = item.get("target")
            label = item.get("label") or item_target
            kind = item.get("kind") or "target"
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            relation = metadata.get("relation")
            relation_suffix = f" [{relation}]" if isinstance(relation, str) else ""
            lines.append(f"- {kind}{relation_suffix}: {label} <{item_target}>")
        lines.append("")

    collaborators = digest.get("collaborators")
    if isinstance(collaborators, list) and collaborators:
        lines.append("## Collaborators")
        for collaborator in collaborators:
            if not isinstance(collaborator, dict):
                continue
            name = collaborator.get("name") or collaborator.get("slug") or collaborator.get("id")
            if name is not None:
                lines.append(f"- {name}")

    return "\n".join(lines).strip()


def _arena_digest_document(
    *,
    target: str,
    context: dict[str, Any],
    settings_key: tuple[Any, ...],
) -> dict[str, Any]:
    digest = arena_digest(target, context)
    channel = digest.get("channel") if isinstance(digest.get("channel"), dict) else {}
    slug = channel.get("slug") if isinstance(channel.get("slug"), str) else target
    title = channel.get("title") if isinstance(channel.get("title"), str) else slug
    return {
        "source": target,
        "label": f"{title} digest",
        "content": _render_arena_digest_document(digest),
        "metadata": {
            "trace_path": f"{slug}/_digest",
            "provider": PLUGIN_NAME,
            "source_ref": "are.na",
            "source_path": slug,
            "context_subpath": f"{slug}/_digest.md",
            "block_id": channel.get("id"),
            "block_type": "Channel",
            "channel_path": slug,
            "source_channel": target,
            "nested_depth": 0,
            "resolution_mode": "digest",
            "settings_key": settings_key,
            "digest": digest,
        },
    }


def arena_targets(target: str, context: dict[str, Any]) -> dict[str, Any]:
    from .arena import (
        _fetch_block,
        build_arena_settings,
        extract_block_id,
        extract_channel_slug,
        is_arena_block_url,
        is_arena_channel_url,
        list_arena_block_targets,
        resolve_channel,
        warmup_arena_network_stack,
    )

    arena_overrides = _arena_overrides(context)
    settings = build_arena_settings(arena_overrides)
    settings = _progressive_channel_settings(
        settings,
        arena_overrides,
        default_depth=0,
    )
    settings = _apply_channel_safety_defaults(settings, arena_overrides)
    include_containing = bool(context.get("include_containing", True))
    use_cache = bool(context.get("use_cache", True))
    cache_ttl = context.get("cache_ttl")
    refresh_cache = bool(context.get("refresh_cache", False))

    if is_arena_channel_url(target):
        slug = extract_channel_slug(target)
        if not slug:
            return {"target": target, "targets": [], "pagination": None}
        warmup_arena_network_stack()
        channel_meta, flat_blocks = resolve_channel(
            slug,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
            settings=settings,
        )
        targets: list[dict[str, Any]] = []
        channel_block = dict(channel_meta) if isinstance(channel_meta, dict) else {}
        channel_block.setdefault("type", "Channel")
        channel_block.setdefault("base_type", "Channel")
        channel_block.setdefault("slug", slug)
        if include_containing:
            targets.extend(
                _containing_channel_targets(
                    channel_block,
                    source_target=_channel_target(slug),
                    settings=settings,
                )
            )
        for channel_path, block in flat_blocks:
            item = _block_target_summary(
                block,
                channel_path=channel_path,
                source_channel=target,
            )
            if item is None:
                continue
            _remember_listed_block(item["target"], block)
            targets.append(item)
        total_count = (channel_meta.get("counts") or {}).get("contents")
        return {
            "target": _channel_target(slug),
            "channel": _channel_summary(channel_meta),
            "targets": targets,
            "pagination": {
                "returned": len(targets),
                "totalCount": total_count if isinstance(total_count, int) else None,
                "hasMore": bool(
                    isinstance(total_count, int)
                    and settings.max_blocks_per_channel is not None
                    and total_count > settings.max_blocks_per_channel
                ),
            },
            "settings": {
                "blockSort": settings.sort_order,
                "recurseDepth": settings.max_depth,
                "maxBlocksPerChannel": settings.max_blocks_per_channel,
            },
        }

    if is_arena_block_url(target):
        block_id = extract_block_id(target)
        if block_id is None:
            return {"target": target, "targets": [], "pagination": None}
        block = _cached_block_for_target(target) or _fetch_block(block_id)
        targets = list_arena_block_targets(block, source_target=target)
        for item in targets:
            metadata = item.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata.setdefault("relation", "resolves_to")
                metadata.setdefault("source_target", target)
        if include_containing:
            targets.extend(
                _containing_channel_targets(
                    block,
                    source_target=target,
                    settings=settings,
                )
            )
        return {
            "target": target,
            "block": _block_target_summary(
                block,
                channel_path="",
                source_channel=target,
                relation="self",
            ),
            "targets": targets,
            "pagination": {"returned": len(targets), "totalCount": len(targets), "hasMore": False},
            "settings": {
                "blockSort": settings.sort_order,
                "recurseDepth": settings.max_depth,
                "maxBlocksPerChannel": settings.max_blocks_per_channel,
            },
        }

    listed = list_targets(target, context)
    listed_targets = listed.get("targets")
    listed_pagination = listed.get("pagination")
    if not isinstance(listed_targets, list):
        listed_targets = []
    return {
        "target": target,
        "targets": listed_targets,
        "pagination": listed_pagination if isinstance(listed_pagination, dict) else None,
        "settings": {
            "blockSort": settings.sort_order,
            "recurseDepth": settings.max_depth,
            "maxBlocksPerChannel": settings.max_blocks_per_channel,
        },
    }


def materialize(target: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    from .arena import (
        materialize_arena_attachment_target,
        parse_arena_block_attachment_target,
    )

    parsed = parse_arena_block_attachment_target(target)
    if parsed is None:
        return []
    block = _cached_block_for_attachment_target(target, parsed)
    return materialize_arena_attachment_target(
        target,
        parsed,
        use_cache=bool(context.get("use_cache", True)),
        cache_ttl=context.get("cache_ttl"),
        refresh_cache=bool(context.get("refresh_cache", False)),
        cache_only=bool(context.get("cache_only", False)),
        block=block,
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
        parse_arena_block_attachment_target,
        resolve_arena_attachment_target,
        warmup_arena_network_stack,
    )

    arena_overrides = _arena_overrides(context)
    settings = build_arena_settings(arena_overrides)
    settings_key = _settings_key(settings)
    use_cache = bool(context.get("use_cache", True))
    cache_ttl = context.get("cache_ttl")
    refresh_cache = bool(context.get("refresh_cache", False))

    out: list[dict[str, Any]] = []
    parsed_attachment = parse_arena_block_attachment_target(target)
    if parsed_attachment is not None:
        block = _cached_block_for_attachment_target(target, parsed_attachment)
        return resolve_arena_attachment_target(
            target,
            parsed_attachment,
            settings_key=settings_key,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
            plugin_overrides=context.get("overrides"),
            block=block,
        )

    owners = _owner_targets(target)
    if owners:
        settings = _apply_channel_safety_defaults(settings, arena_overrides)
        settings_key = _settings_key(settings)
        warmup_arena_network_stack()
        owner_data = _fetch_owner_data_for_target(
            owners,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
            settings=settings,
        )
        if owner_data is None:
            return out
        kind, owner_slug, profile, user_groups, channels = owner_data
        out.append(
            _owner_profile_document(
                target=target,
                kind=kind,
                slug=owner_slug,
                profile=profile,
                user_groups=user_groups,
                settings_key=settings_key,
            )
        )
        context_prefix = _owner_context_prefix(kind, owner_slug)
        for channel in channels:
            channel_slug = channel.get("slug")
            if not isinstance(channel_slug, str) or not channel_slug:
                continue
            out.extend(
                _channel_documents(
                    target=target,
                    slug=channel_slug,
                    settings=settings,
                    settings_key=settings_key,
                    use_cache=use_cache,
                    cache_ttl=cache_ttl,
                    refresh_cache=refresh_cache,
                    context_prefix=context_prefix,
                )
            )
        return out

    if is_arena_channel_url(target):
        if resolution_mode == "digest":
            return [
                _arena_digest_document(
                    target=target,
                    context=context,
                    settings_key=settings_key,
                )
            ]
        if resolution_mode == "richTopLevel":
            settings = _progressive_channel_settings(
                settings,
                arena_overrides,
                default_depth=0,
            )
        settings = _apply_channel_safety_defaults(settings, arena_overrides)
        settings_key = _settings_key(settings)
        slug = extract_channel_slug(target)
        if not slug:
            return out
        return _channel_documents(
            target=target,
            slug=slug,
            settings=settings,
            settings_key=settings_key,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
        )

    if is_arena_block_url(target):
        block_id = extract_block_id(target)
        if block_id is None:
            return out
        block = _cached_block_for_target(target) or _fetch_block(block_id)
        ref = ArenaReference(
            target,
            block=block,
            format="raw",
            inject=False,
            include_descriptions=settings.include_descriptions,
            include_comments=settings.include_comments,
            include_connections=settings.include_connections,
            connections_max_items=settings.connections_max_items,
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
                    "block_id": block_id,
                    "block_type": block.get("type") or block.get("base_type"),
                    "source_channel": target,
                    "nested_depth": 0,
                    **_block_connection_metadata(block),
                    **_block_source_metadata(block),
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
