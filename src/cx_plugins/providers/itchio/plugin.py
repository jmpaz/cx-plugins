from __future__ import annotations

from typing import Any

import click

PLUGIN_API_VERSION = "1"
PLUGIN_NAME = "itchio"
PLUGIN_PRIORITY = 100


def normalize_manifest_config(raw_config: dict[str, Any] | None) -> dict[str, Any] | None:
    if raw_config is None:
        return None
    if not isinstance(raw_config, dict):
        raise ValueError("itchio config must be a mapping")
    return _itchio_runtime_overrides(raw_config)


def _itchio_runtime_overrides(raw: dict[str, Any]) -> dict[str, Any] | None:
    result: dict[str, Any] = {}

    def section(name: str) -> dict[str, Any]:
        value = result.setdefault(name, {})
        if not isinstance(value, dict):
            raise ValueError(f"itchio {name} config must be a mapping")
        return value

    for name in ("theme", "comments", "devlogs", "media"):
        value = raw.get(name)
        if value is None:
            continue
        if isinstance(value, bool) and name == "theme":
            section("theme")["enabled"] = value
            continue
        if not isinstance(value, dict):
            raise ValueError(f"itchio {name} config must be a mapping")
        section(name).update(value)

    for key, path in (
        ("theme-enabled", ("theme", "enabled")),
        ("theme_enabled", ("theme", "enabled")),
        ("comments-enabled", ("comments", "enabled")),
        ("comments_enabled", ("comments", "enabled")),
        ("comments-limit", ("comments", "limit")),
        ("comments_limit", ("comments", "limit")),
        ("comments-offset", ("comments", "offset")),
        ("comments_offset", ("comments", "offset")),
        ("include-devlogs", ("devlogs", "include")),
        ("include_devlogs", ("devlogs", "include")),
        ("devlogs-include", ("devlogs", "include")),
        ("devlogs_include", ("devlogs", "include")),
        ("devlogs-limit", ("devlogs", "limit")),
        ("devlogs_limit", ("devlogs", "limit")),
        ("media-enabled", ("media", "enabled")),
        ("media_enabled", ("media", "enabled")),
        ("media-descriptions", ("media", "describe")),
        ("media_descriptions", ("media", "describe")),
        ("media-describe", ("media", "describe")),
        ("media_describe", ("media", "describe")),
    ):
        if key in raw:
            section(path[0])[path[1]] = raw[key]

    return result or None


def _itchio_overrides(context: dict[str, Any]) -> dict[str, Any] | None:
    overrides = context.get("overrides")
    if not isinstance(overrides, dict):
        return None
    value = overrides.get("itchio")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("itchio overrides must be a mapping")
    return _itchio_runtime_overrides(value)


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
            ["--itch-theme/--no-itch-theme"],
            default=None,
            help="Include itch.io page theme colors in resolved documents.",
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--itch-comments-limit"],
            type=int,
            default=None,
            help="Number of itch.io comments to include per page.",
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--itch-comments-offset"],
            type=int,
            default=None,
            help="Offset into itch.io comments before rendering.",
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--itch-include-devlogs/--no-itch-include-devlogs"],
            default=None,
            help="Fetch and include game devlog post bodies.",
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--itch-devlogs-limit"],
            type=int,
            default=None,
            help="Maximum devlog posts to fetch when devlog inclusion is enabled.",
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--itch-media/--no-itch-media"],
            default=None,
            help="Expose itch.io page media as materializable child targets.",
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--itch-media-descriptions/--no-itch-media-descriptions"],
            default=None,
            help="Request media descriptions when downstream media processors support them.",
        ),
    )


def collect_cli_overrides(
    command_name: str,
    params: dict[str, Any],
) -> dict[str, Any] | None:
    if command_name not in {"cat", "hydrate", "payload"}:
        return None

    raw: dict[str, Any] = {}
    theme = params.get("itch_theme")
    if theme is not None:
        raw.setdefault("theme", {})["enabled"] = bool(theme)

    comments_limit = params.get("itch_comments_limit")
    if comments_limit is not None:
        if int(comments_limit) < 0:
            raise ValueError("--itch-comments-limit must be zero or greater")
        raw.setdefault("comments", {})["limit"] = int(comments_limit)

    comments_offset = params.get("itch_comments_offset")
    if comments_offset is not None:
        if int(comments_offset) < 0:
            raise ValueError("--itch-comments-offset must be zero or greater")
        raw.setdefault("comments", {})["offset"] = int(comments_offset)

    include_devlogs = params.get("itch_include_devlogs")
    if include_devlogs is not None:
        raw.setdefault("devlogs", {})["include"] = bool(include_devlogs)

    devlogs_limit = params.get("itch_devlogs_limit")
    if devlogs_limit is not None:
        if int(devlogs_limit) <= 0:
            raise ValueError("--itch-devlogs-limit must be greater than 0")
        raw.setdefault("devlogs", {})["limit"] = int(devlogs_limit)

    media = params.get("itch_media")
    if media is not None:
        raw.setdefault("media", {})["enabled"] = bool(media)

    media_descriptions = params.get("itch_media_descriptions")
    if media_descriptions is not None:
        raw.setdefault("media", {})["describe"] = bool(media_descriptions)

    return raw or None


def can_resolve(target: str, context: dict[str, Any]) -> bool:
    from .itchio import is_itchio_target

    return is_itchio_target(target)


def classify_target(target: str, context: dict[str, Any]) -> dict[str, Any] | None:
    from .itchio import parse_itchio_media_target, target_kind_from_url

    parsed = parse_itchio_media_target(target)
    if parsed is not None:
        return {
            "provider": PLUGIN_NAME,
            "kind": "media",
            "is_external": True,
            "group_key": "media",
            "metadata": parsed,
            "capabilities": {"resolve": True, "listTargets": False, "materialize": True},
        }
    kind = target_kind_from_url(target)
    if kind is None:
        return None
    return {
        "provider": PLUGIN_NAME,
        "kind": kind,
        "is_external": True,
        "group_key": kind,
        "capabilities": {"resolve": True, "listTargets": True, "materialize": True},
    }


def resolve(target: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    from .itchio import build_itchio_settings, resolve_itchio_target

    settings = build_itchio_settings(_itchio_overrides(context))
    return resolve_itchio_target(
        target,
        settings=settings,
        use_cache=bool(context.get("use_cache", True)),
        cache_ttl=context.get("cache_ttl"),
        refresh_cache=bool(context.get("refresh_cache", False)),
    )


def list_targets(target: str, context: dict[str, Any]) -> dict[str, Any]:
    from .itchio import build_itchio_settings, list_itchio_targets

    settings = build_itchio_settings(_itchio_overrides(context))
    return list_itchio_targets(
        target,
        settings=settings,
        use_cache=bool(context.get("use_cache", True)),
        cache_ttl=context.get("cache_ttl"),
        refresh_cache=bool(context.get("refresh_cache", False)),
    )


def materialize(target: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    from .itchio import materialize_itchio_media_target, parse_itchio_media_target

    parsed = parse_itchio_media_target(target)
    if parsed is None:
        return []
    return materialize_itchio_media_target(
        target,
        parsed,
        use_cache=bool(context.get("use_cache", True)),
        refresh_cache=bool(context.get("refresh_cache", False)),
        cache_only=bool(context.get("cache_only", False)),
    )
