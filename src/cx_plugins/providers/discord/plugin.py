from __future__ import annotations

import sys
from typing import Any

import click

PLUGIN_API_VERSION = "1"
PLUGIN_NAME = "discord"
PLUGIN_PRIORITY = 100
_DISCORD_THREAD_TYPES = frozenset({10, 11, 12})
_DISCORD_CATEGORY_TYPE = 4
_DISCORD_FORUM_TYPE = 15
_DISCORD_ANNOUNCEMENT_TYPE = 5


def normalize_manifest_config(
    raw_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if raw_config is None:
        return None
    if not isinstance(raw_config, dict):
        raise ValueError("discord config must be a mapping")
    return dict(raw_config)


def _append_option(command: click.Command, option: click.Option) -> None:
    existing = {name for param in command.params for name in getattr(param, "opts", ())}
    if any(name in existing for name in option.opts):
        return
    command.params.append(option)


def _discord_runtime_overrides(raw: dict[str, Any]) -> dict[str, Any] | None:
    uses_manifest_style = (
        "window" in raw
        or "media" in raw
        or any(isinstance(key, str) and "-" in key for key in raw)
    )
    if not uses_manifest_style:
        return dict(raw) or None

    from .discord import parse_discord_config_mapping

    return parse_discord_config_mapping(raw, prefix="discord overrides")


def _discord_overrides(context: dict[str, Any]) -> dict[str, Any] | None:
    overrides = context.get("overrides")
    if not isinstance(overrides, dict):
        return None
    value = overrides.get("discord")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("discord overrides must be a mapping")
    return _discord_runtime_overrides(value)


def can_resolve(target: str, context: dict[str, Any]) -> bool:
    from .discord import is_discord_url

    return is_discord_url(target)


def _discord_guild_target(guild_id: str) -> str:
    return f"https://discord.com/channels/{guild_id}"


def _discord_channel_target(guild_id: str, channel_id: str) -> str:
    return f"https://discord.com/channels/{guild_id}/{channel_id}"


def _discord_message_target(guild_id: str, channel_id: str, message_id: str) -> str:
    return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"


def _discord_channel_kind(channel: dict[str, Any]) -> str:
    channel_type = channel.get("type")
    if channel_type in _DISCORD_THREAD_TYPES:
        return "thread"
    if channel_type == _DISCORD_CATEGORY_TYPE:
        return "category"
    if channel_type == _DISCORD_FORUM_TYPE:
        return "forum"
    if channel_type == _DISCORD_ANNOUNCEMENT_TYPE:
        return "announcement"
    return "channel"


def _discord_relation_item(
    *,
    target: str,
    label: str,
    kind: str,
    relation: str,
    source_target: str,
    metadata: dict[str, Any],
    traverse: bool = False,
) -> dict[str, Any]:
    return {
        "target": target,
        "label": label,
        "kind": kind,
        "traverse": traverse,
        "metadata": {
            "relation": relation,
            "source_target": source_target,
            **metadata,
        },
    }


def _discord_guild_relation(
    guild_id: str,
    *,
    source_target: str,
    guild_name: str | None = None,
) -> dict[str, Any]:
    return _discord_relation_item(
        target=_discord_guild_target(guild_id),
        label=guild_name or guild_id,
        kind="guild",
        relation="contained_by",
        source_target=source_target,
        metadata={"guild_id": guild_id, **({"guild_name": guild_name} if guild_name else {})},
    )


def _discord_channel_relation(
    guild_id: str,
    channel_id: str,
    *,
    source_target: str,
    relation: str = "contained_by",
    label: str | None = None,
    kind: str = "channel",
    metadata: dict[str, Any] | None = None,
    traverse: bool = False,
) -> dict[str, Any]:
    return _discord_relation_item(
        target=_discord_channel_target(guild_id, channel_id),
        label=label or channel_id,
        kind=kind,
        relation=relation,
        source_target=source_target,
        metadata={"guild_id": guild_id, "channel_id": channel_id, **(metadata or {})},
        traverse=traverse,
    )


def _discord_category_relation(
    guild_id: str,
    category_id: str,
    *,
    source_target: str,
    category_name: str | None = None,
) -> dict[str, Any]:
    return _discord_channel_relation(
        guild_id,
        category_id,
        source_target=source_target,
        label=category_name or category_id,
        kind="category",
        metadata={
            "category_id": category_id,
            **({"category_name": category_name} if category_name else {}),
        },
    )


def _discord_lookup_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "use_cache": bool(context.get("use_cache", True)),
        "cache_ttl": context.get("cache_ttl"),
        "refresh_cache": bool(context.get("refresh_cache", False)),
    }


def _discord_fetch_guild_label(guild_id: str, context: dict[str, Any]) -> str | None:
    from .discord import _fetch_guild, _guild_label

    try:
        guild = _fetch_guild(guild_id, **_discord_lookup_context(context))
    except Exception:
        return None
    return _guild_label(guild, guild_id)


def _discord_fetch_channel(
    channel_id: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    from .discord import _fetch_channel

    return _fetch_channel(channel_id, **_discord_lookup_context(context))


def _discord_channel_label(channel: dict[str, Any], channel_id: str) -> str:
    from .discord import _channel_label

    return _channel_label(channel, channel_id)


def _discord_channel_metadata(
    guild_id: str,
    channel_id: str,
    channel: dict[str, Any],
) -> dict[str, Any]:
    channel_name = _discord_channel_label(channel, channel_id)
    channel_type = channel.get("type")
    metadata: dict[str, Any] = {
        "guild_id": guild_id,
        "channel_id": channel_id,
        "channel_name": channel_name,
        "channel_type": channel_type,
    }
    parent_id = channel.get("parent_id")
    if parent_id is not None:
        metadata["parent_channel_id"] = str(parent_id)
    return metadata


def _discord_category_name(
    guild_id: str,
    category_id: str,
    context: dict[str, Any],
) -> str | None:
    try:
        category = _discord_fetch_channel(category_id, context)
        if category.get("type") == _DISCORD_CATEGORY_TYPE:
            return _discord_channel_label(category, category_id)
    except Exception:
        pass
    try:
        from .discord import _fetch_guild_channels

        channels = _fetch_guild_channels(guild_id, **_discord_lookup_context(context))
    except Exception:
        return None
    for channel in channels:
        if str(channel.get("id") or "") != category_id:
            continue
        name = channel.get("name")
        return str(name).strip() if isinstance(name, str) and name.strip() else None
    return None


def _discord_channel_relations(
    target: str,
    parsed: dict[str, Any],
    context: dict[str, Any],
    *,
    channel: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    guild_id = str(parsed["guild_id"])
    channel_id = str(parsed["channel_id"])
    guild_name = _discord_fetch_guild_label(guild_id, context)
    relations = [
        _discord_guild_relation(
            guild_id,
            source_target=target,
            guild_name=guild_name,
        )
    ]
    if channel is None:
        channel = _discord_fetch_channel(channel_id, context)
    channel_type = channel.get("type")
    parent_id = str(channel.get("parent_id") or "")
    if channel_type in _DISCORD_THREAD_TYPES and parent_id:
        parent_channel = _discord_fetch_channel(parent_id, context)
        parent_name = _discord_channel_label(parent_channel, parent_id)
        relations.append(
            _discord_channel_relation(
                guild_id,
                parent_id,
                source_target=target,
                label=f"#{parent_name}",
                metadata=_discord_channel_metadata(guild_id, parent_id, parent_channel),
            )
        )
        category_id = str(parent_channel.get("parent_id") or "")
        if category_id:
            relations.append(
                _discord_category_relation(
                    guild_id,
                    category_id,
                    source_target=target,
                    category_name=_discord_category_name(guild_id, category_id, context),
                )
            )
    elif parent_id:
        relations.append(
            _discord_category_relation(
                guild_id,
                parent_id,
                source_target=target,
                category_name=_discord_category_name(guild_id, parent_id, context),
            )
        )
    return relations


def _discord_message_relations(
    target: str,
    parsed: dict[str, Any],
    context: dict[str, Any],
    *,
    message: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    from .discord import _fetch_message

    guild_id = str(parsed["guild_id"])
    channel_id = str(parsed["channel_id"])
    message_id = str(parsed["message_id"])
    channel = _discord_fetch_channel(channel_id, context)
    relations = _discord_channel_relations(
        target,
        parsed,
        context,
        channel=channel,
    )
    channel_kind = _discord_channel_kind(channel)
    channel_name = _discord_channel_label(channel, channel_id)
    relations.append(
        _discord_channel_relation(
            guild_id,
            channel_id,
            source_target=target,
            label=f"#{channel_name}" if channel_kind != "thread" else channel_name,
            kind=channel_kind,
            metadata=_discord_channel_metadata(guild_id, channel_id, channel),
        )
    )
    if message is None:
        message = _fetch_message(
            channel_id,
            message_id,
            **_discord_lookup_context(context),
        )
    thread_data = message.get("thread") if isinstance(message.get("thread"), dict) else None
    thread_id = str(thread_data.get("id") or "") if thread_data else ""
    if thread_id:
        thread_name = thread_data.get("name")
        relations.append(
            _discord_channel_relation(
                guild_id,
                thread_id,
                source_target=target,
                relation="starts_thread",
                label=thread_name if isinstance(thread_name, str) and thread_name else thread_id,
                kind="thread",
                metadata={
                    "guild_id": guild_id,
                    "channel_id": thread_id,
                    "parent_channel_id": channel_id,
                    **(
                        {"thread_name": thread_name}
                        if isinstance(thread_name, str) and thread_name
                        else {}
                    ),
                },
                traverse=True,
            )
        )
    return relations


def classify_target(target: str, context: dict[str, Any]) -> dict[str, Any] | None:
    from .discord import parse_discord_url

    parsed = parse_discord_url(target)
    if not isinstance(parsed, dict):
        return None
    kind = parsed.get("kind")
    if not isinstance(kind, str) or not kind:
        return None
    descriptor: dict[str, Any] = {
        "provider": PLUGIN_NAME,
        "kind": kind,
        "is_external": True,
        "group_key": kind,
        "metadata": dict(parsed),
        "capabilities": {
            "resolve": True,
            "listTargets": True,
            "materialize": kind == "attachment",
        },
    }
    try:
        if kind == "guild":
            guild_name = _discord_fetch_guild_label(str(parsed["guild_id"]), context)
            if guild_name:
                descriptor["metadata"]["guild_name"] = guild_name
        elif kind == "channel":
            channel = _discord_fetch_channel(str(parsed["channel_id"]), context)
            actual_kind = _discord_channel_kind(channel)
            descriptor["kind"] = actual_kind
            descriptor["group_key"] = actual_kind
            descriptor["metadata"].update(
                _discord_channel_metadata(
                    str(parsed["guild_id"]),
                    str(parsed["channel_id"]),
                    channel,
                )
            )
            descriptor["relations"] = _discord_channel_relations(
                target,
                parsed,
                context,
                channel=channel,
            )
        elif kind in {"message", "attachment"}:
            descriptor["relations"] = _discord_message_relations(
                target,
                parsed,
                context,
            )
    except Exception as exc:
        descriptor["error"] = f"{exc.__class__.__name__}: {exc}"
    return descriptor


def _discord_guild_targets(guild_id: str, context: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    from .discord import _fetch_guild_channels

    raw_channels = _fetch_guild_channels(guild_id, **_discord_lookup_context(context))
    categories = {
        str(channel.get("id")): channel
        for channel in raw_channels
        if channel.get("type") == _DISCORD_CATEGORY_TYPE and channel.get("id") is not None
    }
    category_items = [
        _discord_relation_item(
            target=_discord_channel_target(guild_id, str(category.get("id"))),
            label=str(category.get("name") or category.get("id")),
            kind="category",
            relation="contains",
            source_target=_discord_guild_target(guild_id),
            metadata={
                "guild_id": guild_id,
                "category_id": str(category.get("id")),
                "category_name": category.get("name"),
                "position": category.get("position"),
            },
        )
        for category in sorted(categories.values(), key=lambda item: item.get("position", 0))
    ]

    channel_items: list[dict[str, Any]] = []
    for channel in sorted(raw_channels, key=lambda item: item.get("position", 0)):
        channel_id = channel.get("id")
        if channel_id is None or channel.get("type") == _DISCORD_CATEGORY_TYPE:
            continue
        if channel.get("type") not in {0, _DISCORD_ANNOUNCEMENT_TYPE, _DISCORD_FORUM_TYPE}:
            continue
        kind = _discord_channel_kind(channel)
        channel_id_str = str(channel_id)
        parent_id = str(channel.get("parent_id") or "")
        category = categories.get(parent_id)
        metadata = _discord_channel_metadata(guild_id, channel_id_str, channel)
        metadata.update(
            {
                "position": channel.get("position"),
                "category_id": parent_id or None,
                "category_name": category.get("name") if category else None,
            }
        )
        label = _discord_channel_label(channel, channel_id_str)
        channel_items.append(
            _discord_relation_item(
                target=_discord_channel_target(guild_id, channel_id_str),
                label=f"#{label}",
                kind=kind,
                relation="contains",
                source_target=_discord_guild_target(guild_id),
                metadata=metadata,
            )
        )
    return category_items + channel_items, {
        "categoryCount": len(category_items),
        "channelCount": len(channel_items),
    }


def _discord_listing_envelope(
    *,
    target: str,
    kind: str,
    targets: list[dict[str, Any]],
    parsed: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "provider": PLUGIN_NAME,
        "kind": kind,
        "target": target,
    }
    if parsed:
        for key in ("guild_id", "channel_id", "message_id"):
            value = parsed.get(key)
            if value is not None:
                metadata[key] = value
    return {
        "targets": targets,
        "summary": summary or {kind: dict(parsed or {}), "targetCount": len(targets)},
        "pagination": {"returned": len(targets), "totalCount": len(targets), "hasMore": False},
        "metadata": metadata,
        "capabilities": {
            "resolve": True,
            "listTargets": True,
            "materialize": any(
                str(item.get("kind") or "").startswith("attachment:")
                for item in targets
            ),
        },
    }


def list_targets(target: str, context: dict[str, Any]) -> dict[str, Any]:
    from .discord import list_discord_message_targets, parse_discord_url

    parsed = parse_discord_url(target)
    if not isinstance(parsed, dict):
        return _discord_listing_envelope(
            target=target,
            kind="unknown",
            targets=[],
            parsed=None,
        )
    if parsed.get("kind") == "guild":
        items, counts = _discord_guild_targets(str(parsed["guild_id"]), context)
        return _discord_listing_envelope(
            target=target,
            kind="guild",
            targets=items,
            parsed=parsed,
            summary={"guild": dict(parsed), "targetCount": len(items), **counts},
        )

    items = list_discord_message_targets(
        target,
        parsed,
        use_cache=bool(context.get("use_cache", True)),
        cache_ttl=context.get("cache_ttl"),
        refresh_cache=bool(context.get("refresh_cache", False)),
    )
    if bool(context.get("include_containing", False)):
        try:
            if parsed.get("kind") == "channel":
                items = [
                    *_discord_channel_relations(target, parsed, context),
                    *items,
                ]
            elif parsed.get("kind") in {"message", "attachment"}:
                items = [
                    *_discord_message_relations(target, parsed, context),
                    *items,
                ]
        except Exception as exc:
            return {
                **_discord_listing_envelope(
                    target=target,
                    kind=str(parsed.get("kind") or "target"),
                    targets=items,
                    parsed=parsed,
                ),
                "error": f"{exc.__class__.__name__}: {exc}",
            }
    if parsed.get("kind") != "message":
        return _discord_listing_envelope(
            target=target,
            kind=str(parsed.get("kind") or "target"),
            targets=items,
            parsed=parsed,
        )
    return _discord_listing_envelope(
        target=target,
        kind="message",
        targets=items,
        parsed=parsed,
    )


def materialize(target: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    from .discord import materialize_discord_attachment_target, parse_discord_url

    parsed = parse_discord_url(target)
    if not isinstance(parsed, dict):
        return []
    return materialize_discord_attachment_target(
        target,
        parsed,
        use_cache=bool(context.get("use_cache", True)),
        cache_ttl=context.get("cache_ttl"),
        refresh_cache=bool(context.get("refresh_cache", False)),
        cache_only=bool(context.get("cache_only", False)),
    )


def register_cli_options(command_name: str, command: click.Command) -> None:
    if command_name not in {"cat", "hydrate"}:
        return
    _append_option(
        command,
        click.Option(
            ["--discord-file-content/--discord-file-links-only"],
            default=None,
            help=(
                "Inline generic file attachment bodies or leave them as link-only stubs."
            ),
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--discord-skip-file-content-message"],
            multiple=True,
            help=(
                "Skip file attachment body resolution for the given Discord message ID or message URL. "
                "Repeatable."
            ),
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--discord-max-file-content-size"],
            default=None,
            help=(
                "Skip inlining generic file attachments larger than this size. "
                "Accepts bytes or human-size strings like 50K or 50kb."
            ),
        ),
    )


def collect_cli_overrides(
    command_name: str,
    params: dict[str, Any],
) -> dict[str, Any] | None:
    if command_name not in {"cat", "hydrate"}:
        return None

    media: dict[str, Any] = {}

    file_content = params.get("discord_file_content")
    if file_content is not None:
        media["file-content"] = bool(file_content)

    skip_messages = [
        value.strip()
        for value in params.get("discord_skip_file_content_message") or ()
        if isinstance(value, str) and value.strip()
    ]
    if skip_messages:
        media["skip-file-content-messages"] = skip_messages

    max_file_content_size = params.get("discord_max_file_content_size")
    if isinstance(max_file_content_size, str) and max_file_content_size.strip():
        media["max-file-content-size"] = max_file_content_size.strip()

    return {"media": media} if media else None


def _render_documents(
    documents: list[Any],
    source_url: str,
    settings: Any,
    settings_key: str,
    channel_id: str | None = None,
    channel_name: str | None = None,
    category_id: str | None = None,
    category_name: str | None = None,
) -> list[dict[str, Any]]:
    from .discord import (
        discord_document_timestamps,
        render_discord_document_with_metadata,
        split_discord_document_by_utc_day,
        with_discord_document_rendered,
    )

    out: list[dict[str, Any]] = []
    for document in documents:
        if document.kind == "thread":
            day_documents = [document]
        else:
            day_documents = split_discord_document_by_utc_day(
                document, settings=settings
            )
        for day_document in day_documents:
            rendered_document = with_discord_document_rendered(
                day_document,
                rendered=render_discord_document_with_metadata(
                    day_document,
                    settings=settings,
                    source_url=source_url,
                    include_message_bounds=False,
                ),
            )
            source_created, source_modified = discord_document_timestamps(day_document)
            day_slug = source_created[:10] if source_created else "undated"
            scope_id = rendered_document.thread_id or rendered_document.channel_id
            ext = ".yaml" if settings.format == "yaml" else ".md"
            effective_channel_id = (
                channel_id
                or (
                    rendered_document.channel_id
                    if rendered_document.kind != "thread"
                    else None
                )
                or rendered_document.channel_id
            )
            out.append(
                {
                    "source": source_url,
                    "label": rendered_document.label,
                    "content": rendered_document.rendered,
                    "metadata": {
                        "trace_path": rendered_document.trace_path,
                        "provider": PLUGIN_NAME,
                        "source_ref": "discord.com",
                        "source_path": (
                            f"{rendered_document.guild_id}/{rendered_document.channel_id}"
                            + (
                                f"/{rendered_document.thread_id}"
                                if rendered_document.thread_id
                                else ""
                            )
                            + f"/day-{day_slug}"
                        ),
                        "context_subpath": (
                            f"discord/{rendered_document.guild_id}/{scope_id}/{day_slug}{ext}"
                        ),
                        "source_created": source_created,
                        "source_modified": source_modified,
                        "kind": rendered_document.kind,
                        "scope_id": scope_id,
                        "settings_key": settings_key,
                        "channelId": effective_channel_id,
                        "channelName": channel_name,
                        "categoryId": category_id,
                        "categoryName": category_name,
                    },
                }
            )
    return out


def resolve(target: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    from .discord import (
        DiscordResolutionError,
        build_discord_settings,
        discord_settings_cache_key,
        parse_discord_url,
        is_discord_guild_url,
        resolve_discord_attachment_target,
        resolve_discord_url,
    )

    if is_discord_guild_url(target):
        return _resolve_guild(target, context)

    settings = build_discord_settings(_discord_overrides(context))
    settings_key = discord_settings_cache_key(settings)
    parsed = parse_discord_url(target)
    if isinstance(parsed, dict) and parsed.get("kind") == "attachment":
        return resolve_discord_attachment_target(
            target,
            parsed,
            settings=settings,
            settings_key=settings_key,
            use_cache=bool(context.get("use_cache", True)),
            cache_ttl=context.get("cache_ttl"),
            refresh_cache=bool(context.get("refresh_cache", False)),
            plugin_overrides=context.get("overrides"),
        )
    try:
        documents = resolve_discord_url(
            target,
            settings=settings,
            use_cache=bool(context.get("use_cache", True)),
            cache_ttl=context.get("cache_ttl"),
            refresh_cache=bool(context.get("refresh_cache", False)),
        )
    except ValueError as exc:
        if isinstance(exc, DiscordResolutionError) and exc.is_skippable:
            print(
                f"Warning: skipping Discord URL: {target} ({exc})",
                file=sys.stderr,
                flush=True,
            )
            return []
        raise

    return _render_documents(documents, target, settings, settings_key)


_CHANNEL_TYPE_LABELS = {0: "text", 2: "voice", 5: "announcement", 15: "forum"}


def _build_guild_manifest(
    guild_id: str,
    all_channels: list,
    skipped: list[str],
) -> str:
    from .discord import _slugify_channel

    by_category: dict[str | None, list] = {}
    for ch in all_channels:
        key = ch.category_name
        by_category.setdefault(key, []).append(ch)

    lines = ["---", f'guild_id: "{guild_id}"', "kind: manifest", "---", ""]

    for cat_name in sorted(by_category, key=lambda k: (k is None, k or "")):
        heading = cat_name or "(uncategorized)"
        lines.append(f"## {heading}")
        for ch in by_category[cat_name]:
            type_label = _CHANNEL_TYPE_LABELS.get(ch.channel_type, "channel")
            note = ""
            if ch.channel_id in skipped:
                note = " (inaccessible)"
            lines.append(f"- **#{ch.name}** — {type_label}{note}")
        lines.append("")

    return "\n".join(lines)


def _resolve_guild(target: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    from .discord import (
        DiscordResolutionError,
        build_discord_settings,
        discover_guild_channels,
        discord_settings_cache_key,
        parse_discord_url,
        resolve_discord_url,
    )

    parsed = parse_discord_url(target)
    if not parsed or parsed["kind"] != "guild":
        raise ValueError(f"Not a Discord guild URL: {target}")

    guild_id = parsed["guild_id"]
    use_cache = bool(context.get("use_cache", True))
    cache_ttl = context.get("cache_ttl")
    refresh_cache = bool(context.get("refresh_cache", False))

    all_channels = discover_guild_channels(
        guild_id,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )

    channel_batch_limit = None
    overrides = context.get("overrides")
    if isinstance(overrides, dict):
        discord_opts = overrides.get("discord")
        if isinstance(discord_opts, dict):
            raw_limit = (
                discord_opts.get("channel-batch-limit")
                or discord_opts.get("channel_batch_limit")
                or discord_opts.get("channelBatchLimit")
            )
            if isinstance(raw_limit, int) and raw_limit > 0:
                channel_batch_limit = raw_limit

    channels = (
        all_channels[:channel_batch_limit] if channel_batch_limit else all_channels
    )

    settings = build_discord_settings(_discord_overrides(context))
    settings_key = discord_settings_cache_key(settings)

    out: list[dict[str, Any]] = []
    skipped: list[str] = []
    for channel_info in channels:
        try:
            documents = resolve_discord_url(
                channel_info.url,
                settings=settings,
                use_cache=use_cache,
                cache_ttl=cache_ttl,
                refresh_cache=refresh_cache,
            )
        except ValueError as exc:
            if isinstance(exc, DiscordResolutionError) and exc.is_skippable:
                print(
                    f"Warning: skipping #{channel_info.name}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                skipped.append(channel_info.channel_id)
                continue
            if isinstance(exc, DiscordResolutionError):
                print(
                    f"Warning: error resolving #{channel_info.name}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                skipped.append(channel_info.channel_id)
                continue
            raise

        out.extend(
            _render_documents(
                documents,
                channel_info.url,
                settings,
                settings_key,
                channel_id=channel_info.channel_id,
                channel_name=channel_info.name,
                category_id=channel_info.category_id,
                category_name=channel_info.category_name,
            )
        )

    manifest_body = _build_guild_manifest(guild_id, all_channels, skipped)
    out.append(
        {
            "source": target,
            "label": "manifest",
            "content": manifest_body,
            "metadata": {
                "provider": PLUGIN_NAME,
                "kind": "manifest",
                "guild_id": guild_id,
                "channel_count": len(all_channels),
            },
        }
    )

    return out
