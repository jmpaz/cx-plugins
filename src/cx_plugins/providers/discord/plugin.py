from __future__ import annotations

import sys
from typing import Any

PLUGIN_API_VERSION = "1"
PLUGIN_NAME = "discord"
PLUGIN_PRIORITY = 100


def normalize_manifest_config(
    raw_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if raw_config is None:
        return None
    if not isinstance(raw_config, dict):
        raise ValueError("discord config must be a mapping")
    return dict(raw_config)


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


def classify_target(target: str, context: dict[str, Any]) -> dict[str, Any] | None:
    from .discord import parse_discord_url

    parsed = parse_discord_url(target)
    if not isinstance(parsed, dict):
        return None
    kind = parsed.get("kind")
    if not isinstance(kind, str) or not kind:
        return None
    return {
        "provider": PLUGIN_NAME,
        "kind": kind,
        "is_external": True,
        "group_key": kind,
    }


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
            day_documents = split_discord_document_by_utc_day(document, settings=settings)
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
                or (rendered_document.channel_id if rendered_document.kind != "thread" else None)
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
        is_discord_guild_url,
        resolve_discord_url,
    )

    if is_discord_guild_url(target):
        return _resolve_guild(target, context)

    settings = build_discord_settings(_discord_overrides(context))
    settings_key = discord_settings_cache_key(settings)
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

    lines = ["---", f"guild_id: \"{guild_id}\"", "kind: manifest", "---", ""]

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

    channels = all_channels[:channel_batch_limit] if channel_batch_limit else all_channels

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

        out.extend(_render_documents(
            documents,
            channel_info.url,
            settings,
            settings_key,
            channel_id=channel_info.channel_id,
            channel_name=channel_info.name,
            category_id=channel_info.category_id,
            category_name=channel_info.category_name,
        ))

    manifest_body = _build_guild_manifest(guild_id, all_channels, skipped)
    out.append({
        "source": target,
        "label": "manifest",
        "content": manifest_body,
        "metadata": {
            "provider": PLUGIN_NAME,
            "kind": "manifest",
            "guild_id": guild_id,
            "channel_count": len(all_channels),
        },
    })

    return out
