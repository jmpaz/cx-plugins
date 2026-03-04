from __future__ import annotations

import sys
from typing import Any

PLUGIN_API_VERSION = "1"
PLUGIN_NAME = "discord"
PLUGIN_PRIORITY = 100


def _discord_overrides(context: dict[str, Any]) -> dict[str, Any] | None:
    overrides = context.get("overrides")
    if not isinstance(overrides, dict):
        return None
    value = overrides.get("discord")
    return value if isinstance(value, dict) else None


def can_resolve(target: str, context: dict[str, Any]) -> bool:
    from contextualize.references.discord import is_discord_url

    return is_discord_url(target)


def resolve(target: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    from contextualize.references.discord import (
        DiscordResolutionError,
        build_discord_settings,
        discord_document_timestamps,
        discord_settings_cache_key,
        render_discord_document_with_metadata,
        resolve_discord_url,
        split_discord_document_by_utc_day,
        with_discord_document_rendered,
    )

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

    out: list[dict[str, Any]] = []
    for document in documents:
        day_documents = split_discord_document_by_utc_day(document, settings=settings)
        for day_document in day_documents:
            rendered_document = with_discord_document_rendered(
                day_document,
                rendered=render_discord_document_with_metadata(
                    day_document,
                    settings=settings,
                    source_url=target,
                    include_message_bounds=False,
                ),
            )
            source_created, source_modified = discord_document_timestamps(day_document)
            day_slug = source_created[:10] if source_created else "undated"
            scope_id = rendered_document.thread_id or rendered_document.channel_id
            ext = ".yaml" if settings.format == "yaml" else ".md"
            out.append(
                {
                    "source": target,
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
                    },
                }
            )
    return out
