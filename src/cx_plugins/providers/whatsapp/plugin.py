from __future__ import annotations

import sys
from typing import Any

import click

PLUGIN_API_VERSION = "1"
PLUGIN_NAME = "whatsapp"
PLUGIN_PRIORITY = 100


def normalize_manifest_config(
    raw_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if raw_config is None:
        return None
    if not isinstance(raw_config, dict):
        raise ValueError("whatsapp config must be a mapping")
    return dict(raw_config)


def _append_option(command: click.Command, option: click.Option) -> None:
    existing = {name for param in command.params for name in getattr(param, "opts", ())}
    if any(name in existing for name in option.opts):
        return
    command.params.append(option)


def _whatsapp_runtime_overrides(raw: dict[str, Any]) -> dict[str, Any] | None:
    uses_manifest_style = (
        "window" in raw
        or "media" in raw
        or any(isinstance(key, str) and "-" in key for key in raw)
    )
    if not uses_manifest_style:
        return dict(raw) or None
    from .whatsapp import parse_whatsapp_config_mapping

    return parse_whatsapp_config_mapping(raw, prefix="whatsapp overrides")


def _whatsapp_overrides(context: dict[str, Any]) -> dict[str, Any] | None:
    overrides = context.get("overrides")
    if not isinstance(overrides, dict):
        return None
    value = overrides.get("whatsapp")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("whatsapp overrides must be a mapping")
    return _whatsapp_runtime_overrides(value)


def can_resolve(target: str, context: dict[str, Any]) -> bool:
    from .whatsapp import is_whatsapp_target

    return is_whatsapp_target(target)


def classify_target(target: str, context: dict[str, Any]) -> dict[str, Any] | None:
    from .whatsapp import parse_whatsapp_target

    parsed = parse_whatsapp_target(target)
    if not isinstance(parsed, dict):
        return None
    kind = str(parsed.get("kind") or "")
    if not kind:
        return None
    return {
        "provider": PLUGIN_NAME,
        "kind": kind,
        "is_external": True,
        "group_key": kind,
    }


def register_cli_options(command_name: str, command: click.Command) -> None:
    if command_name not in {"cat", "hydrate", "payload"}:
        return
    _append_option(
        command,
        click.Option(
            ["--whatsapp-file-content/--whatsapp-file-links-only"],
            default=None,
            help=(
                "Inline generic file attachment bodies or leave them as link-only stubs."
            ),
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--whatsapp-skip-file-content-message"],
            multiple=True,
            help=(
                "Skip file attachment body resolution for the given WhatsApp message ID. "
                "Repeatable."
            ),
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--whatsapp-max-file-content-size"],
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
    if command_name not in {"cat", "hydrate", "payload"}:
        return None
    media: dict[str, Any] = {}
    file_content = params.get("whatsapp_file_content")
    if file_content is not None:
        media["file-content"] = bool(file_content)
    skip_messages = [
        value.strip()
        for value in params.get("whatsapp_skip_file_content_message") or ()
        if isinstance(value, str) and value.strip()
    ]
    if skip_messages:
        media["skip-file-content-messages"] = skip_messages
    max_file_content_size = params.get("whatsapp_max_file_content_size")
    if isinstance(max_file_content_size, str) and max_file_content_size.strip():
        media["max-file-content-size"] = max_file_content_size.strip()
    return {"media": media} if media else None


def list_targets(target: str, context: dict[str, Any]) -> dict[str, Any]:
    from .whatsapp import list_whatsapp_targets

    items = list_whatsapp_targets(target)
    return {
        "targets": items,
        "summary": {"targetCount": len(items)},
        "pagination": {"returned": len(items), "totalCount": len(items), "hasMore": False},
        "metadata": {"provider": PLUGIN_NAME, "target": target},
        "capabilities": {
            "resolve": True,
            "listTargets": True,
            "materialize": False,
        },
    }


def _render_documents(
    documents: list[Any],
    settings: Any,
    settings_key: tuple[Any, ...],
) -> list[dict[str, Any]]:
    from .whatsapp import (
        render_whatsapp_document_with_metadata,
        split_whatsapp_document_by_day,
        whatsapp_document_prose,
        whatsapp_document_timestamps,
        with_whatsapp_document_rendered,
    )

    out: list[dict[str, Any]] = []
    for document in documents:
        day_documents = split_whatsapp_document_by_day(document, settings=settings)
        for day_document in day_documents:
            rendered_document = with_whatsapp_document_rendered(
                day_document,
                rendered=render_whatsapp_document_with_metadata(
                    day_document,
                    settings=settings,
                    include_message_bounds=False,
                ),
            )
            source_created, source_modified = whatsapp_document_timestamps(day_document)
            prose, prose_authors = whatsapp_document_prose(day_document)
            day_slug = source_created[:10] if source_created else "undated"
            ext = ".yaml" if settings.format == "yaml" else ".md"
            out.append(
                {
                    "source": rendered_document.source_url,
                    "label": rendered_document.label,
                    "content": rendered_document.rendered,
                    "prose": prose,
                    "prose_authors": prose_authors,
                    "metadata": {
                        "trace_path": rendered_document.trace_path,
                        "provider": PLUGIN_NAME,
                        "source_ref": "whatsapp",
                        "source_path": f"{rendered_document.chat_id}/day-{day_slug}",
                        "context_subpath": (
                            f"whatsapp/{rendered_document.chat_id}/{day_slug}{ext}"
                        ),
                        "source_created": source_created,
                        "source_modified": source_modified,
                        "kind": rendered_document.kind,
                        "scope_id": rendered_document.chat_id,
                        "settings_key": settings_key,
                        "chatId": rendered_document.chat_id,
                        "chatName": rendered_document.chat_name,
                        "archivePath": rendered_document.source_path,
                    },
                }
            )
    return out


def resolve(target: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    from .whatsapp import (
        build_whatsapp_settings,
        resolve_whatsapp_target,
        whatsapp_settings_cache_key,
    )

    settings = build_whatsapp_settings(_whatsapp_overrides(context))
    settings_key = whatsapp_settings_cache_key(settings)
    try:
        documents = resolve_whatsapp_target(
            target,
            settings=settings,
            use_cache=bool(context.get("use_cache", True)),
            cache_ttl=context.get("cache_ttl"),
            refresh_cache=bool(context.get("refresh_cache", False)),
        )
    except ValueError as exc:
        print(
            f"Warning: skipping WhatsApp target: {target} ({exc})",
            file=sys.stderr,
            flush=True,
        )
        return []
    return _render_documents(documents, settings, settings_key)
