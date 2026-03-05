from __future__ import annotations

from typing import Any

PLUGIN_API_VERSION = "1"
PLUGIN_NAME = "ytdlp"
PLUGIN_PRIORITY = 10


def normalize_manifest_config(
    raw_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if raw_config is None:
        return None
    if not isinstance(raw_config, dict):
        raise ValueError("ytdlp config must be a mapping")
    return dict(raw_config)


def can_resolve(target: str, context: dict[str, Any]) -> bool:
    from .ytdlp import probe_ytdlp_url

    return probe_ytdlp_url(target, timeout_seconds=5)


def classify_target(target: str, context: dict[str, Any]) -> dict[str, Any] | None:
    from .ytdlp import probe_ytdlp_metadata

    metadata = probe_ytdlp_metadata(target, timeout_seconds=5)
    if metadata is None:
        return None
    duration = metadata.get("duration")
    kind = (
        "video" if isinstance(duration, (int, float)) and duration > 0 else "resource"
    )
    return {
        "provider": PLUGIN_NAME,
        "kind": kind,
        "is_external": True,
        "group_key": kind,
    }


def resolve(target: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    from .ytdlp import YtDlpReference

    reference = YtDlpReference(
        target,
        format="raw",
        label="relative",
        inject=False,
        use_cache=bool(context.get("use_cache", True)),
        cache_ttl=context.get("cache_ttl"),
        refresh_cache=bool(context.get("refresh_cache", False)),
    )
    return [
        {
            "source": target,
            "label": reference.get_label(),
            "content": reference.read(),
            "metadata": {
                "trace_path": reference.get_label(),
                "provider": PLUGIN_NAME,
                "source_ref": reference.source_ref(),
                "source_path": reference.source_path(),
                "context_subpath": reference.context_subpath(),
                "kind": reference.get_kind(),
            },
        }
    ]
