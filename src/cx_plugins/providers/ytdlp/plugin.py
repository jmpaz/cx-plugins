from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

PLUGIN_API_VERSION = "1"
PLUGIN_NAME = "ytdlp"
PLUGIN_PRIORITY = 10

_EXPECTED_RESOLUTION_ERRORS = (RuntimeError, ValueError, OSError)


def normalize_manifest_config(
    raw_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if raw_config is None:
        return None
    if not isinstance(raw_config, dict):
        raise ValueError("ytdlp config must be a mapping")
    return dict(raw_config)


def can_resolve(target: str, context: dict[str, Any]) -> bool:
    from .ytdlp import looks_like_ytdlp_url

    return looks_like_ytdlp_url(target)


def classify_target(target: str, context: dict[str, Any]) -> dict[str, Any] | None:
    from .ytdlp import looks_like_ytdlp_url, probe_ytdlp_metadata

    metadata = probe_ytdlp_metadata(target, timeout_seconds=5)
    if metadata is None:
        if not looks_like_ytdlp_url(target):
            return None
        return {
            "provider": PLUGIN_NAME,
            "kind": "video",
            "is_external": True,
            "group_key": "video",
        }
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
    from .ytdlp import YtDlpReference, looks_like_ytdlp_url

    try:
        reference = YtDlpReference(
            target,
            format="raw",
            label="relative",
            inject=False,
            use_cache=bool(context.get("use_cache", True)),
            cache_ttl=context.get("cache_ttl"),
            refresh_cache=bool(context.get("refresh_cache", False)),
            plugin_overrides=context.get("overrides"),
        )
    except _EXPECTED_RESOLUTION_ERRORS as exc:
        if not looks_like_ytdlp_url(target):
            raise
        netloc = urlparse(target).netloc or "ytdlp"
        return [
            {
                "source": target,
                "label": target,
                "content": (
                    f"yt-dlp claimed this media URL but failed to resolve it: {exc}"
                ),
                "metadata": {
                    "trace_path": target,
                    "provider": PLUGIN_NAME,
                    "source_ref": netloc,
                    "source_path": target,
                    "context_subpath": "ytdlp-error.md",
                    "kind": "video",
                    "resolution_error": str(exc),
                },
            }
        ]
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
