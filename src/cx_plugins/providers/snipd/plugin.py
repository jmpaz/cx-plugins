from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

PLUGIN_API_VERSION = "1"
PLUGIN_NAME = "snipd"
PLUGIN_PRIORITY = 100

_EXPECTED_RESOLUTION_ERRORS = (RuntimeError, ValueError, OSError)


def normalize_manifest_config(
    raw_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if raw_config is None:
        return None
    if not isinstance(raw_config, dict):
        raise ValueError("snipd config must be a mapping")
    return dict(raw_config)


def can_resolve(target: str, context: dict[str, Any]) -> bool:
    from .snipd import parse_snipd_target

    return parse_snipd_target(target) is not None


def classify_target(target: str, context: dict[str, Any]) -> dict[str, Any] | None:
    from .snipd import parse_snipd_target

    if parse_snipd_target(target) is None:
        return None
    return {
        "provider": PLUGIN_NAME,
        "kind": "audio",
        "is_external": True,
        "group_key": "audio",
    }


def _failure_document(target: str, exc: BaseException) -> dict[str, Any]:
    from .snipd import parse_snipd_target

    netloc = urlparse(target).netloc or "snipd"
    parsed = parse_snipd_target(target)
    source_path = f"snipd:{parsed.clip_id}" if parsed else target
    return {
        "source": target,
        "label": target,
        "content": f"Snipd claimed this public snip but failed to resolve it: {exc}",
        "metadata": {
            "trace_path": target,
            "provider": PLUGIN_NAME,
            "source_ref": netloc,
            "source_path": source_path,
            "context_subpath": "snipd-error.md",
            "kind": "audio",
            "clip_id": parsed.clip_id if parsed else None,
            "resolution_error": str(exc),
        },
    }


def resolve(target: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    from .snipd import SnipdReference, parse_snipd_target

    try:
        reference = SnipdReference(
            target,
            use_cache=bool(context.get("use_cache", True)),
            cache_ttl=context.get("cache_ttl"),
            refresh_cache=bool(context.get("refresh_cache", False)),
            plugin_overrides=context.get("overrides"),
        )
        content = reference.read()
    except _EXPECTED_RESOLUTION_ERRORS as exc:
        if parse_snipd_target(target) is None:
            raise
        return [_failure_document(target, exc)]

    clip = reference.loaded_clip()
    clip_id = reference.clip_id()
    return [
        {
            "source": target,
            "label": reference.get_label(),
            "content": content,
            "prose": reference.prose_text(),
            "metadata": {
                "trace_path": reference.get_label(),
                "provider": PLUGIN_NAME,
                "source_ref": reference.source_ref(),
                "source_path": reference.source_path(),
                "context_subpath": reference.context_subpath(),
                "kind": reference.get_kind(),
                "clip_id": clip_id,
                "public_clip_id": clip.public_clip_id if clip else None,
                "episode_id": clip.episode_id if clip else None,
                "episode_title": clip.episode_title if clip else None,
                "clip_start_seconds": clip.start_seconds if clip else None,
                "clip_end_seconds": clip.end_seconds if clip else None,
                "clip_duration_seconds": clip.duration_seconds if clip else None,
            },
        }
    ]
