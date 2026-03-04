from __future__ import annotations

from typing import Any

PLUGIN_API_VERSION = "1"
PLUGIN_NAME = "youtube"
PLUGIN_PRIORITY = 100


def normalize_manifest_config(raw_config: dict[str, Any] | None) -> dict[str, Any] | None:
    if raw_config is None:
        return None
    if not isinstance(raw_config, dict):
        raise ValueError("youtube config must be a mapping")
    return dict(raw_config)


def can_resolve(target: str, context: dict[str, Any]) -> bool:
    from .youtube import is_youtube_url

    return is_youtube_url(target)


def classify_target(target: str, context: dict[str, Any]) -> dict[str, Any] | None:
    from .youtube import extract_video_id, is_youtube_url

    if not is_youtube_url(target):
        return None
    kind = "video" if extract_video_id(target) else "resource"
    return {
        "provider": PLUGIN_NAME,
        "kind": kind,
        "is_external": True,
        "group_key": kind,
    }


def resolve(target: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    from .youtube import YouTubeReference, extract_video_id

    reference = YouTubeReference(
        target,
        format="raw",
        label="relative",
        inject=False,
        use_cache=bool(context.get("use_cache", True)),
        cache_ttl=context.get("cache_ttl"),
        refresh_cache=bool(context.get("refresh_cache", False)),
    )
    video_id = extract_video_id(target) or "video"
    return [
        {
            "source": target,
            "label": reference.get_label(),
            "content": reference.read(),
            "metadata": {
                "trace_path": reference.get_label(),
                "provider": PLUGIN_NAME,
                "source_ref": "youtube.com",
                "source_path": video_id,
                "context_subpath": f"youtube-{video_id}.md",
            },
        }
    ]
