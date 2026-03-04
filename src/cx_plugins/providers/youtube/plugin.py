from __future__ import annotations

from typing import Any

PLUGIN_API_VERSION = "1"
PLUGIN_NAME = "youtube"
PLUGIN_PRIORITY = 100


def can_resolve(target: str, context: dict[str, Any]) -> bool:
    from contextualize.references.youtube import is_youtube_url

    return is_youtube_url(target)


def resolve(target: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    from contextualize.references.youtube import YouTubeReference, extract_video_id

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
