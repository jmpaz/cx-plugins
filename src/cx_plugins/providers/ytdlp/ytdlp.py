from __future__ import annotations

import hashlib
import json
import os
import random
import re
import selectors
import shutil
import string
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse, urlunparse


def is_url_target(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _normalized_host(url: str) -> str:
    parsed = urlparse(url.strip())
    host = parsed.hostname or ""
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


_TIKTOK_PHOTO_PATH_RE = re.compile(
    r"^/@(?P<user_id>[\w.-]+)/photo/(?P<id>\d+)(?:/)?$"
)
_TIKTOK_SHORT_PATH_RE = re.compile(r"^/t/[^/]+/?$")
_INSTAGRAM_MEDIA_PATH_RE = re.compile(
    r"^/(?:[^/]+/)?(?P<kind>p|tv|reel|reels)/"
    r"(?!audio(?:/|$))(?P<shortcode>[^/?#&/]+)/?$"
)


def _is_tiktok_host(url: str) -> bool:
    host = _normalized_host(url)
    return host == "tiktok.com" or host.endswith(".tiktok.com")


def _is_instagram_host(url: str) -> bool:
    host = _normalized_host(url)
    return host == "instagram.com" or host.endswith(".instagram.com")


def _instagram_shortcode(url: str) -> str | None:
    if not _is_instagram_host(url):
        return None
    parsed = urlparse(url.strip())
    match = _INSTAGRAM_MEDIA_PATH_RE.match(parsed.path)
    if match is None:
        return None
    return match.group("shortcode")


def is_instagram_media_url(url: str) -> bool:
    return _instagram_shortcode(url) is not None


def _tiktok_photo_match(url: str) -> re.Match[str] | None:
    if not _is_tiktok_host(url):
        return None
    parsed = urlparse(url.strip())
    return _TIKTOK_PHOTO_PATH_RE.match(parsed.path)


def is_tiktok_photo_url(url: str) -> bool:
    return _tiktok_photo_match(url) is not None


def _is_tiktok_short_url(url: str) -> bool:
    if not _is_tiktok_host(url):
        return False
    parsed = urlparse(url.strip())
    return _TIKTOK_SHORT_PATH_RE.match(parsed.path) is not None


def _resolve_tiktok_short_url(
    url: str, *, timeout_seconds: int | float
) -> str:
    if not _is_tiktok_short_url(url):
        return url
    try:
        import requests

        response = requests.get(
            url,
            headers=_TIKTOK_API_HEADERS,
            timeout=max(1.0, min(float(timeout_seconds), 10.0)),
            allow_redirects=False,
        )
    except Exception as exc:
        _log(f"TikTok short link resolution failed for {url}: {exc}")
        return url
    location = response.headers.get("Location")
    if not isinstance(location, str) or not location.strip():
        return url
    resolved = urljoin(url, location.strip())
    if _is_tiktok_host(resolved):
        return resolved
    return url


def _tiktok_photo_as_video_url(url: str) -> str:
    match = _tiktok_photo_match(url)
    if match is None:
        return url
    parsed = urlparse(url.strip())
    video_path = f"/@{match.group('user_id')}/video/{match.group('id')}"
    return urlunparse(parsed._replace(path=video_path))


def is_excluded_ytdlp_url(url: str) -> bool:
    return _normalized_host(url) in {
        "x.com",
        "twitter.com",
        "mobile.x.com",
        "mobile.twitter.com",
    }


@lru_cache(maxsize=1)
def _ytdlp_extractors() -> tuple[Any, ...]:
    try:
        from yt_dlp.extractor import gen_extractors
    except Exception:
        return ()
    try:
        return tuple(gen_extractors())
    except Exception:
        return ()


@lru_cache(maxsize=512)
def matching_ytdlp_extractors(url: str) -> tuple[str, ...]:
    if not is_url_target(url):
        return ()
    names: list[str] = []
    for extractor in _ytdlp_extractors():
        name = getattr(extractor, "IE_NAME", "")
        if name == "generic":
            continue
        try:
            if extractor.suitable(url):
                names.append(name)
        except Exception:
            continue
    return tuple(names)


def requires_ytdlp_probe_for_claim(url: str) -> bool:
    return "Substack" in matching_ytdlp_extractors(url)


@lru_cache(maxsize=512)
def looks_like_ytdlp_url(url: str) -> bool:
    if is_tiktok_photo_url(url) or is_instagram_media_url(url):
        return True
    return bool(matching_ytdlp_extractors(url))


def _log(message: str) -> None:
    try:
        from contextualize.runtime import get_verbose_logging

        if get_verbose_logging():
            print(f"[ytdlp] {message}", file=sys.stderr, flush=True)
    except Exception:
        return


def _clean_identity_part(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _normalize_url_for_key(url: str) -> str:
    parsed = urlparse(url.strip())
    return urlunparse(parsed._replace(fragment=""))


def _youtube_video_id_from_url(url: str) -> str | None:
    parsed = urlparse(url.strip())
    host = _normalized_host(url)
    path_parts = [part for part in parsed.path.split("/") if part]

    if host == "youtu.be":
        return path_parts[0] if path_parts else None

    if host not in {
        "youtube.com",
        "m.youtube.com",
        "music.youtube.com",
    }:
        return None

    if parsed.path == "/watch":
        values = parse_qs(parsed.query).get("v") or []
        return values[0] if values and values[0] else None

    if len(path_parts) >= 2 and path_parts[0] in {"embed", "shorts", "live", "v"}:
        return path_parts[1] or None

    return None


def _url_cache_identity(url: str) -> str:
    normalized = _normalize_url_for_key(url)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"url:{digest}"


def _candidate_render_base_identities(url: str) -> tuple[str, ...]:
    identities: list[str] = []
    youtube_id = _youtube_video_id_from_url(url)
    if youtube_id:
        identities.append(f"youtube:{youtube_id}")
    identities.append(_url_cache_identity(url))
    return tuple(dict.fromkeys(identities))


def _slugify(value: str, *, default: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_.").lower()
    return slug or default


def _yt_dlp_module_available() -> bool:
    try:
        import yt_dlp  # noqa: F401
    except Exception:
        return False
    return True


def _yt_dlp_command() -> list[str]:
    if _yt_dlp_module_available():
        return [sys.executable, "-m", "yt_dlp"]
    executable = shutil.which("yt-dlp")
    if executable:
        return [executable]
    raise RuntimeError(
        "yt-dlp processing requires the yt_dlp Python package or yt-dlp executable: "
        "https://github.com/yt-dlp/yt-dlp"
    )


def _check_ytdlp() -> None:
    _yt_dlp_command()


def _run_ytdlp(
    args: list[str],
    *,
    timeout_seconds: int | float | None,
    idle_timeout_seconds: int | float | None = None,
    stream_progress: bool = False,
) -> subprocess.CompletedProcess:
    command = [*_yt_dlp_command(), *args]
    if stream_progress:
        return _run_ytdlp_streaming(
            command,
            timeout_seconds=timeout_seconds,
            idle_timeout_seconds=idle_timeout_seconds,
        )
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def _run_ytdlp_streaming(
    command: list[str],
    *,
    timeout_seconds: int | float | None,
    idle_timeout_seconds: int | float | None,
) -> subprocess.CompletedProcess:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if process.stdout is None:
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="yt-dlp did not expose an output pipe",
        )

    started = time.monotonic()
    last_activity = started
    output = bytearray()
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)

    def append_output(chunk: bytes) -> None:
        nonlocal last_activity
        if not chunk:
            return
        last_activity = time.monotonic()
        output.extend(chunk)
        text = chunk.decode("utf-8", errors="replace").replace("\r", "\n")
        for line in text.splitlines():
            cleaned = line.strip()
            if cleaned:
                _log(f"yt-dlp: {cleaned}")

    def terminate(reason: str) -> subprocess.CompletedProcess:
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        detail = output.decode("utf-8", errors="replace")
        message = f"{reason}\n{detail}" if detail else reason
        return subprocess.CompletedProcess(
            command,
            process.returncode if process.returncode is not None else -9,
            stdout="",
            stderr=message,
        )

    try:
        while True:
            now = time.monotonic()
            if timeout_seconds is not None and now - started >= timeout_seconds:
                return terminate(
                    f"yt-dlp timed out after {timeout_seconds} seconds"
                )
            if (
                idle_timeout_seconds is not None
                and now - last_activity >= idle_timeout_seconds
            ):
                return terminate(
                    "yt-dlp produced no output for "
                    f"{idle_timeout_seconds} seconds"
                )

            if process.poll() is not None:
                remaining = process.stdout.read()
                if remaining:
                    append_output(remaining)
                break

            deadlines: list[float] = []
            if timeout_seconds is not None:
                deadlines.append(started + timeout_seconds)
            if idle_timeout_seconds is not None:
                deadlines.append(last_activity + idle_timeout_seconds)
            wait_for = 1.0
            if deadlines:
                wait_for = max(0.05, min(1.0, min(deadlines) - time.monotonic()))
            events = selector.select(wait_for)
            for key, _ in events:
                chunk = os.read(key.fileobj.fileno(), 65536)
                if chunk:
                    append_output(chunk)

        return subprocess.CompletedProcess(
            command,
            process.returncode or 0,
            stdout="",
            stderr=output.decode("utf-8", errors="replace"),
        )
    finally:
        selector.close()


def probe_ytdlp_metadata(
    url: str, *, timeout_seconds: int = 10
) -> dict[str, Any] | None:
    if not is_url_target(url):
        return None
    ytdlp_url = _tiktok_photo_as_video_url(
        _resolve_tiktok_short_url(url, timeout_seconds=timeout_seconds)
    )
    try:
        _check_ytdlp()
        result = _run_ytdlp(
            [
                "--dump-single-json",
                "--no-download",
                "--no-playlist",
                "--socket-timeout",
                str(timeout_seconds),
                "--",
                ytdlp_url,
            ],
            timeout_seconds=timeout_seconds,
        )
    except (RuntimeError, OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        if is_instagram_media_url(url):
            return _fetch_instagram_metadata(url)
        return None

    payload = result.stdout.strip()
    if not payload:
        if is_instagram_media_url(url):
            return _fetch_instagram_metadata(url)
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        if is_instagram_media_url(url):
            return _fetch_instagram_metadata(url)
        return None
    if not isinstance(data, dict):
        return None
    if is_instagram_media_url(url) or _is_instagram_metadata(data):
        return _enrich_instagram_metadata(url, data)
    return data


def probe_ytdlp_url(url: str, *, timeout_seconds: int = 10) -> bool:
    return probe_ytdlp_metadata(url, timeout_seconds=timeout_seconds) is not None


def _get_timestamp_interval(duration_seconds: int) -> int:
    if duration_seconds < 300:
        return 60
    if duration_seconds < 1800:
        return 180
    if duration_seconds < 3600:
        return 300
    return 600


def _format_timestamp(seconds: int) -> str:
    minutes, secs = divmod(seconds, 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"[{hours}:{minutes:02d}:{secs:02d}]"
    return f"[{minutes}:{secs:02d}]"


def _insert_timestamps(text: str, duration_seconds: int) -> str:
    if duration_seconds <= 0:
        return text

    interval = _get_timestamp_interval(duration_seconds)
    paragraphs = text.split("\n\n")
    if len(paragraphs) <= 1:
        return text

    total_chars = sum(len(p) for p in paragraphs)
    if total_chars == 0:
        return text

    result = []
    cumulative_chars = 0

    for i, paragraph in enumerate(paragraphs):
        position_ratio = cumulative_chars / total_chars
        current_time = int(position_ratio * duration_seconds)
        marker_time = (current_time // interval) * interval

        if i > 0 and marker_time > 0:
            prev_ratio = (cumulative_chars - len(paragraphs[i - 1])) / total_chars
            prev_time = int(prev_ratio * duration_seconds)
            prev_marker = (prev_time // interval) * interval
            if marker_time > prev_marker:
                paragraph = f"{_format_timestamp(marker_time)}\n{paragraph}"

        result.append(paragraph)
        cumulative_chars += len(paragraph)

    return "\n\n".join(result)


def _split_into_paragraphs(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    paragraphs = []
    current = []
    for sent in sentences:
        current.append(sent)
        if len(current) >= 4:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def _escape_yaml_string(s: str) -> str:
    if not s:
        return '""'
    if "\n" in s:
        indented = "\n".join("  " + line for line in s.split("\n"))
        return f"|\n{indented}"
    needs_quotes = any(c in s for c in ":{}[],\"'|>&*!?#%@`") or s.startswith(" ")
    if needs_quotes:
        escaped = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return s


@dataclass
class _YtDlpIdentity:
    extractor: str | None
    media_id: str | None
    cache_identity: str
    display_name: str
    slug: str


def _build_identity(url: str, metadata: dict[str, Any]) -> _YtDlpIdentity:
    extractor_raw = _clean_identity_part(
        metadata.get("extractor_key")
    ) or _clean_identity_part(metadata.get("extractor"))
    media_id = _clean_identity_part(metadata.get("id"))
    extractor = extractor_raw.lower() if extractor_raw else None
    if extractor and media_id:
        cache_identity = f"{extractor}:{media_id}"
        slug = f"{_slugify(extractor, default='extractor')}-{_slugify(media_id, default='id')}"
        return _YtDlpIdentity(
            extractor=extractor,
            media_id=media_id,
            cache_identity=cache_identity,
            display_name=cache_identity,
            slug=slug,
        )

    cache_identity = _url_cache_identity(url)
    digest = cache_identity.removeprefix("url:")
    short_digest = digest[:12]
    return _YtDlpIdentity(
        extractor=extractor,
        media_id=media_id,
        cache_identity=cache_identity,
        display_name=f"url:{short_digest}",
        slug=f"url-{short_digest}",
    )


def _identity_from_cache_identity(cache_identity: str) -> _YtDlpIdentity:
    if cache_identity.startswith("url:"):
        digest = cache_identity.removeprefix("url:")
        short_digest = digest[:12]
        return _YtDlpIdentity(
            extractor=None,
            media_id=None,
            cache_identity=cache_identity,
            display_name=f"url:{short_digest}",
            slug=f"url-{short_digest}",
        )

    extractor, _, media_id = cache_identity.partition(":")
    extractor = extractor or None
    media_id = media_id or None
    slug_parts = [
        _slugify(part, default="id")
        for part in (extractor, media_id)
        if part is not None
    ]
    return _YtDlpIdentity(
        extractor=extractor,
        media_id=media_id,
        cache_identity=cache_identity,
        display_name=cache_identity,
        slug="-".join(slug_parts) or "media",
    )


def _transcription_routing_identity(
    plugin_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    from contextualize.transcription import transcription_routing_identity

    return transcription_routing_identity(
        filename="media.mp3",
        content_type="audio/mpeg",
        plugin_overrides=plugin_overrides,
    )


def _render_cache_identity(
    base_identity: str, plugin_overrides: dict[str, Any] | None
) -> str:
    transcribe_overrides = None
    video_overrides = None
    if isinstance(plugin_overrides, dict):
        value = plugin_overrides.get("transcribe")
        if isinstance(value, dict):
            transcribe_overrides = dict(value)
        value = plugin_overrides.get("video")
        if isinstance(value, dict):
            video_overrides = dict(value)
    payload = {
        "version": "video-frames-v2",
        "transcribe_overrides": transcribe_overrides or {},
        "video_overrides": video_overrides or {},
        "routing": _transcription_routing_identity(
            _effective_video_transcription_overrides(plugin_overrides)
        ),
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:16]
    return f"{base_identity}:transcribe:{digest}"


def _fast_render_cache_identity(
    base_identity: str, plugin_overrides: dict[str, Any] | None
) -> str:
    transcribe_overrides = None
    video_overrides = None
    if isinstance(plugin_overrides, dict):
        value = plugin_overrides.get("transcribe")
        if isinstance(value, dict):
            transcribe_overrides = dict(value)
        value = plugin_overrides.get("video")
        if isinstance(value, dict):
            video_overrides = dict(value)
    payload = {
        "version": "video-frames-v2",
        "transcribe_overrides": transcribe_overrides or {},
        "video_overrides": video_overrides or {},
        "routing": "fast-v1",
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:16]
    return f"{base_identity}:transcribe-fast:{digest}"


def _effective_video_transcription_overrides(
    plugin_overrides: dict[str, Any] | None,
) -> dict[str, Any] | None:
    try:
        from contextualize.references.video_context import video_transcription_overrides

        return video_transcription_overrides(plugin_overrides)
    except Exception:
        return plugin_overrides


def _source_ref(url: str, metadata: dict[str, Any], extractor: str | None) -> str:
    webpage_url = _clean_identity_part(
        metadata.get("webpage_url")
    ) or _clean_identity_part(metadata.get("original_url"))
    netloc = urlparse((webpage_url or url).strip()).netloc.strip().lower()
    if netloc:
        return netloc
    if extractor:
        return extractor
    return "ytdlp"


def _is_tiktok_metadata(metadata: dict[str, Any]) -> bool:
    extractor = str(
        metadata.get("extractor_key") or metadata.get("extractor") or ""
    ).lower()
    if extractor == "tiktok":
        return True
    webpage_url = metadata.get("webpage_url") or metadata.get("original_url")
    return isinstance(webpage_url, str) and _is_tiktok_host(webpage_url)


def _is_instagram_metadata(metadata: dict[str, Any]) -> bool:
    extractor = str(
        metadata.get("extractor_key") or metadata.get("extractor") or ""
    ).lower()
    if extractor == "instagram":
        return True
    webpage_url = metadata.get("webpage_url") or metadata.get("original_url")
    return isinstance(webpage_url, str) and _is_instagram_host(webpage_url)


def _metadata_has_tiktok_image_post(metadata: dict[str, Any]) -> bool:
    if isinstance(metadata.get("imagePost"), dict):
        return True
    if not _is_tiktok_metadata(metadata):
        return False
    for thumbnail in metadata.get("thumbnails") or ():
        if not isinstance(thumbnail, dict):
            continue
        url = thumbnail.get("url")
        if isinstance(url, str) and "photomode" in url.lower():
            return True
    thumbnail = metadata.get("thumbnail")
    return isinstance(thumbnail, str) and "photomode" in thumbnail.lower()


def _instagram_media_from_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
    media = metadata.get("__instagram_media")
    return media if isinstance(media, dict) else None


def _metadata_has_instagram_image_post(metadata: dict[str, Any]) -> bool:
    if not _is_instagram_metadata(metadata):
        return False
    media = _instagram_media_from_metadata(metadata)
    return media is not None and bool(_instagram_image_entries(media))


def kind_from_ytdlp_metadata(url: str, metadata: dict[str, Any]) -> str:
    if (
        is_tiktok_photo_url(url)
        or _metadata_has_tiktok_image_post(metadata)
        or _metadata_has_instagram_image_post(metadata)
    ):
        return "image"
    duration = metadata.get("duration")
    if isinstance(duration, (int, float)) and duration > 0:
        return "video"
    return "resource"


def _first_image_url(value: Any) -> str | None:
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        return value
    if not isinstance(value, dict):
        return None
    for key in ("urlList", "url_list", "urls"):
        urls = value.get(key)
        if isinstance(urls, list):
            for url in urls:
                if isinstance(url, str) and url.startswith(("http://", "https://")):
                    return url
    for key in ("url", "src"):
        url = value.get(key)
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            return url
    return None


def _tiktok_image_entries(image_post: dict[str, Any]) -> list[dict[str, Any]]:
    raw_images = image_post.get("images")
    if not isinstance(raw_images, list):
        raw_images = []
    entries: list[dict[str, Any]] = []
    for index, image in enumerate(raw_images, start=1):
        if not isinstance(image, dict):
            continue
        url = _first_image_url(image.get("imageURL") or image.get("image_url"))
        if not url:
            continue
        width = image.get("imageWidth") or image.get("width")
        height = image.get("imageHeight") or image.get("height")
        alt = (
            image.get("alt")
            or image.get("altText")
            or image.get("description")
            or image.get("desc")
        )
        entries.append(
            {
                "index": index,
                "url": url,
                "width": width if isinstance(width, int) else None,
                "height": height if isinstance(height, int) else None,
                "alt": alt.strip() if isinstance(alt, str) and alt.strip() else None,
            }
        )
    return entries


def _normalize_image_alttext(markdown: str) -> str:
    text = markdown.strip()
    if not text:
        return ""
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("ImageSize:"):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    if lines:
        first = lines[0].strip().lower()
        if first in {
            "# description (auto-generated):",
            "## description (auto-generated):",
            "# description:",
            "## description:",
        }:
            lines = lines[1:]
            while lines and not lines[0].strip():
                lines = lines[1:]
    return "\n".join(line.rstrip() for line in lines).strip()


def _tiktok_image_post_render_cache_identity(base_identity: str) -> str:
    return f"{base_identity}:image-post:v2"


def _tiktok_transcript_render_base_identity(base_identity: str) -> str:
    return f"{base_identity}:tiktok-sound:v1"


def _instagram_image_post_render_cache_identity(base_identity: str) -> str:
    return f"{base_identity}:instagram-image-post:v1"


def _instagram_transcript_render_base_identity(base_identity: str) -> str:
    return f"{base_identity}:instagram-sound:v1"


def _image_cache_identity(base_identity: str, image: dict[str, Any]) -> str:
    url = str(image.get("url") or "")
    index = image.get("index")
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"image:{base_identity}:{index}:{digest}"


def _tiktok_image_cache_identity(base_identity: str, image: dict[str, Any]) -> str:
    return _image_cache_identity(base_identity, image)


def _image_suffix(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix
    return suffix if suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"


def _tiktok_image_suffix(url: str) -> str:
    return _image_suffix(url)


def _should_refresh_social_images() -> bool:
    try:
        from contextualize.runtime import get_refresh_images, get_refresh_media

        return get_refresh_images() or get_refresh_media()
    except Exception:
        return False


def _should_refresh_tiktok_images() -> bool:
    return _should_refresh_social_images()


_TIKTOK_API_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.tiktok.com/",
}

_TIKTOK_IMAGE_HEADERS = {
    "User-Agent": _TIKTOK_API_HEADERS["User-Agent"],
    "Accept": "image/*,*/*;q=0.8",
    "Referer": "https://www.tiktok.com/",
}

_INSTAGRAM_API_HEADERS = {
    "User-Agent": _TIKTOK_API_HEADERS["User-Agent"],
    "Accept": "*/*",
    "X-Requested-With": "XMLHttpRequest",
}

_INSTAGRAM_IMAGE_HEADERS = {
    "User-Agent": _TIKTOK_API_HEADERS["User-Agent"],
    "Accept": "image/*,*/*;q=0.8",
    "Referer": "https://www.instagram.com/",
}


def _tiktok_item_list_params(sec_uid: str, cursor_ms: int) -> dict[str, str]:
    return {
        "aid": "1988",
        "app_language": "en",
        "app_name": "tiktok_web",
        "browser_language": "en-US",
        "browser_name": "Mozilla",
        "browser_online": "true",
        "browser_platform": "Win32",
        "browser_version": "5.0 (Windows)",
        "channel": "tiktok_web",
        "cookie_enabled": "true",
        "count": "15",
        "cursor": str(cursor_ms),
        "device_id": str(random.randint(7250000000000000000, 7325099899999994577)),
        "device_platform": "web_pc",
        "focus_state": "true",
        "from_page": "user",
        "history_len": "2",
        "is_fullscreen": "false",
        "is_page_visible": "true",
        "language": "en",
        "os": "windows",
        "priority_region": "",
        "referer": "",
        "region": "US",
        "screen_height": "1080",
        "screen_width": "1920",
        "secUid": sec_uid,
        "type": "1",
        "tz_name": "America/New_York",
        "verifyFp": f"verify_{''.join(random.choices(string.hexdigits, k=7))}",
        "webcast_language": "en",
    }


def _tiktok_item_list_cursors(metadata: dict[str, Any]) -> tuple[int, ...]:
    timestamp = metadata.get("timestamp")
    cursors: list[int] = []
    if isinstance(timestamp, (int, float)) and timestamp > 0:
        base = int(timestamp * 1000)
        cursors.extend(
            [
                base + 1000,
                base + 86_400_000,
                base + 7 * 86_400_000,
            ]
        )
    cursors.append(int(time.time() * 1000))
    return tuple(dict.fromkeys(cursors))


def _fetch_tiktok_item(metadata: dict[str, Any]) -> dict[str, Any] | None:
    if not _is_tiktok_metadata(metadata):
        return None
    sec_uid = metadata.get("channel_id")
    item_id = metadata.get("id")
    if not isinstance(sec_uid, str) or not sec_uid:
        return None
    if not isinstance(item_id, str) or not item_id:
        return None

    import requests

    for cursor_ms in _tiktok_item_list_cursors(metadata):
        try:
            response = requests.get(
                "https://www.tiktok.com/api/creator/item_list/",
                headers=_TIKTOK_API_HEADERS,
                params=_tiktok_item_list_params(sec_uid, cursor_ms),
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            _log(f"TikTok item lookup failed for {item_id}: {exc}")
            continue
        for item in payload.get("itemList") or ():
            if not isinstance(item, dict) or item.get("id") != item_id:
                continue
            return item
    return None


def _tiktok_image_post_from_item(
    metadata: dict[str, Any],
    item: dict[str, Any] | None,
) -> dict[str, Any] | None:
    image_post = metadata.get("imagePost")
    if isinstance(image_post, dict) and _tiktok_image_entries(image_post):
        return image_post
    if not isinstance(item, dict):
        return None
    image_post = item.get("imagePost")
    if isinstance(image_post, dict) and _tiktok_image_entries(image_post):
        return image_post
    return None


def _fetch_tiktok_image_post(metadata: dict[str, Any]) -> dict[str, Any] | None:
    return _tiktok_image_post_from_item(metadata, _fetch_tiktok_item(metadata))


def _instagram_graphql_variables(shortcode: str) -> dict[str, Any]:
    return {
        "shortcode": shortcode,
        "child_comment_count": 3,
        "fetch_comment_count": 40,
        "parent_comment_count": 24,
        "has_threaded_comments": True,
    }


def _fetch_instagram_media(url: str) -> dict[str, Any] | None:
    shortcode = _instagram_shortcode(url)
    if not shortcode:
        return None

    import requests

    try:
        response = requests.get(
            "https://www.instagram.com/graphql/query/",
            headers={
                **_INSTAGRAM_API_HEADERS,
                "Referer": url,
                "X-CSRFToken": "",
            },
            params={
                "doc_id": "8845758582119845",
                "variables": json.dumps(
                    _instagram_graphql_variables(shortcode),
                    separators=(",", ":"),
                ),
            },
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        _log(f"Instagram media lookup failed for {shortcode}: {exc}")
        return None

    data = payload.get("data") if isinstance(payload, dict) else None
    media = data.get("xdt_shortcode_media") if isinstance(data, dict) else None
    return media if isinstance(media, dict) else None


def _clean_metadata_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _metadata_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _metadata_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _metadata_artists(value: Any) -> str | None:
    if isinstance(value, list):
        names = [
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        ]
        return ", ".join(names) or None
    return _clean_metadata_text(value)


def _instagram_owner(media: dict[str, Any]) -> dict[str, Any]:
    owner = media.get("owner")
    if isinstance(owner, dict):
        return owner
    user = media.get("user")
    return user if isinstance(user, dict) else {}


def _instagram_caption_text(media: dict[str, Any]) -> str | None:
    edge = media.get("edge_media_to_caption")
    if isinstance(edge, dict):
        edges = edge.get("edges")
        if isinstance(edges, list):
            for item in edges:
                if not isinstance(item, dict):
                    continue
                node = item.get("node")
                if not isinstance(node, dict):
                    continue
                text = _clean_metadata_text(node.get("text"))
                if text:
                    return text
    caption = media.get("caption")
    if isinstance(caption, dict):
        return _clean_metadata_text(caption.get("text"))
    return _clean_metadata_text(caption)


def _instagram_display_image_url(media: dict[str, Any]) -> str | None:
    url = _first_image_url(media.get("display_url")) or _first_image_url(
        media.get("thumbnail_src")
    )
    if url:
        return url
    resources = media.get("display_resources")
    if isinstance(resources, list):
        for resource in reversed(resources):
            if isinstance(resource, dict):
                url = _first_image_url(resource)
                if url:
                    return url
    image_versions = media.get("image_versions2")
    if isinstance(image_versions, dict):
        candidates = image_versions.get("candidates")
        if isinstance(candidates, list):
            for candidate in candidates:
                if isinstance(candidate, dict):
                    url = _first_image_url(candidate)
                    if url:
                        return url
    return None


def _instagram_dimensions(media: dict[str, Any]) -> tuple[int | None, int | None]:
    dimensions = media.get("dimensions")
    if isinstance(dimensions, dict):
        width = dimensions.get("width")
        height = dimensions.get("height")
    else:
        width = media.get("width")
        height = media.get("height")
    return (
        width if isinstance(width, int) else None,
        height if isinstance(height, int) else None,
    )


def _instagram_image_entries(media: dict[str, Any]) -> list[dict[str, Any]]:
    sidecar = media.get("edge_sidecar_to_children")
    edges = sidecar.get("edges") if isinstance(sidecar, dict) else None
    if isinstance(edges, list) and edges:
        nodes = [
            edge.get("node")
            for edge in edges
            if isinstance(edge, dict) and isinstance(edge.get("node"), dict)
        ]
    else:
        nodes = [media]

    entries: list[dict[str, Any]] = []
    for index, node in enumerate(nodes, start=1):
        if not isinstance(node, dict):
            continue
        if node.get("is_video") is True or str(node.get("__typename", "")).endswith(
            "Video"
        ):
            continue
        url = _instagram_display_image_url(node)
        if not url:
            continue
        width, height = _instagram_dimensions(node)
        alt = _clean_metadata_text(node.get("accessibility_caption"))
        entries.append(
            {
                "index": index,
                "url": url,
                "width": width,
                "height": height,
                "alt": alt,
            }
        )
    return entries


def _instagram_metadata_from_media(
    url: str, media: dict[str, Any]
) -> dict[str, Any]:
    owner = _instagram_owner(media)
    username = _clean_metadata_text(owner.get("username"))
    full_name = _clean_metadata_text(owner.get("full_name"))
    description = _instagram_caption_text(media)
    shortcode = _clean_metadata_text(media.get("shortcode")) or _instagram_shortcode(
        url
    )
    is_video = media.get("is_video") is True or str(
        media.get("__typename", "")
    ).endswith("Video")
    title = _clean_metadata_text(media.get("title"))
    if not title:
        if username:
            title = f"{'Video' if is_video else 'Post'} by {username}"
        else:
            title = "Instagram media"

    metadata: dict[str, Any] = {
        "extractor": "Instagram",
        "extractor_key": "Instagram",
        "id": shortcode or _clean_metadata_text(media.get("id")),
        "webpage_url": url,
        "original_url": url,
        "title": title,
        "__instagram_media": media,
    }
    if description:
        metadata["description"] = description
    if username:
        metadata["channel"] = username
    if full_name:
        metadata["uploader"] = full_name
    uploader_id = _clean_metadata_text(owner.get("id")) or _clean_metadata_text(
        owner.get("pk")
    )
    if uploader_id:
        metadata["uploader_id"] = uploader_id
    timestamp = _metadata_int(media.get("taken_at_timestamp") or media.get("taken_at"))
    if timestamp is not None:
        metadata["timestamp"] = timestamp
    duration = _metadata_float(media.get("video_duration"))
    if duration is not None:
        metadata["duration"] = duration
    thumbnail = _instagram_display_image_url(media)
    if thumbnail:
        metadata["thumbnail"] = thumbnail
    resources = media.get("display_resources")
    if isinstance(resources, list):
        thumbnails = []
        for resource in resources:
            if not isinstance(resource, dict):
                continue
            thumb_url = _first_image_url(resource)
            if not thumb_url:
                continue
            thumbnails.append(
                {
                    "url": thumb_url,
                    "width": resource.get("config_width"),
                    "height": resource.get("config_height"),
                }
            )
        if thumbnails:
            metadata["thumbnails"] = thumbnails
    return metadata


def _enrich_instagram_metadata(
    url: str, metadata: dict[str, Any]
) -> dict[str, Any]:
    media = _fetch_instagram_media(url)
    if media is None:
        return metadata
    graph_metadata = _instagram_metadata_from_media(url, media)
    enriched = dict(metadata)
    for key, value in graph_metadata.items():
        if key == "__instagram_media" or enriched.get(key) in (None, "", []):
            enriched[key] = value
    return enriched


def _fetch_instagram_metadata(url: str) -> dict[str, Any] | None:
    media = _fetch_instagram_media(url)
    if media is None:
        return None
    return _instagram_metadata_from_media(url, media)


def _tiktok_music_metadata(
    metadata: dict[str, Any],
    item: dict[str, Any] | None,
) -> dict[str, Any]:
    music = item.get("music") if isinstance(item, dict) else None
    if not isinstance(music, dict):
        music = {}

    title = _clean_metadata_text(music.get("title")) or _clean_metadata_text(
        metadata.get("track")
    )
    artist = (
        _clean_metadata_text(music.get("authorName"))
        or _clean_metadata_text(music.get("author"))
        or _clean_metadata_text(metadata.get("artist"))
        or _metadata_artists(metadata.get("artists"))
    )
    album = _clean_metadata_text(music.get("album")) or _clean_metadata_text(
        metadata.get("album")
    )
    music_id = _clean_metadata_text(music.get("id")) or _clean_metadata_text(
        metadata.get("music_id")
    )
    duration_seconds = _metadata_int(music.get("duration"))
    original = music.get("original")
    if not isinstance(original, bool):
        original = None

    return {
        "title": title,
        "artist": artist,
        "album": album,
        "id": music_id,
        "duration_seconds": duration_seconds,
        "original": original,
        "url": _tiktok_music_url(title, music_id),
    }


def _tiktok_music_url(title: str | None, music_id: str | None) -> str | None:
    if not music_id:
        return None
    slug = _slugify(title or "sound", default="sound")
    return f"https://www.tiktok.com/music/{slug}-{music_id}"


def _instagram_audio_url(audio_id: str | None) -> str | None:
    if not audio_id or audio_id == "0":
        return None
    return f"https://www.instagram.com/reels/audio/{audio_id}/"


def _instagram_sound_metadata(
    metadata: dict[str, Any],
    media: dict[str, Any] | None,
) -> dict[str, Any]:
    music = (
        media.get("clips_music_attribution_info")
        if isinstance(media, dict)
        else None
    )
    if not isinstance(music, dict):
        music = {}

    original = music.get("uses_original_audio")
    if not isinstance(original, bool):
        original = None
    muted = music.get("should_mute_audio")
    if not isinstance(muted, bool):
        muted = None

    title = _clean_metadata_text(music.get("song_name")) or _clean_metadata_text(
        metadata.get("track")
    )
    artist = (
        _clean_metadata_text(music.get("artist_name"))
        or _clean_metadata_text(metadata.get("artist"))
        or _metadata_artists(metadata.get("artists"))
    )
    if original is True:
        title = title or "original audio"
        artist = artist or _clean_metadata_text(
            metadata.get("channel") or metadata.get("uploader")
        )
    audio_id = _clean_metadata_text(music.get("audio_id"))

    return {
        "title": title,
        "artist": artist,
        "album": _clean_metadata_text(metadata.get("album")),
        "duration_seconds": None,
        "original": original,
        "url": _instagram_audio_url(audio_id),
        "muted": muted,
        "mute_reason": _clean_metadata_text(music.get("should_mute_audio_reason")),
    }


def _append_sound_field(lines: list[str], key: str, value: Any) -> None:
    if isinstance(value, bool):
        lines.append(f"- {key}: {str(value).lower()}")
        return
    if isinstance(value, int):
        lines.append(f"- {key}: {value}")
        return
    cleaned = _clean_metadata_text(value)
    if cleaned:
        lines.append(f"- {key}: {cleaned}")


def _sound_has_renderable_metadata(sound: dict[str, Any]) -> bool:
    for key in ("title", "artist", "duration_seconds", "url", "album", "mute_reason"):
        if sound.get(key) is not None:
            return True
    return sound.get("original") is True or sound.get("muted") is True


def _format_sound_section(sound: dict[str, Any]) -> list[str]:
    if not _sound_has_renderable_metadata(sound):
        return []

    lines = ["## Sound", ""]
    _append_sound_field(lines, "title", sound.get("title"))
    _append_sound_field(lines, "artist", sound.get("artist"))
    _append_sound_field(lines, "duration_seconds", sound.get("duration_seconds"))
    _append_sound_field(lines, "url", sound.get("url"))
    _append_sound_field(lines, "album", sound.get("album"))
    _append_sound_field(lines, "original", sound.get("original"))
    if sound.get("muted") is True or sound.get("mute_reason") is not None:
        _append_sound_field(lines, "muted", sound.get("muted"))
    _append_sound_field(lines, "mute_reason", sound.get("mute_reason"))
    return lines


def _format_tiktok_sound_section(
    metadata: dict[str, Any],
    item: dict[str, Any] | None,
) -> list[str]:
    if not _is_tiktok_metadata(metadata):
        return []
    return _format_sound_section(_tiktok_music_metadata(metadata, item))


def _format_instagram_sound_section(
    metadata: dict[str, Any],
    media: dict[str, Any] | None,
) -> list[str]:
    if not _is_instagram_metadata(metadata):
        return []
    return _format_sound_section(_instagram_sound_metadata(metadata, media))


@dataclass
class YtDlpReference:
    url: str
    format: str = "md"
    label: str = "relative"
    token_target: str = "cl100k_base"
    include_token_count: bool = False
    label_suffix: str | None = None
    inject: bool = False
    depth: int = 5
    trace_collector: list[Any] | None = None
    use_cache: bool = True
    cache_ttl: timedelta | None = None
    refresh_cache: bool = False
    plugin_overrides: dict[str, Any] | None = None
    _metadata: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _identity: _YtDlpIdentity | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        _check_ytdlp()
        if not is_url_target(self.url):
            raise ValueError(f"Unsupported URL target: {self.url}")
        self.file_content = ""
        self.original_file_content = ""
        self.output = self._get_contents()

    @property
    def path(self) -> str:
        return self.url

    def _metadata_url(self) -> str:
        return _tiktok_photo_as_video_url(self.url)

    def read(self) -> str:
        return self.original_file_content

    def exists(self) -> bool:
        try:
            self._fetch_metadata()
            return True
        except Exception:
            return False

    def token_count(self, encoding: str = "cl100k_base") -> int:
        from contextualize.utils import count_tokens

        return count_tokens(self.original_file_content, target=encoding)["count"]

    def get_label(self) -> str:
        if self.label == "relative":
            return self.url
        if self.label == "name":
            return self._get_identity().display_name
        if self.label == "ext":
            return ""
        return self.label

    def source_ref(self) -> str:
        if self._metadata is None and self._identity is not None:
            return urlparse(self.url.strip()).netloc.strip().lower() or (
                self._identity.extractor or "ytdlp"
            )
        metadata = self._fetch_metadata()
        return _source_ref(self.url, metadata, self._get_identity().extractor)

    def source_path(self) -> str:
        return self._get_identity().cache_identity

    def context_subpath(self) -> str:
        return f"ytdlp-{self._get_identity().slug}.md"

    def get_kind(self) -> str:
        if self._metadata is None and self._identity is not None:
            return "video"
        metadata = self._fetch_metadata()
        return kind_from_ytdlp_metadata(self.url, metadata)

    def _fetch_metadata(self) -> dict[str, Any]:
        if self._metadata is not None:
            return self._metadata

        _log(f"fetching metadata for {self.url}")
        result = _run_ytdlp(
            [
                "--dump-single-json",
                "--no-download",
                "--no-playlist",
                "--",
                self._metadata_url(),
            ],
            timeout_seconds=60,
        )
        if result.returncode != 0:
            if is_instagram_media_url(self.url):
                metadata = _fetch_instagram_metadata(self.url)
                if metadata is not None:
                    self._metadata = metadata
                    return self._metadata
            raise RuntimeError(f"yt-dlp metadata failed: {result.stderr}")

        try:
            metadata = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            if is_instagram_media_url(self.url):
                metadata = _fetch_instagram_metadata(self.url)
                if metadata is not None:
                    self._metadata = metadata
                    return self._metadata
            raise RuntimeError(f"yt-dlp metadata parse failed: {exc}") from exc
        if not isinstance(metadata, dict):
            raise RuntimeError("yt-dlp metadata returned unexpected payload")
        if is_instagram_media_url(self.url) or _is_instagram_metadata(metadata):
            metadata = _enrich_instagram_metadata(self.url, metadata)
        self._metadata = metadata
        title = metadata.get("title") if isinstance(metadata, dict) else None
        _log(f"metadata fetched for {title or self.url}")
        return self._metadata

    def _get_identity(self) -> _YtDlpIdentity:
        if self._identity is None:
            self._identity = _build_identity(self.url, self._fetch_metadata())
        return self._identity

    def _extract_audio(self) -> Path:
        from .cache import (
            get_cached_media_bytes,
            store_media_bytes,
        )
        from contextualize.runtime import get_refresh_audio

        identity = self._get_identity()
        cache_identity = f"audio:{identity.cache_identity}"
        if self.use_cache and not get_refresh_audio():
            cached = get_cached_media_bytes(cache_identity)
            if cached:
                _log(f"audio cache hit for {identity.display_name}")
                tmpdir = Path(tempfile.mkdtemp(prefix="ytdlp-audio-cache-"))
                path = tmpdir / "audio.mp3"
                path.write_bytes(cached)
                return path
        _log(f"extracting audio with yt-dlp for {identity.display_name}")
        tmpdir = tempfile.mkdtemp(prefix="ytdlp-")
        output_template = os.path.join(tmpdir, f"{identity.slug}.%(ext)s")

        result = _run_ytdlp(
            [
                "-f",
                "bestaudio/best",
                "--concurrent-fragments",
                "16",
                "-x",
                "--newline",
                "--progress",
                "--socket-timeout",
                "30",
                "--audio-format",
                "mp3",
                "--audio-quality",
                "5",
                "--no-playlist",
                "-o",
                output_template,
                "--",
                self._metadata_url(),
            ],
            timeout_seconds=None,
            idle_timeout_seconds=180,
            stream_progress=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp audio extraction failed: {result.stderr}")

        audio_dir = Path(tmpdir)
        audio_files = sorted(audio_dir.glob("*.mp3"))
        if not audio_files:
            audio_files = sorted(path for path in audio_dir.iterdir() if path.is_file())
        if not audio_files:
            raise RuntimeError("yt-dlp audio extraction produced no audio file")
        _log(f"audio extraction finished for {identity.display_name}")
        if self.use_cache:
            try:
                store_media_bytes(cache_identity, audio_files[0].read_bytes())
                _log(f"stored extracted audio cache for {identity.display_name}")
            except OSError:
                pass
        return audio_files[0]

    def _extract_video(self) -> Path:
        from .cache import (
            get_cached_media_bytes,
            store_media_bytes,
        )
        from contextualize.runtime import get_refresh_videos

        identity = self._get_identity()
        cache_identity = f"video:{identity.cache_identity}"
        if self.use_cache and not get_refresh_videos():
            cached = get_cached_media_bytes(cache_identity)
            if cached:
                _log(f"video cache hit for {identity.display_name}")
                tmpdir = Path(tempfile.mkdtemp(prefix="ytdlp-video-cache-"))
                path = tmpdir / "video.mp4"
                path.write_bytes(cached)
                return path
        _log(f"extracting video with yt-dlp for {identity.display_name}")
        tmpdir = tempfile.mkdtemp(prefix="ytdlp-video-")
        output_template = os.path.join(tmpdir, f"{identity.slug}.%(ext)s")

        result = _run_ytdlp(
            [
                "-f",
                "bv*[height<=720][ext=mp4]/bv*[height<=720]/best[height<=720]/best",
                "--merge-output-format",
                "mp4",
                "--concurrent-fragments",
                "16",
                "--newline",
                "--progress",
                "--socket-timeout",
                "30",
                "--no-playlist",
                "-o",
                output_template,
                "--",
                self._metadata_url(),
            ],
            timeout_seconds=None,
            idle_timeout_seconds=180,
            stream_progress=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp video extraction failed: {result.stderr}")

        video_dir = Path(tmpdir)
        video_files = sorted(video_dir.glob("*.mp4"))
        if not video_files:
            video_files = sorted(path for path in video_dir.iterdir() if path.is_file())
        if not video_files:
            raise RuntimeError("yt-dlp video extraction produced no video file")
        _log(f"video extraction finished for {identity.display_name}")
        if self.use_cache:
            try:
                store_media_bytes(cache_identity, video_files[0].read_bytes())
                _log(f"stored extracted video cache for {identity.display_name}")
            except OSError:
                pass
        return video_files[0]

    def _get_transcript_result(self, _duration: int):
        audio_path = None
        try:
            audio_path = self._extract_audio()
            from contextualize.transcription import (
                transcribe_media_file_result,
            )

            identity = getattr(self, "_identity", None)
            display_name = getattr(identity, "display_name", audio_path.name)
            _log(f"transcribing extracted audio for {display_name}")
            result = transcribe_media_file_result(
                audio_path,
                use_cache=self.use_cache,
                refresh_cache=None,
                plugin_overrides=_effective_video_transcription_overrides(
                    self.plugin_overrides
                ),
            )
            _log(f"transcription finished for {display_name}")
            return result, "transcription"
        finally:
            if audio_path and audio_path.exists():
                shutil.rmtree(audio_path.parent, ignore_errors=True)

    def _get_transcript(self, _duration: int) -> tuple[str, str]:
        result, source = self._get_transcript_result(_duration)
        return result.text, source

    def _render_video_frames(self, transcript_result) -> str:
        try:
            from contextualize.references.video_context import (
                render_video_frame_section,
                resolve_video_frame_settings,
            )
            from contextualize.runtime import get_skip_media

            if get_skip_media():
                return ""
            if not resolve_video_frame_settings(self.plugin_overrides).frames:
                return ""

            video_path = None
            try:
                video_path = self._extract_video()
                return render_video_frame_section(
                    video_path,
                    transcript_result=transcript_result,
                    use_cache=self.use_cache,
                    refresh_cache=None,
                    plugin_overrides=self.plugin_overrides,
                    source_url=self.url,
                )
            finally:
                if video_path and video_path.exists():
                    shutil.rmtree(video_path.parent, ignore_errors=True)
        except Exception as exc:
            identity = getattr(self, "_identity", None)
            display_name = getattr(identity, "display_name", self.url)
            _log(f"video frame rendering failed for {display_name}: {exc}")
            return ""

    def _get_transcript_and_frames(self, duration: int) -> tuple[str, str, str]:
        transcript_result = None
        transcript = ""
        source = "none"
        try:
            transcript_result, source = self._get_transcript_result(duration)
            transcript = transcript_result.text
        except Exception as exc:
            identity = getattr(self, "_identity", None)
            display_name = getattr(identity, "display_name", self.url)
            _log(f"transcription unavailable for {display_name}: {exc}")
        return transcript, source, self._render_video_frames(transcript_result)

    def _render_output_text(self, text: str) -> str:
        from contextualize.render.text import process_text

        if self.inject:
            from contextualize.render.inject import inject_content_in_text

            text = inject_content_in_text(
                text, self.depth, self.trace_collector, self.url
            )
            self.file_content = text

        return process_text(
            text,
            format=self.format,
            label=self.get_label(),
            label_suffix=self.label_suffix,
            token_target=self.token_target,
            include_token_count=self.include_token_count,
        )

    def _record_render_cache_hit(self, text: str, identity: _YtDlpIdentity) -> str:
        from contextualize.transcription import (
            record_transcription_routing_summary,
        )

        self._identity = identity
        record_transcription_routing_summary(
            filename="media.mp3",
            content_type="audio/mpeg",
            plugin_overrides=self.plugin_overrides,
            source="render-cache",
        )
        _log(f"render cache hit for {identity.display_name}")
        self.original_file_content = text
        self.file_content = text
        return self._render_output_text(text)

    def _cached_render_output(
        self,
        base_identity: str,
        whisper_available: bool,
        *,
        fast: bool = False,
    ) -> str | None:
        if not self.use_cache or self.refresh_cache:
            return None

        from .cache import get_cached_transcript

        if fast:
            render_cache_identity = _fast_render_cache_identity(
                base_identity,
                self.plugin_overrides,
            )
        else:
            render_cache_identity = _render_cache_identity(
                base_identity,
                self.plugin_overrides,
            )
        cached = get_cached_transcript(
            render_cache_identity,
            self.cache_ttl,
            whisper_available=whisper_available,
        )
        if cached is None:
            return None
        if not fast:
            from .cache import store_transcript

            store_transcript(
                _fast_render_cache_identity(base_identity, self.plugin_overrides),
                cached,
                source="render-cache",
            )
        return self._record_render_cache_hit(
            cached,
            _identity_from_cache_identity(base_identity),
        )

    def _store_render_cache_aliases(
        self,
        *,
        primary_base_identity: str,
        text: str,
        source: str,
    ) -> None:
        if not self.use_cache or source == "none":
            return

        from .cache import store_transcript

        identities = (
            primary_base_identity,
            *_candidate_render_base_identities(self.url),
        )
        for base_identity in tuple(dict.fromkeys(identities)):
            if base_identity == primary_base_identity:
                full_identity = _render_cache_identity(
                    base_identity,
                    self.plugin_overrides,
                )
                store_transcript(full_identity, text, source=source)
            else:
                store_transcript(
                    _render_cache_identity(base_identity, self.plugin_overrides),
                    text,
                    source=source,
                )
            store_transcript(
                _fast_render_cache_identity(base_identity, self.plugin_overrides),
                text,
                source=source,
            )

    def _tiktok_transcript_base_identities(
        self, identity: _YtDlpIdentity
    ) -> tuple[str, ...]:
        identities = (
            identity.cache_identity,
            *_candidate_render_base_identities(self.url),
        )
        return tuple(dict.fromkeys(identities))

    def _cached_tiktok_transcript_output(
        self,
        identity: _YtDlpIdentity,
        whisper_available: bool,
    ) -> str | None:
        if not self.use_cache or self.refresh_cache:
            return None

        from .cache import get_cached_transcript

        for base_identity in self._tiktok_transcript_base_identities(identity):
            cached = get_cached_transcript(
                _render_cache_identity(
                    _tiktok_transcript_render_base_identity(base_identity),
                    self.plugin_overrides,
                ),
                self.cache_ttl,
                whisper_available=whisper_available,
            )
            if cached is not None:
                return self._record_render_cache_hit(cached, identity)
        return None

    def _store_tiktok_transcript_output(
        self,
        identity: _YtDlpIdentity,
        *,
        text: str,
        source: str,
    ) -> None:
        if not self.use_cache or source == "none":
            return

        from .cache import store_transcript

        for base_identity in self._tiktok_transcript_base_identities(identity):
            store_transcript(
                _render_cache_identity(
                    _tiktok_transcript_render_base_identity(base_identity),
                    self.plugin_overrides,
                ),
                text,
                source=source,
            )

    def _instagram_transcript_base_identities(
        self, identity: _YtDlpIdentity
    ) -> tuple[str, ...]:
        identities = (
            identity.cache_identity,
            *_candidate_render_base_identities(self.url),
        )
        return tuple(dict.fromkeys(identities))

    def _cached_instagram_transcript_output(
        self,
        identity: _YtDlpIdentity,
        whisper_available: bool,
    ) -> str | None:
        if not self.use_cache or self.refresh_cache:
            return None

        from .cache import get_cached_transcript

        for base_identity in self._instagram_transcript_base_identities(identity):
            cached = get_cached_transcript(
                _render_cache_identity(
                    _instagram_transcript_render_base_identity(base_identity),
                    self.plugin_overrides,
                ),
                self.cache_ttl,
                whisper_available=whisper_available,
            )
            if cached is not None:
                return self._record_render_cache_hit(cached, identity)
        return None

    def _store_instagram_transcript_output(
        self,
        identity: _YtDlpIdentity,
        *,
        text: str,
        source: str,
    ) -> None:
        if not self.use_cache or source == "none":
            return

        from .cache import store_transcript

        for base_identity in self._instagram_transcript_base_identities(identity):
            store_transcript(
                _render_cache_identity(
                    _instagram_transcript_render_base_identity(base_identity),
                    self.plugin_overrides,
                ),
                text,
                source=source,
            )

    def _cached_tiktok_image_post_output(self, base_identity: str) -> str | None:
        if not self.use_cache or self.refresh_cache or _should_refresh_tiktok_images():
            return None

        from .cache import get_cached_transcript

        cached = get_cached_transcript(
            _tiktok_image_post_render_cache_identity(base_identity),
            self.cache_ttl,
        )
        if cached is None:
            return None
        _log(f"image post render cache hit for {base_identity}")
        self.original_file_content = cached
        self.file_content = cached
        return self._render_output_text(cached)

    def _store_tiktok_image_post_output(self, base_identity: str, text: str) -> None:
        if not self.use_cache:
            return

        from .cache import store_transcript

        store_transcript(
            _tiktok_image_post_render_cache_identity(base_identity),
            text,
            source="image-post",
        )

    def _cached_instagram_image_post_output(self, base_identity: str) -> str | None:
        if not self.use_cache or self.refresh_cache or _should_refresh_social_images():
            return None

        from .cache import get_cached_transcript

        cached = get_cached_transcript(
            _instagram_image_post_render_cache_identity(base_identity),
            self.cache_ttl,
        )
        if cached is None:
            return None
        _log(f"Instagram image post render cache hit for {base_identity}")
        self.original_file_content = cached
        self.file_content = cached
        return self._render_output_text(cached)

    def _store_instagram_image_post_output(
        self, base_identity: str, text: str
    ) -> None:
        if not self.use_cache:
            return

        from .cache import store_transcript

        store_transcript(
            _instagram_image_post_render_cache_identity(base_identity),
            text,
            source="image-post",
        )

    def _describe_social_image(
        self,
        image: dict[str, Any],
        *,
        metadata: dict[str, Any],
        total_images: int,
        platform: str,
        headers: dict[str, str],
    ) -> str:
        native_alt = image.get("alt")
        if isinstance(native_alt, str) and native_alt.strip():
            return native_alt.strip()

        from .cache import get_cached_media_bytes, store_media_bytes
        from contextualize.render.markitdown import convert_path_to_markdown
        from ..shared.media import download_cached_media_to_temp

        url = str(image.get("url") or "")
        if not url:
            return ""
        identity = self._get_identity()
        refresh_images = _should_refresh_social_images()
        tmp = download_cached_media_to_temp(
            url,
            suffix=_image_suffix(url),
            headers=headers,
            cache_identity=_image_cache_identity(identity.cache_identity, image),
            get_cached_media_bytes=get_cached_media_bytes,
            store_media_bytes=store_media_bytes,
            refresh_cache=refresh_images,
        )
        if tmp is None:
            return ""

        caption = metadata.get("description")
        if not isinstance(caption, str) or not caption.strip():
            caption = metadata.get("title")
        prompt_parts = [
            (
                f"Image {image.get('index')} of {total_images}. "
                "Write concise alt text for this image."
            )
        ]
        if isinstance(caption, str) and caption.strip():
            prompt_parts.insert(0, f"{platform} image post caption: {caption.strip()}")
        prompt_append = "\n".join(prompt_parts)

        try:
            result = convert_path_to_markdown(
                str(tmp),
                refresh_images=refresh_images,
                prompt_append=prompt_append,
                source_url=url,
            )
            return _normalize_image_alttext(result.markdown or "")
        except Exception as exc:
            _log(f"{platform} image description failed for {url}: {exc}")
            return ""
        finally:
            tmp.unlink(missing_ok=True)

    def _describe_tiktok_image(
        self,
        image: dict[str, Any],
        *,
        metadata: dict[str, Any],
        total_images: int,
    ) -> str:
        return self._describe_social_image(
            image,
            metadata=metadata,
            total_images=total_images,
            platform="TikTok",
            headers=_TIKTOK_IMAGE_HEADERS,
        )

    def _describe_instagram_image(
        self,
        image: dict[str, Any],
        *,
        metadata: dict[str, Any],
        total_images: int,
    ) -> str:
        return self._describe_social_image(
            image,
            metadata=metadata,
            total_images=total_images,
            platform="Instagram",
            headers=_INSTAGRAM_IMAGE_HEADERS,
        )

    def _format_tiktok_image_post_output(
        self,
        metadata: dict[str, Any],
        image_post: dict[str, Any],
        tiktok_item: dict[str, Any] | None = None,
    ) -> str:
        entries = _tiktok_image_entries(image_post)
        title = metadata.get("title") or "TikTok image post"
        channel = metadata.get("channel") or metadata.get("uploader")
        description = metadata.get("description")
        uploader = metadata.get("uploader")
        track = metadata.get("track")

        lines = [
            "---",
            f"title: {_escape_yaml_string(str(title))}",
            "kind: image",
            f"image_count: {len(entries)}",
        ]
        if channel:
            lines.append(f"channel: {_escape_yaml_string(str(channel))}")
        if uploader:
            lines.append(f"uploader: {_escape_yaml_string(str(uploader))}")
        if track:
            lines.append(f"track: {_escape_yaml_string(str(track))}")
        if description:
            lines.append(f"description: {_escape_yaml_string(str(description))}")
        lines.extend(["---", ""])
        sound_lines = _format_tiktok_sound_section(metadata, tiktok_item)
        if sound_lines:
            lines.extend(sound_lines)
            lines.append("")
        lines.append("## Images")

        total_images = len(entries)
        for image in entries:
            attrs = [f'index="{image["index"]}"']
            if image.get("width") and image.get("height"):
                attrs.append(f'width="{image["width"]}"')
                attrs.append(f'height="{image["height"]}"')
            lines.append("")
            lines.append(f"<image {' '.join(attrs)}>")
            alttext = self._describe_tiktok_image(
                image,
                metadata=metadata,
                total_images=total_images,
            )
            lines.extend((alttext or "Alt text unavailable.").splitlines())
            lines.append("</image>")

        return "\n".join(lines)

    def _render_tiktok_image_post(
        self,
        metadata: dict[str, Any],
        image_post: dict[str, Any],
        tiktok_item: dict[str, Any] | None = None,
    ) -> str:
        identity = self._get_identity()
        cached = self._cached_tiktok_image_post_output(identity.cache_identity)
        if cached is not None:
            return cached

        text = self._format_tiktok_image_post_output(
            metadata,
            image_post,
            tiktok_item=tiktok_item,
        )
        self.original_file_content = text
        self.file_content = text
        self._store_tiktok_image_post_output(identity.cache_identity, text)
        return self._render_output_text(text)

    def _format_instagram_image_post_output(
        self,
        metadata: dict[str, Any],
        media: dict[str, Any],
    ) -> str:
        entries = _instagram_image_entries(media)
        title = metadata.get("title") or "Instagram image post"
        channel = metadata.get("channel") or metadata.get("uploader")
        description = metadata.get("description")
        uploader = metadata.get("uploader")

        lines = [
            "---",
            f"title: {_escape_yaml_string(str(title))}",
            "kind: image",
            f"image_count: {len(entries)}",
        ]
        if channel:
            lines.append(f"channel: {_escape_yaml_string(str(channel))}")
        if uploader:
            lines.append(f"uploader: {_escape_yaml_string(str(uploader))}")
        if description:
            lines.append(f"description: {_escape_yaml_string(str(description))}")
        lines.extend(["---", ""])
        sound_lines = _format_instagram_sound_section(metadata, media)
        if sound_lines:
            lines.extend(sound_lines)
            lines.append("")
        lines.append("## Images")

        total_images = len(entries)
        for image in entries:
            attrs = [f'index="{image["index"]}"']
            if image.get("width") and image.get("height"):
                attrs.append(f'width="{image["width"]}"')
                attrs.append(f'height="{image["height"]}"')
            lines.append("")
            lines.append(f"<image {' '.join(attrs)}>")
            alttext = self._describe_instagram_image(
                image,
                metadata=metadata,
                total_images=total_images,
            )
            lines.extend((alttext or "Alt text unavailable.").splitlines())
            lines.append("</image>")

        return "\n".join(lines)

    def _render_instagram_image_post(
        self,
        metadata: dict[str, Any],
        media: dict[str, Any],
    ) -> str:
        identity = self._get_identity()
        cached = self._cached_instagram_image_post_output(identity.cache_identity)
        if cached is not None:
            return cached

        text = self._format_instagram_image_post_output(metadata, media)
        self.original_file_content = text
        self.file_content = text
        self._store_instagram_image_post_output(identity.cache_identity, text)
        return self._render_output_text(text)

    def _format_output(
        self,
        metadata: dict,
        transcript: str,
        source: str,
        *,
        frames: str = "",
        tiktok_item: dict[str, Any] | None = None,
        instagram_media: dict[str, Any] | None = None,
    ) -> str:
        title = metadata.get("title", "Untitled")
        channel = metadata.get("channel") or metadata.get("uploader")
        description = metadata.get("description")
        duration = metadata.get("duration", 0)

        has_rich_metadata = channel or description

        lines = []

        if has_rich_metadata:
            lines.append("---")
            lines.append(f"title: {_escape_yaml_string(title)}")
            if channel:
                lines.append(f"channel: {_escape_yaml_string(channel)}")
            if description:
                lines.append(f"description: {_escape_yaml_string(description)}")
            lines.append("---")
            lines.append("")
        else:
            lines.append(f"# {title}")
            lines.append("")

        sound_lines = _format_tiktok_sound_section(metadata, tiktok_item)
        if not sound_lines:
            sound_lines = _format_instagram_sound_section(
                metadata,
                instagram_media,
            )
        if sound_lines:
            lines.extend(sound_lines)
            lines.append("")

        if frames:
            lines.append(frames)
            lines.append("")

        if transcript:
            formatted_transcript = _insert_timestamps(transcript, duration)
            lines.append(formatted_transcript)
        elif source == "none":
            lines.append("*No transcript available.*")

        return "\n".join(lines)

    def _get_contents(self) -> str:
        whisper_available = True

        if _is_tiktok_host(self._metadata_url()):
            metadata = self._fetch_metadata()
            identity = self._get_identity()
            if is_tiktok_photo_url(self.url) or _metadata_has_tiktok_image_post(
                metadata
            ):
                cached_image = self._cached_tiktok_image_post_output(
                    identity.cache_identity
                )
                if cached_image is not None:
                    return cached_image
                tiktok_item = _fetch_tiktok_item(metadata)
                image_post = _tiktok_image_post_from_item(metadata, tiktok_item)
                if image_post is not None:
                    return self._render_tiktok_image_post(
                        metadata,
                        image_post,
                        tiktok_item=tiktok_item,
                    )

            cached_tiktok = self._cached_tiktok_transcript_output(
                identity,
                whisper_available,
            )
            if cached_tiktok is not None:
                return cached_tiktok

            tiktok_item = _fetch_tiktok_item(metadata)
            duration = int(metadata.get("duration", 0) or 0)
            _log(
                f"building transcript for {identity.display_name} "
                f"(duration={duration}s, refresh_cache={self.refresh_cache})"
            )
            transcript, source, frames = self._get_transcript_and_frames(duration)
            text = self._format_output(
                metadata,
                transcript,
                source,
                frames=frames,
                tiktok_item=tiktok_item,
            )

            self.original_file_content = text
            self.file_content = text
            self._store_tiktok_transcript_output(identity, text=text, source=source)
            return self._render_output_text(text)

        if _is_instagram_host(self._metadata_url()):
            metadata = self._fetch_metadata()
            identity = self._get_identity()
            instagram_media = _instagram_media_from_metadata(metadata)
            if instagram_media is not None and _instagram_image_entries(
                instagram_media
            ):
                cached_image = self._cached_instagram_image_post_output(
                    identity.cache_identity
                )
                if cached_image is not None:
                    return cached_image
                return self._render_instagram_image_post(metadata, instagram_media)

            cached_instagram = self._cached_instagram_transcript_output(
                identity,
                whisper_available,
            )
            if cached_instagram is not None:
                return cached_instagram

            duration = int(metadata.get("duration", 0) or 0)
            _log(
                f"building transcript for {identity.display_name} "
                f"(duration={duration}s, refresh_cache={self.refresh_cache})"
            )
            transcript, source, frames = self._get_transcript_and_frames(duration)
            text = self._format_output(
                metadata,
                transcript,
                source,
                frames=frames,
                instagram_media=instagram_media,
            )

            self.original_file_content = text
            self.file_content = text
            self._store_instagram_transcript_output(
                identity,
                text=text,
                source=source,
            )
            return self._render_output_text(text)

        for base_identity in _candidate_render_base_identities(self.url):
            cached_output = self._cached_render_output(
                base_identity,
                whisper_available,
                fast=True,
            )
            if cached_output is not None:
                return cached_output

        for base_identity in _candidate_render_base_identities(self.url):
            cached_output = self._cached_render_output(
                base_identity,
                whisper_available,
            )
            if cached_output is not None:
                return cached_output

        identity = self._get_identity()
        render_cache_identity = _render_cache_identity(
            identity.cache_identity,
            self.plugin_overrides,
        )

        if self.use_cache and not self.refresh_cache:
            from .cache import get_cached_transcript

            cached = get_cached_transcript(
                render_cache_identity,
                self.cache_ttl,
                whisper_available=whisper_available,
            )
            if cached is not None:
                self._store_render_cache_aliases(
                    primary_base_identity=identity.cache_identity,
                    text=cached,
                    source="render-cache",
                )
                return self._record_render_cache_hit(cached, identity)

        metadata = self._fetch_metadata()
        duration = int(metadata.get("duration", 0) or 0)
        _log(
            f"building transcript for {identity.display_name} "
            f"(duration={duration}s, refresh_cache={self.refresh_cache})"
        )
        transcript, source, frames = self._get_transcript_and_frames(duration)
        text = self._format_output(
            metadata,
            transcript,
            source,
            frames=frames,
        )

        self.original_file_content = text
        self.file_content = text

        if self.use_cache:
            self._store_render_cache_aliases(
                primary_base_identity=identity.cache_identity,
                text=text,
                source=source,
            )

        return self._render_output_text(text)

    def get_contents(self) -> str:
        return self.output
