from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse, urlunparse


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
                url,
            ],
            timeout_seconds=timeout_seconds,
        )
    except (RuntimeError, OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None

    payload = result.stdout.strip()
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
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
    if isinstance(plugin_overrides, dict):
        value = plugin_overrides.get("transcribe")
        if isinstance(value, dict):
            transcribe_overrides = dict(value)
    payload = {
        "overrides": transcribe_overrides or {},
        "routing": _transcription_routing_identity(plugin_overrides),
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:16]
    return f"{base_identity}:transcribe:{digest}"


def _fast_render_cache_identity(
    base_identity: str, plugin_overrides: dict[str, Any] | None
) -> str:
    transcribe_overrides = None
    if isinstance(plugin_overrides, dict):
        value = plugin_overrides.get("transcribe")
        if isinstance(value, dict):
            transcribe_overrides = dict(value)
    payload = {
        "overrides": transcribe_overrides or {},
        "routing": "fast-v1",
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:16]
    return f"{base_identity}:transcribe-fast:{digest}"


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
        duration = metadata.get("duration")
        if isinstance(duration, (int, float)) and duration > 0:
            return "video"
        return "resource"

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
                self.url,
            ],
            timeout_seconds=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp metadata failed: {result.stderr}")

        try:
            metadata = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"yt-dlp metadata parse failed: {exc}") from exc
        if not isinstance(metadata, dict):
            raise RuntimeError("yt-dlp metadata returned unexpected payload")
        self._metadata = metadata
        title = metadata.get("title") if isinstance(metadata, dict) else None
        _log(f"metadata fetched for {title or self.url}")
        return self._metadata

    def _get_identity(self) -> _YtDlpIdentity:
        if self._identity is None:
            self._identity = _build_identity(self.url, self._fetch_metadata())
        return self._identity

    def _extract_audio(self) -> Path:
        from contextualize.cache.youtube import (
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
                fd, path = tempfile.mkstemp(prefix="ytdlp-audio-", suffix=".mp3")
                try:
                    os.write(fd, cached)
                finally:
                    os.close(fd)
                return Path(path)
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
                self.url,
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

    def _get_transcript(self, _duration: int) -> tuple[str, str]:
        audio_path = None
        try:
            audio_path = self._extract_audio()
            from contextualize.transcription import (
                transcribe_media_file,
            )

            identity = getattr(self, "_identity", None)
            display_name = getattr(identity, "display_name", audio_path.name)
            _log(f"transcribing extracted audio for {display_name}")
            transcript = transcribe_media_file(
                audio_path,
                use_cache=self.use_cache,
                refresh_cache=self.refresh_cache,
                plugin_overrides=self.plugin_overrides,
            )
            _log(f"transcription finished for {display_name}")
            return transcript, "transcription"
        finally:
            if audio_path and audio_path.exists():
                shutil.rmtree(audio_path.parent, ignore_errors=True)

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

        from contextualize.cache.youtube import get_cached_transcript

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
            from contextualize.cache.youtube import store_transcript

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
        if not self.use_cache:
            return

        from contextualize.cache.youtube import store_transcript

        identities = (primary_base_identity, *_candidate_render_base_identities(self.url))
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

    def _format_output(self, metadata: dict, transcript: str, source: str) -> str:
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

        if transcript:
            formatted_transcript = _insert_timestamps(transcript, duration)
            lines.append(formatted_transcript)
        elif source == "none":
            lines.append("*No transcript available.*")

        return "\n".join(lines)

    def _get_contents(self) -> str:
        whisper_available = True

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
            from contextualize.cache.youtube import get_cached_transcript

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
        transcript, source = self._get_transcript(duration)
        text = self._format_output(metadata, transcript, source)

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
