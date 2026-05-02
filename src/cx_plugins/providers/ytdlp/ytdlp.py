from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse


def is_url_target(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


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


def _slugify(value: str, *, default: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_.").lower()
    return slug or default


def _check_ytdlp() -> None:
    if not shutil.which("yt-dlp"):
        raise RuntimeError(
            "yt-dlp processing requires yt-dlp: https://github.com/yt-dlp/yt-dlp"
        )


def _run_ytdlp(
    command: list[str], *, timeout_seconds: int
) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def probe_ytdlp_metadata(
    url: str, *, timeout_seconds: int = 10
) -> dict[str, Any] | None:
    if not is_url_target(url):
        return None
    try:
        _check_ytdlp()
        result = _run_ytdlp(
            [
                "yt-dlp",
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

    normalized = _normalize_url_for_key(url)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    short_digest = digest[:12]
    return _YtDlpIdentity(
        extractor=extractor,
        media_id=media_id,
        cache_identity=f"url:{digest}",
        display_name=f"url:{short_digest}",
        slug=f"url-{short_digest}",
    )


def _render_cache_identity(
    base_identity: str, plugin_overrides: dict[str, Any] | None
) -> str:
    transcribe_overrides = None
    if isinstance(plugin_overrides, dict):
        value = plugin_overrides.get("transcribe")
        if isinstance(value, dict):
            transcribe_overrides = dict(value)
    if not transcribe_overrides:
        return base_identity
    payload = json.dumps(transcribe_overrides, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{base_identity}:transcribe:{digest}"


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
        metadata = self._fetch_metadata()
        return _source_ref(self.url, metadata, self._get_identity().extractor)

    def source_path(self) -> str:
        return self._get_identity().cache_identity

    def context_subpath(self) -> str:
        return f"ytdlp-{self._get_identity().slug}.md"

    def get_kind(self) -> str:
        metadata = self._fetch_metadata()
        duration = metadata.get("duration")
        if isinstance(duration, (int, float)) and duration > 0:
            return "video"
        return "resource"

    def _fetch_metadata(self) -> dict[str, Any]:
        if self._metadata is not None:
            return self._metadata

        result = _run_ytdlp(
            [
                "yt-dlp",
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
                "yt-dlp",
                "-x",
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
            timeout_seconds=600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp audio extraction failed: {result.stderr}")

        audio_dir = Path(tmpdir)
        audio_files = sorted(audio_dir.glob("*.mp3"))
        if not audio_files:
            audio_files = sorted(path for path in audio_dir.iterdir() if path.is_file())
        if not audio_files:
            raise RuntimeError("yt-dlp audio extraction produced no audio file")
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
            from contextualize.references.audio_transcription import (
                transcribe_media_file,
            )

            transcript = transcribe_media_file(
                audio_path,
                use_cache=self.use_cache,
                refresh_cache=self.refresh_cache,
                plugin_overrides=self.plugin_overrides,
            )
            return transcript, "transcription"
        finally:
            if audio_path and audio_path.exists():
                shutil.rmtree(audio_path.parent, ignore_errors=True)

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
        from contextualize.render.text import process_text

        whisper_available = True
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
                _log(f"render cache hit for {identity.display_name}")
                self.original_file_content = cached
                self.file_content = cached
                text = cached
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
            from contextualize.cache.youtube import store_transcript

            store_transcript(
                render_cache_identity,
                text,
                source=source,
            )

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

    def get_contents(self) -> str:
        return self.output
