from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

_SNIPD_HOSTS = frozenset({"share.snipd.com"})
_CLIP_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    flags=re.IGNORECASE,
)
_STATE_RE = re.compile(
    r'<script\s+id=["\']serverApp-state["\']\s+type=["\']application/json["\']\s*>(.*?)</script>',
    re.DOTALL | re.IGNORECASE,
)
_PAGE_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL | re.IGNORECASE)
_NOTE_HEADING_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)
_USER_AGENT = "contextualize/snipd"
_RENDER_CACHE_SCHEMA = 2


@dataclass(frozen=True)
class SnipdTarget:
    original: str
    clip_id: str


@dataclass(frozen=True)
class SnipdClip:
    clip_id: str
    public_clip_id: str | None
    episode_id: str | None
    title: str
    episode_title: str | None
    show_title: str | None
    audio_url: str
    start_seconds: float
    end_seconds: float
    note_md: str | None = None

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


def _log(message: str) -> None:
    try:
        from contextualize.runtime import get_verbose_logging

        if get_verbose_logging():
            print(f"[snipd] {message}", file=sys.stderr, flush=True)
    except Exception:
        return


def parse_snipd_target(target: str) -> SnipdTarget | None:
    parsed = urlparse(target.strip())
    if parsed.scheme not in {"http", "https"}:
        return None
    if parsed.netloc.lower() not in _SNIPD_HOSTS:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2 or parts[0].lower() != "snip":
        return None
    clip_id = unquote(parts[1]).lower()
    if _CLIP_ID_RE.fullmatch(clip_id) is None:
        return None
    return SnipdTarget(original=target, clip_id=clip_id)


def _fetch_html(url: str, *, timeout_seconds: int = 30) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to fetch Snipd page: {exc}") from exc


def _extract_state(page_html: str) -> dict[str, Any]:
    match = _STATE_RE.search(page_html)
    if match is None:
        raise ValueError("could not find Snipd serverApp-state JSON")
    try:
        state = json.loads(html.unescape(match.group(1)))
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse Snipd serverApp-state JSON: {exc}") from exc
    if not isinstance(state, dict):
        raise ValueError("Snipd serverApp-state JSON must be a mapping")
    return state


def _extract_page_title(page_html: str) -> str | None:
    match = _PAGE_TITLE_RE.search(page_html)
    if match is None:
        return None
    title = re.sub(r"\s+", " ", html.unescape(match.group(1))).strip()
    return title or None


def _apollo_state(state: dict[str, Any]) -> dict[str, Any]:
    apollo = state.get("apollo.state")
    if not isinstance(apollo, dict):
        raise ValueError("Snipd state does not contain apollo.state")
    return apollo


def _find_public_clip(apollo: dict[str, Any], clip_id: str) -> dict[str, Any]:
    root = apollo.get("ROOT_QUERY")
    if not isinstance(root, dict):
        raise ValueError("Snipd Apollo state does not contain ROOT_QUERY")
    for key, value in root.items():
        if not isinstance(key, str) or not key.startswith("public_clips("):
            continue
        if not isinstance(value, list):
            continue
        for candidate in value:
            if isinstance(candidate, dict) and candidate.get("clip_id") == clip_id:
                return candidate
    raise ValueError(f"Snipd Apollo state does not contain clip {clip_id}")


def _resolve_ref(apollo: dict[str, Any], value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    ref = value.get("__ref")
    if not isinstance(ref, str):
        return None
    resolved = apollo.get(ref)
    return resolved if isinstance(resolved, dict) else None


def _first_note_heading(note_md: str | None) -> str | None:
    if not note_md:
        return None
    match = _NOTE_HEADING_RE.search(note_md)
    if match is None:
        return None
    heading = match.group(1).strip().strip("*").strip()
    return heading or None


def _clean_title(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = re.sub(r"\s+", " ", value).strip()
    return text or None


def _float_field(mapping: dict[str, Any], field: str) -> float:
    try:
        return float(mapping[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Snipd clip is missing numeric {field}") from exc


def _build_clip(
    apollo: dict[str, Any],
    clip: dict[str, Any],
    *,
    clip_id: str,
    page_title: str | None,
) -> SnipdClip:
    episode = _resolve_ref(apollo, clip.get("episode"))
    if episode is None:
        raise ValueError("Snipd clip does not resolve to an episode")

    audio_url = _clean_title(episode.get("audio_url"))
    if audio_url is None:
        raise ValueError("Snipd episode is missing audio_url")

    start_seconds = _float_field(clip, "user_start_time_seconds")
    end_seconds = _float_field(clip, "user_end_time_seconds")
    if end_seconds <= start_seconds:
        raise ValueError("Snipd clip end must be greater than start")

    note_md = clip.get("note_md") if isinstance(clip.get("note_md"), str) else None
    title = (
        _clean_title(clip.get("generated_title"))
        or _clean_title(clip.get("user_title"))
        or _first_note_heading(note_md)
        or page_title
        or _clean_title(episode.get("title"))
        or "Snipd clip"
    )
    show = _resolve_ref(apollo, episode.get("show"))

    return SnipdClip(
        clip_id=clip_id,
        public_clip_id=_clean_title(clip.get("public_clip_id")),
        episode_id=_clean_title(episode.get("id")),
        title=title,
        episode_title=_clean_title(episode.get("title")),
        show_title=_clean_title(show.get("title")) if show else None,
        audio_url=audio_url,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        note_md=note_md,
    )


def resolve_snipd_clip(url: str) -> SnipdClip:
    target = parse_snipd_target(url)
    if target is None:
        raise ValueError(f"Unsupported Snipd target: {url}")
    _log(f"fetching public snip metadata for {target.clip_id}")
    page_html = _fetch_html(url)
    state = _extract_state(page_html)
    apollo = _apollo_state(state)
    clip = _find_public_clip(apollo, target.clip_id)
    return _build_clip(
        apollo,
        clip,
        clip_id=target.clip_id,
        page_title=_extract_page_title(page_html),
    )


def _escape_yaml_string(value: str) -> str:
    if not value:
        return '""'
    if "\n" in value:
        indented = "\n".join("  " + line for line in value.split("\n"))
        return f"|\n{indented}"
    needs_quotes = any(c in value for c in ":{}[],\"'|>&*!?#%@`") or value.startswith(
        " "
    )
    if needs_quotes:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


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
        "output_schema": _RENDER_CACHE_SCHEMA,
        "overrides": transcribe_overrides or {},
        "routing": _transcription_routing_identity(plugin_overrides),
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:16]
    return f"{base_identity}:transcribe:{digest}"


def _format_output(clip: SnipdClip, transcript: str) -> str:
    lines = [
        "---",
        f"title: {_escape_yaml_string(clip.title)}",
    ]
    if clip.episode_title:
        lines.append(f"episode: {_escape_yaml_string(clip.episode_title)}")
    if clip.show_title:
        lines.append(f"show: {_escape_yaml_string(clip.show_title)}")
    lines.extend(
        [
            f"start_seconds: {clip.start_seconds:.3f}",
            f"end_seconds: {clip.end_seconds:.3f}",
            f"duration_seconds: {clip.duration_seconds:.3f}",
            "---",
            "",
        ]
    )
    if transcript.strip():
        lines.append(transcript.strip())
    else:
        lines.append("*No transcript available.*")
    return "\n".join(lines)


@dataclass
class SnipdReference:
    url: str
    use_cache: bool = True
    cache_ttl: timedelta | None = None
    refresh_cache: bool = False
    plugin_overrides: dict[str, Any] | None = None
    _clip: SnipdClip | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if parse_snipd_target(self.url) is None:
            raise ValueError(f"Unsupported Snipd target: {self.url}")
        self.file_content = ""
        self.original_file_content = ""
        self.output = self._get_contents()

    @property
    def clip(self) -> SnipdClip:
        if self._clip is None:
            self._clip = resolve_snipd_clip(self.url)
        return self._clip

    def loaded_clip(self) -> SnipdClip | None:
        return self._clip

    def clip_id(self) -> str:
        target = parse_snipd_target(self.url)
        if target is None:
            raise ValueError(f"Unsupported Snipd target: {self.url}")
        return target.clip_id

    @property
    def path(self) -> str:
        return self.url

    def read(self) -> str:
        return self.original_file_content

    def exists(self) -> bool:
        try:
            self.clip
            return True
        except Exception:
            return False

    def get_label(self) -> str:
        return self.url

    def source_ref(self) -> str:
        return "share.snipd.com"

    def source_path(self) -> str:
        return f"snipd:{self.clip_id()}"

    def context_subpath(self) -> str:
        return f"snipd-{self.clip_id()}.md"

    def get_kind(self) -> str:
        return "audio"

    def _cached_render_output(self) -> str | None:
        if not self.use_cache or self.refresh_cache:
            return None
        from .cache import get_cached_transcript

        target = parse_snipd_target(self.url)
        if target is None:
            return None
        cached = get_cached_transcript(
            _render_cache_identity(f"snipd:{target.clip_id}", self.plugin_overrides),
            self.cache_ttl,
        )
        if cached is None:
            return None
        from contextualize.transcription import record_transcription_routing_summary

        record_transcription_routing_summary(
            filename="media.mp3",
            content_type="audio/mpeg",
            plugin_overrides=self.plugin_overrides,
            source="render-cache",
        )
        self.original_file_content = cached
        self.file_content = cached
        return cached

    def _extract_audio(self) -> Path:
        from .cache import get_cached_media_bytes, store_media_bytes
        from contextualize.runtime import get_refresh_audio

        clip = self.clip
        media_identity = f"audio:snipd:{clip.clip_id}"
        if self.use_cache and not get_refresh_audio():
            cached = get_cached_media_bytes(media_identity)
            if cached:
                tmpdir = tempfile.mkdtemp(prefix="snipd-audio-")
                path = Path(tmpdir) / f"{clip.clip_id}.mp3"
                path.write_bytes(cached)
                return Path(path)

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("Snipd clip extraction requires ffmpeg")

        tmpdir = tempfile.mkdtemp(prefix="snipd-")
        output_path = Path(tmpdir) / f"{clip.clip_id}.mp3"
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-ss",
            f"{clip.start_seconds:.3f}",
            "-i",
            clip.audio_url,
            "-t",
            f"{clip.duration_seconds:.3f}",
            "-vn",
            "-acodec",
            "copy",
            str(output_path),
        ]
        _log(f"cutting public snip audio for {clip.clip_id}")
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=None,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg Snipd clip extraction failed: {result.stderr}"
                )
            if not output_path.exists():
                raise RuntimeError("ffmpeg Snipd clip extraction produced no audio file")
            if self.use_cache:
                try:
                    store_media_bytes(media_identity, output_path.read_bytes())
                except OSError:
                    pass
            return output_path
        except Exception:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise

    def _get_transcript(self) -> tuple[str, str]:
        audio_path = None
        try:
            audio_path = self._extract_audio()
            from contextualize.transcription import transcribe_media_file

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

    def _get_contents(self) -> str:
        cached = self._cached_render_output()
        if cached is not None:
            return cached

        clip = self.clip
        transcript, source = self._get_transcript()
        text = _format_output(clip, transcript)
        self.original_file_content = text
        self.file_content = text

        if self.use_cache:
            from .cache import store_transcript

            store_transcript(
                _render_cache_identity(f"snipd:{clip.clip_id}", self.plugin_overrides),
                text,
                source=source,
            )

        return text

    def get_contents(self) -> str:
        return self.output
