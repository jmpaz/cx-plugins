from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import sys
import tempfile
import unicodedata
import zipfile
from dataclasses import asdict, dataclass, replace as dataclass_replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote

from contextualize.references.helpers import (
    parse_compound_duration,
    parse_timestamp_or_duration,
)
from contextualize.utils import count_tokens

_CHAT_TEXT_NAME = "_chat.txt"
_WHATSAPP_ZIP_PREFIX = "whatsapp:zip:"
_WHATSAPP_ARCHIVE_QUERY_PREFIX = "whatsapp:archive?"
_VALID_FORMATS = frozenset({"transcript", "yaml"})
_VALID_MEDIA_MODES = frozenset({"describe", "transcribe"})
_VALID_AUTHOR_NAME_MODES = frozenset({"full", "first"})
_MEDIA_IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif", ".heic", ".heif"}
)
_MEDIA_VIDEO_SUFFIXES = frozenset(
    {".mp4", ".mov", ".webm", ".mkv", ".avi", ".mpeg", ".mpg", ".m4v"}
)
_MEDIA_AUDIO_SUFFIXES = frozenset(
    {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".aiff"}
)
_FILE_SIZE_LIMIT_RE = re.compile(r"^(?P<value>\d+)\s*(?P<unit>[kmg]?b?)?$", re.I)
_MESSAGE_START_RE = re.compile(
    r"^[\u200e\u200f\ufeff]*(?:\u202a|\u202c)?"
    r"\[(?P<stamp>[^\]]+)\]\s(?P<body>.*)$"
)
_ATTACHMENT_RE = re.compile(
    r"[\u200e\u200f\s]*<attached:\s*(?P<name>[^>]+)>", flags=re.I
)
_PIN_EVENT_RE = re.compile(r"^(?:(?P<actor>.+?)\s+)?pinned a message$", re.I)
_PNG_CHUNK_MARKER_RE = re.compile(r"\bIHDR\b.*\bIDAT\b.*\bIEND\b", flags=re.S)
_REPLY_QUOTE_MAX_CHARS = 600
_TIMESTAMP_FORMATS = (
    "%m/%d/%y, %I:%M:%S %p",
    "%m/%d/%Y, %I:%M:%S %p",
    "%d/%m/%y, %H:%M:%S",
    "%d/%m/%Y, %H:%M:%S",
    "%Y-%m-%d, %H:%M:%S",
)

WHATSAPP_CACHE_ROOT = Path(
    os.environ.get(
        "CONTEXTUALIZE_WHATSAPP_CACHE",
        os.path.expanduser("~/.local/share/contextualize/cache/whatsapp/v1"),
    )
)
RENDER_CACHE_ROOT = WHATSAPP_CACHE_ROOT / "render"


def _log(message: str) -> None:
    from contextualize.runtime import get_verbose_logging

    if get_verbose_logging():
        print(message, file=sys.stderr, flush=True)


def _cache_key(identity: str) -> str:
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _get_cached_rendered(identity: str) -> str | None:
    path = RENDER_CACHE_ROOT / f"{_cache_key(identity)}.txt"
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _store_rendered(identity: str, content: str) -> None:
    if not content:
        return
    try:
        RENDER_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        path = RENDER_CACHE_ROOT / f"{_cache_key(identity)}.txt"
        path.write_text(content, encoding="utf-8")
    except OSError:
        return


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:60] or "chat"


def _clean_timestamp_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", normalized).strip()


def _parse_archive_timestamp(value: str) -> datetime | None:
    cleaned = _clean_timestamp_text(value)
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            pass
    return None


def _parse_datetime(value: str) -> datetime | None:
    parsed = parse_timestamp_or_duration(value)
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        return parsed
    return parsed.replace(tzinfo=None)


def _format_local_iso(value: datetime) -> str:
    return value.replace(microsecond=0).isoformat()


def _parse_bool(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_optional_int(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _parse_duration_compound(raw: Any) -> timedelta | None:
    if raw is None:
        return None
    return parse_compound_duration(
        str(raw).strip(),
        unit_seconds={
            "y": 365 * 24 * 60 * 60,
            "mo": 30 * 24 * 60 * 60,
            "w": 7 * 24 * 60 * 60,
            "d": 24 * 60 * 60,
            "h": 60 * 60,
            "m": 60,
            "s": 1,
            "i": 60,
        },
    )


def _parse_gap_threshold(raw: Any, *, default: timedelta) -> timedelta:
    parsed = _parse_duration_compound(raw)
    if parsed is None or parsed.total_seconds() <= 0:
        return default
    return parsed


def _parse_media_mode(raw: str, *, default: str) -> str:
    mode = str(raw or "").strip().lower()
    return mode if mode in _VALID_MEDIA_MODES else default


def _parse_author_name_mode(raw: Any, *, default: str) -> str:
    mode = str(raw or "").strip().lower()
    return mode if mode in _VALID_AUTHOR_NAME_MODES else default


def _parse_file_size_limit(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw if raw > 0 else None
    text = str(raw).strip()
    if not text:
        return None
    match = _FILE_SIZE_LIMIT_RE.match(text)
    if not match:
        return None
    value = int(match.group("value"))
    unit = (match.group("unit") or "b").lower()
    if unit in {"k", "kb"}:
        value *= 1024
    elif unit in {"m", "mb"}:
        value *= 1024 * 1024
    elif unit in {"g", "gb"}:
        value *= 1024 * 1024 * 1024
    return value if value > 0 else None


def _format_file_size_human(value: int | None) -> str | None:
    if value is None or value < 0:
        return None
    if value < 1024:
        return f"{value}B"
    if value < 1024 * 1024:
        return f"{round(value / 1024)}K"
    return f"{round(value / (1024 * 1024))}M"


def _normalize_message_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parsed = parse_whatsapp_target(text)
    if parsed and parsed.get("message_id"):
        return str(parsed["message_id"])
    if re.fullmatch(r"m\d{6}", text):
        return text
    if text.isdigit():
        return f"m{int(text):06d}"
    return text


def _normalize_message_id_set(value: Any) -> frozenset[str] | None:
    if value is None:
        return None
    items: list[Any]
    if isinstance(value, str):
        items = [part for part in re.split(r"[,\s]+", value) if part]
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
    else:
        return None
    normalized = {
        parsed for item in items if (parsed := _normalize_message_id(item)) is not None
    }
    return frozenset(normalized) if normalized else frozenset()


def _distribute_message_context(
    *,
    message_context: int | None,
    before_messages: int | None,
    after_messages: int | None,
) -> tuple[int | None, int | None]:
    if message_context is None:
        return before_messages, after_messages
    total = max(0, int(message_context))
    if before_messages is None and after_messages is None:
        return (total + 1) // 2, total // 2
    if before_messages is None:
        after = max(0, int(after_messages or 0))
        return max(0, total - after), after
    if after_messages is None:
        before = max(0, int(before_messages))
        return before, max(0, total - before)
    return before_messages, after_messages


@dataclass(frozen=True)
class WhatsAppSettings:
    format: str = "transcript"
    include_system: bool = True
    gap_threshold: timedelta = timedelta(hours=6)
    before_messages: int | None = 0
    after_messages: int | None = 0
    around_messages: int | None = None
    message_context: int | None = None
    channel_limit: int = 0
    start: datetime | None = None
    end: datetime | None = None
    start_message_id: str | None = None
    end_message_id: str | None = None
    before_duration: timedelta | None = None
    after_duration: timedelta | None = None
    around_duration: timedelta | None = None
    include_media_descriptions: bool = True
    include_file_content: bool = True
    media_mode: str = "describe"
    author_name_mode: str = "full"
    attachment_max_tokens: int | None = None
    skip_file_content_messages: frozenset[str] | None = None
    max_file_content_size_bytes: int | None = None
    hard_min_timestamp: datetime | None = None
    hard_max_timestamp: datetime | None = None


@dataclass(frozen=True)
class WhatsAppAttachment:
    filename: str
    member_name: str
    size: int | None
    content_type: str | None


@dataclass(frozen=True)
class WhatsAppMessage:
    id: str
    ordinal: int
    timestamp: datetime
    sender: str | None
    content: str
    attachments: list[WhatsAppAttachment]
    is_system: bool = False
    reply_to_id: str | None = None
    reply_to_content: str | None = None
    forward: dict[str, Any] | None = None
    reactions: list[dict[str, Any]] | None = None
    pins: list[dict[str, Any]] | None = None
    edits: list[dict[str, Any]] | None = None
    deletes: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class WhatsAppChatInfo:
    id: str
    title: str
    source_path: str
    kind: str = "chat"


@dataclass(frozen=True)
class WhatsAppDocument:
    source_url: str
    label: str
    trace_path: str
    chat_id: str
    chat_name: str
    kind: str
    messages: list[dict[str, Any]]
    rendered: str
    source_path: str


class WhatsAppSource:
    capabilities: frozenset[str] = frozenset()

    def list_chats(self) -> list[WhatsAppChatInfo]:
        raise NotImplementedError

    def iter_messages(self, chat_id: str) -> list[WhatsAppMessage]:
        raise NotImplementedError

    def read_media(self, attachment: WhatsAppAttachment) -> bytes | None:
        raise NotImplementedError

    def media_identity(self, attachment: WhatsAppAttachment) -> str:
        raise NotImplementedError


class WhatsAppArchiveSource(WhatsAppSource):
    capabilities = frozenset({"archive", "media"})

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        if not self.path.exists():
            raise ValueError(f"WhatsApp archive not found: {self.path}")
        self._chat_text_name: str | None = None
        self._chat_info: WhatsAppChatInfo | None = None

    def list_chats(self) -> list[WhatsAppChatInfo]:
        return [self._load_chat_info()]

    def iter_messages(self, chat_id: str) -> list[WhatsAppMessage]:
        chat = self._load_chat_info()
        if chat_id != chat.id:
            raise ValueError(f"WhatsApp chat not found in archive: {chat_id}")
        return self._parse_messages(chat)

    def read_media(self, attachment: WhatsAppAttachment) -> bytes | None:
        try:
            with zipfile.ZipFile(self.path) as zf:
                return zf.read(attachment.member_name)
        except (OSError, KeyError, zipfile.BadZipFile):
            return None

    def media_identity(self, attachment: WhatsAppAttachment) -> str:
        data = self.read_media(attachment)
        if data is not None:
            media_key = hashlib.sha256(data).hexdigest()
            return f"whatsapp:media:sha256:{media_key}"
        return f"whatsapp:archive:{self.path}:media:{attachment.member_name}"

    def _load_chat_info(self) -> WhatsAppChatInfo:
        if self._chat_info is not None:
            return self._chat_info
        chat_text_name = self._find_chat_text_name()
        self._chat_text_name = chat_text_name
        title = self._title_from_path()
        self._chat_info = WhatsAppChatInfo(
            id=_slugify(title),
            title=title,
            source_path=str(self.path),
        )
        return self._chat_info

    def _find_chat_text_name(self) -> str:
        try:
            with zipfile.ZipFile(self.path) as zf:
                names = [info.filename for info in zf.infolist() if not info.is_dir()]
        except (OSError, zipfile.BadZipFile) as exc:
            raise ValueError(f"Not a readable WhatsApp archive: {self.path}") from exc
        if _CHAT_TEXT_NAME in names:
            return _CHAT_TEXT_NAME
        text_names = [name for name in names if name.lower().endswith(".txt")]
        if len(text_names) == 1:
            return text_names[0]
        raise ValueError(f"WhatsApp archive has no unique chat text file: {self.path}")

    def _title_from_path(self) -> str:
        stem = self.path.stem.strip()
        prefix = "WhatsApp Chat - "
        if stem.startswith(prefix):
            return stem.removeprefix(prefix).strip() or stem
        return stem or "WhatsApp Chat"

    def _read_chat_text(self) -> str:
        name = self._chat_text_name or self._find_chat_text_name()
        try:
            with zipfile.ZipFile(self.path) as zf:
                return zf.read(name).decode("utf-8-sig", errors="replace")
        except (OSError, KeyError, zipfile.BadZipFile) as exc:
            raise ValueError(f"Unable to read WhatsApp chat text: {self.path}") from exc

    def _media_index(self) -> dict[str, zipfile.ZipInfo]:
        try:
            with zipfile.ZipFile(self.path) as zf:
                return {
                    info.filename: info
                    for info in zf.infolist()
                    if not info.is_dir() and info.filename != (self._chat_text_name or "")
                }
        except (OSError, zipfile.BadZipFile):
            return {}

    def _parse_messages(self, chat: WhatsAppChatInfo) -> list[WhatsAppMessage]:
        media_index = self._media_index()
        parsed: list[tuple[str, str | None, bool, str]] = []
        current_stamp: str | None = None
        current_sender: str | None = None
        current_system = False
        current_content = ""

        for raw_line in self._read_chat_text().splitlines():
            match = _MESSAGE_START_RE.match(raw_line)
            if match:
                if current_stamp is not None:
                    parsed.append(
                        (
                            current_stamp,
                            current_sender,
                            current_system,
                            current_content,
                        )
                    )
                current_stamp = match.group("stamp")
                current_sender, current_content, current_system = _split_sender(
                    match.group("body")
                )
                continue
            if current_stamp is None:
                continue
            current_content = f"{current_content}\n{raw_line}"

        if current_stamp is not None:
            parsed.append((current_stamp, current_sender, current_system, current_content))

        messages: list[WhatsAppMessage] = []
        for index, (stamp, sender, is_system, content) in enumerate(parsed, start=1):
            timestamp = _parse_archive_timestamp(stamp)
            if timestamp is None:
                continue
            attachments = _attachments_from_content(content, media_index)
            cleaned_content = _clean_message_content(content)
            pin_event = _pin_event_from_content(cleaned_content, sender=sender)
            pins = [pin_event] if pin_event is not None else None
            if pin_event is not None:
                sender = None
                is_system = True
            messages.append(
                WhatsAppMessage(
                    id=f"m{index:06d}",
                    ordinal=index,
                    timestamp=timestamp,
                    sender=sender,
                    content=cleaned_content,
                    attachments=attachments,
                    is_system=is_system,
                    pins=pins,
                )
            )
        return messages


def _split_sender(body: str) -> tuple[str | None, str, bool]:
    text = body.lstrip("\u200e\u200f")
    colon = text.find(":")
    if colon > 0:
        prefix = text[:colon].strip()
        after = text[colon + 1 :]
        if prefix and (after == "" or after.startswith(" ")):
            return prefix, after[1:] if after.startswith(" ") else "", False
    return None, text, True


def _attachments_from_content(
    content: str,
    media_index: dict[str, zipfile.ZipInfo],
) -> list[WhatsAppAttachment]:
    attachments: list[WhatsAppAttachment] = []
    for match in _ATTACHMENT_RE.finditer(content):
        filename = match.group("name").strip()
        if not filename:
            continue
        info = media_index.get(filename)
        attachments.append(
            WhatsAppAttachment(
                filename=Path(filename).name or filename,
                member_name=filename,
                size=info.file_size if info is not None else None,
                content_type=mimetypes.guess_type(filename)[0],
            )
        )
    return attachments


def _clean_message_content(content: str) -> str:
    text = _ATTACHMENT_RE.sub("", content)
    text = text.replace("\u200e", "").replace("\u200f", "")
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _pin_event_from_content(
    content: str, *, sender: str | None
) -> dict[str, Any] | None:
    text = " ".join(content.split())
    match = _PIN_EVENT_RE.fullmatch(text)
    if match is None:
        return None
    event: dict[str, Any] = {"type": "message_pin"}
    actor = (match.group("actor") or "").strip()
    if actor:
        event["actor"] = actor
    if sender:
        event["archive_sender"] = sender
    return event


def is_whatsapp_archive_path(target: str) -> bool:
    path = Path(os.path.expanduser(target))
    if not path.is_file() or path.suffix.lower() != ".zip":
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            names = [info.filename for info in zf.infolist() if not info.is_dir()]
    except (OSError, zipfile.BadZipFile):
        return False
    if _CHAT_TEXT_NAME in names:
        return True
    return sum(1 for name in names if name.lower().endswith(".txt")) == 1


def _target_from_archive_path(path: str, **query: str) -> str:
    encoded_path = quote(str(Path(path).expanduser()), safe="/:")
    query_items = [(key, value) for key, value in query.items() if value]
    if not query_items:
        return f"{_WHATSAPP_ZIP_PREFIX}{encoded_path}"
    suffix = "&".join(f"{key}={quote(value, safe='')}" for key, value in query_items)
    return f"{_WHATSAPP_ZIP_PREFIX}{encoded_path}?{suffix}"


def parse_whatsapp_target(target: str) -> dict[str, str] | None:
    if not isinstance(target, str) or not target.strip():
        return None
    text = target.strip()
    if text.startswith(_WHATSAPP_ZIP_PREFIX):
        rest = text.removeprefix(_WHATSAPP_ZIP_PREFIX)
        path_part, _, query_text = rest.partition("?")
        query = parse_qs(query_text)
        return {
            "kind": "archive",
            "path": unquote(path_part),
            **_first_query_values(query),
        }
    if text.startswith(_WHATSAPP_ARCHIVE_QUERY_PREFIX):
        query = parse_qs(text.removeprefix(_WHATSAPP_ARCHIVE_QUERY_PREFIX))
        path = next((value for value in query.get("path", ()) if value), None)
        if not path:
            return None
        return {
            "kind": "archive",
            "path": unquote(path),
            **_first_query_values(query),
        }
    if text.startswith("whatsapp:"):
        rest = text.removeprefix("whatsapp:")
        path = unquote(rest)
        if path and is_whatsapp_archive_path(path):
            return {"kind": "archive", "path": path}
        return None
    if is_whatsapp_archive_path(text):
        return {"kind": "archive", "path": text}
    return None


def _first_query_values(query: dict[str, list[str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for source_key, result_key in (
        ("chat", "chat_id"),
        ("message", "message_id"),
        ("attachment", "attachment"),
    ):
        value = next((item for item in query.get(source_key, ()) if item), None)
        if value:
            out[result_key] = unquote(value)
    return out


def is_whatsapp_target(target: str) -> bool:
    return parse_whatsapp_target(target) is not None


def list_whatsapp_targets(target: str) -> list[dict[str, Any]]:
    parsed = parse_whatsapp_target(target)
    if not parsed or parsed.get("kind") != "archive":
        return []
    source = WhatsAppArchiveSource(parsed["path"])
    return [
        {
            "target": _target_from_archive_path(chat.source_path, chat=chat.id),
            "label": chat.title,
            "kind": chat.kind,
            "metadata": {
                "provider": "whatsapp",
                "chatId": chat.id,
                "sourcePath": chat.source_path,
            },
        }
        for chat in source.list_chats()
    ]


def _whatsapp_settings_from_env() -> WhatsAppSettings:
    raw_format = (os.environ.get("WHATSAPP_FORMAT") or "transcript").strip().lower()
    out_format = raw_format if raw_format in _VALID_FORMATS else "transcript"
    media_desc = _parse_bool(os.environ.get("WHATSAPP_MEDIA_DESC"), default=True)
    include_file_content = _parse_bool(
        os.environ.get("WHATSAPP_FILE_CONTENT"), default=True
    )
    media_mode = _parse_media_mode(
        os.environ.get("WHATSAPP_MEDIA_MODE", ""), default="describe"
    )
    around_messages = _parse_optional_int(os.environ.get("WHATSAPP_AROUND_MESSAGES"))
    before_messages = _parse_optional_int(os.environ.get("WHATSAPP_BEFORE_MESSAGES"))
    after_messages = _parse_optional_int(os.environ.get("WHATSAPP_AFTER_MESSAGES"))
    message_context = _parse_optional_int(os.environ.get("WHATSAPP_MESSAGE_CONTEXT"))
    if around_messages is not None:
        before_messages = around_messages
        after_messages = around_messages
    else:
        before_messages, after_messages = _distribute_message_context(
            message_context=message_context,
            before_messages=before_messages,
            after_messages=after_messages,
        )
    channel_limit = _parse_optional_int(os.environ.get("WHATSAPP_CHANNEL_LIMIT")) or 0
    return _clamp_settings(
        WhatsAppSettings(
            format=out_format,
            include_system=_parse_bool(
                os.environ.get("WHATSAPP_INCLUDE_SYSTEM"), default=True
            ),
            gap_threshold=_parse_gap_threshold(
                os.environ.get("WHATSAPP_GAP_THRESHOLD", "6h"),
                default=timedelta(hours=6),
            ),
            before_messages=before_messages if before_messages is not None else 0,
            after_messages=after_messages if after_messages is not None else 0,
            around_messages=around_messages,
            message_context=message_context,
            channel_limit=channel_limit,
            start=_parse_datetime(os.environ.get("WHATSAPP_START_TIMESTAMP", "")),
            end=_parse_datetime(os.environ.get("WHATSAPP_END_TIMESTAMP", "")),
            start_message_id=_normalize_message_id(
                os.environ.get("WHATSAPP_START_MESSAGE", "")
            ),
            end_message_id=_normalize_message_id(
                os.environ.get("WHATSAPP_END_MESSAGE", "")
            ),
            before_duration=_parse_duration_compound(
                os.environ.get("WHATSAPP_BEFORE_DURATION")
            ),
            after_duration=_parse_duration_compound(
                os.environ.get("WHATSAPP_AFTER_DURATION")
            ),
            around_duration=_parse_duration_compound(
                os.environ.get("WHATSAPP_AROUND_DURATION")
            ),
            include_media_descriptions=media_desc,
            include_file_content=include_file_content,
            media_mode=media_mode,
            author_name_mode=_parse_author_name_mode(
                os.environ.get("WHATSAPP_AUTHOR_NAME")
                or os.environ.get("WHATSAPP_AUTHOR_NAME_MODE"),
                default="full",
            ),
            attachment_max_tokens=_parse_optional_int(
                os.environ.get("WHATSAPP_ATTACHMENT_MAX_TOKENS")
            ),
            skip_file_content_messages=_normalize_message_id_set(
                os.environ.get("WHATSAPP_SKIP_FILE_CONTENT_MESSAGES")
            ),
            max_file_content_size_bytes=_parse_file_size_limit(
                os.environ.get("WHATSAPP_MAX_FILE_CONTENT_SIZE")
            ),
            hard_min_timestamp=_parse_datetime(
                os.environ.get("WHATSAPP_MIN_TIMESTAMP", "")
            ),
            hard_max_timestamp=_parse_datetime(
                os.environ.get("WHATSAPP_MAX_TIMESTAMP", "")
            ),
        )
    )


def _clamp_settings(settings: WhatsAppSettings) -> WhatsAppSettings:
    start = settings.start
    end = settings.end
    if settings.hard_min_timestamp and (
        start is None or start < settings.hard_min_timestamp
    ):
        start = settings.hard_min_timestamp
    if settings.hard_max_timestamp and (
        end is None or end > settings.hard_max_timestamp
    ):
        end = settings.hard_max_timestamp
    if start and end and start > end:
        start = end
    return dataclass_replace(settings, start=start, end=end)


def build_whatsapp_settings(
    overrides: dict[str, Any] | None = None,
) -> WhatsAppSettings:
    env = _whatsapp_settings_from_env()
    if not overrides:
        return env
    format_value = str(overrides.get("format", env.format)).strip().lower()
    if format_value not in _VALID_FORMATS:
        format_value = env.format
    before_messages = overrides.get("before_messages", env.before_messages)
    after_messages = overrides.get("after_messages", env.after_messages)
    around_messages = overrides.get("around_messages", env.around_messages)
    message_context = overrides.get("message_context", env.message_context)
    if around_messages is not None:
        before_messages = around_messages
        after_messages = around_messages
    else:
        before_messages, after_messages = _distribute_message_context(
            message_context=message_context,
            before_messages=before_messages,
            after_messages=after_messages,
        )
    channel_limit = _parse_optional_int(overrides.get("channel_limit"))
    result = WhatsAppSettings(
        format=format_value,
        include_system=bool(overrides.get("include_system", env.include_system)),
        gap_threshold=overrides.get("gap_threshold", env.gap_threshold)
        if isinstance(overrides.get("gap_threshold", env.gap_threshold), timedelta)
        else env.gap_threshold,
        before_messages=before_messages,
        after_messages=after_messages,
        around_messages=around_messages,
        message_context=message_context,
        channel_limit=channel_limit if channel_limit is not None else env.channel_limit,
        start=_normalize_override_datetime(overrides.get("start", env.start)),
        end=_normalize_override_datetime(overrides.get("end", env.end)),
        start_message_id=_normalize_message_id(
            overrides.get("start_message", env.start_message_id)
        ),
        end_message_id=_normalize_message_id(
            overrides.get("end_message", env.end_message_id)
        ),
        before_duration=overrides.get("before_duration", env.before_duration),
        after_duration=overrides.get("after_duration", env.after_duration),
        around_duration=overrides.get("around_duration", env.around_duration),
        include_media_descriptions=bool(
            overrides.get("include_media_descriptions", env.include_media_descriptions)
        ),
        include_file_content=bool(
            overrides.get("include_file_content", env.include_file_content)
        ),
        media_mode=_parse_media_mode(
            str(overrides.get("media_mode", env.media_mode) or ""),
            default=env.media_mode,
        ),
        author_name_mode=_parse_author_name_mode(
            overrides.get("author_name_mode", env.author_name_mode),
            default=env.author_name_mode,
        ),
        attachment_max_tokens=_parse_optional_int(
            overrides.get("attachment_max_tokens", env.attachment_max_tokens)
        ),
        skip_file_content_messages=_normalize_message_id_set(
            overrides.get("skip_file_content_messages", env.skip_file_content_messages)
        ),
        max_file_content_size_bytes=_parse_file_size_limit(
            overrides.get(
                "max_file_content_size_bytes", env.max_file_content_size_bytes
            )
        ),
        hard_min_timestamp=env.hard_min_timestamp,
        hard_max_timestamp=env.hard_max_timestamp,
    )
    return _clamp_settings(result)


def _normalize_override_datetime(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        if isinstance(value, datetime) and value.tzinfo is not None:
            return value.replace(tzinfo=None)
        return value
    if isinstance(value, str):
        return _parse_datetime(value)
    return None


def _time_window_enabled(settings: WhatsAppSettings) -> bool:
    return any(
        value is not None
        for value in (
            settings.start,
            settings.end,
            settings.start_message_id,
            settings.end_message_id,
            settings.before_duration,
            settings.after_duration,
            settings.around_duration,
        )
    )


def _window_bounds_for_target(
    timestamp: datetime, settings: WhatsAppSettings
) -> tuple[datetime | None, datetime | None]:
    start = settings.start
    end = settings.end
    if settings.around_duration is not None:
        start = timestamp - settings.around_duration
        end = timestamp + settings.around_duration
    else:
        if settings.before_duration is not None:
            candidate = timestamp - settings.before_duration
            start = candidate if start is None or candidate > start else start
        if settings.after_duration is not None:
            candidate = timestamp + settings.after_duration
            end = candidate if end is None or candidate < end else end
    if settings.hard_min_timestamp and (
        start is None or start < settings.hard_min_timestamp
    ):
        start = settings.hard_min_timestamp
    if settings.hard_max_timestamp and (
        end is None or end > settings.hard_max_timestamp
    ):
        end = settings.hard_max_timestamp
    if start and end and start > end:
        start = end
    return start, end


def _window_bounds_no_target(
    messages: list[WhatsAppMessage], settings: WhatsAppSettings
) -> tuple[datetime | None, datetime | None]:
    anchor = max((message.timestamp for message in messages), default=datetime.now())
    return _window_bounds_for_target(anchor, settings)


def _message_window_counts(settings: WhatsAppSettings) -> tuple[int, int]:
    before = settings.before_messages if settings.before_messages is not None else 0
    after = settings.after_messages if settings.after_messages is not None else 0
    return max(0, int(before)), max(0, int(after))


def _filter_messages(
    messages: list[WhatsAppMessage],
    *,
    settings: WhatsAppSettings,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[WhatsAppMessage]:
    start_id = _normalize_message_id(settings.start_message_id)
    end_id = _normalize_message_id(settings.end_message_id)
    out: list[WhatsAppMessage] = []
    for message in messages:
        if start is not None and message.timestamp < start:
            continue
        if end is not None and message.timestamp > end:
            continue
        if start_id is not None and message.id < start_id:
            continue
        if end_id is not None and message.id > end_id:
            continue
        if message.is_system and not settings.include_system:
            continue
        out.append(message)
    return out


def _select_messages(
    messages: list[WhatsAppMessage],
    *,
    anchor_message_id: str | None,
    settings: WhatsAppSettings,
) -> list[WhatsAppMessage]:
    ordered = sorted(messages, key=lambda message: (message.timestamp, message.ordinal))
    if anchor_message_id:
        anchor_id = _normalize_message_id(anchor_message_id)
        anchor_index = next(
            (index for index, message in enumerate(ordered) if message.id == anchor_id),
            None,
        )
        if anchor_index is None:
            return []
        anchor = ordered[anchor_index]
        if _time_window_enabled(settings):
            start, end = _window_bounds_for_target(anchor.timestamp, settings)
            return _filter_messages(ordered, settings=settings, start=start, end=end)
        before_count, after_count = _message_window_counts(settings)
        lo = max(0, anchor_index - before_count)
        hi = min(len(ordered), anchor_index + after_count + 1)
        return _filter_messages(ordered[lo:hi], settings=settings)

    if _time_window_enabled(settings):
        start, end = _window_bounds_no_target(ordered, settings)
        return _filter_messages(ordered, settings=settings, start=start, end=end)

    filtered = _filter_messages(ordered, settings=settings)
    if settings.channel_limit > 0:
        return filtered[-settings.channel_limit :]
    return filtered


def _parse_media_kind(*, filename: str, content_type: str | None) -> str:
    ctype = (content_type or "").lower().strip()
    if ctype.startswith("image/"):
        return "image"
    if ctype.startswith("video/"):
        return "video"
    if ctype.startswith("audio/"):
        return "audio"
    suffix = Path(filename).suffix.lower()
    if suffix in _MEDIA_IMAGE_SUFFIXES:
        return "image"
    if suffix in _MEDIA_VIDEO_SUFFIXES:
        return "video"
    if suffix in _MEDIA_AUDIO_SUFFIXES:
        return "audio"
    return "file"


def _media_render_cache_identity(
    *,
    media_sha256: str,
    suffix: str,
    mode: str,
    media_kind: str,
) -> str:
    return "whatsapp-media-render:" + json.dumps(
        {
            "media_sha256": media_sha256,
            "suffix": suffix,
            "mode": mode,
            "media_kind": media_kind,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _describe_media_bytes(
    data: bytes,
    *,
    filename: str,
    kind: str,
    content_type: str | None,
    mode: str,
) -> str | None:
    if not data:
        return None
    media_sha256 = hashlib.sha256(data).hexdigest()
    suffix = Path(filename).suffix.lower() or ".bin"
    render_identity = _media_render_cache_identity(
        media_sha256=media_sha256,
        suffix=suffix,
        mode=mode,
        media_kind=kind,
    )
    from contextualize.runtime import get_refresh_media

    if not get_refresh_media():
        cached = _get_cached_rendered(render_identity)
        if cached and cached.strip() and not _is_invalid_media_description(cached):
            _log(f"  whatsapp media render cache hit: {filename}")
            return cached
        _log(f"  whatsapp media render cache miss: {filename}")

    fd, raw_path = tempfile.mkstemp(suffix=suffix)
    path = Path(raw_path)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        if mode == "transcribe" and kind in {"audio", "video"}:
            from contextualize.references.audio_transcription import transcribe_media_bytes

            markdown = transcribe_media_bytes(
                data,
                filename=filename,
                content_type=content_type,
            )
        else:
            from contextualize.render.markitdown import convert_path_to_markdown
            from contextualize.runtime import get_refresh_images

            result = convert_path_to_markdown(
                str(path),
                refresh_images=get_refresh_images(),
            )
            markdown = result.markdown
    except Exception:
        return None
    finally:
        path.unlink(missing_ok=True)

    markdown = (markdown or "").strip()
    if not markdown or _is_invalid_media_description(markdown):
        return None
    _store_rendered(render_identity, markdown)
    return markdown


def _extract_utf8_media(data: bytes) -> str | None:
    if not data:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return cleaned or None


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    token_count = count_tokens(text, target="cl100k_base")["count"]
    if token_count <= max_tokens:
        return text
    ratio = max_tokens / max(1, token_count)
    return text[: max(1, int(len(text) * ratio))].rstrip()


def _is_invalid_media_description(value: str) -> bool:
    text = value.strip()
    if not text:
        return True
    replacement_count = text.count("\ufffd")
    if replacement_count >= 128:
        return True
    if replacement_count >= 16 and (replacement_count / max(1, len(text))) >= 0.01:
        return True
    if text.startswith("�PNG") and "IHDR" in text:
        return True
    if _PNG_CHUNK_MARKER_RE.search(text) and replacement_count >= 2:
        return True
    longest_line = max((len(line) for line in text.splitlines()), default=0)
    return replacement_count >= 8 and longest_line >= 500


def _normalize_media_description(value: str) -> str:
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and re.fullmatch(r"ImageSize:\s*\d+x\d+", lines[0].strip(), flags=re.I):
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _normalize_attachment_nodes(
    message: WhatsAppMessage,
    *,
    source: WhatsAppSource,
    chat_id: str,
    settings: WhatsAppSettings,
) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for index, attachment in enumerate(message.attachments):
        kind = _parse_media_kind(
            filename=attachment.filename,
            content_type=attachment.content_type,
        )
        attachment_ref = _target_from_archive_path(
            getattr(source, "path", ""),
            chat=chat_id,
            message=message.id,
            attachment=attachment.member_name,
        )
        node: dict[str, Any] = {
            "type": kind,
            "filename": attachment.filename,
            "content_type": attachment.content_type,
            "bytes": attachment.size,
            "size": _format_file_size_human(attachment.size),
            "ref": attachment_ref,
        }
        data = source.read_media(attachment)
        if data is None:
            node["missing"] = True
            nodes.append({k: v for k, v in node.items() if v not in (None, "", [])})
            continue
        skip_file_content = kind == "file" and (
            not settings.include_file_content
            or (
                settings.skip_file_content_messages is not None
                and message.id in settings.skip_file_content_messages
            )
            or (
                attachment.size is not None
                and settings.max_file_content_size_bytes is not None
                and attachment.size > settings.max_file_content_size_bytes
            )
        )
        if kind == "file" and not skip_file_content:
            text_content = _extract_utf8_media(data)
            if text_content and settings.attachment_max_tokens is not None:
                text_content = _truncate_to_tokens(
                    text_content, settings.attachment_max_tokens
                )
            if text_content:
                node["text_content"] = text_content
        has_text_content = isinstance(node.get("text_content"), str) and bool(
            str(node.get("text_content")).strip()
        )
        if (
            settings.include_media_descriptions
            and not (kind == "file" and (skip_file_content or has_text_content))
        ):
            media_desc = _describe_media_bytes(
                data,
                filename=attachment.filename,
                kind=kind,
                content_type=attachment.content_type,
                mode=settings.media_mode if kind in {"audio", "video"} else "describe",
            )
            if media_desc:
                node["media_description"] = media_desc
        nodes.append({k: v for k, v in node.items() if v not in (None, "", [])})
    return nodes


def _normalize_message(
    message: WhatsAppMessage,
    *,
    source: WhatsAppSource,
    chat_id: str,
    settings: WhatsAppSettings,
) -> dict[str, Any]:
    sender = _display_author_name(message.sender, mode=settings.author_name_mode)
    normalized: dict[str, Any] = {
        "id": message.id,
        "ordinal": message.ordinal,
        "timestamp": _format_local_iso(message.timestamp),
        "sender": sender,
        "archive_sender": (
            message.sender if message.sender and message.sender != sender else None
        ),
        "content": message.content,
        "is_system": message.is_system,
        "chat_id": chat_id,
        "attachments": _normalize_attachment_nodes(
            message, source=source, chat_id=chat_id, settings=settings
        ),
        "reply_to_id": message.reply_to_id,
        "reply_to_content": message.reply_to_content,
        "forward": message.forward,
        "reactions": message.reactions,
        "pins": message.pins,
        "edits": message.edits,
        "deletes": message.deletes,
    }
    return {k: v for k, v in normalized.items() if v not in (None, [], "")}


def _display_author_name(sender: str | None, *, mode: str) -> str:
    if not sender:
        return "system"
    if mode != "first":
        return sender
    parts = sender.strip().split()
    return parts[0] if parts else sender


def _escape_xml_attr(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;")


def _format_attachment_idiomatic(
    node: dict[str, Any], *, indent: str = ""
) -> list[str]:
    node_type = str(node.get("type") or "attachment")
    filename = str(node.get("filename") or "")
    text_content = node.get("text_content")
    media_desc = node.get("media_description")
    normalized_desc = (
        _normalize_media_description(media_desc)
        if isinstance(media_desc, str) and media_desc.strip()
        else ""
    )
    if node_type == "file" and isinstance(text_content, str) and text_content.strip():
        name_attr = f' name="{_escape_xml_attr(filename)}"' if filename else ""
        lines = [f"{indent}<file{name_attr}>"]
        lines.extend(f"{indent}{line}" for line in text_content.splitlines())
        lines.append(f"{indent}</file>")
        return lines

    attrs = [f'type="{_escape_xml_attr(node_type)}"']
    if filename:
        attrs.append(f'filename="{_escape_xml_attr(filename)}"')
    ref = node.get("ref")
    if isinstance(ref, str) and ref:
        attrs.append(f'ref="{_escape_xml_attr(ref)}"')
    size = node.get("size")
    if isinstance(size, str) and size:
        attrs.append(f'size="{_escape_xml_attr(size)}"')
    raw_bytes = node.get("bytes")
    if isinstance(raw_bytes, int) and raw_bytes >= 0:
        attrs.append(f'bytes="{raw_bytes}"')
    if node.get("missing") is True:
        attrs.append('missing="true"')
    if normalized_desc:
        lines = [f"{indent}<attachment {' '.join(attrs)}>"]
        lines.extend(f"{indent}{line}" for line in normalized_desc.splitlines())
        lines.append(f"{indent}</attachment>")
        return lines
    return [f"{indent}<attachment {' '.join(attrs)} />"]


def _format_reply_quote(content: str) -> list[str]:
    normalized = content.strip()
    if not normalized:
        return []
    if len(normalized) > _REPLY_QUOTE_MAX_CHARS:
        normalized = normalized[: _REPLY_QUOTE_MAX_CHARS - 1].rstrip() + "..."
    return [f"> {line}" if line else ">" for line in normalized.splitlines()]


def _render_idiomatic(document: WhatsAppDocument, settings: WhatsAppSettings) -> str:
    lines: list[str] = []
    previous_date: str | None = None
    previous_dt: datetime | None = None
    for message in document.messages:
        timestamp = _parse_archive_timestamp_from_iso(str(message.get("timestamp") or ""))
        if timestamp is None:
            continue
        date_str = timestamp.date().isoformat()
        if previous_date != date_str:
            lines.append("")
            lines.append(f"<{date_str}>")
            previous_date = date_str
        elif previous_dt is not None:
            gap = timestamp - previous_dt
            if gap >= settings.gap_threshold:
                lines.append("")
                lines.append("---")
        sender = str(message.get("sender") or "system")
        content = str(message.get("content") or "")
        reply_to_content = (
            message.get("reply_to_content")
            if isinstance(message.get("reply_to_content"), str)
            else ""
        )
        attachments = (
            message.get("attachments")
            if isinstance(message.get("attachments"), list)
            else []
        )
        lines.append(f"[{sender}]")
        if reply_to_content:
            lines.extend(_format_reply_quote(reply_to_content))
        if content:
            lines.append(content)
        elif not attachments and not reply_to_content:
            lines.append("")
        for node in attachments:
            if isinstance(node, dict):
                lines.extend(_format_attachment_idiomatic(node))
        lines.append("")
        previous_dt = timestamp
    rendered = "\n".join(lines).strip()
    return rendered + "\n" if rendered else ""


def _parse_archive_timestamp_from_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class _LiteralString(str):
    pass


def _render_yaml(document: WhatsAppDocument, settings: WhatsAppSettings) -> str:
    import yaml

    class _WhatsAppDumper(yaml.SafeDumper):
        pass

    def _repr_literal(dumper, value):
        return dumper.represent_scalar("tag:yaml.org,2002:str", str(value), style="|")

    _WhatsAppDumper.add_representer(_LiteralString, _repr_literal)
    participants = sorted(
        {
            str(message.get("sender"))
            for message in document.messages
            if isinstance(message.get("sender"), str)
            and str(message.get("sender")) != "system"
        }
    )
    stream: list[dict[str, Any]] = []
    previous_date: str | None = None
    previous_dt: datetime | None = None
    for message in document.messages:
        timestamp = _parse_archive_timestamp_from_iso(str(message.get("timestamp") or ""))
        if timestamp is None:
            continue
        date_str = timestamp.date().isoformat()
        if previous_date != date_str:
            stream.append({"type": "timestamp", "date": date_str})
            previous_date = date_str
        elif previous_dt is not None:
            gap = timestamp - previous_dt
            if gap >= settings.gap_threshold:
                stream.append({"type": "gap", "duration": _format_duration_human(gap)})
        entry: dict[str, Any] = {
            "type": "message",
            "id": message.get("id"),
            "ordinal": message.get("ordinal"),
            "timestamp": message.get("timestamp"),
            "sender": message.get("sender"),
            "archive_sender": message.get("archive_sender"),
            "is_system": message.get("is_system"),
            "chat_id": message.get("chat_id"),
            "reply_to_id": message.get("reply_to_id"),
            "reply_to_content": message.get("reply_to_content"),
            "forward": message.get("forward"),
            "reactions": message.get("reactions"),
            "pins": message.get("pins"),
            "edits": message.get("edits"),
            "deletes": message.get("deletes"),
            "content": _LiteralString(str(message.get("content") or "")),
        }
        attachments = message.get("attachments")
        if isinstance(attachments, list) and attachments:
            entry["attachments"] = attachments
        stream.append({k: v for k, v in entry.items() if v not in (None, [], "")})
        previous_dt = timestamp
    return yaml.dump(
        {
            "platform": "WhatsApp",
            "whatsapp": {
                "chat": {
                    "id": document.chat_id,
                    "name": document.chat_name,
                    "source_path": document.source_path,
                }
            },
            "participants": [{"alias": participant} for participant in participants],
            "stream": stream,
        },
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        Dumper=_WhatsAppDumper,
    )


def _format_duration_human(value: timedelta) -> str:
    total = int(value.total_seconds())
    if total < 60:
        return f"{total}s"
    if total < 3600:
        minutes, seconds = divmod(total, 60)
        return f"{minutes}m{seconds}s" if seconds else f"{minutes}m"
    if total < 86400:
        hours, rest = divmod(total, 3600)
        minutes = rest // 60
        return f"{hours}h{minutes}m" if minutes else f"{hours}h"
    days, rest = divmod(total, 86400)
    hours = rest // 3600
    return f"{days}d{hours}h" if hours else f"{days}d"


def _render_document(
    document: WhatsAppDocument, settings: WhatsAppSettings
) -> WhatsAppDocument:
    rendered = (
        _render_yaml(document, settings)
        if settings.format == "yaml"
        else _render_idiomatic(document, settings)
    )
    return dataclass_replace(document, rendered=rendered)


def _document_label(chat: WhatsAppChatInfo, message_id: str | None = None) -> str:
    core = f"whatsapp:{chat.id}"
    return f"{core}/{message_id}" if message_id else core


def _document_trace_path(chat: WhatsAppChatInfo, message_id: str | None = None) -> str:
    core = f"whatsapp/{chat.id}"
    return f"{core}/{message_id}" if message_id else core


def _build_document(
    *,
    source_url: str,
    source: WhatsAppSource,
    chat: WhatsAppChatInfo,
    messages: list[WhatsAppMessage],
    settings: WhatsAppSettings,
    message_id: str | None = None,
) -> WhatsAppDocument:
    normalized_messages = [
        _normalize_message(message, source=source, chat_id=chat.id, settings=settings)
        for message in messages
    ]
    document = WhatsAppDocument(
        source_url=source_url,
        label=_document_label(chat, message_id),
        trace_path=_document_trace_path(chat, message_id),
        chat_id=chat.id,
        chat_name=chat.title,
        kind="message" if message_id else chat.kind,
        messages=normalized_messages,
        rendered="",
        source_path=chat.source_path,
    )
    return _render_document(document, settings)


def resolve_whatsapp_target(
    target: str,
    *,
    settings: WhatsAppSettings | None = None,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
) -> list[WhatsAppDocument]:
    del use_cache, cache_ttl, refresh_cache
    parsed = parse_whatsapp_target(target)
    if not parsed or parsed.get("kind") != "archive":
        raise ValueError(f"Not a WhatsApp target: {target}")
    source = WhatsAppArchiveSource(parsed["path"])
    chats = source.list_chats()
    if not chats:
        return []
    chat_id = parsed.get("chat_id")
    chat = next((item for item in chats if item.id == chat_id), None) if chat_id else chats[0]
    if chat is None:
        raise ValueError(f"WhatsApp chat not found: {chat_id}")
    effective_settings = settings or _whatsapp_settings_from_env()
    message_id = _normalize_message_id(parsed.get("message_id"))
    messages = _select_messages(
        source.iter_messages(chat.id),
        anchor_message_id=message_id,
        settings=effective_settings,
    )
    _log(
        f"Resolved WhatsApp archive {parsed['path']} chat={chat.id} messages={len(messages)}"
    )
    return [
        _build_document(
            source_url=target,
            source=source,
            chat=chat,
            messages=messages,
            settings=effective_settings,
            message_id=message_id,
        )
    ]


def whatsapp_document_timestamps(
    document: WhatsAppDocument,
) -> tuple[str | None, str | None]:
    parsed: list[datetime] = []
    for message in document.messages:
        timestamp = _parse_archive_timestamp_from_iso(str(message.get("timestamp") or ""))
        if timestamp is not None:
            parsed.append(timestamp)
    if not parsed:
        return None, None
    return _format_local_iso(min(parsed)), _format_local_iso(max(parsed))


def whatsapp_document_prose(
    document: WhatsAppDocument,
) -> tuple[str, list[str]]:
    bodies: list[str] = []
    authors: list[str] = []
    seen_authors: set[str] = set()
    for message in document.messages:
        if message.get("is_system"):
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        bodies.append(content)
        sender = message.get("sender")
        if isinstance(sender, str) and sender and sender not in seen_authors:
            seen_authors.add(sender)
            authors.append(sender)
    return "\n".join(bodies), authors


def whatsapp_scope_title(document: WhatsAppDocument) -> str:
    return f"WhatsApp -> {document.chat_name}"


def whatsapp_scope_url(document: WhatsAppDocument) -> str:
    return _target_from_archive_path(document.source_path, chat=document.chat_id)


def whatsapp_document_metadata(
    document: WhatsAppDocument,
    *,
    first_message: str | None = None,
    last_message: str | None = None,
    include_message_bounds: bool = True,
) -> dict[str, str]:
    resolved_first = first_message
    resolved_last = last_message
    if resolved_first is None or resolved_last is None:
        doc_first, doc_last = whatsapp_document_timestamps(document)
        resolved_first = resolved_first or doc_first
        resolved_last = resolved_last or doc_last
    metadata = {
        "title": whatsapp_scope_title(document),
        "url": whatsapp_scope_url(document),
        "kind": document.kind,
    }
    if include_message_bounds:
        if resolved_first:
            metadata["first_message"] = resolved_first
        if resolved_last:
            metadata["last_message"] = resolved_last
    return metadata


def render_whatsapp_document_with_metadata(
    document: WhatsAppDocument,
    *,
    settings: WhatsAppSettings,
    first_message: str | None = None,
    last_message: str | None = None,
    include_message_bounds: bool = True,
) -> str:
    import yaml

    metadata = whatsapp_document_metadata(
        document,
        first_message=first_message,
        last_message=last_message,
        include_message_bounds=include_message_bounds,
    )
    if settings.format == "yaml":
        metadata_text = yaml.safe_dump(
            {"metadata": metadata},
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
        return metadata_text + document.rendered.lstrip()
    frontmatter = yaml.safe_dump(
        metadata,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    ).strip()
    return f"---\n{frontmatter}\n---\n\n{document.rendered.lstrip()}"


def with_whatsapp_document_rendered(
    document: WhatsAppDocument, *, rendered: str
) -> WhatsAppDocument:
    return dataclass_replace(document, rendered=rendered)


def split_whatsapp_document_by_day(
    document: WhatsAppDocument, *, settings: WhatsAppSettings
) -> list[WhatsAppDocument]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for message in document.messages:
        timestamp = _parse_archive_timestamp_from_iso(str(message.get("timestamp") or ""))
        if timestamp is None:
            continue
        grouped.setdefault(timestamp.date().isoformat(), []).append(message)
    if not grouped:
        return [document]
    multi_day = len(grouped) > 1
    out: list[WhatsAppDocument] = []
    for day_key in sorted(grouped):
        day_doc = dataclass_replace(
            document,
            trace_path=f"{document.trace_path}@{day_key}"
            if multi_day
            else document.trace_path,
            messages=grouped[day_key],
            rendered="",
        )
        out.append(_render_document(day_doc, settings))
    return out


def _parse_window_mapping(raw: Any, *, prefix: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{prefix}.window must be a mapping")
    allowed = {
        "before-messages",
        "after-messages",
        "around-messages",
        "message-context",
        "start",
        "end",
        "start-message",
        "end-message",
        "before-duration",
        "after-duration",
        "around-duration",
        "channel-limit",
    }
    unknown = sorted(str(key) for key in raw if key not in allowed)
    if unknown:
        raise ValueError(f"{prefix}.window has invalid keys: {', '.join(unknown)}")
    result: dict[str, Any] = {}
    for config_key, result_key in (
        ("before-messages", "before_messages"),
        ("after-messages", "after_messages"),
        ("around-messages", "around_messages"),
        ("message-context", "message_context"),
        ("channel-limit", "channel_limit"),
    ):
        if config_key not in raw:
            continue
        value = raw.get(config_key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                f"{prefix}.window.{config_key} must be a non-negative integer"
            )
        result[result_key] = value
    for config_key, result_key in (("start", "start"), ("end", "end")):
        if config_key not in raw:
            continue
        value = raw.get(config_key)
        if not isinstance(value, str):
            raise ValueError(f"{prefix}.window.{config_key} must be a timestamp string")
        parsed = _parse_datetime(value)
        if parsed is None:
            raise ValueError(f"{prefix}.window.{config_key} is not a valid timestamp")
        result[result_key] = parsed
    for config_key, result_key in (
        ("start-message", "start_message"),
        ("end-message", "end_message"),
    ):
        if config_key not in raw:
            continue
        parsed = _normalize_message_id(raw.get(config_key))
        if parsed is None:
            raise ValueError(f"{prefix}.window.{config_key} must be a message id")
        result[result_key] = parsed
    for config_key, result_key in (
        ("before-duration", "before_duration"),
        ("after-duration", "after_duration"),
        ("around-duration", "around_duration"),
    ):
        if config_key not in raw:
            continue
        value = raw.get(config_key)
        if not isinstance(value, str):
            raise ValueError(f"{prefix}.window.{config_key} must be a duration string")
        parsed = _parse_duration_compound(value)
        if parsed is None:
            raise ValueError(f"{prefix}.window.{config_key} is not a valid duration")
        result[result_key] = parsed
    return result or None


def parse_whatsapp_config_mapping(raw: Any, *, prefix: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{prefix} must be a mapping")
    allowed = {
        "format",
        "author-name",
        "include-system",
        "gap-threshold",
        "window",
        "media",
    }
    unknown = sorted(str(key) for key in raw if key not in allowed)
    if unknown:
        raise ValueError(f"{prefix} has invalid keys: {', '.join(unknown)}")
    result: dict[str, Any] = {}
    if "format" in raw:
        value = raw.get("format")
        if not isinstance(value, str) or value.strip().lower() not in _VALID_FORMATS:
            raise ValueError(
                f"{prefix}.format must be one of: {', '.join(sorted(_VALID_FORMATS))}"
            )
        result["format"] = value.strip().lower()
    if "author-name" in raw:
        value = raw.get("author-name")
        if (
            not isinstance(value, str)
            or value.strip().lower() not in _VALID_AUTHOR_NAME_MODES
        ):
            raise ValueError(
                f"{prefix}.author-name must be one of: {', '.join(sorted(_VALID_AUTHOR_NAME_MODES))}"
            )
        result["author_name_mode"] = value.strip().lower()
    if "include-system" in raw:
        value = raw.get("include-system")
        if not isinstance(value, bool):
            raise ValueError(f"{prefix}.include-system must be a boolean")
        result["include_system"] = value
    if "gap-threshold" in raw:
        value = raw.get("gap-threshold")
        if not isinstance(value, str):
            raise ValueError(f"{prefix}.gap-threshold must be a duration string")
        parsed = _parse_duration_compound(value)
        if parsed is None:
            raise ValueError(f"{prefix}.gap-threshold is not a valid duration")
        result["gap_threshold"] = parsed
    window = _parse_window_mapping(raw.get("window"), prefix=prefix)
    if window:
        result.update(window)
    media = raw.get("media")
    if media is not None:
        if not isinstance(media, dict):
            raise ValueError(f"{prefix}.media must be a mapping")
        allowed_media = {
            "describe",
            "file-content",
            "skip-file-content-messages",
            "max-file-content-size",
            "mode",
        }
        unknown_media = sorted(str(key) for key in media if key not in allowed_media)
        if unknown_media:
            raise ValueError(
                f"{prefix}.media has invalid keys: {', '.join(unknown_media)}"
            )
        if "describe" in media:
            describe = media.get("describe")
            if not isinstance(describe, bool):
                raise ValueError(f"{prefix}.media.describe must be a boolean")
            result["include_media_descriptions"] = describe
        if "file-content" in media:
            file_content = media.get("file-content")
            if not isinstance(file_content, bool):
                raise ValueError(f"{prefix}.media.file-content must be a boolean")
            result["include_file_content"] = file_content
        if "skip-file-content-messages" in media:
            skip_messages = _normalize_message_id_set(
                media.get("skip-file-content-messages")
            )
            if skip_messages is None:
                raise ValueError(
                    f"{prefix}.media.skip-file-content-messages must be a string or list"
                )
            result["skip_file_content_messages"] = skip_messages
        if "max-file-content-size" in media:
            max_size = _parse_file_size_limit(media.get("max-file-content-size"))
            if max_size is None:
                raise ValueError(
                    f"{prefix}.media.max-file-content-size must be a size string"
                )
            result["max_file_content_size_bytes"] = max_size
        if "mode" in media:
            mode = media.get("mode")
            if not isinstance(mode, str) or mode.strip().lower() not in _VALID_MEDIA_MODES:
                raise ValueError(
                    f"{prefix}.media.mode must be one of: {', '.join(sorted(_VALID_MEDIA_MODES))}"
                )
            result["media_mode"] = mode.strip().lower()
    return result or None


def whatsapp_settings_cache_key(settings: WhatsAppSettings) -> tuple[Any, ...]:
    return (
        settings.format,
        settings.include_system,
        int(settings.gap_threshold.total_seconds()),
        settings.before_messages,
        settings.after_messages,
        settings.around_messages,
        settings.message_context,
        settings.channel_limit,
        settings.start.isoformat() if settings.start else None,
        settings.end.isoformat() if settings.end else None,
        settings.start_message_id,
        settings.end_message_id,
        int(settings.before_duration.total_seconds())
        if settings.before_duration
        else None,
        int(settings.after_duration.total_seconds())
        if settings.after_duration
        else None,
        int(settings.around_duration.total_seconds())
        if settings.around_duration
        else None,
        settings.include_media_descriptions,
        settings.include_file_content,
        settings.media_mode,
        settings.author_name_mode,
        tuple(sorted(settings.skip_file_content_messages))
        if settings.skip_file_content_messages
        else None,
        settings.max_file_content_size_bytes,
        settings.hard_min_timestamp.isoformat()
        if settings.hard_min_timestamp
        else None,
        settings.hard_max_timestamp.isoformat()
        if settings.hard_max_timestamp
        else None,
    )


def document_to_payload(document: WhatsAppDocument) -> dict[str, Any]:
    return asdict(document)
