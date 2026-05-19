from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any
from urllib.parse import parse_qs, urlencode, unquote, urlparse
import zipfile

_DEFAULT_TIMEOUT_SECONDS = 30.0
_PROVIDER_CACHE_ROOT = Path(
    os.environ.get(
        "CONTEXTUALIZE_GOOGLEDOCS_CACHE",
        os.path.expanduser("~/.local/share/contextualize/cache/googledocs/v1"),
    )
)
_DOC_URL_RE = re.compile(
    r"^/(?:document/(?:u/\d+/)?)?d/(?P<doc_id>[A-Za-z0-9_-]+)(?:/|$)"
)
_DOC_PUBLISHED_URL_RE = re.compile(
    r"^/(?:document/(?:u/\d+/)?)?d/e/(?P<doc_id>[A-Za-z0-9_-]+)(?:/|$)"
)
_DOC_ID_RE = re.compile(r"^[A-Za-z0-9_-]{20,}$")
_IMAGE_RE = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\((?P<target>[^)\s]+)(?:\s+\"[^\"]*\")?\)(?:\{[^}\n]*\})?"
)
_REFERENCE_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\[(?P<ref>[^\]]+)\]")
_REFERENCE_DEF_RE = re.compile(
    r"^\[(?P<ref>[^\]]+)\]:\s*(?P<target><[^>]+>|\S+)(?:\s+\"[^\"]*\")?\s*$"
)
_MEDIA_TAG_RE = re.compile(r"^#?\s*description\s*\(auto-generated\)\s*:?\s*$", re.I)


@dataclass(frozen=True)
class ParsedGoogleDocTarget:
    raw_target: str
    doc_id: str
    tab: str | None = None
    published: bool = False

    @property
    def canonical_id(self) -> str:
        if self.tab:
            return f"{self.doc_id}:tab:{self.tab}"
        return self.doc_id

    @property
    def canonical_url(self) -> str:
        if self.published:
            return f"https://docs.google.com/document/d/e/{self.doc_id}/pub"
        return f"https://docs.google.com/document/d/{self.doc_id}/edit"


@dataclass(frozen=True)
class GoogleDocsSettings:
    include_media: bool = True
    include_media_descriptions: bool = True
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True)
class GoogleDocsMedia:
    kind: str
    url: str | None
    filename: str | None
    caption: str | None
    description: str | None = None


@dataclass(frozen=True)
class GoogleDocsResolvedDocument:
    label: str
    rendered: str
    source_ref: str
    source_path: str
    context_subpath: str
    kind: str
    canonical_id: str
    title: str | None
    media_count: int = 0


@dataclass(frozen=True)
class _RenderedMarkdown:
    markdown: str
    media_count: int


@dataclass(frozen=True)
class _ExportBundle:
    markdown: str
    media: dict[str, bytes] = field(default_factory=dict)
    markdown_path: str | None = None


def _load_dotenv() -> None:
    try:
        from contextualize.auth.common import load_dotenv_optional

        load_dotenv_optional()
    except Exception:
        return


def _parse_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if not cleaned:
            return default
        return cleaned not in {"0", "false", "no", "off"}
    return default


def _parse_timeout(value: Any, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return max(1.0, float(value))
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return default
        try:
            return max(1.0, float(cleaned))
        except ValueError:
            return default
    return default


def _parse_optional_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def build_google_docs_settings(
    overrides: dict[str, Any] | None = None,
) -> GoogleDocsSettings:
    _load_dotenv()
    overrides = overrides or {}
    return GoogleDocsSettings(
        include_media=_parse_bool(
            overrides.get("include_media", os.environ.get("GOOGLEDOCS_INCLUDE_MEDIA")),
            default=True,
        ),
        include_media_descriptions=_parse_bool(
            overrides.get(
                "include_media_descriptions",
                os.environ.get("GOOGLEDOCS_MEDIA_DESCRIPTIONS"),
            ),
            default=True,
        ),
        timeout_seconds=_parse_timeout(
            overrides.get("timeout_seconds", os.environ.get("GOOGLEDOCS_TIMEOUT")),
            default=_DEFAULT_TIMEOUT_SECONDS,
        ),
    )


def google_docs_settings_cache_key(settings: GoogleDocsSettings) -> str:
    payload = {
        "include_media": settings.include_media,
        "include_media_descriptions": settings.include_media_descriptions,
        "timeout_seconds": settings.timeout_seconds,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def parse_google_doc_target(target: str) -> ParsedGoogleDocTarget | None:
    cleaned = target.strip()
    if not cleaned:
        return None

    parsed = urlparse(cleaned)
    scheme = parsed.scheme.lower()
    if scheme in {"http", "https"}:
        host = (parsed.hostname or "").lower()
        if host != "docs.google.com":
            return None
        published_match = _DOC_PUBLISHED_URL_RE.match(parsed.path)
        if published_match is not None:
            return ParsedGoogleDocTarget(
                raw_target=target,
                doc_id=published_match.group("doc_id"),
                tab=_tab_from_url(parsed.query),
                published=True,
            )
        match = _DOC_URL_RE.match(parsed.path)
        if match is None:
            return None
        return ParsedGoogleDocTarget(
            raw_target=target,
            doc_id=match.group("doc_id"),
            tab=_tab_from_url(parsed.query),
        )

    if scheme in {"gdoc", "googledoc", "googledocs", "google-doc"}:
        candidate = parsed.netloc or parsed.path.lstrip("/")
        if _DOC_ID_RE.fullmatch(candidate):
            return ParsedGoogleDocTarget(raw_target=target, doc_id=candidate)
        return None

    if cleaned.lower().startswith(("gdoc:", "googledoc:", "googledocs:", "google-doc:")):
        candidate = cleaned.split(":", 1)[1].strip().lstrip("/")
        if _DOC_ID_RE.fullmatch(candidate):
            return ParsedGoogleDocTarget(raw_target=target, doc_id=candidate)

    return None


def is_google_doc_target(target: str) -> bool:
    return parse_google_doc_target(target) is not None


def resolve_google_doc(
    target: str,
    *,
    settings: GoogleDocsSettings,
    use_cache: bool,
    cache_ttl: Any,
    refresh_cache: bool,
) -> GoogleDocsResolvedDocument:
    parsed = parse_google_doc_target(target)
    if parsed is None:
        raise ValueError(f"not a Google Docs target: {target}")

    settings_key = google_docs_settings_cache_key(settings)
    cache_key = _resolution_cache_key(parsed, settings_key)
    if use_cache and not refresh_cache:
        cached = _read_cached_document(cache_key, cache_ttl)
        if cached is not None:
            return cached

    exported = _fetch_published_html(parsed, settings) if parsed.published else _fetch_public_markdown(parsed, settings)
    rendered = _render_export(exported, settings=settings)
    body = rendered.markdown.strip()
    title = _title_from_markdown(body) or parsed.doc_id
    safe_title = _safe_path_segment(title, fallback=parsed.doc_id)
    document = GoogleDocsResolvedDocument(
        label=f"googledocs/{safe_title}",
        rendered=_render_document(
            canonical_url=parsed.canonical_url,
            doc_id=parsed.doc_id,
            content=body,
        ),
        source_ref="docs.google.com",
        source_path=parsed.canonical_id,
        context_subpath=f"googledocs/{safe_title}.md",
        kind="document",
        canonical_id=parsed.canonical_id,
        title=title,
        media_count=rendered.media_count,
    )
    if use_cache:
        _store_cached_document(cache_key, document)
    return document


def _tab_from_url(query: str) -> str | None:
    values = parse_qs(query).get("tab") or []
    for value in values:
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return None


def _fetch_public_markdown(
    parsed: ParsedGoogleDocTarget,
    settings: GoogleDocsSettings,
) -> bytes:
    try:
        import requests

        response = requests.get(
            _public_markdown_export_url(parsed),
            timeout=settings.timeout_seconds,
        )
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"public Google Docs markdown export failed for {parsed.doc_id}: {exc}"
        ) from exc

    content_type = response.headers.get("Content-Type", "").lower()
    content = response.content
    if "text/html" in content_type:
        snippet = content.decode("utf-8", errors="replace").strip()[:500]
        raise RuntimeError(
            "public Google Docs markdown export returned HTML instead of markdown; "
            f"document may not be public: {snippet}"
        )
    if not content:
        raise RuntimeError("public Google Docs markdown export was empty")
    return content


def _public_markdown_export_url(parsed: ParsedGoogleDocTarget) -> str:
    params = {"format": "md"}
    if parsed.tab:
        params["tab"] = parsed.tab
    return f"https://docs.google.com/document/d/{parsed.doc_id}/export?{urlencode(params)}"


def _fetch_published_html(
    parsed: ParsedGoogleDocTarget,
    settings: GoogleDocsSettings,
) -> bytes:
    try:
        import requests

        response = requests.get(parsed.canonical_url, timeout=settings.timeout_seconds)
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"fetch published Google Doc failed: {exc}") from exc
    markdown, title = _convert_published_response(response)
    if title and not _title_from_markdown(markdown):
        markdown = f"# {title}\n\n{markdown}"
    return markdown.encode("utf-8")


def _convert_published_response(response: Any) -> tuple[str, str | None]:
    try:
        from contextualize.render.markitdown import convert_response_to_markdown

        result = convert_response_to_markdown(response)
        markdown = getattr(result, "markdown", "") or ""
        title = _parse_optional_str(getattr(result, "title", None))
        if markdown.strip():
            return markdown, title
    except Exception:
        pass
    text = getattr(response, "text", "") or ""
    return _html_fallback_to_markdown(text), _title_from_html(text)


def _title_from_html(text: str) -> str | None:
    match = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.I | re.S)
    if match is None:
        return None
    return _clean_text(re.sub(r"<[^>]+>", " ", match.group(1)))


def _html_fallback_to_markdown(text: str) -> str:
    body_match = re.search(r"<body[^>]*>(.*?)</body>", text, flags=re.I | re.S)
    body = body_match.group(1) if body_match else text
    body = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", body)
    body = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", body)
    body = re.sub(
        r"(?is)<img[^>]*alt=[\"']([^\"']*)[\"'][^>]*src=[\"']([^\"']*)[\"'][^>]*>",
        lambda match: f"![{html.unescape(match.group(1)).strip()}]({html.unescape(match.group(2)).strip()})",
        body,
    )
    for level in range(1, 7):
        body = re.sub(
            rf"(?is)<h{level}[^>]*>(.*?)</h{level}>",
            lambda match, level=level: "\n\n"
            + "#" * level
            + " "
            + _clean_text(re.sub(r"<[^>]+>", " ", match.group(1)))
            + "\n\n",
            body,
        )
    body = re.sub(r"(?is)<br\s*/?>", "\n", body)
    body = re.sub(r"(?is)</p\s*>", "\n\n", body)
    body = re.sub(r"(?is)</li\s*>", "\n", body)
    body = re.sub(r"(?is)<li[^>]*>", "- ", body)
    body = re.sub(r"(?is)<[^>]+>", " ", body)
    return _clean_text(body, preserve_newlines=True)


def _render_export(data: bytes, *, settings: GoogleDocsSettings) -> _RenderedMarkdown:
    bundle = _decode_export_bundle(data)
    if not settings.include_media:
        return _RenderedMarkdown(
            markdown=_strip_markdown_images(bundle.markdown),
            media_count=0,
        )
    return _rewrite_markdown_media(bundle, settings=settings)


def _decode_export_bundle(data: bytes) -> _ExportBundle:
    if _is_zip(data):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            markdown_name = _select_markdown_member(archive)
            if markdown_name is None:
                raise ValueError("Google Docs markdown export zip did not contain a markdown file")
            markdown = archive.read(markdown_name).decode("utf-8", errors="replace")
            media: dict[str, bytes] = {}
            markdown_parent = str(PurePosixPath(markdown_name).parent)
            for info in archive.infolist():
                if info.is_dir() or info.filename == markdown_name:
                    continue
                normalized = _normalize_zip_member(info.filename, markdown_parent)
                if normalized:
                    media[normalized] = archive.read(info.filename)
            return _ExportBundle(markdown=markdown, media=media, markdown_path=markdown_name)
    return _ExportBundle(markdown=data.decode("utf-8", errors="replace"))


def _is_zip(data: bytes) -> bool:
    if not data.startswith(b"PK"):
        return False
    return zipfile.is_zipfile(io.BytesIO(data))


def _select_markdown_member(archive: zipfile.ZipFile) -> str | None:
    names = [info.filename for info in archive.infolist() if not info.is_dir()]
    markdown_names = [name for name in names if name.lower().endswith((".md", ".markdown"))]
    if not markdown_names:
        return None
    markdown_names.sort(key=lambda value: (value.count("/"), len(value), value))
    return markdown_names[0]


def _normalize_zip_member(filename: str, markdown_parent: str) -> str | None:
    path = PurePosixPath(filename)
    if path.is_absolute() or ".." in path.parts:
        return None
    direct = str(path)
    if markdown_parent and markdown_parent != ".":
        parent = PurePosixPath(markdown_parent)
        try:
            return str(path.relative_to(parent))
        except ValueError:
            return direct
    return direct


def _rewrite_markdown_media(
    bundle: _ExportBundle,
    *,
    settings: GoogleDocsSettings,
) -> _RenderedMarkdown:
    media_count = 0
    consumed_refs: set[str] = set()
    reference_targets = _markdown_reference_targets(bundle.markdown)

    def render_media(
        raw_target: str,
        alt: str | None,
        *,
        fallback_filename: str | None = None,
    ) -> str | None:
        nonlocal media_count
        target = html.unescape(raw_target).strip().strip("<>")
        media_bytes = _media_bytes_for_target(target, bundle.media)
        filename = (
            _filename_from_markdown_target(target)
            or _filename_from_data_url(target, fallback=fallback_filename or "media")
            or _filename_from_url(target)
        )
        description = None
        url = None
        if media_bytes is not None:
            if settings.include_media_descriptions:
                description = _describe_media_bytes(
                    media_bytes,
                    filename=filename,
                    caption=alt,
                )
        elif _is_http_url(target):
            url = target
            if settings.include_media_descriptions:
                description = _describe_media_url(target, filename, alt)
        else:
            return None
        media_count += 1
        return _format_media_tag(
            GoogleDocsMedia(
                kind=_media_kind_from_filename(filename),
                url=url,
                filename=filename,
                caption=alt,
                description=description,
            )
        )

    def replace_inline(match: re.Match[str]) -> str:
        alt = html.unescape(match.group("alt")).strip() or None
        rendered = render_media(match.group("target"), alt)
        return rendered if rendered is not None else match.group(0)

    def replace_reference(match: re.Match[str]) -> str:
        ref = html.unescape(match.group("ref")).strip()
        raw_target = reference_targets.get(ref)
        if raw_target is None:
            return match.group(0)
        alt = html.unescape(match.group("alt")).strip() or None
        rendered = render_media(raw_target, alt, fallback_filename=ref)
        if rendered is None:
            return match.group(0)
        consumed_refs.add(ref)
        return rendered

    markdown = _IMAGE_RE.sub(replace_inline, bundle.markdown)
    markdown = _REFERENCE_IMAGE_RE.sub(replace_reference, markdown)
    if consumed_refs:
        markdown = _drop_consumed_reference_definitions(markdown, consumed_refs)
    return _RenderedMarkdown(markdown=markdown, media_count=media_count)


def _media_bytes_for_target(target: str, media: dict[str, bytes]) -> bytes | None:
    data_bytes = _bytes_from_data_url(target)
    if data_bytes is not None:
        return data_bytes
    parsed = urlparse(target)
    if parsed.scheme or target.startswith("#"):
        return None
    normalized = _normalize_markdown_media_path(target)
    return media.get(normalized)


def _is_http_url(target: str) -> bool:
    parsed = urlparse(target)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _bytes_from_data_url(target: str) -> bytes | None:
    if not target.startswith("data:"):
        return None
    header, separator, payload = target.partition(",")
    if not separator:
        return None
    try:
        if header.lower().endswith(";base64"):
            return base64.b64decode(payload, validate=True)
        return unquote(payload).encode("utf-8")
    except Exception:
        return None


def _normalize_markdown_media_path(target: str) -> str:
    path = PurePosixPath(unquote(target).split("?", 1)[0].split("#", 1)[0])
    parts = [part for part in path.parts if part not in {"", "."}]
    if ".." in parts:
        return ""
    return str(PurePosixPath(*parts)) if parts else ""


def _strip_markdown_images(markdown: str) -> str:
    reference_targets = _markdown_reference_targets(markdown)
    consumed_refs: set[str] = set()

    def replace_inline(match: re.Match[str]) -> str:
        return html.unescape(match.group("alt")).strip()

    def replace_reference(match: re.Match[str]) -> str:
        ref = html.unescape(match.group("ref")).strip()
        if ref in reference_targets:
            consumed_refs.add(ref)
        return html.unescape(match.group("alt")).strip()

    stripped = _IMAGE_RE.sub(replace_inline, markdown)
    stripped = _REFERENCE_IMAGE_RE.sub(replace_reference, stripped)
    if consumed_refs:
        stripped = _drop_consumed_reference_definitions(stripped, consumed_refs)
    return stripped


def _markdown_reference_targets(markdown: str) -> dict[str, str]:
    targets: dict[str, str] = {}
    for line in markdown.splitlines():
        match = _REFERENCE_DEF_RE.match(line.strip())
        if match is None:
            continue
        target = html.unescape(match.group("target")).strip().strip("<>")
        if target:
            targets[html.unescape(match.group("ref")).strip()] = target
    return targets


def _drop_consumed_reference_definitions(markdown: str, refs: set[str]) -> str:
    lines: list[str] = []
    for line in markdown.splitlines():
        match = _REFERENCE_DEF_RE.match(line.strip())
        if match is not None and html.unescape(match.group("ref")).strip() in refs:
            continue
        lines.append(line)
    return "\n".join(lines)


def _describe_media_url(url: str, filename: str | None, caption: str | None) -> str | None:
    try:
        import requests

        response = requests.get(url, timeout=30.0)
        response.raise_for_status()
        content = response.content
    except Exception:
        return None
    if not content:
        return None
    return _describe_media_bytes(content, filename=filename, caption=caption)


def _describe_media_bytes(
    content: bytes,
    *,
    filename: str | None,
    caption: str | None,
) -> str | None:
    if not content:
        return None
    suffix = Path(filename or "media.bin").suffix or ".bin"
    fd, path_str = tempfile.mkstemp(suffix=suffix)
    path = Path(path_str)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
        try:
            from contextualize.render.markitdown import convert_path_to_markdown
        except Exception:
            return None
        prompt_append = f"Caption context: {caption}" if caption else "Describe the media."
        result = convert_path_to_markdown(str(path), prompt_append=prompt_append)
        return _normalize_media_description(getattr(result, "markdown", "") or "") or None
    except Exception:
        return None
    finally:
        path.unlink(missing_ok=True)


def _normalize_media_description(value: str) -> str:
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and _MEDIA_TAG_RE.fullmatch(lines[0].strip()):
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return _clean_text("\n".join(lines), preserve_newlines=True)


def _format_media_tag(media: GoogleDocsMedia) -> str:
    tag = "video" if media.kind == "video" else "image"
    attrs: list[str] = []
    if media.filename:
        attrs.append(f'filename="{_escape_xml_attr(media.filename)}"')
    if media.url:
        attrs.append(f'url="{_escape_xml_attr(media.url)}"')
    if media.caption:
        attrs.append(f'caption="{_escape_xml_attr(media.caption)}"')
    suffix = f" {' '.join(attrs)}" if attrs else ""
    if media.description:
        lines = [f"<{tag}{suffix}>"]
        lines.extend(_clean_text(media.description, preserve_newlines=True).splitlines())
        lines.append(f"</{tag}>")
        return "\n".join(lines)
    return f"<{tag}{suffix} />"


def _render_document(
    *,
    canonical_url: str,
    doc_id: str,
    content: str,
) -> str:
    lines = ["---", f"url: {_escape_yaml_string(canonical_url)}", f"doc_id: {doc_id}"]
    lines.extend(["---", ""])
    body = content.strip()
    if body:
        lines.append(body)
    lines.append("")
    return "\n".join(lines)


def _title_from_markdown(markdown: str) -> str | None:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or None
    return None


def _safe_path_segment(value: str, *, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
    return cleaned or fallback


def _escape_xml_attr(value: str) -> str:
    return html.escape(value, quote=True)


def _escape_yaml_string(value: str) -> str:
    if not value:
        return '""'
    if "\n" in value:
        indented = "\n".join(f"  {line}" for line in value.splitlines())
        return f"|\n{indented}"
    needs_quotes = any(c in value for c in ":{}[],\"'|>&*!?#%@`")
    if needs_quotes or value.startswith(" "):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def _clean_text(value: str, *, preserve_newlines: bool = False) -> str:
    cleaned = html.unescape(value).replace("\xa0", " ")
    if preserve_newlines:
        cleaned = re.sub(r"[^\S\n]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    else:
        cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _filename_from_markdown_target(target: str) -> str | None:
    if urlparse(target).scheme:
        return None
    normalized = _normalize_markdown_media_path(target)
    if not normalized:
        return None
    return PurePosixPath(normalized).name or None


def _filename_from_url(url: str) -> str | None:
    path = unquote(urlparse(url).path)
    name = Path(path).name
    return name or None


def _filename_from_data_url(target: str, *, fallback: str) -> str | None:
    if not target.startswith("data:"):
        return None
    mime = target[5:].split(",", 1)[0].split(";", 1)[0].lower()
    suffix = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/svg+xml": ".svg",
        "video/mp4": ".mp4",
        "video/webm": ".webm",
    }.get(mime, ".bin")
    return f"{_safe_path_segment(fallback, fallback='media')}{suffix}"


def _media_kind_from_filename(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in {".mp4", ".mov", ".webm", ".ogv"}:
        return "video"
    return "image"


def _resolution_cache_key(parsed: ParsedGoogleDocTarget, settings_key: str) -> str:
    payload = {
        "v": 2,
        "doc_id": parsed.doc_id,
        "tab": parsed.tab,
        "published": parsed.published,
        "settings": settings_key,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _cache_path(key: str) -> Path:
    return _PROVIDER_CACHE_ROOT / f"{key}.json"


def _read_cached_document(key: str, cache_ttl: Any) -> GoogleDocsResolvedDocument | None:
    path = _cache_path(key)
    try:
        if not path.exists():
            return None
        ttl_seconds = _cache_ttl_seconds(cache_ttl)
        if ttl_seconds is not None:
            age = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
            if age > ttl_seconds:
                return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return GoogleDocsResolvedDocument(
            label=str(payload["label"]),
            rendered=str(payload["rendered"]),
            source_ref=str(payload["source_ref"]),
            source_path=str(payload["source_path"]),
            context_subpath=str(payload["context_subpath"]),
            kind=str(payload["kind"]),
            canonical_id=str(payload["canonical_id"]),
            title=_parse_optional_str(payload.get("title")),
            media_count=int(payload.get("media_count") or 0),
        )
    except Exception:
        return None


def _store_cached_document(key: str, document: GoogleDocsResolvedDocument) -> None:
    try:
        _PROVIDER_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        payload = {
            "label": document.label,
            "rendered": document.rendered,
            "source_ref": document.source_ref,
            "source_path": document.source_path,
            "context_subpath": document.context_subpath,
            "kind": document.kind,
            "canonical_id": document.canonical_id,
            "title": document.title,
            "media_count": document.media_count,
        }
        _cache_path(key).write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
    except Exception:
        return


def _cache_ttl_seconds(cache_ttl: Any) -> float | None:
    if cache_ttl is None:
        return None
    total_seconds = getattr(cache_ttl, "total_seconds", None)
    if callable(total_seconds):
        try:
            seconds = total_seconds()
            if isinstance(seconds, (int, float)):
                return max(0.0, float(seconds))
            return max(0.0, float(str(seconds)))
        except (TypeError, ValueError):
            return None
    try:
        return max(0.0, float(cache_ttl))
    except (TypeError, ValueError):
        return None
