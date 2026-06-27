from __future__ import annotations

import hashlib
import html
import json
import re
import secrets
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from urllib.parse import urlencode, urlparse

from ..shared.cache import (
    provider_cache_root,
    read_json_entry,
    write_json_entry,
)

_PROVIDER = "airtable"
_CACHE_ENV = "CONTEXTUALIZE_AIRTABLE_CACHE"
_DEFAULT_TIMEOUT_SECONDS = 30.0
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
# Any well-formed page-bundle id satisfies the readForPages auth gate; the real
# value is neither exposed in the page nor validated against the share.
_SYNTHETIC_PAGE_BUNDLE_ID = "pbd0000000000XXXX"

_ID_BODY = r"[A-Za-z0-9]{14,}"
_APP_RE = re.compile(rf"^app{_ID_BODY}$")
_SHARE_RE = re.compile(rf"^shr{_ID_BODY}$")
_PAGE_RE = re.compile(rf"^pag{_ID_BODY}$")
_TABLE_RE = re.compile(rf"^tbl{_ID_BODY}$")
_VIEW_RE = re.compile(rf"^viw{_ID_BODY}$")

_PAGE_LOAD_ID_RE = re.compile(r'"pageLoadId":"(\w+)"')
_CODE_VERSION_RE = re.compile(r'"codeVersion":"([a-f0-9]+)"')
_APP_ID_RE = re.compile(r'"(?:applicationId|singleApplicationId)":"(app[A-Za-z0-9]+)"')
_PREFETCH_RE = re.compile(r'urlWithParams:\s*"((?:[^"\\]|\\.)*)"')
_ACCESS_POLICY_RE = re.compile(r'"accessPolicy":"((?:\\.|[^"\\])*)"')
_OG_TITLE_RE = re.compile(
    r'<meta\s+property="og:title"\s+content="([^"]*)"', re.IGNORECASE
)
_HTML_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)

_VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".ogv", ".m4v", ".avi", ".mkv"}
_ATTACHMENT_TYPES = {"multipleAttachment", "multipleAttachments"}
_VALUE_KEYS = (
    "name",
    "text",
    "displayName",
    "foreignRowDisplayName",
    "label",
    "title",
    "email",
    "value",
    "url",
    "filename",
)
_PRETTY_FIELD_TYPES = {
    "text": "Short text",
    "multilineText": "Long text",
    "richText": "Rich text",
    "select": "Single select",
    "singleSelect": "Single select",
    "multiSelect": "Multiple select",
    "multipleSelects": "Multiple select",
    "number": "Number",
    "currency": "Currency",
    "percent": "Percent",
    "checkbox": "Checkbox",
    "date": "Date",
    "dateTime": "Date and time",
    "datetime": "Date and time",
    "phone": "Phone number",
    "email": "Email",
    "url": "URL",
    "rating": "Rating",
    "duration": "Duration",
    "multipleAttachment": "Attachments",
    "multipleAttachments": "Attachments",
    "foreignKey": "Linked records",
    "multipleRecordLinks": "Linked records",
    "barcode": "Barcode",
    "formula": "Formula",
    "rollup": "Rollup",
    "lookup": "Lookup",
    "autoNumber": "Auto number",
    "collaborator": "Collaborator",
    "multipleCollaborators": "Collaborators",
}


@dataclass(frozen=True)
class ParsedAirtableTarget:
    raw_target: str
    app_id: str | None
    share_id: str | None
    page_id: str | None
    table_id: str | None
    view_id: str | None
    is_embed: bool
    is_form_route: bool

    @property
    def primary_id(self) -> str:
        return self.share_id or self.page_id or self.app_id or "unknown"

    @property
    def _path_segments(self) -> list[str]:
        segments: list[str] = []
        if self.is_embed:
            segments.append("embed")
        if self.app_id:
            segments.append(self.app_id)
        primary = self.share_id or self.page_id
        if primary:
            segments.append(primary)
        if self.table_id:
            segments.append(self.table_id)
        if self.view_id:
            segments.append(self.view_id)
        if self.is_form_route:
            segments.append("form")
        return segments

    @property
    def page_url(self) -> str:
        return "https://airtable.com/" + "/".join(self._path_segments)

    @property
    def canonical_url(self) -> str:
        return self.page_url

    @property
    def canonical_id(self) -> str:
        parts = [
            part
            for part in (self.app_id, self.share_id, self.page_id, self.table_id, self.view_id)
            if part
        ]
        identity = ":".join(parts) or self.raw_target
        if self.is_form_route:
            identity = f"{identity}:form"
        return identity


@dataclass(frozen=True)
class AirtableSettings:
    include_media: bool = True
    max_rows: int = 0
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True)
class AirtableResolvedDocument:
    label: str
    rendered: str
    prose: str
    source_ref: str
    source_path: str
    context_subpath: str
    kind: str
    canonical_id: str
    title: str | None
    media_count: int = 0


def _parse_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if not cleaned:
            return default
        return cleaned not in {"0", "false", "no", "off"}
    return default


def _parse_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return default
        try:
            return max(0, int(cleaned))
        except ValueError:
            return default
    return default


def _parse_timeout(value: Any, *, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(1.0, float(value))
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            try:
                return max(1.0, float(cleaned))
            except ValueError:
                return default
    return default


def build_airtable_settings(overrides: dict[str, Any] | None = None) -> AirtableSettings:
    import os

    overrides = overrides or {}
    return AirtableSettings(
        include_media=_parse_bool(
            overrides.get("include_media", os.environ.get("AIRTABLE_INCLUDE_MEDIA")),
            default=True,
        ),
        max_rows=_parse_int(
            overrides.get("max_rows", os.environ.get("AIRTABLE_MAX_ROWS")),
            default=0,
        ),
        timeout_seconds=_parse_timeout(
            overrides.get("timeout_seconds", os.environ.get("AIRTABLE_TIMEOUT")),
            default=_DEFAULT_TIMEOUT_SECONDS,
        ),
    )


def airtable_settings_cache_key(settings: AirtableSettings) -> str:
    payload = {
        "include_media": settings.include_media,
        "max_rows": settings.max_rows,
        "timeout_seconds": settings.timeout_seconds,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def parse_airtable_target(target: str) -> ParsedAirtableTarget | None:
    cleaned = target.strip()
    if not cleaned:
        return None

    parsed = urlparse(cleaned)
    scheme = parsed.scheme.lower()

    if scheme in {"http", "https"}:
        host = (parsed.hostname or "").lower()
        if host not in {"airtable.com", "www.airtable.com"}:
            return None
        segments = [segment for segment in parsed.path.split("/") if segment]
    elif scheme == "airtable":
        remainder = (parsed.netloc + parsed.path).strip("/")
        segments = [segment for segment in remainder.split("/") if segment]
    else:
        return None

    is_embed = bool(segments) and segments[0].lower() == "embed"
    if is_embed:
        segments = segments[1:]

    is_form_route = bool(segments) and segments[-1].lower() == "form"
    if is_form_route:
        segments = segments[:-1]

    app_id = share_id = page_id = table_id = view_id = None
    for segment in segments:
        if app_id is None and _APP_RE.match(segment):
            app_id = segment
        elif share_id is None and _SHARE_RE.match(segment):
            share_id = segment
        elif page_id is None and _PAGE_RE.match(segment):
            page_id = segment
        elif table_id is None and _TABLE_RE.match(segment):
            table_id = segment
        elif view_id is None and _VIEW_RE.match(segment):
            view_id = segment

    if share_id is None and page_id is None:
        return None

    return ParsedAirtableTarget(
        raw_target=target,
        app_id=app_id,
        share_id=share_id,
        page_id=page_id,
        table_id=table_id,
        view_id=view_id,
        is_embed=is_embed,
        is_form_route=is_form_route,
    )


def is_airtable_target(target: str) -> bool:
    return parse_airtable_target(target) is not None


# --------------------------------------------------------------------------- #
# HTTP                                                                         #
# --------------------------------------------------------------------------- #


def _js_unescape(value: str) -> str:
    return bytes(value, "utf-8").decode("unicode_escape")


def _api_headers(
    app_id: str | None,
    page_load_id: str,
    code_version: str | None,
    *,
    pages_context: str | None = None,
    referer: str | None = None,
) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "x-airtable-accept-msgpack": "false",
        "x-airtable-application-id": app_id or "",
        "x-airtable-page-load-id": page_load_id,
        "x-airtable-inter-service-client": "webClient",
        "x-time-zone": "UTC",
        "x-user-locale": "en",
        "x-requested-with": "XMLHttpRequest",
    }
    if code_version:
        headers["x-airtable-inter-service-client-code-version"] = code_version
    if pages_context:
        headers["x-airtable-pages-context"] = pages_context
    if referer:
        headers["Referer"] = referer
    return headers


def _fetch_page(session: Any, url: str, settings: AirtableSettings) -> tuple[str, str, str]:
    try:
        response = session.get(url, timeout=settings.timeout_seconds)
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"failed to load Airtable page {url}: {exc}") from exc

    text = response.text or ""
    page_load_id = _search(_PAGE_LOAD_ID_RE, text)
    if not page_load_id:
        raise RuntimeError(
            "Airtable page did not expose a session bootstrap; the link may be "
            "private, expired, or an unsupported page type"
        )
    code_version = _search(_CODE_VERSION_RE, text) or ""
    return text, page_load_id, code_version


def _get_json(
    session: Any,
    url: str,
    headers: dict[str, str],
    settings: AirtableSettings,
) -> dict[str, Any]:
    try:
        response = session.get(url, headers=headers, timeout=settings.timeout_seconds)
    except Exception as exc:
        raise RuntimeError(f"Airtable data request failed: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"Airtable data request returned {response.status_code}: "
            f"{_error_message(response)}"
        )
    try:
        payload = response.json()
    except Exception as exc:
        raise RuntimeError("Airtable data response was not JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Airtable data response had an unexpected shape")
    return payload


def _error_message(response: Any) -> str:
    try:
        payload = response.json()
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            return str(error.get("message") or error.get("type") or payload)
        if isinstance(payload, dict) and payload.get("errorMessage"):
            return str(payload["errorMessage"])
        return str(payload)[:200]
    except Exception:
        return (getattr(response, "text", "") or "")[:200]


def _object_params(page_id: str) -> dict[str, Any]:
    return {
        "includeDataForPageId": page_id,
        "shouldIncludeSchemaChecksum": True,
        "expectedPageLayoutSchemaVersion": 26,
        "shouldPreloadQueries": True,
        "shouldPreloadAllPossibleContainerElementQueries": False,
        "urlSearch": "",
        "includePageLayoutTypeInfo": True,
        "includeDataForExpandedRowPageFromQueryContainer": True,
        "includeDataForAllReferencedExpandedRowPagesInLayout": False,
        "navigationMode": "view",
        "allowMsgpackOfResultIfEnabled": False,
    }


def _request_id() -> str:
    return "req" + secrets.token_hex(8)


def _read_for_pages(
    session: Any,
    app_id: str,
    page_id: str,
    page_load_id: str,
    code_version: str,
    referer: str,
    settings: AirtableSettings,
    *,
    shared: bool,
    access_policy: str | None = None,
) -> dict[str, Any]:
    query = {
        "stringifiedObjectParams": json.dumps(
            _object_params(page_id), separators=(",", ":")
        ),
        "requestId": _request_id(),
    }
    if access_policy is not None:
        query["accessPolicy"] = access_policy
    endpoint = "readForSharedPages" if shared else "readForPages"
    url = f"https://airtable.com/v0.3/application/{app_id}/{endpoint}?{urlencode(query)}"
    headers = _api_headers(
        app_id,
        page_load_id,
        code_version,
        pages_context=f"view,{_SYNTHETIC_PAGE_BUNDLE_ID},{page_id}",
        referer=referer,
    )
    headers["x-airtable-integration-id"] = page_id
    return _get_json(session, url, headers, settings)


def _extract_prefetch_path(html_text: str) -> str | None:
    match = _PREFETCH_RE.search(html_text)
    if match is None:
        return None
    rel = _js_unescape(match.group(1))
    if "readShared" not in rel:
        return None
    return rel


def _extract_access_policy(html_text: str) -> str | None:
    match = _ACCESS_POLICY_RE.search(html_text)
    if match is None:
        return None
    try:
        policy = json.loads('"' + match.group(1) + '"')
    except Exception:
        return None
    if not isinstance(policy, str) or "allowedActions" not in policy:
        return None
    return policy


def _page_id_from_access_policy(policy: str) -> str | None:
    try:
        parsed = json.loads(policy)
    except Exception:
        return None
    actions = parsed.get("allowedActions") if isinstance(parsed, dict) else None
    if not isinstance(actions, list):
        return None
    for action in actions:
        if not isinstance(action, dict):
            continue
        if action.get("modelClassName") == "page":
            selector = action.get("modelIdSelector")
            if isinstance(selector, str) and _PAGE_RE.match(selector):
                return selector
    return None


# --------------------------------------------------------------------------- #
# Rendering                                                                    #
# --------------------------------------------------------------------------- #


def _clean_text(value: str, *, preserve_newlines: bool = False) -> str:
    cleaned = html.unescape(value).replace("\xa0", " ")
    if preserve_newlines:
        cleaned = re.sub(r"[^\S\n]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    else:
        cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


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


def _format_number(value: float) -> str:
    if isinstance(value, bool):
        return "Yes" if value else ""
    if isinstance(value, int):
        return str(value)
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


def _render_value(value: Any, choices: dict[str, Any] | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "Yes" if value else ""
    if isinstance(value, (int, float)):
        return _format_number(value)
    if isinstance(value, str):
        if choices and value in choices:
            choice = choices[value]
            if isinstance(choice, dict):
                return str(choice.get("name") or value).strip()
        return value.strip()
    if isinstance(value, list):
        parts = [_render_value(item, choices) for item in value]
        return ", ".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in _VALUE_KEYS:
            candidate = value.get(key)
            if isinstance(candidate, (str, int, float)) and str(candidate).strip():
                return str(candidate).strip()
        return ""
    return str(value)


def _choices_map(column: dict[str, Any]) -> dict[str, Any] | None:
    type_options = column.get("typeOptions")
    if isinstance(type_options, dict):
        choices = type_options.get("choices")
        if isinstance(choices, dict):
            return choices
    return None


def _is_attachment(column: dict[str, Any], value: Any) -> bool:
    if column.get("type") in _ATTACHMENT_TYPES:
        return True
    return (
        isinstance(value, list)
        and bool(value)
        and isinstance(value[0], dict)
        and "url" in value[0]
        and ("filename" in value[0] or "type" in value[0])
    )


def _attachment_kind(attachment: dict[str, Any]) -> str:
    mime = str(attachment.get("type") or "").lower()
    if mime.startswith("video"):
        return "video"
    if mime.startswith("image"):
        return "image"
    filename = str(attachment.get("filename") or "")
    suffix = filename[filename.rfind(".") :].lower() if "." in filename else ""
    return "video" if suffix in _VIDEO_SUFFIXES else "image"


def _attachment_tag(attachment: dict[str, Any], caption: str | None) -> str:
    kind = _attachment_kind(attachment)
    attrs: list[str] = []
    filename = attachment.get("filename")
    url = attachment.get("url")
    if isinstance(filename, str) and filename.strip():
        attrs.append(f'filename="{_escape_xml_attr(filename.strip())}"')
    if isinstance(url, str) and url.strip():
        attrs.append(f'url="{_escape_xml_attr(url.strip())}"')
    if caption:
        attrs.append(f'caption="{_escape_xml_attr(caption)}"')
    suffix = f" {' '.join(attrs)}" if attrs else ""
    return f"<{kind}{suffix} />"


@dataclass
class _Record:
    title: str
    lines: list[str]
    media: list[str]
    prose: str
    media_count: int


def _format_record(
    columns: list[dict[str, Any]],
    cells: dict[str, Any],
    primary_column_id: str | None,
    settings: AirtableSettings,
    *,
    index: int,
) -> _Record:
    lines: list[str] = []
    media: list[str] = []
    prose_parts: list[str] = []
    media_count = 0
    title = ""

    for column in columns:
        column_id = column.get("id")
        if column_id is None or column_id not in cells:
            continue
        name = str(column.get("name") or column_id)
        value = cells[column_id]

        if _is_attachment(column, value):
            attachments = [item for item in (value or []) if isinstance(item, dict)]
            filenames = [
                str(item.get("filename") or "").strip()
                for item in attachments
                if str(item.get("filename") or "").strip()
            ]
            if settings.include_media:
                for attachment in attachments:
                    media.append(_attachment_tag(attachment, name))
                    media_count += 1
            if filenames:
                prose_parts.append(f"{name}: {', '.join(filenames)}")
                if not settings.include_media:
                    lines.append(f"- **{name}**: {', '.join(filenames)}")
            continue

        text = _render_value(value, _choices_map(column))
        if not text:
            continue
        inline = _clean_text(text)
        lines.append(f"- **{name}**: {inline}")
        prose_parts.append(f"{name}: {inline}")
        if column_id == primary_column_id and not title:
            title = inline

    if not title:
        title = f"Record {index}"

    prose = title
    body_parts = [part for part in prose_parts if not part.startswith(f"{title}:")]
    if body_parts:
        prose = title + "\n" + "\n".join(body_parts)
    return _Record(
        title=title,
        lines=lines,
        media=media,
        prose=prose,
        media_count=media_count,
    )


def _frontmatter(parsed: ParsedAirtableTarget, fields: list[tuple[str, str]]) -> list[str]:
    lines = ["---", f"url: {_escape_yaml_string(parsed.canonical_url)}"]
    for key, value in fields:
        if value:
            lines.append(f"{key}: {_escape_yaml_string(value)}")
    lines.append("---")
    return lines


def _render_table(
    *,
    columns: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    primary_column_id: str | None,
    settings: AirtableSettings,
    heading_level: int,
) -> tuple[list[str], list[str], int, int, int]:
    total = len(rows)
    limit = settings.max_rows
    shown = rows if limit <= 0 else rows[:limit]

    lines: list[str] = []
    prose_blocks: list[str] = []
    media_count = 0

    for index, row in enumerate(shown, start=1):
        cells = row.get("cellValuesByColumnId")
        if not isinstance(cells, dict):
            continue
        record = _format_record(
            columns, cells, primary_column_id, settings, index=index
        )
        media_count += record.media_count
        lines.append("")
        lines.append(f"{'#' * heading_level} {record.title}")
        if record.lines:
            lines.append("")
            lines.extend(record.lines)
        if record.media:
            lines.append("")
            lines.extend(record.media)
        prose_blocks.append(record.prose)

    return lines, prose_blocks, media_count, total, len(shown)


def _primary_column_id(table: dict[str, Any], columns: list[dict[str, Any]]) -> str | None:
    primary = table.get("primaryColumnId")
    if isinstance(primary, str) and primary:
        return primary
    if columns:
        first = columns[0].get("id")
        return first if isinstance(first, str) else None
    return None


def _build_document(
    parsed: ParsedAirtableTarget,
    *,
    kind: str,
    title: str,
    rendered_body: list[str],
    prose: str,
    media_count: int,
) -> AirtableResolvedDocument:
    safe_title = _safe_path_segment(title, fallback=parsed.primary_id)
    rendered = "\n".join(rendered_body).rstrip() + "\n"
    return AirtableResolvedDocument(
        label=f"airtable/{safe_title}",
        rendered=rendered,
        prose=prose.strip(),
        source_ref="airtable.com",
        source_path=parsed.canonical_id,
        context_subpath=f"airtable/{safe_title}.md",
        kind=kind,
        canonical_id=parsed.canonical_id,
        title=title or None,
        media_count=media_count,
    )


def _render_grid_document(
    parsed: ParsedAirtableTarget,
    page_title: str | None,
    data: dict[str, Any],
    settings: AirtableSettings,
    app_id: str | None,
) -> AirtableResolvedDocument:
    table = data.get("table")
    if not isinstance(table, dict):
        raise RuntimeError("Airtable shared view payload did not contain a table")
    columns = [c for c in (table.get("columns") or []) if isinstance(c, dict)]
    rows = [r for r in (table.get("rows") or []) if isinstance(r, dict)]
    title = page_title or _clean_text(str(table.get("name") or "")) or parsed.primary_id
    primary_column_id = _primary_column_id(table, columns)

    body_lines, prose_blocks, media_count, total, shown = _render_table(
        columns=columns,
        rows=rows,
        primary_column_id=primary_column_id,
        settings=settings,
        heading_level=2,
    )

    fields = [("app_id", app_id or ""), ("records", str(total))]
    if shown < total:
        fields.append(("records_shown", str(shown)))
    lines = _frontmatter(parsed, fields)
    lines.extend(["", f"# {title}"])
    if not rows:
        lines.extend(["", "_This shared view returned no records._"])
    lines.extend(body_lines)
    if shown < total:
        lines.extend(
            ["", f"_Showing {shown} of {total} records (truncated by max_rows)._"]
        )

    return _build_document(
        parsed,
        kind="table",
        title=title,
        rendered_body=lines,
        prose="\n\n".join(prose_blocks),
        media_count=media_count,
    )


def _render_form_document(
    parsed: ParsedAirtableTarget,
    page_title: str | None,
    data: dict[str, Any],
    app_id: str | None,
) -> AirtableResolvedDocument:
    form_table = data.get("formTable")
    if not isinstance(form_table, dict):
        raise RuntimeError("Airtable form payload did not contain a form table")
    columns = {
        column["id"]: column
        for column in (form_table.get("columns") or [])
        if isinstance(column, dict) and isinstance(column.get("id"), str)
    }
    view = _form_view(form_table)
    raw_meta = view.get("metadata")
    form_meta = raw_meta.get("form") if isinstance(raw_meta, dict) else None
    form_meta = form_meta if isinstance(form_meta, dict) else {}
    field_config = form_meta.get("fieldsByColumnId")
    field_config = field_config if isinstance(field_config, dict) else {}
    column_order = view.get("columnOrder")
    column_order = column_order if isinstance(column_order, list) else []

    title = (
        page_title
        or _clean_text(str(view.get("name") or ""))
        or _clean_text(str(form_table.get("name") or ""))
        or parsed.primary_id
    )
    description = _text_or_quill(form_meta.get("description"))

    items: list[dict[str, Any]] = []
    for entry in column_order:
        if not isinstance(entry, dict) or entry.get("visibility") is False:
            continue
        column_id = entry.get("columnId")
        if not isinstance(column_id, str):
            continue
        column = columns.get(column_id) or {}
        config = field_config.get(column_id)
        config = config if isinstance(config, dict) else {}
        items.append(
            {
                "kind": "field",
                "question": _clean_text(
                    str(config.get("title") or column.get("name") or column_id)
                ),
                "required": bool(config.get("required")),
                "help": _text_or_quill(config.get("description")),
                "column": column,
            }
        )

    if not items:
        for column in form_table.get("columns") or []:
            if not isinstance(column, dict):
                continue
            name = _clean_text(str(column.get("name") or column.get("id") or ""))
            if not name:
                continue
            items.append(
                {"kind": "field", "question": name, "required": False, "help": "", "column": column}
            )

    footer = _text_or_quill(form_meta.get("afterSubmitMessage"))
    return _render_form_markdown(parsed, app_id, title, description, items, footer=footer)


def _quill_to_markdown(delta: Any) -> str:
    if not isinstance(delta, list):
        return ""
    parts: list[str] = []
    for op in delta:
        if not isinstance(op, dict):
            continue
        text = op.get("insert")
        if not isinstance(text, str):
            continue
        raw_attrs = op.get("attributes")
        attrs = raw_attrs if isinstance(raw_attrs, dict) else {}
        link = attrs.get("link")
        if isinstance(link, str) and link.strip() and text.strip():
            lead = text[: len(text) - len(text.lstrip())]
            trail = text[len(text.rstrip()) :]
            text = f"{lead}[{text.strip()}]({link.strip()}){trail}"
        parts.append(text)
    return _clean_text("".join(parts), preserve_newlines=True)


def _text_or_quill(value: Any) -> str:
    if isinstance(value, list):
        return _quill_to_markdown(value)
    if isinstance(value, str):
        return _clean_text(value)
    return ""


def _form_view(form_table: dict[str, Any]) -> dict[str, Any]:
    views = form_table.get("views")
    if isinstance(views, list):
        for view in views:
            if isinstance(view, dict) and view.get("type") == "form":
                return view
        for view in views:
            if isinstance(view, dict):
                return view
    return {}


def _columns_by_id(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    columns: dict[str, dict[str, Any]] = {}
    for schema in data.get("tableSchemas") or []:
        if not isinstance(schema, dict):
            continue
        for column in schema.get("columns") or []:
            if isinstance(column, dict) and isinstance(column.get("id"), str):
                columns[column["id"]] = column
    return columns


def _form_container(elements: dict[str, Any]) -> dict[str, Any] | None:
    for element in elements.values():
        if isinstance(element, dict) and element.get("type") == "formContainer":
            return element
    return None


def _page_layout(data: dict[str, Any], page_id: str | None) -> dict[str, Any] | None:
    pages = data.get("pages")
    if not isinstance(pages, list):
        return None
    chosen: dict[str, Any] | None = None
    for page in pages:
        if isinstance(page, dict) and page.get("id") == page_id:
            chosen = page
            break
    if chosen is None:
        chosen = next((p for p in pages if isinstance(p, dict)), None)
    if chosen is None:
        return None
    layout = chosen.get("publishedLayout")
    return layout if isinstance(layout, dict) else None


_MULTI_SELECT_TYPES = {"multiSelect", "multipleSelects"}


def _pretty_type(raw_type: Any) -> str:
    raw = str(raw_type or "").strip()
    return _PRETTY_FIELD_TYPES.get(raw, raw)


def _choice_names(column: dict[str, Any]) -> list[str]:
    choices = _choices_map(column)
    if not choices:
        return []
    type_options = column.get("typeOptions")
    order = type_options.get("choiceOrder") if isinstance(type_options, dict) else None
    choice_ids = order if isinstance(order, list) else list(choices.keys())
    names: list[str] = []
    for choice_id in choice_ids:
        choice = choices.get(choice_id)
        if isinstance(choice, dict):
            name = str(choice.get("name") or "").strip()
            if name:
                names.append(name)
    return names


def _field_value_lines(column: dict[str, Any]) -> list[str]:
    names = _choice_names(column)
    if names:
        raw_type = str(column.get("type") or "")
        label = "Multiple select" if raw_type in _MULTI_SELECT_TYPES else "Single select"
        return [f"*{label}:*", *(f"- [ ] {name}" for name in names)]
    pretty = _pretty_type(column.get("type"))
    return [f"*{pretty}*"] if pretty else []


def _render_form_markdown(
    parsed: ParsedAirtableTarget,
    app_id: str | None,
    title: str,
    description: str,
    items: list[dict[str, Any]],
    footer: str = "",
) -> AirtableResolvedDocument:
    lines = _frontmatter(parsed, [("app_id", app_id or ""), ("type", "form")])
    lines.extend(["", f"# {title}"])
    prose_parts: list[str] = [title]
    if description:
        lines.extend(["", description])
        prose_parts.append(description)

    previous = "head"
    for item in items:
        if item.get("kind") == "section":
            section_title = str(item.get("title") or "")
            lines.extend(["", "", f"## {section_title}"])
            prose_parts.append(section_title)
            previous = "section"
            continue
        lines.extend([""] if previous == "section" else ["", ""])
        question = str(item.get("question") or "")
        heading = f"### {question}"
        if item.get("required"):
            heading += " *(required)*"
        lines.append(heading)
        help_text = str(item.get("help") or "")
        if help_text:
            lines.append(f"_{' '.join(help_text.split())}_")
        lines.append("")
        column = item.get("column")
        lines.extend(_field_value_lines(column if isinstance(column, dict) else {}))
        prose_parts.append(f"{question} — {help_text}" if help_text else question)
        previous = "field"

    if footer:
        lines.extend(["", "", "---", "", f"*After submitting:* {footer}"])

    return _build_document(
        parsed,
        kind="form",
        title=title,
        rendered_body=lines,
        prose="\n\n".join(prose_parts),
        media_count=0,
    )


def _interface_field_item(
    element: dict[str, Any],
    columns: dict[str, dict[str, Any]],
    required: set[str],
) -> dict[str, Any]:
    source = element.get("source")
    column_id = source.get("columnId") if isinstance(source, dict) else None
    column = columns.get(column_id) if isinstance(column_id, str) else None
    column = column if isinstance(column, dict) else {}

    label = element.get("label")
    question = ""
    if isinstance(label, dict) and label.get("isEnabled") and isinstance(label.get("value"), str):
        question = _clean_text(label["value"])
    if not question:
        question = _clean_text(str(column.get("name") or column_id or "Field"))

    return {
        "kind": "field",
        "question": question,
        "required": isinstance(column_id, str) and column_id in required,
        "help": _quill_to_markdown(element.get("description")),
        "column": column,
    }


def _render_interface_form(
    parsed: ParsedAirtableTarget,
    page_title: str | None,
    data: dict[str, Any],
    layout: dict[str, Any],
    app_id: str | None,
) -> AirtableResolvedDocument:
    elements = layout.get("elementById")
    elements = elements if isinstance(elements, dict) else {}
    slots = layout.get("slotElementsById")
    slots = slots if isinstance(slots, dict) else {}

    children: dict[str, list[tuple[str, str]]] = {}
    for slot in slots.values():
        if not isinstance(slot, dict):
            continue
        parent = slot.get("parentId")
        child = slot.get("elementId")
        if isinstance(parent, str) and isinstance(child, str):
            children.setdefault(parent, []).append((str(slot.get("index") or ""), child))
    for ordered in children.values():
        ordered.sort(key=lambda item: item[0])

    columns = _columns_by_id(data)
    container = _form_container(elements)
    required: set[str] = set()
    description = ""
    if container is not None:
        raw_required = container.get("requiredColumnIds")
        if isinstance(raw_required, list):
            required = {c for c in raw_required if isinstance(c, str)}
        description = _quill_to_markdown(container.get("description"))

    title = page_title or _interface_title(data, parsed.primary_id)

    items: list[dict[str, Any]] = []
    visited: set[str] = set()

    def walk(element_id: str) -> None:
        if element_id in visited:
            return
        visited.add(element_id)
        element = elements.get(element_id)
        if isinstance(element, dict):
            element_type = element.get("type")
            if element_type == "section":
                section_title = _clean_text(str(element.get("title") or ""))
                if section_title and element.get("shouldDisplayTitle", True):
                    items.append({"kind": "section", "title": section_title})
            elif element_type == "cellEditor":
                items.append(_interface_field_item(element, columns, required))
        for _, child in children.get(element_id, []):
            walk(child)

    root_id = container.get("id") if container is not None else None
    if isinstance(root_id, str):
        for _, child in children.get(root_id, []):
            walk(child)

    return _render_form_markdown(parsed, app_id, title, description, items)


def _render_interface_document(
    parsed: ParsedAirtableTarget,
    page_title: str | None,
    data: dict[str, Any],
    settings: AirtableSettings,
    app_id: str | None,
    page_id: str | None,
) -> AirtableResolvedDocument:
    layout = _page_layout(data, page_id)
    if layout is not None and _form_container(layout.get("elementById") or {}) is not None:
        return _render_interface_form(parsed, page_title, data, layout, app_id)

    schemas = [s for s in (data.get("tableSchemas") or []) if isinstance(s, dict)]
    datas = {
        d.get("id"): d
        for d in (data.get("tableDatas") or [])
        if isinstance(d, dict)
    }
    title = page_title or _interface_title(data, parsed.primary_id)

    lines = _frontmatter(parsed, [("app_id", app_id or ""), ("type", "interface")])
    lines.extend(["", f"# {title}"])
    prose_blocks: list[str] = []
    media_count = 0
    rendered_any = False

    for schema in schemas:
        columns = [c for c in (schema.get("columns") or []) if isinstance(c, dict)]
        if not columns:
            continue
        table_data = datas.get(schema.get("id")) or {}
        rows = [r for r in (table_data.get("rows") or []) if isinstance(r, dict)]
        if not rows:
            continue
        table_name = _clean_text(str(schema.get("name") or ""))
        primary_column_id = _primary_column_id(schema, columns)
        if table_name:
            lines.extend(["", f"## {table_name}"])
        body_lines, blocks, mc, total, shown = _render_table(
            columns=columns,
            rows=rows,
            primary_column_id=primary_column_id,
            settings=settings,
            heading_level=3,
        )
        lines.extend(body_lines)
        if shown < total:
            lines.extend(
                ["", f"_Showing {shown} of {total} records (truncated by max_rows)._"]
            )
        prose_blocks.extend(blocks)
        media_count += mc
        rendered_any = True

    if not rendered_any:
        lines.extend(
            [
                "",
                "_Interface record data is not included (it loads per page element); "
                "showing the underlying table field schema._",
            ]
        )
        lines.extend(_interface_schema_lines(schemas, prose_blocks))

    return _build_document(
        parsed,
        kind="interface",
        title=title,
        rendered_body=lines,
        prose="\n\n".join(prose_blocks),
        media_count=media_count,
    )


def _interface_schema_lines(
    schemas: list[dict[str, Any]],
    prose_blocks: list[str],
) -> list[str]:
    lines: list[str] = []
    for schema in schemas:
        columns = [c for c in (schema.get("columns") or []) if isinstance(c, dict)]
        if not columns:
            continue
        table_name = _clean_text(str(schema.get("name") or "")) or "Fields"
        lines.extend(["", f"## {table_name}", ""])
        field_names: list[str] = []
        for column in columns:
            name = _clean_text(str(column.get("name") or ""))
            if not name:
                continue
            column_type = str(column.get("type") or "").strip()
            lines.append(f"- **{name}**" + (f" ({column_type})" if column_type else ""))
            field_names.append(name)
        if field_names:
            prose_blocks.append(f"{table_name}: {', '.join(field_names)}")
    return lines


def _page_title(html_text: str) -> str | None:
    match = _OG_TITLE_RE.search(html_text)
    raw = match.group(1) if match else None
    if not raw:
        match = _HTML_TITLE_RE.search(html_text)
        raw = re.sub(r"<[^>]+>", " ", match.group(1)) if match else None
    if not raw:
        return None
    title = _clean_text(raw)
    for suffix in (" - Airtable", " | Airtable"):
        if title.endswith(suffix):
            title = title[: -len(suffix)].strip()
    lowered = title.lower()
    if not title or lowered == "airtable" or "everyone's app platform" in lowered:
        return None
    return title


def _interface_title(data: dict[str, Any], fallback: str) -> str:
    blob = json.dumps(data)
    match = re.search(r'"title":\s*"([^"]{1,80})"\s*,\s*"formCoverImageUrl"', blob)
    if match:
        cleaned = _clean_text(match.group(1))
        if cleaned:
            return cleaned
    bundles = data.get("pageBundles")
    if isinstance(bundles, list):
        for bundle in bundles:
            if isinstance(bundle, dict):
                name = bundle.get("name")
                if isinstance(name, str) and name.strip():
                    return _clean_text(name)
    return fallback


def _search(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    return match.group(1) if match else ""


# --------------------------------------------------------------------------- #
# Resolution                                                                   #
# --------------------------------------------------------------------------- #


def _resolve_live(
    parsed: ParsedAirtableTarget,
    settings: AirtableSettings,
) -> AirtableResolvedDocument:
    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})

    html_text, page_load_id, code_version = _fetch_page(
        session, parsed.page_url, settings
    )
    app_id = parsed.app_id or _search(_APP_ID_RE, html_text) or None
    title = _page_title(html_text)

    prefetch = _extract_prefetch_path(html_text)
    if prefetch is not None:
        payload = _get_json(
            session,
            "https://airtable.com" + prefetch,
            _api_headers(app_id, page_load_id, code_version, referer=parsed.page_url),
            settings,
        )
        data = payload.get("data") if isinstance(payload, dict) else None
        data = data if isinstance(data, dict) else {}
        if isinstance(data.get("table"), dict):
            return _render_grid_document(parsed, title, data, settings, app_id)
        if isinstance(data.get("formTable"), dict):
            return _render_form_document(parsed, title, data, app_id)

    access_policy = _extract_access_policy(html_text)
    if access_policy is not None and app_id:
        page_id = parsed.page_id or _page_id_from_access_policy(access_policy)
        if page_id:
            payload = _read_for_pages(
                session,
                app_id,
                page_id,
                page_load_id,
                code_version,
                parsed.page_url,
                settings,
                shared=True,
                access_policy=access_policy,
            )
            data = payload.get("data") if isinstance(payload, dict) else None
            return _render_interface_document(
                parsed,
                title,
                data if isinstance(data, dict) else {},
                settings,
                app_id,
                page_id,
            )

    if parsed.page_id and app_id:
        payload = _read_for_pages(
            session,
            app_id,
            parsed.page_id,
            page_load_id,
            code_version,
            parsed.page_url,
            settings,
            shared=False,
        )
        data = payload.get("data") if isinstance(payload, dict) else None
        return _render_interface_document(
            parsed,
            title,
            data if isinstance(data, dict) else {},
            settings,
            app_id,
            parsed.page_id,
        )

    raise RuntimeError(
        "could not locate a public Airtable data endpoint for this link; it may be "
        "private, expired, or an unsupported page type"
    )


def _as_timedelta(cache_ttl: Any) -> timedelta | None:
    if cache_ttl is None:
        return None
    if isinstance(cache_ttl, timedelta):
        return cache_ttl
    if isinstance(cache_ttl, bool):
        return None
    if isinstance(cache_ttl, (int, float)):
        return timedelta(seconds=float(cache_ttl))
    if isinstance(cache_ttl, str):
        try:
            return timedelta(seconds=float(cache_ttl.strip()))
        except ValueError:
            return None
    return None


def _document_to_payload(document: AirtableResolvedDocument) -> dict[str, Any]:
    return {
        "label": document.label,
        "rendered": document.rendered,
        "prose": document.prose,
        "source_ref": document.source_ref,
        "source_path": document.source_path,
        "context_subpath": document.context_subpath,
        "kind": document.kind,
        "canonical_id": document.canonical_id,
        "title": document.title,
        "media_count": document.media_count,
    }


def _document_from_payload(payload: Any) -> AirtableResolvedDocument | None:
    if not isinstance(payload, dict):
        return None
    try:
        return AirtableResolvedDocument(
            label=str(payload["label"]),
            rendered=str(payload["rendered"]),
            prose=str(payload.get("prose") or ""),
            source_ref=str(payload["source_ref"]),
            source_path=str(payload["source_path"]),
            context_subpath=str(payload["context_subpath"]),
            kind=str(payload["kind"]),
            canonical_id=str(payload["canonical_id"]),
            title=(str(payload["title"]) if payload.get("title") else None),
            media_count=int(payload.get("media_count") or 0),
        )
    except Exception:
        return None


def resolve_airtable(
    target: str,
    *,
    settings: AirtableSettings,
    use_cache: bool,
    cache_ttl: Any,
    refresh_cache: bool,
) -> AirtableResolvedDocument:
    parsed = parse_airtable_target(target)
    if parsed is None:
        raise ValueError(f"not an Airtable target: {target}")

    settings_key = airtable_settings_cache_key(settings)
    identity = f"{parsed.canonical_id}|{settings_key}"
    root = provider_cache_root(_CACHE_ENV, _PROVIDER)

    if use_cache and not refresh_cache:
        entry = read_json_entry(root, identity, ttl=_as_timedelta(cache_ttl))
        if entry is not None:
            cached = _document_from_payload(entry.value)
            if cached is not None:
                return cached

    document = _resolve_live(parsed, settings)

    if use_cache:
        write_json_entry(
            root,
            identity,
            _document_to_payload(document),
            identity_field="canonical_id",
            extra_metadata={"provider": _PROVIDER},
        )
    return document
