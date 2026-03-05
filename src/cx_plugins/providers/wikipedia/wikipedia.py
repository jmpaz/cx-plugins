from __future__ import annotations

from dataclasses import dataclass
import html
import re
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

_DEFAULT_LANGUAGE = "en"
_DEFAULT_TIMEOUT_SECONDS = 20
_USER_AGENT = "contextualize/wikipedia"
_WIKI_HOST_RE = re.compile(r"^(?P<lang>[a-z]{2,12})(?:\.m)?\.wikipedia\.org$")
_WIKI_PATH_RE = re.compile(r"^/wiki/(?P<title>[^#?]+)")
_LANG_PREFIX_RE = re.compile(r"^[a-z]{2,12}$")
_TAG_RE = re.compile(r"<[^>]+>")
_STRIP_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"[\u2018\u2019]"), "'"),
    (re.compile(r"[\u201C\u201D]"), '"'),
    (re.compile(r"\u2013"), "-"),
    (re.compile(r"\u2014"), "--"),
    (re.compile(r"\u2026"), "..."),
    (re.compile(r"\[\s*\d+\s*\]"), ""),
    (re.compile(r"\[\s*citation needed\s*\]", flags=re.IGNORECASE), ""),
    (re.compile(r"\[\s*edit\s*\]", flags=re.IGNORECASE), ""),
    (re.compile(r"\[\s*note\s+\d+\s*\]", flags=re.IGNORECASE), ""),
)
_META_SECTION_TITLES = frozenset(
    {
        "see also",
        "notes",
        "references",
        "further reading",
        "external links",
        "bibliography",
    }
)
_EXCLUDED_CLASS_FRAGMENTS = (
    "infobox",
    "sidebar",
    "navbox",
    "reflist",
    "reference",
    "gallery",
    "metadata",
)


@dataclass(frozen=True)
class ParsedWikipediaTarget:
    raw_target: str
    language: str
    title: str
    revision_id: int | None

    @property
    def canonical_title(self) -> str:
        return self.title.replace(" ", "_")

    @property
    def canonical_id(self) -> str:
        base = f"{self.language}:{self.canonical_title}"
        if self.revision_id is None:
            return base
        return f"{base}@oldid={self.revision_id}"


@dataclass(frozen=True)
class WikipediaSettings:
    default_lang: str = _DEFAULT_LANGUAGE
    include_media: bool = True
    include_media_descriptions: bool = True
    include_references: bool = True
    include_external_links: bool = True
    include_categories: bool = True


@dataclass(frozen=True)
class WikipediaSection:
    index: int
    level: int
    title: str
    content: str


@dataclass(frozen=True)
class WikipediaMedia:
    url: str
    caption: str | None
    description: str | None
    width: int
    height: int
    section_index: int | None


@dataclass(frozen=True)
class WikipediaResolvedDocument:
    label: str
    rendered: str
    source_ref: str
    source_path: str
    context_subpath: str
    kind: str
    canonical_id: str


@dataclass(frozen=True)
class _ExtractedArticle:
    sections: tuple[WikipediaSection, ...]
    references: tuple[str, ...]
    external_links: tuple[tuple[str, str], ...]


def _clean_text(value: str, *, preserve_newlines: bool = False) -> str:
    cleaned = html.unescape(value).replace("\xa0", " ")
    for pattern, replacement in _STRIP_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    if preserve_newlines:
        cleaned = re.sub(r"[^\S\n]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    else:
        cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _parse_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if not cleaned:
            return default
        return cleaned not in {"0", "false", "no", "off"}
    return default


def _parse_lang(value: Any, *, default: str) -> str:
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if _LANG_PREFIX_RE.fullmatch(cleaned):
            return cleaned
    return default


def _normalize_revision(value: str | None) -> int | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if not cleaned.isdigit():
        return None
    parsed = int(cleaned)
    return parsed if parsed > 0 else None


def _decode_title(value: str) -> str:
    return _clean_text(unquote(value).replace("_", " "))


def _safe_path_segment(value: str, *, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
    return cleaned or fallback


def _escape_xml_attr(value: str) -> str:
    return html.escape(value, quote=True)


def _strip_html_tags(text: str) -> str:
    return _clean_text(_TAG_RE.sub(" ", text))


def _target_from_http_url(target: str) -> ParsedWikipediaTarget | None:
    parsed = urlparse(target)
    if parsed.scheme not in {"http", "https"}:
        return None
    host = parsed.netloc.lower()
    host_match = _WIKI_HOST_RE.fullmatch(host)
    if host_match is None:
        return None
    path_match = _WIKI_PATH_RE.match(parsed.path)
    if path_match is None:
        return None

    title = _decode_title(path_match.group("title"))
    if not title:
        return None

    query = parse_qs(parsed.query)
    revision_id = _normalize_revision((query.get("oldid") or [None])[0])
    return ParsedWikipediaTarget(
        raw_target=target,
        language=host_match.group("lang"),
        title=title,
        revision_id=revision_id,
    )


def _target_from_wikipedia_scheme(
    target: str,
    *,
    default_lang: str,
) -> ParsedWikipediaTarget | None:
    parsed = urlparse(target)
    if parsed.scheme.lower() != "wikipedia":
        return None

    language = _parse_lang(parsed.netloc, default=default_lang)
    title_path = parsed.path.lstrip("/")
    if title_path.startswith("wiki/"):
        title_path = title_path[5:]
    title = _decode_title(title_path)
    if not title:
        return None

    query = parse_qs(parsed.query)
    revision_id = _normalize_revision((query.get("oldid") or [None])[0])
    return ParsedWikipediaTarget(
        raw_target=target,
        language=language,
        title=title,
        revision_id=revision_id,
    )


def _target_from_wiki_scheme(
    target: str,
    *,
    default_lang: str,
) -> ParsedWikipediaTarget | None:
    if not target.lower().startswith("wiki:"):
        return None

    remainder = target.split(":", 1)[1].strip()
    if not remainder:
        return None

    if "?" in remainder:
        path_part, query_part = remainder.split("?", 1)
        query = parse_qs(query_part)
    else:
        path_part = remainder
        query = {}

    language = default_lang
    title_part = path_part.strip().lstrip("/")
    if "/" in title_part:
        maybe_lang, maybe_title = title_part.split("/", 1)
        maybe_lang = maybe_lang.strip().lower()
        if _LANG_PREFIX_RE.fullmatch(maybe_lang) and maybe_title.strip():
            language = maybe_lang
            title_part = maybe_title

    title = _decode_title(title_part)
    if not title:
        return None

    revision_id = _normalize_revision((query.get("oldid") or [None])[0])
    return ParsedWikipediaTarget(
        raw_target=target,
        language=language,
        title=title,
        revision_id=revision_id,
    )


def parse_wikipedia_target(
    target: str,
    *,
    default_lang: str = _DEFAULT_LANGUAGE,
) -> ParsedWikipediaTarget | None:
    cleaned = target.strip()
    if not cleaned:
        return None

    parsed = _target_from_http_url(cleaned)
    if parsed is not None:
        return parsed

    parsed = _target_from_wikipedia_scheme(cleaned, default_lang=default_lang)
    if parsed is not None:
        return parsed

    parsed = _target_from_wiki_scheme(cleaned, default_lang=default_lang)
    if parsed is not None:
        return parsed

    return None


def is_wikipedia_target(target: str, *, default_lang: str = _DEFAULT_LANGUAGE) -> bool:
    return parse_wikipedia_target(target, default_lang=default_lang) is not None


def _http_get(url: str, *, timeout: int, headers: dict[str, str]) -> Any:
    import requests

    response = requests.get(url, timeout=timeout, headers=headers)
    response.raise_for_status()
    return response


def _wiki_api_request(
    *,
    base_url: str,
    params: dict[str, str],
    timeout_seconds: int,
) -> dict[str, Any]:
    query = "&".join(f"{quote(k)}={quote(v)}" for k, v in params.items())
    url = f"{base_url}/w/api.php?{query}"
    response = _http_get(
        url,
        timeout=timeout_seconds,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
    )
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Wikipedia API returned an invalid payload")
    return data


def _resolve_parse_payload(
    parsed: ParsedWikipediaTarget,
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    base_url = f"https://{parsed.language}.wikipedia.org"
    params: dict[str, str] = {
        "action": "parse",
        "prop": "text|categories|displaytitle",
        "format": "json",
        "disabletoc": "true",
        "disableeditsection": "true",
    }
    if parsed.revision_id is not None:
        params["oldid"] = str(parsed.revision_id)
    else:
        params["page"] = parsed.title

    payload = _wiki_api_request(
        base_url=base_url,
        params=params,
        timeout_seconds=timeout_seconds,
    )
    error = payload.get("error")
    if isinstance(error, dict):
        info = error.get("info")
        if isinstance(info, str) and info.strip():
            raise ValueError(f"Wikipedia API error: {info}")
        raise ValueError("Wikipedia API returned an error")

    parse_obj = payload.get("parse")
    if not isinstance(parse_obj, dict):
        raise ValueError("Wikipedia parse payload is missing")
    return parse_obj


def _resolve_intro(
    parsed: ParsedWikipediaTarget,
    *,
    timeout_seconds: int,
) -> str:
    base_url = f"https://{parsed.language}.wikipedia.org"
    params: dict[str, str] = {
        "action": "query",
        "prop": "extracts",
        "exintro": "1",
        "format": "json",
        "formatversion": "2",
    }
    if parsed.revision_id is not None:
        params["revids"] = str(parsed.revision_id)
    else:
        params["titles"] = parsed.title

    payload = _wiki_api_request(
        base_url=base_url,
        params=params,
        timeout_seconds=timeout_seconds,
    )
    query = payload.get("query")
    if not isinstance(query, dict):
        return ""
    pages = query.get("pages")
    if not isinstance(pages, list) or not pages:
        return ""
    page = pages[0]
    if not isinstance(page, dict):
        return ""
    extract = page.get("extract")
    if not isinstance(extract, str) or not extract.strip():
        return ""
    return _strip_html_tags(extract)


def _node_tag(node: Any) -> str:
    return str(getattr(node, "tag", "")).lower()


def _node_attr(node: Any, key: str) -> str:
    attrs = getattr(node, "attributes", {})
    if not isinstance(attrs, dict):
        return ""
    value = attrs.get(key)
    return str(value).strip() if isinstance(value, str) else ""


def _iter_children(node: Any):
    child = getattr(node, "child", None)
    while child is not None:
        yield child
        child = getattr(child, "next", None)


def _render_inline_nodes(node: Any) -> str:
    return "".join(_render_inline_node(child) for child in _iter_children(node))


def _render_link_node(node: Any) -> str:
    href = _node_attr(node, "href")
    display = _clean_text(_render_inline_nodes(node))
    if not href:
        return display

    if href.startswith("/wiki/"):
        path = href.split("#", 1)[0].split("?", 1)[0]
        target = path.replace("/wiki/", "", 1)
        if ":" in target:
            return display
        title = _decode_title(target)
        if not title:
            return display
        if display and display != title:
            return f"[[{title}|{display}]]"
        return f"[[{title}]]"

    if href.startswith(("http://", "https://")):
        if display and display != href:
            return f"[{href} {display}]"
        return f"[{href}]"

    return display


def _render_inline_node(node: Any) -> str:
    tag = _node_tag(node)
    if tag == "-text":
        return str(getattr(node, "text", lambda *args, **kwargs: "")())

    if tag in {"style", "script", "noscript"}:
        return ""

    class_name = _node_attr(node, "class").lower()
    href = _node_attr(node, "href")
    if "reference" in class_name or href.startswith("#cite_note"):
        return ""

    if tag == "br":
        return "\n"
    if tag == "a":
        return _render_link_node(node)

    return _render_inline_nodes(node)


def _serialize_list(node: Any, *, ordered: bool) -> str:
    lines: list[str] = []
    index = 1
    for child in _iter_children(node):
        if _node_tag(child) != "li":
            continue
        inline_parts: list[str] = []
        nested_blocks: list[str] = []
        for inner in _iter_children(child):
            inner_tag = _node_tag(inner)
            if inner_tag in {"ul", "ol"}:
                nested = _serialize_list(inner, ordered=inner_tag == "ol")
                if nested:
                    nested_blocks.append(nested)
                continue
            if inner_tag == "dl":
                nested = _serialize_definition_list(inner)
                if nested:
                    nested_blocks.append(nested)
                continue
            inline_parts.append(_render_inline_node(inner))

        text = _clean_text("".join(inline_parts))
        if text:
            marker = f"{index}." if ordered else "-"
            lines.append(f"{marker} {text}")
        index += 1

        for nested in nested_blocks:
            lines.extend(f"  {line}" if line else "" for line in nested.splitlines())

    return "\n".join(line for line in lines if line.strip())


def _serialize_definition_list(node: Any) -> str:
    lines: list[str] = []
    for child in _iter_children(node):
        tag = _node_tag(child)
        text = _clean_text(_render_inline_nodes(child))
        if not text:
            continue
        if tag == "dt":
            lines.append(f"**{text}**")
        elif tag == "dd":
            lines.append(f"  {text}")
    return "\n".join(lines)


def _serialize_block(node: Any) -> str:
    tag = _node_tag(node)
    if tag == "p":
        return _clean_text(_render_inline_nodes(node))
    if tag == "ul":
        return _serialize_list(node, ordered=False)
    if tag == "ol":
        return _serialize_list(node, ordered=True)
    if tag == "dl":
        return _serialize_definition_list(node)
    return ""


def _is_excluded(node: Any) -> bool:
    current = node
    while current is not None:
        tag = _node_tag(current)
        if tag in {"style", "script", "noscript"}:
            return True
        class_name = _node_attr(current, "class").lower()
        style = _node_attr(current, "style").lower()
        if any(fragment in class_name for fragment in _EXCLUDED_CLASS_FRAGMENTS):
            return True
        if (
            "background-color" in style or "background:" in style
        ) and "infobox" not in class_name:
            return True
        current = getattr(current, "parent", None)
    return False


def _extract_article_with_selectolax(html_text: str) -> _ExtractedArticle:
    from selectolax.parser import HTMLParser

    tree = HTMLParser(html_text)
    root = tree.body or tree
    root_html = str(getattr(root, "html", ""))

    positioned: list[tuple[int, str, Any]] = []

    for heading in root.css("h2, h3, h4, h5, h6"):
        if _is_excluded(heading):
            continue
        outer = str(getattr(heading, "html", ""))
        pos = root_html.find(outer) if outer else -1
        positioned.append((pos if pos >= 0 else 10**12, "heading", heading))

    for block in root.css("p, ul, ol, dl"):
        if _is_excluded(block):
            continue
        block_text = _serialize_block(block)
        if not block_text:
            continue
        if _node_tag(block) == "p" and len(_clean_text(block_text)) < 20:
            continue
        outer = str(getattr(block, "html", ""))
        pos = root_html.find(outer) if outer else -1
        positioned.append((pos if pos >= 0 else 10**12, "content", block))

    positioned.sort(key=lambda item: item[0])

    sections: list[WikipediaSection] = []
    references: list[str] = []
    external_links: list[tuple[str, str]] = []

    current_title = "Introduction"
    current_level = 1
    current_content: list[str] = []

    def _flush() -> None:
        nonlocal current_content
        content = "\n\n".join(part for part in current_content if part).strip()
        if not content:
            current_content = []
            return
        sections.append(
            WikipediaSection(
                index=len(sections),
                level=current_level,
                title=current_title,
                content=_clean_text(content, preserve_newlines=True),
            )
        )
        current_content = []

    for _, kind, node in positioned:
        if kind == "heading":
            heading_text = _clean_text(_render_inline_nodes(node))
            if not heading_text:
                continue
            _flush()
            tag = _node_tag(node)
            level = 2
            if len(tag) > 1 and tag[1].isdigit():
                level = int(tag[1])
            current_title = heading_text
            current_level = level
            continue

        block_text = _serialize_block(node)
        if not block_text:
            continue
        separator = "\n" if _node_tag(node) in {"ul", "ol", "dl"} else "\n\n"
        if not current_content:
            separator = ""
        current_content.append(f"{separator}{block_text}")

    _flush()

    seen_refs: set[str] = set()
    for node in root.css("li[id^='cite_note']"):
        if _is_excluded(node):
            continue
        text = _clean_text(str(getattr(node, "text", lambda *args, **kwargs: "")()))
        text = re.sub(r"^\^+\s*", "", text).strip()
        if len(text) <= 5 or text in seen_refs:
            continue
        seen_refs.add(text)
        references.append(text)

    seen_external: set[str] = set()
    for anchor in root.css("a.external"):
        href = _node_attr(anchor, "href")
        if not href.startswith(("http://", "https://")):
            continue
        if "wikipedia.org" in href or "wikimedia.org" in href:
            continue
        if href in seen_external:
            continue
        seen_external.add(href)
        text = _clean_text(_render_inline_nodes(anchor)) or href
        external_links.append((text, href))

    return _ExtractedArticle(
        sections=tuple(sections),
        references=tuple(references),
        external_links=tuple(external_links),
    )


def _extract_article_fallback(html_text: str) -> _ExtractedArticle:
    stripped = _strip_html_tags(html_text)
    section = WikipediaSection(index=0, level=1, title="Introduction", content=stripped)
    return _ExtractedArticle(
        sections=(section,) if stripped else (),
        references=(),
        external_links=(),
    )


def extract_article_data(html_text: str) -> _ExtractedArticle:
    try:
        return _extract_article_with_selectolax(html_text)
    except Exception:
        return _extract_article_fallback(html_text)


def _resolve_media_list(
    parsed: ParsedWikipediaTarget,
    *,
    include_descriptions: bool,
    timeout_seconds: int,
) -> tuple[WikipediaMedia, ...]:
    if parsed.revision_id is not None:
        return ()

    encoded_title = quote(parsed.title.replace(" ", "_"), safe="")
    url = f"https://{parsed.language}.wikipedia.org/api/rest_v1/page/media-list/{encoded_title}"

    try:
        response = _http_get(
            url,
            timeout=timeout_seconds,
            headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
        )
    except Exception:
        return ()

    data = response.json()
    if not isinstance(data, dict):
        return ()

    items = data.get("items")
    if not isinstance(items, list):
        return ()

    out: list[WikipediaMedia] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get("type", "")).lower() != "image":
            continue

        src = ""
        srcset = item.get("srcset")
        if isinstance(srcset, list) and srcset:
            first = srcset[0]
            if isinstance(first, dict):
                maybe_src = first.get("src")
                if isinstance(maybe_src, str):
                    src = maybe_src
        if not src:
            original = item.get("original")
            if isinstance(original, dict):
                maybe_src = original.get("source")
                if isinstance(maybe_src, str):
                    src = maybe_src

        src = src.strip()
        if not src:
            continue
        if src.startswith("//"):
            src = f"https:{src}"

        caption = None
        caption_obj = item.get("caption")
        if isinstance(caption_obj, dict):
            text = caption_obj.get("text")
            if isinstance(text, str):
                caption = _clean_text(text)
        if not caption:
            maybe_title = item.get("title")
            if isinstance(maybe_title, str):
                caption = _clean_text(maybe_title.replace("_", " ")) or None

        description = caption if include_descriptions else None

        width = item.get("original_width")
        height = item.get("original_height")
        width_val = width if isinstance(width, int) and width > 0 else 0
        height_val = height if isinstance(height, int) and height > 0 else 0

        section_raw = item.get("section_id")
        section_index = section_raw if isinstance(section_raw, int) else None

        out.append(
            WikipediaMedia(
                url=src,
                caption=caption,
                description=description,
                width=width_val,
                height=height_val,
                section_index=section_index,
            )
        )

    return tuple(out)


def _extract_categories(parse_payload: dict[str, Any]) -> tuple[str, ...]:
    categories_raw = parse_payload.get("categories")
    if not isinstance(categories_raw, list):
        return ()

    out: list[str] = []
    seen: set[str] = set()
    for item in categories_raw:
        if not isinstance(item, dict):
            continue
        value = item.get("*")
        if not isinstance(value, str):
            continue
        category = _clean_text(value.replace("_", " "))
        if not category or category in seen:
            continue
        seen.add(category)
        out.append(category)
    return tuple(out)


def _clean_display_title(display_title: Any, *, fallback: str) -> str:
    if not isinstance(display_title, str) or not display_title.strip():
        return fallback
    cleaned = _strip_html_tags(display_title)
    return cleaned or fallback


def build_wikipedia_url(
    *,
    language: str,
    title: str,
    revision_id: int | None,
) -> str:
    encoded_title = quote(title.replace(" ", "_"), safe="")
    base = f"https://{language}.wikipedia.org/wiki/{encoded_title}"
    if revision_id is None:
        return base
    return f"{base}?oldid={revision_id}"


def build_wikipedia_settings(overrides: dict[str, Any] | None) -> WikipediaSettings:
    raw = overrides or {}
    default_lang = _parse_lang(raw.get("default_lang"), default=_DEFAULT_LANGUAGE)
    include_media = _parse_bool(raw.get("include_media"), default=True)
    include_media_descriptions = _parse_bool(
        raw.get("include_media_descriptions"),
        default=True,
    )
    include_references = _parse_bool(raw.get("include_references"), default=True)
    include_external_links = _parse_bool(
        raw.get("include_external_links"),
        default=True,
    )
    include_categories = _parse_bool(raw.get("include_categories"), default=True)

    return WikipediaSettings(
        default_lang=default_lang,
        include_media=include_media,
        include_media_descriptions=include_media_descriptions,
        include_references=include_references,
        include_external_links=include_external_links,
        include_categories=include_categories,
    )


def wikipedia_settings_cache_key(settings: WikipediaSettings) -> tuple[Any, ...]:
    return (
        settings.default_lang,
        settings.include_media,
        settings.include_media_descriptions,
        settings.include_references,
        settings.include_external_links,
        settings.include_categories,
    )


def _attachment_filename(url: str) -> str | None:
    parsed = urlparse(url)
    tail = parsed.path.rsplit("/", 1)[-1].strip()
    return tail or None


def _format_attachment(media: WikipediaMedia) -> list[str]:
    attrs: list[str] = ['type="image"']
    attrs.append(f'url="{_escape_xml_attr(media.url)}"')
    filename = _attachment_filename(media.url)
    if filename:
        attrs.append(f'filename="{_escape_xml_attr(filename)}"')
    if media.width > 0 and media.height > 0:
        attrs.append(f'dimensions="{media.width}x{media.height}"')

    if media.description:
        lines = [f"<attachment {' '.join(attrs)}>"]
        lines.extend(
            _clean_text(media.description, preserve_newlines=True).splitlines()
        )
        lines.append("</attachment>")
        return lines

    return [f"<attachment {' '.join(attrs)} />"]


def _render_article_document(
    *,
    title: str,
    canonical_url: str,
    language: str,
    revision_id: int | None,
    intro: str,
    sections: tuple[WikipediaSection, ...],
    media: tuple[WikipediaMedia, ...],
    references: tuple[str, ...],
    categories: tuple[str, ...],
    settings: WikipediaSettings,
) -> str:
    lines: list[str] = [
        f"# {title}",
        "",
        f"- URL: {canonical_url}",
        f"- Language: {language}",
    ]
    if revision_id is not None:
        lines.append(f"- Revision: {revision_id}")

    media_by_section: dict[int | None, list[WikipediaMedia]] = {}
    for item in media:
        media_by_section.setdefault(item.section_index, []).append(item)

    if intro:
        lines.extend(["", intro])

    if settings.include_media:
        intro_media = media_by_section.get(0, []) + media_by_section.get(None, [])
        if intro_media:
            lines.append("")
            for item in intro_media:
                lines.extend(_format_attachment(item))

    for section in sections:
        heading = section.title.strip()
        content = section.content.strip()
        if not heading or not content:
            continue
        heading_lower = heading.lower()
        if heading_lower == "introduction":
            continue
        if heading_lower in _META_SECTION_TITLES:
            continue

        level = section.level if section.level >= 2 else 2
        lines.extend(["", f"{'#' * level} {heading}", "", content])

        if settings.include_media:
            section_media = media_by_section.get(section.index, [])
            if section_media:
                lines.append("")
                for item in section_media:
                    lines.extend(_format_attachment(item))

    if settings.include_references and references:
        lines.extend(["", "# References", ""])
        for index, reference in enumerate(references, start=1):
            lines.append(f"{index}. {reference}")

    if settings.include_categories and categories:
        lines.extend(["", "# Categories", ""])
        for category in categories:
            lines.append(f"- {category}")

    while lines and not lines[-1].strip():
        lines.pop()

    return "\n".join(lines)


def resolve_wikipedia_article(
    target: str,
    *,
    settings: WikipediaSettings,
    use_cache: bool,
    cache_ttl: Any,
    refresh_cache: bool,
) -> WikipediaResolvedDocument:
    del use_cache
    del cache_ttl
    del refresh_cache

    parsed = parse_wikipedia_target(target, default_lang=settings.default_lang)
    if parsed is None:
        raise ValueError(f"Unsupported Wikipedia target: {target}")

    parse_payload = _resolve_parse_payload(
        parsed, timeout_seconds=_DEFAULT_TIMEOUT_SECONDS
    )
    html_payload = parse_payload.get("text")
    html_text = ""
    if isinstance(html_payload, dict):
        star = html_payload.get("*")
        if isinstance(star, str):
            html_text = star

    extracted = extract_article_data(html_text)
    intro = _resolve_intro(parsed, timeout_seconds=_DEFAULT_TIMEOUT_SECONDS)
    media = ()
    if settings.include_media:
        media = _resolve_media_list(
            parsed,
            include_descriptions=settings.include_media_descriptions,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )

    categories = _extract_categories(parse_payload)
    raw_title = parse_payload.get("title")
    fallback_title = parsed.title
    title = _clean_text(raw_title) if isinstance(raw_title, str) else fallback_title
    if not title:
        title = fallback_title

    display_title = _clean_display_title(
        parse_payload.get("displaytitle"),
        fallback=title,
    )

    canonical_url = build_wikipedia_url(
        language=parsed.language,
        title=title,
        revision_id=parsed.revision_id,
    )

    rendered = _render_article_document(
        title=display_title,
        canonical_url=canonical_url,
        language=parsed.language,
        revision_id=parsed.revision_id,
        intro=intro,
        sections=extracted.sections,
        media=media,
        references=extracted.references,
        categories=categories,
        settings=settings,
    )

    safe_title = _safe_path_segment(title.replace(" ", "_"), fallback="article")
    context_subpath = f"wikipedia/{parsed.language}/{safe_title}.md"
    source_path = f"{parsed.language}/{title.replace(' ', '_')}"
    if parsed.revision_id is not None:
        context_subpath = (
            f"wikipedia/{parsed.language}/{safe_title}-oldid-{parsed.revision_id}.md"
        )
        source_path = f"{source_path}@oldid={parsed.revision_id}"

    return WikipediaResolvedDocument(
        label=f"Wikipedia/{parsed.language}/{title.replace(' ', '_')}",
        rendered=rendered,
        source_ref=f"{parsed.language}.wikipedia.org",
        source_path=source_path,
        context_subpath=context_subpath,
        kind="article",
        canonical_id=parsed.canonical_id,
    )
