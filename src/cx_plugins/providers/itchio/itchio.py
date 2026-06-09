from __future__ import annotations

import base64
from dataclasses import dataclass
import html
import json
import mimetypes
from pathlib import Path
import re
from typing import Any, Iterable
from urllib.parse import quote, unquote, urljoin, urlparse, urlunparse

import requests
from selectolax.parser import HTMLParser

_DEFAULT_TIMEOUT_SECONDS = 25
_USER_AGENT = "contextualize/itchio"
_MEDIA_TARGET_PREFIX = "itchio:media:"
_ASSET_HOST_SUFFIXES = (
    ".itch.zone",
    ".itch.ovh",
)
_ASSET_HOSTS = {
    "img.itch.zone",
    "static.itch.io",
    "w3g3a5v6.ssl.hwcdn.net",
}
_EXCLUDED_PATH_PREFIXES = (
    "/login",
    "/logout",
    "/register",
    "/dashboard",
    "/download",
    "/purchase",
    "/report",
    "/embed",
    "/jam/new",
)
_SKIP_RENDER_TAGS = {
    "script",
    "style",
    "noscript",
    "template",
    "svg",
}
_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "dd",
    "details",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
}
_REQUEST_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "User-Agent": _USER_AGENT,
}
_THEME_LABELS = {
    "bg_color": "bg-color",
    "fg_color": "fg-color",
    "link_color": "link-color",
    "border_color": "border-color",
    "button_color": "button-color",
    "button_fg_color": "button-fg-color",
    "itchio_primary": "primary",
    "itchio_text": "text",
    "itchio_bg": "background",
}


@dataclass(frozen=True)
class ItchioSettings:
    theme_enabled: bool = True
    comments_enabled: bool = True
    comments_limit: int = 15
    comments_offset: int = 0
    include_devlogs: bool = False
    devlogs_limit: int | None = None
    media_enabled: bool = True
    media_descriptions: bool = True


@dataclass(frozen=True)
class ItchioMedia:
    kind: str
    url: str
    filename: str | None = None
    caption: str | None = None
    description: str | None = None
    content_type: str | None = None
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class ItchioLink:
    target: str
    label: str
    kind: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ItchioGameCard:
    target: str
    label: str
    description: str | None
    author: str | None
    author_url: str | None
    genre: str | None
    platforms: tuple[str, ...]
    metadata: tuple[tuple[str, str], ...]
    thumbnail: ItchioMedia | None


@dataclass(frozen=True)
class ItchioComment:
    author: str | None
    content: str
    published_at: str | None
    url: str | None = None


@dataclass(frozen=True)
class ItchioPage:
    source_url: str
    canonical_url: str
    kind: str
    title: str
    itch_path: str | None
    description: str
    theme: dict[str, str]
    more_info: tuple[tuple[str, str], ...]
    files: tuple[ItchioLink, ...]
    media: tuple[ItchioMedia, ...]
    devlogs: tuple[ItchioLink, ...]
    comments: tuple[ItchioComment, ...]
    games: tuple[ItchioGameCard, ...]
    collections: tuple[ItchioLink, ...]
    queue_items: tuple[ItchioLink, ...]
    public_links: tuple[ItchioLink, ...]
    pagination_next: str | None


def build_itchio_settings(raw_config: dict[str, Any] | None = None) -> ItchioSettings:
    raw = raw_config or {}
    return ItchioSettings(
        theme_enabled=_nested_bool(raw, ("theme", "enabled"), True),
        comments_enabled=_nested_bool(raw, ("comments", "enabled"), True),
        comments_limit=_nested_int(raw, ("comments", "limit"), 15, minimum=0),
        comments_offset=_nested_int(raw, ("comments", "offset"), 0, minimum=0),
        include_devlogs=_nested_bool(raw, ("devlogs", "include"), False),
        devlogs_limit=_nested_optional_int(raw, ("devlogs", "limit"), minimum=1),
        media_enabled=_nested_bool(raw, ("media", "enabled"), True),
        media_descriptions=_nested_bool(raw, ("media", "describe"), True),
    )


def is_itchio_target(target: str) -> bool:
    if parse_itchio_media_target(target) is not None:
        return True
    parsed = urlparse(target)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = parsed.netloc.lower()
    if not _is_itchio_page_host(host):
        return False
    return _is_public_path(parsed.path)


def target_kind_from_url(target: str) -> str | None:
    if parse_itchio_media_target(target) is not None:
        return "media"
    parsed = urlparse(target)
    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    if not _is_itchio_page_host(host):
        return None
    if host in {"itch.io", "www.itch.io"}:
        if path.startswith("/queue/c/"):
            return "queue"
        if path.startswith("/c/"):
            return "collection"
        if "/devlog/" in path:
            return "devlog-post"
        if path.endswith("/devlog"):
            return "devlog-index"
        return "index" if path else "home"
    if not path:
        return "profile"
    if "/devlog/" in path:
        return "devlog-post"
    if path.endswith("/devlog"):
        return "devlog-index"
    return "game"


def fetch_itchio_html(
    url: str,
    *,
    use_cache: bool = True,
    cache_ttl: Any = None,
    refresh_cache: bool = False,
) -> str:
    from . import cache

    cached = None
    if use_cache and not refresh_cache:
        cached = cache.get_cached_html(url, ttl=cache_ttl)
    if cached is not None:
        return cached

    response = requests.get(url, headers=_REQUEST_HEADERS, timeout=_DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    text = _decode_response_text(response)
    if use_cache:
        cache.store_html(url, text)
    return text


def parse_itchio_page(
    url: str,
    html_text: str,
    *,
    settings: ItchioSettings | None = None,
    describe_media: bool = True,
) -> ItchioPage:
    settings = settings or ItchioSettings()
    tree = HTMLParser(html_text)
    requested_kind = target_kind_from_url(url)
    canonical_url = _without_fragment(url) if requested_kind == "queue" else _canonical_url(tree, url)
    itch_path = _meta_content(tree, "itch:path")
    kind = "queue" if requested_kind == "queue" else _detect_kind(tree, canonical_url, itch_path)
    title = _extract_title(tree, kind, canonical_url)
    description_node = _description_node(tree, kind)
    description = _render_html_fragment(
        description_node,
        canonical_url,
        describe_media=describe_media
        and settings.media_enabled
        and settings.media_descriptions,
    )
    if not description:
        description = _clean_text(_meta_content(tree, "description") or "")
    description_media_urls = _extract_node_media_urls(description_node, canonical_url)
    theme = _extract_theme(tree) if settings.theme_enabled else {}
    more_info = tuple(_dedupe_pairs([*_extract_header_info(tree, kind), *_extract_more_info(tree)]))
    files = tuple(_extract_files(tree, canonical_url))
    media = (
        tuple(
            _describe_media_items(
                _extract_media(
                    tree,
                    canonical_url,
                    exclude_urls=description_media_urls,
                ),
                enabled=describe_media and settings.media_descriptions,
            )
        )
        if settings.media_enabled
        else ()
    )
    devlogs = tuple(_extract_devlogs(tree, canonical_url))
    comments = (
        tuple(_extract_comments(tree, settings))
        if settings.comments_enabled
        else ()
    )
    games = tuple(
        _extract_game_cells(
            tree,
            canonical_url,
            include_images=_game_card_images_enabled(kind) and settings.media_enabled,
        )
    )
    collections = tuple(_extract_collection_links(tree, canonical_url))
    queue_items = tuple(_extract_queue_items(html_text, canonical_url))
    public_links = tuple(_extract_public_links(tree, canonical_url))
    return ItchioPage(
        source_url=url,
        canonical_url=canonical_url,
        kind=kind,
        title=title,
        itch_path=itch_path,
        description=description,
        theme=theme,
        more_info=more_info,
        files=files,
        media=media,
        devlogs=devlogs,
        comments=comments,
        games=games,
        collections=collections,
        queue_items=queue_items,
        public_links=public_links,
        pagination_next=_extract_next_page(tree, canonical_url),
    )


def resolve_itchio_target(
    target: str,
    *,
    settings: ItchioSettings,
    use_cache: bool = True,
    cache_ttl: Any = None,
    refresh_cache: bool = False,
) -> list[dict[str, Any]]:
    parsed_media = parse_itchio_media_target(target)
    if parsed_media is not None:
        return [_media_stub_document(target, parsed_media)]

    html_text = fetch_itchio_html(
        target,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )
    page = parse_itchio_page(target, html_text, settings=settings, describe_media=True)
    settings_key = _settings_key(settings)
    documents = [_page_document(page, settings_key=settings_key)]
    if page.kind == "game" and settings.include_devlogs and page.devlogs:
        devlogs = page.devlogs
        if settings.devlogs_limit is not None:
            devlogs = devlogs[: settings.devlogs_limit]
        child_settings = ItchioSettings(
            theme_enabled=settings.theme_enabled,
            comments_enabled=settings.comments_enabled,
            comments_limit=settings.comments_limit,
            comments_offset=settings.comments_offset,
            include_devlogs=False,
            devlogs_limit=None,
            media_enabled=settings.media_enabled,
            media_descriptions=settings.media_descriptions,
        )
        for devlog in devlogs:
            child_html = fetch_itchio_html(
                devlog.target,
                use_cache=use_cache,
                cache_ttl=cache_ttl,
                refresh_cache=refresh_cache,
            )
            child_page = parse_itchio_page(
                devlog.target,
                child_html,
                settings=child_settings,
                describe_media=True,
            )
            documents.append(_page_document(child_page, settings_key=settings_key))
    return documents


def list_itchio_targets(
    target: str,
    *,
    settings: ItchioSettings,
    use_cache: bool = True,
    cache_ttl: Any = None,
    refresh_cache: bool = False,
) -> dict[str, Any]:
    parsed_media = parse_itchio_media_target(target)
    if parsed_media is not None:
        return _listing_envelope(
            target=target,
            kind="media",
            targets=[],
            settings=settings,
            capabilities={"resolve": True, "listTargets": False, "materialize": True},
        )

    html_text = fetch_itchio_html(
        target,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )
    page = parse_itchio_page(target, html_text, settings=settings, describe_media=False)
    items: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(link: ItchioLink, *, traverse: bool = True) -> None:
        if not link.target or link.target in seen:
            return
        seen.add(link.target)
        item: dict[str, Any] = {
            "target": link.target,
            "label": link.label,
            "kind": link.kind,
            "metadata": dict(link.metadata),
        }
        if not traverse:
            item["traverse"] = False
        items.append(item)

    source_kind = page.kind
    source_id = _page_source_id(page)
    for index, media in enumerate(page.media):
        media_target = build_itchio_media_target(
            source_kind=source_kind,
            source_id=source_id,
            index=index,
            media=media,
            source_url=page.canonical_url,
        )
        add(
            ItchioLink(
                target=media_target,
                label=media.caption or media.filename or media.url,
                kind=f"media:{media.kind}",
                metadata={
                    "relation": "media",
                    "url": media.url,
                    "contentType": media.content_type,
                    "filename": media.filename,
                    "sourceUrl": page.canonical_url,
                },
            )
        )
    for devlog in page.devlogs:
        add(devlog)
    for game in page.games:
        add(_game_card_link(game, source_url=page.canonical_url))
    for collection in (page.collections, page.queue_items):
        for link in collection:
            add(link)
    for link in page.public_links:
        add(link)

    return _listing_envelope(
        target=target,
        kind=page.kind,
        targets=items,
        settings=settings,
        summary={
            "title": page.title,
            "counts": {
                "media": len(page.media),
                "devlogs": len(page.devlogs),
                "games": len(page.games),
                "collections": len(page.collections),
                "queueItems": len(page.queue_items),
                "comments": len(page.comments),
            },
        },
        pagination={
            "returned": len(items),
            "totalCount": len(items),
            "hasMore": page.pagination_next is not None,
            "next": page.pagination_next,
        },
        metadata={
            "sourceUrl": page.canonical_url,
            "itchPath": page.itch_path,
            "theme": page.theme,
        },
        capabilities={"resolve": True, "listTargets": True, "materialize": True},
    )


def materialize_itchio_media_target(
    target: str,
    parsed: dict[str, Any],
    *,
    use_cache: bool,
    refresh_cache: bool,
    cache_only: bool = False,
) -> list[dict[str, Any]]:
    url = str(parsed["url"])
    filename = str(parsed.get("filename") or _filename_from_url(url) or "itchio-media")
    content_type = parsed.get("content_type") or mimetypes.guess_type(filename)[0]
    suffix = Path(filename).suffix or Path(urlparse(url).path).suffix or ".bin"
    cache_identity = _media_cache_identity(parsed)

    from . import cache

    content: bytes | None = None
    if cache_only:
        content = cache.get_cached_media_bytes(cache_identity)
    else:
        from ..shared.media import download_cached_media_to_temp

        try:
            from contextualize.runtime import get_refresh_media
        except Exception:
            refresh_media = False
        else:
            refresh_media = get_refresh_media()

        tmp = download_cached_media_to_temp(
            url,
            suffix=suffix,
            headers=_REQUEST_HEADERS,
            cache_identity=cache_identity,
            get_cached_media_bytes=(
                cache.get_cached_media_bytes if use_cache else lambda _identity: None
            ),
            store_media_bytes=(
                cache.store_media_bytes
                if use_cache
                else lambda _identity, _content: None
            ),
            refresh_cache=refresh_cache or refresh_media,
        )
        if tmp is not None:
            try:
                content = tmp.read_bytes()
            finally:
                tmp.unlink(missing_ok=True)

    if not content:
        return []
    return [
        {
            "source": target,
            "label": parsed.get("caption") or filename,
            "filename": filename,
            "content": content,
            "content_type": content_type,
            "metadata": {
                "provider": "itchio",
                "kind": parsed.get("kind"),
                "mediaUrl": url,
                "sourceUrl": parsed.get("source_url"),
                "sourceKind": parsed.get("source_kind"),
                "sourceId": parsed.get("source_id"),
                "bytes": len(content),
            },
        }
    ]


def build_itchio_media_target(
    *,
    source_kind: str,
    source_id: str,
    index: int,
    media: ItchioMedia,
    source_url: str,
) -> str:
    payload = {
        "kind": media.kind,
        "url": media.url,
        "filename": media.filename,
        "caption": media.caption,
        "content_type": media.content_type,
        "source_url": source_url,
    }
    encoded = _base64url_json(payload)
    return (
        f"{_MEDIA_TARGET_PREFIX}{_safe_token(source_kind, fallback='page')}:"
        f"{_safe_token(source_id, fallback='source')}:{index}:{encoded}"
    )


def parse_itchio_media_target(target: str) -> dict[str, Any] | None:
    if not target.startswith(_MEDIA_TARGET_PREFIX):
        return None
    parts = target.split(":", 5)
    if len(parts) != 6 or parts[0] != "itchio" or parts[1] != "media":
        return None
    source_kind, source_id, index_raw, encoded = parts[2:]
    try:
        index = int(index_raw)
    except ValueError:
        return None
    payload = _decode_base64url_json(encoded)
    if not isinstance(payload, dict):
        return None
    url = payload.get("url")
    if not isinstance(url, str) or not url:
        return None
    return {
        "source_kind": source_kind,
        "source_id": source_id,
        "index": index,
        "kind": str(payload.get("kind") or "media"),
        "url": url,
        "filename": _string_or_none(payload.get("filename")),
        "caption": _string_or_none(payload.get("caption")),
        "content_type": _string_or_none(payload.get("content_type")),
        "source_url": _string_or_none(payload.get("source_url")),
    }


def _game_card_link(game: ItchioGameCard, *, source_url: str) -> ItchioLink:
    metadata: dict[str, Any] = {
        "relation": "game",
        "sourceUrl": source_url,
    }
    if game.description:
        metadata["description"] = game.description
    if game.author:
        metadata["author"] = game.author
    if game.author_url:
        metadata["authorUrl"] = game.author_url
    if game.genre:
        metadata["genre"] = game.genre
    if game.platforms:
        metadata["platforms"] = list(game.platforms)
    if game.thumbnail:
        metadata["image"] = game.thumbnail.url
        metadata["imageFilename"] = game.thumbnail.filename
    for key, value in game.metadata:
        metadata[_metadata_key(key)] = value
    return ItchioLink(
        target=game.target,
        label=game.label,
        kind="game",
        metadata=metadata,
    )


def _page_document(page: ItchioPage, *, settings_key: tuple[Any, ...]) -> dict[str, Any]:
    source_ref, source_path, context_subpath = _page_paths(page)
    return {
        "source": page.source_url,
        "label": page.canonical_url,
        "content": _render_page(page),
        "metadata": {
            "trace_path": source_path,
            "provider": "itchio",
            "source_ref": source_ref,
            "source_path": source_path,
            "context_subpath": context_subpath,
            "kind": page.kind,
            "canonical_id": page.itch_path or page.canonical_url,
            "canonical_url": page.canonical_url,
            "theme": page.theme,
            "settings_key": settings_key,
        },
    }


def _media_stub_document(target: str, parsed: dict[str, Any]) -> dict[str, Any]:
    filename = parsed.get("filename") or _filename_from_url(str(parsed["url"])) or "media"
    lines = [
        f"# {parsed.get('caption') or filename}",
        "",
        f"url: {parsed['url']}",
        f"kind: {parsed.get('kind') or 'media'}",
    ]
    if parsed.get("source_url"):
        lines.append(f"source_url: {parsed['source_url']}")
    return {
        "source": target,
        "label": filename,
        "content": "\n".join(lines),
        "metadata": {
            "trace_path": f"itchio/media/{filename}",
            "provider": "itchio",
            "source_ref": "itch.io",
            "source_path": f"media/{filename}",
            "context_subpath": f"itchio/media/{_safe_filename(filename)}.md",
            "kind": "media",
            "canonical_id": target,
        },
    }


def _render_page(page: ItchioPage) -> str:
    lines = [f"# {page.title}", "", f"url: {page.canonical_url}", f"kind: {page.kind}"]
    if page.itch_path:
        lines.append(f"itch_path: {page.itch_path}")
    lines.append("")

    _append_section(lines, "Description", page.description)
    if page.theme:
        lines.append("## Theme")
        for key, value in page.theme.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    if page.more_info:
        lines.append("## More Information")
        for key, value in page.more_info:
            lines.append(f"- {key}: {value}")
        lines.append("")
    if page.files:
        lines.append("## Files")
        for file in page.files:
            if file.target:
                lines.append(f"- [{file.label}]({file.target})")
            else:
                lines.append(f"- {file.label}")
        lines.append("")
    if page.media:
        lines.append("## Media")
        for media in page.media:
            lines.extend(_format_media_tag(media))
        lines.append("")
    if page.devlogs:
        lines.append("## Development Log")
        for devlog in page.devlogs:
            lines.append(f"- [{devlog.label}]({_devlog_display_target(devlog.target)})")
        lines.append("")
    if page.comments:
        lines.append("***")
        lines.append("")
        lines.append("## Comments")
        for comment in page.comments:
            prefix = "- "
            if comment.author:
                prefix += comment.author
                if comment.published_at:
                    prefix += f" ({comment.published_at})"
                prefix += ": "
            lines.append(prefix + comment.content.replace("\n", "\n  "))
        lines.append("")
    if page.games:
        lines.append("## Games")
        for game in page.games:
            lines.extend(_format_game_card(game))
        lines.append("")
    if page.collections:
        lines.append("## Collections")
        for collection in page.collections:
            lines.append(f"- [{collection.label}]({collection.target})")
        lines.append("")
    if page.queue_items:
        lines.append("## Queue")
        for item in page.queue_items:
            relation = item.metadata.get("relation")
            suffix = f" [{relation}]" if isinstance(relation, str) else ""
            lines.append(f"- {item.label}{suffix}: {item.target}")
        lines.append("")
    return _normalize_markdown("\n".join(lines))


def _append_section(lines: list[str], title: str, content: str) -> None:
    if not content:
        return
    lines.append(f"## {title}")
    lines.append(content)
    lines.append("")


def _devlog_display_target(target: str) -> str:
    parsed = urlparse(target)
    segments = [segment for segment in parsed.path.split("/") if segment]
    try:
        index = segments.index("devlog")
    except ValueError:
        return target
    tail = segments[index + 1 :]
    return f"devlog/{'/'.join(tail)}" if tail else "devlog"


def _format_game_card(game: ItchioGameCard) -> list[str]:
    lines = [f"### [{game.label}]({game.target})", ""]
    if game.thumbnail:
        lines.extend(_format_media_tag(game.thumbnail))
        lines.append("")
    if game.description:
        lines.append(game.description)
        lines.append("")

    details: list[tuple[str, str]] = []
    if game.author:
        author = (
            f"[{game.author}]({game.author_url})"
            if game.author_url
            else game.author
        )
        details.append(("Author", author))
    if game.genre:
        details.append(("Genre", game.genre))
    if game.platforms:
        details.append(("Platforms", ", ".join(game.platforms)))
    details.extend(game.metadata)
    if details:
        for key, value in details:
            lines.append(f"- {key}: {value}")
        lines.append("")
    return lines


def _format_media_tag(media: ItchioMedia) -> list[str]:
    tag = _media_tag_name(media.kind)
    attrs: list[str] = []
    if media.filename:
        attrs.append(f'filename="{_escape_xml_attr(media.filename)}"')
    caption = _semantic_caption(media.caption)
    if caption:
        attrs.append(f'caption="{_escape_xml_attr(caption)}"')
    attr_suffix = f" {' '.join(attrs)}" if attrs else ""

    if media.description:
        lines = [f"<{tag}{attr_suffix}>"]
        lines.extend(_clean_text(media.description, preserve_newlines=True).splitlines())
        lines.append(f"</{tag}>")
        return lines
    return [f"<{tag}{attr_suffix} />"]


def _media_tag_name(kind: str) -> str:
    normalized = kind.strip().lower()
    if normalized in {"image", "video", "audio"}:
        return normalized
    if normalized == "html-game":
        return "video"
    return "media"


def _extract_title(tree: HTMLParser, kind: str, url: str) -> str:
    for selector in (
        "meta[property='og:title']",
        "meta[name='twitter:title']",
    ):
        value = _node_attr(tree.css_first(selector), "content")
        if value:
            return _clean_title(value)
    for selector in (
        ".game_title",
        ".profile_header .name",
        ".user_name",
        ".collection_title",
        ".post_title",
        "h1",
        "title",
    ):
        node = tree.css_first(selector)
        if node is not None:
            text = _clean_title(node.text(separator=" "))
            if text:
                return text
    parsed = urlparse(url)
    if kind == "profile":
        return parsed.netloc.split(".")[0]
    return unquote(Path(parsed.path.rstrip("/")).name) or parsed.netloc


def _clean_title(value: str) -> str:
    value = _clean_text(value)
    return re.sub(r"\s+-\s+itch\.io$", "", value, flags=re.I).strip()


def _description_node(tree: HTMLParser, kind: str) -> Any | None:
    if kind == "devlog-post":
        selectors = (
            ".post_body",
            ".formatted_post_body",
            "article",
        )
    elif kind == "profile":
        selectors = (
            ".user_profile.formatted",
            ".user_profile",
            ".formatted_description",
            ".description",
        )
    elif kind == "collection":
        selectors = (
            ".collection_description",
            ".formatted_description",
            ".description",
        )
    else:
        selectors = (
            ".formatted_description",
            ".game_description",
            ".description",
            "article",
        )
    for selector in selectors:
        node = tree.css_first(selector)
        if node is not None:
            return node
    return None


def _extract_theme(tree: HTMLParser) -> dict[str, str]:
    theme: dict[str, str] = {}
    meta_color = _meta_content(tree, "theme-color")
    if meta_color:
        theme["theme-color"] = meta_color.strip()
    for selector in ("#game_theme", "#user_theme"):
        node = tree.css_first(selector)
        if node is None:
            continue
        raw = node.text(separator="\n") or node.html or ""
        for key, value in re.findall(r"--itchio_([A-Za-z0-9_]+)\s*:\s*([^;{}]+)", raw):
            label = _THEME_LABELS.get(key, key.replace("_", "-"))
            theme[label] = _clean_text(value)
    return theme


def _extract_header_info(tree: HTMLParser, kind: str) -> list[tuple[str, str]]:
    if kind not in {"collection", "profile", "index"}:
        return []
    rows: list[tuple[str, str]] = []
    for header in tree.css(".grid_header .sub_header, .collection_page .sub_header"):
        author = header.css_first("a[href]")
        author_text = _clean_text(author.text(separator=" ")) if author else ""
        if author_text:
            rows.append(("Author", author_text))
        updated = header.css_first(".date_format")
        updated_text = _clean_text(updated.text(separator=" ")) if updated else ""
        if updated_text:
            rows.append(("Last updated", updated_text))
    return rows


def _extract_more_info(tree: HTMLParser) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for table in tree.css(".game_info_panel_widget table, .more_information table, table.game_info_panel_widget"):
        for tr in table.css("tr"):
            cells = tr.css("td, th")
            if len(cells) < 2:
                continue
            key = _clean_text(cells[0].text(separator=" "))
            value = _clean_text(cells[1].text(separator=" "))
            if not key or not value:
                continue
            identity = f"{key}\0{value}"
            if identity in seen:
                continue
            seen.add(identity)
            rows.append((key.rstrip(":"), value))
    return rows


def _dedupe_pairs(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key, value in rows:
        if not key or not value:
            continue
        identity = f"{key.lower()}\0{value}"
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append((key, value))
    return deduped


def _extract_files(tree: HTMLParser, base_url: str) -> list[ItchioLink]:
    files: list[ItchioLink] = []
    seen: set[str] = set()
    for selector in (
        ".upload",
        ".downloadable",
        ".game_downloads li",
        ".post_files li",
        ".post_files a",
    ):
        for node in tree.css(selector):
            link = node if _node_tag(node) == "a" else node.css_first("a[href]")
            href = _node_attr(link, "href")
            label = _clean_text(node.text(separator=" "))
            if not label and href:
                label = _filename_from_url(href) or href
            if not label:
                continue
            target = _absolute_url(base_url, href) if href else ""
            identity = target or label
            if identity in seen:
                continue
            seen.add(identity)
            files.append(
                ItchioLink(
                    target=target,
                    label=label,
                    kind="file",
                    metadata={"relation": "file", "sourceUrl": base_url},
                )
            )
    return files


def _extract_media(
    tree: HTMLParser,
    base_url: str,
    *,
    exclude_urls: set[str] | None = None,
) -> list[ItchioMedia]:
    media: list[ItchioMedia] = []
    seen: set[str] = set(exclude_urls or set())

    def add(
        kind: str,
        url: str | None,
        *,
        caption: str | None = None,
        width: int | None = None,
        height: int | None = None,
        content_type: str | None = None,
    ) -> None:
        if not url:
            return
        absolute = _absolute_url(base_url, url)
        if not absolute or absolute in seen:
            return
        seen.add(absolute)
        filename = _filename_from_url(absolute)
        media.append(
            ItchioMedia(
                kind=kind,
                url=absolute,
                filename=filename,
                caption=_semantic_caption(caption),
                description=None,
                content_type=content_type or mimetypes.guess_type(filename or absolute)[0],
                width=width,
                height=height,
            )
        )

    for anchor in tree.css(".screenshot_list a[href], a.screenshot[href], .screenshot a[href]"):
        img = anchor.css_first("img")
        caption = _node_attr(img, "alt") or _node_attr(img, "title") or anchor.text(separator=" ")
        add(
            "image",
            _node_attr(anchor, "href"),
            caption=caption,
            width=_node_int_attr(img, "width"),
            height=_node_int_attr(img, "height"),
        )
    for img in tree.css(".screenshot_list img, .screenshot img, .game_thumb img, meta[property='og:image']"):
        if _node_tag(img) == "meta":
            add("image", _node_attr(img, "content"), caption="OpenGraph image")
        elif _has_ancestor(img, "a"):
            continue
        else:
            add(
                "image",
                _first_attr(img, ("data-lazy_src", "data-lazy-src", "data-src", "src")),
                caption=_node_attr(img, "alt") or _node_attr(img, "title"),
                width=_node_int_attr(img, "width"),
                height=_node_int_attr(img, "height"),
            )
    for node in tree.css(".iframe_placeholder[data-iframe]"):
        iframe_html = _node_attr(node, "data-iframe") or ""
        match = re.search(r"\bsrc=[\"']([^\"']+)[\"']", html.unescape(iframe_html))
        add("html-game", match.group(1) if match else None, caption="Embedded game")
    for iframe in tree.css("iframe[src]"):
        add("video", _node_attr(iframe, "src"), caption=_node_attr(iframe, "title"))
    for source in tree.css("video source[src], audio source[src], video[src], audio[src]"):
        tag = _node_tag(source)
        parent = _node_tag(getattr(source, "parent", None))
        kind = "audio" if tag == "audio" or parent == "audio" else "video"
        add(kind, _node_attr(source, "src"), content_type=_node_attr(source, "type"))
    return media


def _describe_media_items(
    media: list[ItchioMedia],
    *,
    enabled: bool,
) -> list[ItchioMedia]:
    if not enabled:
        return media
    return [
        ItchioMedia(
            kind=item.kind,
            url=item.url,
            filename=item.filename,
            caption=item.caption,
            description=_describe_media(item),
            content_type=item.content_type,
            width=item.width,
            height=item.height,
        )
        for item in media
    ]


def _describe_media(media: ItchioMedia) -> str | None:
    if not media.url.startswith(("http://", "https://")):
        return None
    try:
        from contextualize.render.markitdown import convert_path_to_markdown
        from ..shared.media import download_cached_media_to_temp
    except Exception:
        return None

    from . import cache

    tmp = download_cached_media_to_temp(
        media.url,
        suffix=_media_suffix(media.url, media.filename),
        headers=_REQUEST_HEADERS,
        cache_identity=f"itchio:{media.kind}:{media.url}",
        get_cached_media_bytes=cache.get_cached_media_bytes,
        store_media_bytes=cache.store_media_bytes,
        refresh_cache=False,
    )
    if tmp is None:
        return None

    try:
        caption = _semantic_caption(media.caption)
        prompt_append = (
            f"Caption context: {caption}" if caption else "Describe the media."
        )
        result = convert_path_to_markdown(str(tmp), prompt_append=prompt_append)
        markdown = _normalize_media_description(result.markdown or "")
    except Exception:
        return None
    finally:
        tmp.unlink(missing_ok=True)

    return markdown or None


def _extract_devlogs(tree: HTMLParser, base_url: str) -> list[ItchioLink]:
    devlogs: list[ItchioLink] = []
    seen: set[str] = set()
    selectors = (
        ".devlog_post a[href]",
        ".devlog_post_row a[href]",
        ".game_devlog_widget a[href]",
        ".more_posts a[href]",
        "a[href*='/devlog/']",
    )
    for selector in selectors:
        for anchor in tree.css(selector):
            href = _node_attr(anchor, "href")
            target = _absolute_url(base_url, href)
            if not target or target in seen or target_kind_from_url(target) != "devlog-post":
                continue
            label = _clean_text(anchor.text(separator=" ")) or target
            seen.add(target)
            devlogs.append(
                ItchioLink(
                    target=target,
                    label=label,
                    kind="devlog-post",
                    metadata={"relation": "devlog", "sourceUrl": base_url},
                )
            )
    return devlogs


def _extract_comments(tree: HTMLParser, settings: ItchioSettings) -> list[ItchioComment]:
    raw_nodes = tree.css(".community_post, .post_comment, .comment")
    comments: list[ItchioComment] = []
    for node in raw_nodes:
        body = _first_node(
            node,
            (
                ".post_body",
                ".formatted_body",
                ".comment_body",
                ".body",
                ".post_content",
            ),
        )
        content = _render_html_fragment(body, "") if body is not None else ""
        if not content:
            content = _clean_text(node.text(separator=" "))
        if not content:
            continue
        author_node = _first_node(
            node,
            (
                ".post_author a",
                ".post_author",
                ".author a",
                ".author",
                ".username",
            ),
        )
        time_node = node.css_first("time")
        comments.append(
            ItchioComment(
                author=_clean_text(author_node.text(separator=" ")) if author_node else None,
                content=content,
                published_at=_node_attr(time_node, "datetime") or _clean_text(time_node.text(separator=" ")) if time_node else None,
                url=_node_attr(node.css_first("a[href*='#']"), "href"),
            )
        )
    start = settings.comments_offset
    stop = start + settings.comments_limit if settings.comments_limit else start
    return comments[start:stop]


def _game_card_images_enabled(kind: str) -> bool:
    return kind not in {"collection", "profile", "index", "home", "queue"}


def _extract_game_cells(
    tree: HTMLParser,
    base_url: str,
    *,
    include_images: bool,
) -> list[ItchioGameCard]:
    games: list[ItchioGameCard] = []
    seen: set[str] = set()
    for node in tree.css(".game_cell, .index_game_cell, .game_item"):
        anchor = _first_node(
            node,
            (
                "a.title[href]",
                ".game_title a[href]",
                ".thumb_link[href]",
                "a.game_link[href]",
                "a[href]",
            ),
        )
        href = _node_attr(anchor, "href")
        target = _absolute_url(base_url, href)
        if not target or target in seen or target_kind_from_url(target) != "game":
            continue
        title_anchor = _first_node(node, ("a.title[href]", ".game_title a[href]")) or anchor
        title = _clean_text(title_anchor.text(separator=" ")) or _clean_text(node.text(separator=" "))
        author_node = _first_node(node, (".game_author a[href]", ".game_author", ".user_link", ".author"))
        author = _clean_text(author_node.text(separator=" ")) if author_node else ""
        author_url = _absolute_url(base_url, _node_attr(author_node, "href"))
        description = _text_or_title(_first_node(node, (".game_text", ".game_description", ".description")))
        genre = _clean_text(
            (_first_node(node, (".game_genre", ".genre")) or _EmptyNode()).text(separator=" ")
        )
        platforms = tuple(_extract_platforms(node))
        metadata = tuple(_extract_game_card_metadata(node))
        thumbnail = (
            _extract_game_thumbnail(node, base_url, title or target)
            if include_images
            else None
        )
        seen.add(target)
        games.append(
            ItchioGameCard(
                target=target,
                label=title or target,
                description=description or None,
                author=author or None,
                author_url=author_url,
                genre=genre or None,
                platforms=platforms,
                metadata=metadata,
                thumbnail=thumbnail,
            )
        )
    return games


def _extract_game_thumbnail(
    node: Any,
    base_url: str,
    title: str,
) -> ItchioMedia | None:
    img = _first_node(
        node,
        (
            ".game_thumb img",
            "img[data-lazy_src]",
            "img[data-lazy-src]",
            "img[data-src]",
            "img[src]",
        ),
    )
    if img is None:
        return None
    raw_url = _first_attr(img, ("data-lazy_src", "data-lazy-src", "data-src", "src"))
    url = _absolute_url(base_url, raw_url)
    if not url:
        return None
    filename = _filename_from_url(url)
    caption = _semantic_caption(_node_attr(img, "alt") or title)
    return ItchioMedia(
        kind="image",
        url=url,
        filename=filename,
        caption=caption,
        content_type=mimetypes.guess_type(filename or url)[0],
        width=_node_int_attr(img, "width"),
        height=_node_int_attr(img, "height"),
    )


def _extract_platforms(node: Any) -> list[str]:
    platforms: list[str] = []
    for platform_node in node.css(".game_platform, .game_platforms, .platforms"):
        text = _clean_text(platform_node.text(separator=" "))
        if text:
            platforms.extend(_split_platform_text(text))
        for titled in platform_node.css("[title]"):
            title = _platform_label(_node_attr(titled, "title"))
            if title:
                platforms.append(title)
    return _dedupe_strings(platforms)


def _extract_game_card_metadata(node: Any) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    price_node = node.css_first(".price_tag")
    price_value_node = price_node.css_first(".price_value") if price_node else None
    price = _clean_text(
        (price_value_node or price_node or _EmptyNode()).text(separator=" ")
    )
    if price:
        rows.append(("Price", price))
    price_note = _clean_text(_node_attr(price_node, "title") or "") if price_node else ""
    if price_note and price_note != price:
        rows.append(("Price note", price_note))
    tags = _dedupe_strings(
        _clean_text(tag.text(separator=" "))
        for tag in node.css(".game_tags a, .tags a, a.tag, .tag")
    )
    if tags:
        rows.append(("Tags", ", ".join(tags)))

    for selector, label in (
        (".game_status", "Status"),
        (".game_category", "Category"),
        (".game_rating", "Rating"),
        (".game_updated", "Last updated"),
        (".game_published", "Published"),
        (".game_release_date", "Released"),
    ):
        value = _clean_text((_first_node(node, (selector,)) or _EmptyNode()).text(separator=" "))
        if value:
            rows.append((label, value))

    date = _clean_text((node.css_first(".date_format") or _EmptyNode()).text(separator=" "))
    if date and not any(key == "Last updated" for key, _value in rows):
        rows.append(("Last updated", date))
    return _dedupe_pairs(rows)


def _extract_collection_links(tree: HTMLParser, base_url: str) -> list[ItchioLink]:
    links: list[ItchioLink] = []
    seen: set[str] = set()
    for node in tree.css(".collection_row, .collection, .user_collection"):
        anchor = node if _node_tag(node) == "a" else node.css_first("a[href]")
        href = _node_attr(anchor, "href")
        target = _absolute_url(base_url, href)
        if not target or target in seen or target_kind_from_url(target) != "collection":
            continue
        label = _clean_text(anchor.text(separator=" ")) or _clean_text(node.text(separator=" ")) or target
        seen.add(target)
        links.append(
            ItchioLink(
                target=target,
                label=label,
                kind="collection",
                metadata={"relation": "collection", "sourceUrl": base_url},
            )
        )
    return links


def _extract_queue_items(html_text: str, base_url: str) -> list[ItchioLink]:
    payload = _extract_call_payload(html_text, "R.Queue.Viewer(")
    if not payload:
        return []
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    items: list[ItchioLink] = []
    for key, relation in (
        ("games_before", "before"),
        ("game", "current"),
        ("current_item", "current"),
        ("next_item", "next"),
        ("games_after", "after"),
    ):
        value = data.get(key)
        values = value if isinstance(value, list) else [value]
        for entry in values:
            link = _queue_entry_link(entry, base_url, relation)
            if link is not None:
                items.append(link)
    return _dedupe_links(items)


def _queue_entry_link(entry: Any, base_url: str, relation: str) -> ItchioLink | None:
    if not isinstance(entry, dict):
        return None
    raw_url = entry.get("url") or entry.get("game_url") or entry.get("link")
    target = _absolute_url(base_url, raw_url if isinstance(raw_url, str) else None)
    if not target:
        user = entry.get("user") if isinstance(entry.get("user"), dict) else {}
        user_url = user.get("url") or user.get("username")
        slug = entry.get("slug") or entry.get("title")
        if isinstance(user_url, str) and isinstance(slug, str):
            target = _absolute_url(base_url, f"{user_url.rstrip('/')}/{quote(slug)}")
    if not target:
        return None
    title = entry.get("title") or entry.get("name") or target
    return ItchioLink(
        target=target,
        label=_clean_text(str(title)),
        kind=target_kind_from_url(target) or "game",
        metadata={"relation": relation, "sourceUrl": base_url},
    )


def _extract_public_links(tree: HTMLParser, base_url: str) -> list[ItchioLink]:
    links: list[ItchioLink] = []
    seen: set[str] = {base_url.rstrip("/")}
    for anchor in tree.css("a[href]"):
        href = _node_attr(anchor, "href")
        target = _absolute_url(base_url, href)
        if not target or target.rstrip("/") in seen or not is_itchio_target(target):
            continue
        kind = target_kind_from_url(target)
        if kind in {None, "media", "home"}:
            continue
        label = _clean_text(anchor.text(separator=" ")) or target
        seen.add(target.rstrip("/"))
        links.append(
            ItchioLink(
                target=target,
                label=label,
                kind=kind,
                metadata={"relation": "link", "sourceUrl": base_url},
            )
        )
    return links


def _extract_next_page(tree: HTMLParser, base_url: str) -> str | None:
    for selector in ("link[rel='next']", "a.next_page[href]", ".pagination .next a[href]"):
        node = tree.css_first(selector)
        href = _node_attr(node, "href")
        target = _absolute_url(base_url, href)
        if target:
            return target
    return None


def _detect_kind(tree: HTMLParser, url: str, itch_path: str | None) -> str:
    if tree.css_first(".game_devlog_post_page, .post_body") is not None and "/devlog/" in urlparse(url).path:
        return "devlog-post"
    if itch_path:
        if itch_path.startswith("games/"):
            return "game"
        if itch_path.startswith("users/"):
            return "profile"
        if itch_path.startswith("collections/"):
            return "collection"
    return target_kind_from_url(url) or "page"


def _canonical_url(tree: HTMLParser, fallback: str) -> str:
    for selector in ("link[rel='canonical']", "meta[property='og:url']"):
        node = tree.css_first(selector)
        value = _node_attr(node, "href") or _node_attr(node, "content")
        if value:
            return _without_fragment(_absolute_url(fallback, value) or fallback)
    return _without_fragment(fallback)


def _render_html_fragment(
    node: Any,
    base_url: str,
    *,
    describe_media: bool = False,
) -> str:
    if node is None:
        return ""
    return _normalize_markdown(
        _render_node(node, base_url, describe_media=describe_media)
    )


def _render_node(node: Any, base_url: str, *, describe_media: bool = False) -> str:
    tag = _node_tag(node)
    if tag == "-text":
        return html.unescape(node.html or node.text() or "")
    if tag in _SKIP_RENDER_TAGS:
        return ""
    if tag == "br":
        return "\n"
    if tag == "hr":
        return "\n\n---\n\n"
    if tag in {"strong", "b"}:
        text = _clean_inline(_render_children(node, base_url, describe_media=describe_media))
        return f"**{text}**" if text else ""
    if tag in {"em", "i"}:
        text = _clean_inline(_render_children(node, base_url, describe_media=describe_media))
        return f"*{text}*" if text else ""
    if tag == "code":
        text = _clean_inline(node.text(separator=" "))
        return f"`{text}`" if text else ""
    if tag == "pre":
        text = (node.text(separator="\n") or "").strip("\n")
        return f"\n\n```\n{text}\n```\n\n" if text else ""
    if tag == "a":
        rendered_children = _render_children(
            node,
            base_url,
            describe_media=describe_media,
        )
        if _contains_media_tag(rendered_children):
            return rendered_children
        text = _clean_inline(rendered_children) or _node_attr(node, "href")
        href = _absolute_url(base_url, _node_attr(node, "href"))
        if text and href and _is_public_url(href):
            return f"[{text}]({href})"
        return text or ""
    if tag == "img":
        src = _absolute_url(
            base_url,
            _first_attr(node, ("data-lazy_src", "data-lazy-src", "data-src", "src")),
        )
        alt = _semantic_caption(_node_attr(node, "alt") or _node_attr(node, "title"))
        if not src:
            return alt or ""
        filename = _filename_from_url(src)
        media = ItchioMedia(
            kind="image",
            url=src,
            filename=filename,
            caption=alt,
            description=None,
            content_type=mimetypes.guess_type(filename or src)[0],
        )
        if describe_media:
            media = ItchioMedia(
                kind=media.kind,
                url=media.url,
                filename=media.filename,
                caption=media.caption,
                description=_describe_media(media),
                content_type=media.content_type,
                width=media.width,
                height=media.height,
            )
        return "\n\n" + "\n".join(_format_media_tag(media)) + "\n\n"
    if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        level = int(tag[1])
        text = _clean_inline(_render_children(node, base_url, describe_media=describe_media))
        if level >= 5:
            if not text:
                return ""
            if "](" in text:
                return f"\n\n{text}\n\n"
            return f"\n\n**{text}**\n\n"
        return f"\n\n{'#' * level} {text}\n\n" if text else ""
    if tag == "blockquote":
        rendered = _normalize_markdown(_render_children(node, base_url, describe_media=describe_media))
        quoted = "\n".join(f"> {line}" if line else ">" for line in rendered.splitlines())
        return f"\n\n{quoted}\n\n" if quoted else ""
    if tag in {"ul", "ol"}:
        ordered = tag == "ol"
        lines: list[str] = []
        index = 1
        for child in _children(node):
            if _node_tag(child) != "li":
                continue
            rendered = _normalize_markdown(_render_children(child, base_url, describe_media=describe_media))
            if not rendered:
                continue
            prefix = f"{index}. " if ordered else "- "
            indented = rendered.replace("\n", "\n  ")
            lines.append(prefix + indented)
            index += 1
        return "\n\n" + "\n".join(lines) + "\n\n" if lines else ""
    if tag == "table":
        rows: list[str] = []
        for tr in node.css("tr"):
            cells = [
                _normalize_markdown(
                    _render_children(cell, base_url, describe_media=describe_media)
                )
                for cell in tr.css("td, th")
            ]
            cells = [cell for cell in cells if cell]
            if len(cells) >= 2:
                first = _clean_text(cells[0])
                if (
                    len(cells) == 2
                    and "\n" not in cells[0]
                    and len(first) <= 48
                    and not cells[1].lstrip().startswith(("#", "-", "1. "))
                ):
                    rows.append(f"- {first}: {cells[1]}")
                else:
                    rows.extend(cells)
            elif cells:
                rows.append(f"- {cells[0]}")
        return "\n\n" + "\n\n".join(rows) + "\n\n" if rows else ""
    rendered = _render_children(node, base_url, describe_media=describe_media)
    if tag in _BLOCK_TAGS:
        return f"\n\n{rendered}\n\n"
    return rendered


def _render_children(
    node: Any,
    base_url: str,
    *,
    describe_media: bool = False,
) -> str:
    return "".join(
        _render_node(child, base_url, describe_media=describe_media)
        for child in _children(node)
    )


def _children(node: Any) -> list[Any]:
    children: list[Any] = []
    child = getattr(node, "child", None)
    while child is not None:
        children.append(child)
        child = getattr(child, "next", None)
    return children


def _has_ancestor(node: Any, tag_name: str) -> bool:
    current = getattr(node, "parent", None)
    while current is not None:
        if _node_tag(current) == tag_name:
            return True
        current = getattr(current, "parent", None)
    return False


def _node_tag(node: Any) -> str:
    return str(getattr(node, "tag", "") or "").lower()


def _node_attr(node: Any, name: str) -> str | None:
    if node is None:
        return None
    value = getattr(node, "attributes", {}).get(name)
    if value is None:
        return None
    value = html.unescape(str(value)).strip()
    return value or None


def _node_int_attr(node: Any, name: str) -> int | None:
    value = _node_attr(node, name)
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _first_attr(node: Any, names: tuple[str, ...]) -> str | None:
    for name in names:
        value = _node_attr(node, name)
        if value:
            return value
    return None


def _first_node(node: Any, selectors: tuple[str, ...]) -> Any | None:
    for selector in selectors:
        found = node.css_first(selector)
        if found is not None:
            return found
    return None


def _text_or_title(node: Any | None) -> str:
    if node is None:
        return ""
    return _clean_text(_node_attr(node, "title") or node.text(separator=" "))


def _extract_node_media_urls(node: Any | None, base_url: str) -> set[str]:
    if node is None:
        return set()
    urls: set[str] = set()
    for img in node.css("img[src], img[data-src], img[data-lazy_src], img[data-lazy-src]"):
        url = _absolute_url(
            base_url,
            _first_attr(img, ("data-lazy_src", "data-lazy-src", "data-src", "src")),
        )
        if url:
            urls.add(url)
    for source in node.css("video source[src], audio source[src], video[src], audio[src], iframe[src]"):
        url = _absolute_url(base_url, _node_attr(source, "src"))
        if url:
            urls.add(url)
    return urls


def _meta_content(tree: HTMLParser, name: str) -> str | None:
    escaped = name.replace("'", "\\'")
    for selector in (
        f"meta[name='{escaped}']",
        f"meta[property='{escaped}']",
    ):
        value = _node_attr(tree.css_first(selector), "content")
        if value:
            return value
    return None


def _clean_text(value: str, *, preserve_newlines: bool = False) -> str:
    cleaned = html.unescape(value).replace("\xa0", " ")
    if preserve_newlines:
        cleaned = re.sub(r"[^\S\n]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    else:
        cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _clean_inline(value: str | None) -> str:
    if not value:
        return ""
    return _clean_text(value.replace("\n", " "))


def _split_platform_text(value: str) -> list[str]:
    cleaned = _clean_text(value)
    if not cleaned:
        return []
    parts = re.split(r"\s{2,}|,\s*|/\s*", cleaned)
    return [_platform_label(part) for part in parts if _platform_label(part)]


def _platform_label(value: str | None) -> str:
    cleaned = _clean_text(value or "")
    if cleaned.lower().startswith("download for "):
        cleaned = cleaned[13:].strip()
    return cleaned


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean_text(value)
        if not cleaned:
            continue
        identity = cleaned.lower()
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(cleaned)
    return deduped


def _metadata_key(value: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", value)
    if not words:
        return "metadata"
    first, *rest = words
    return first.lower() + "".join(word[:1].upper() + word[1:].lower() for word in rest)


def _semantic_caption(value: str | None) -> str | None:
    cleaned = _clean_text(value or "")
    if not cleaned:
        return None
    normalized = re.sub(r"[\W_]+", " ", cleaned, flags=re.UNICODE).strip().lower()
    if normalized in {
        "image",
        "img",
        "screenshot",
        "thumbnail",
        "open graph image",
        "opengraph image",
        "embedded game",
    }:
        return None
    if re.fullmatch(r"screenshot\s+\d+", normalized):
        return None
    if re.fullmatch(r"screenshot\s+(one|two|three|four|five|six|seven|eight|nine|ten)", normalized):
        return None
    return cleaned


def _contains_media_tag(value: str) -> bool:
    return re.search(r"<(?:image|video|audio|media)\b", value) is not None


def _normalize_markdown(value: str) -> str:
    value = html.unescape(value).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+$", "", line) for line in value.splitlines()]
    value = "\n".join(lines)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _escape_xml_attr(value: str) -> str:
    return html.escape(value, quote=True)


def _decode_response_text(response: requests.Response) -> str:
    content = response.content
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        encoding = response.encoding or response.apparent_encoding or "utf-8"
        return content.decode(encoding, errors="replace")


def _media_suffix(url: str, filename: str | None) -> str:
    candidate = filename or urlparse(url).path.rsplit("/", 1)[-1]
    suffix = Path(candidate).suffix.lower()
    return suffix if re.fullmatch(r"\.[A-Za-z0-9]+", suffix) else ""


def _normalize_media_description(value: str) -> str:
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and re.fullmatch(
        r"#?\s*description\s*\(auto-generated\)\s*:?\s*",
        lines[0].strip(),
        flags=re.I,
    ):
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return _clean_text("\n".join(lines), preserve_newlines=True)


def _absolute_url(base_url: str, value: str | None) -> str | None:
    if not value:
        return None
    value = html.unescape(value).strip()
    if not value or value.startswith(("javascript:", "mailto:", "tel:")):
        return None
    if base_url:
        value = urljoin(base_url, value)
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return _without_fragment(value)


def _without_fragment(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, ""))


def _is_public_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _is_itchio_page_host(host: str) -> bool:
    if not host:
        return False
    if host in _ASSET_HOSTS or any(host.endswith(suffix) for suffix in _ASSET_HOST_SUFFIXES):
        return False
    return host in {"itch.io", "www.itch.io"} or host.endswith(".itch.io")


def _is_public_path(path: str) -> bool:
    if any(path.startswith(prefix) for prefix in _EXCLUDED_PATH_PREFIXES):
        return False
    blocked_segments = {
        "-",
        "add-to-collection",
        "download",
        "download_url",
        "follow",
        "purchase",
        "login",
        "logout",
        "register",
        "report",
        "embed",
        "unfollow",
    }
    segments = {segment for segment in path.split("/") if segment}
    return not bool(segments & blocked_segments)


def _filename_from_url(url: str) -> str | None:
    path = urlparse(url).path
    if not path:
        return None
    filename = unquote(Path(path).name).replace("/", "_")
    filename = filename.strip("._")
    return filename or None


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "item"


def _safe_token(value: str, *, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or fallback


def _page_source_id(page: ItchioPage) -> str:
    if page.itch_path:
        return page.itch_path
    parsed = urlparse(page.canonical_url)
    return f"{parsed.netloc}{parsed.path}".strip("/")


def _page_paths(page: ItchioPage) -> tuple[str, str, str]:
    parsed = urlparse(page.canonical_url)
    source_ref = parsed.netloc
    path = page.itch_path or f"{parsed.netloc}{parsed.path}".strip("/")
    source_path = _safe_context_path(path, fallback=page.kind)
    basename = "_profile.md" if page.kind == "profile" else f"{_safe_filename(page.title)}.md"
    if page.kind in {"game", "devlog-post"}:
        context_subpath = f"itchio/{source_path}/{basename}"
    elif page.kind == "collection":
        context_subpath = f"itchio/{source_path}/_collection.md"
    else:
        context_subpath = f"itchio/{source_path}.md"
    return source_ref, source_path, context_subpath


def _safe_context_path(value: str, *, fallback: str) -> str:
    parts = [_safe_filename(unquote(part)) for part in value.split("/") if part]
    return "/".join(parts) or fallback


def _settings_key(settings: ItchioSettings) -> tuple[Any, ...]:
    return (
        settings.theme_enabled,
        settings.comments_enabled,
        settings.comments_limit,
        settings.comments_offset,
        settings.include_devlogs,
        settings.devlogs_limit,
        settings.media_enabled,
        settings.media_descriptions,
    )


def _listing_settings(settings: ItchioSettings) -> dict[str, Any]:
    return {
        "theme": {"enabled": settings.theme_enabled},
        "comments": {
            "enabled": settings.comments_enabled,
            "limit": settings.comments_limit,
            "offset": settings.comments_offset,
        },
        "devlogs": {
            "include": settings.include_devlogs,
            "limit": settings.devlogs_limit,
        },
        "media": {
            "enabled": settings.media_enabled,
            "describe": settings.media_descriptions,
        },
    }


def _listing_envelope(
    *,
    target: str,
    kind: str,
    targets: list[dict[str, Any]],
    settings: ItchioSettings,
    summary: dict[str, Any] | None = None,
    pagination: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    capabilities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    envelope_metadata = {
        "provider": "itchio",
        "kind": kind,
        "target": target,
        "settings": _listing_settings(settings),
    }
    if metadata:
        envelope_metadata.update(metadata)
    return {
        "targets": targets,
        "summary": summary or {},
        "pagination": pagination
        or {"returned": len(targets), "totalCount": len(targets), "hasMore": False},
        "metadata": envelope_metadata,
        "capabilities": capabilities
        or {"resolve": True, "listTargets": True, "materialize": False},
    }


def _base64url_json(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_base64url_json(value: str) -> Any:
    padding = "=" * (-len(value) % 4)
    try:
        raw = base64.urlsafe_b64decode((value + padding).encode("ascii"))
        return json.loads(raw.decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        return None


def _media_cache_identity(parsed: dict[str, Any]) -> str:
    return str(parsed.get("url") or parsed)


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _nested_bool(raw: dict[str, Any], path: tuple[str, str], default: bool) -> bool:
    value = _nested_value(raw, path)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"1", "true", "yes", "on"}:
            return True
        if cleaned in {"0", "false", "no", "off"}:
            return False
    return default


def _nested_int(
    raw: dict[str, Any],
    path: tuple[str, str],
    default: int,
    *,
    minimum: int,
) -> int:
    value = _nested_value(raw, path)
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


def _nested_optional_int(
    raw: dict[str, Any],
    path: tuple[str, str],
    *,
    minimum: int,
) -> int | None:
    value = _nested_value(raw, path)
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= minimum else None


def _nested_value(raw: dict[str, Any], path: tuple[str, str]) -> Any:
    section = raw.get(path[0])
    if isinstance(section, dict) and path[1] in section:
        return section[path[1]]
    snake = f"{path[0]}_{path[1]}"
    hyphen = f"{path[0]}-{path[1]}"
    if snake in raw:
        return raw[snake]
    if hyphen in raw:
        return raw[hyphen]
    return raw.get(path[1])


def _extract_call_payload(html_text: str, marker: str) -> str | None:
    start = html_text.find(marker)
    if start < 0:
        return None
    index = start + len(marker)
    depth = 0
    in_string: str | None = None
    escaped = False
    payload_start = index
    while index < len(html_text):
        char = html_text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = None
        elif char in {'"', "'"}:
            in_string = char
        elif char in "[{(":
            depth += 1
        elif char in "]})":
            if depth == 0:
                return html_text[payload_start:index]
            depth -= 1
        index += 1
    return None


def _dedupe_links(links: list[ItchioLink]) -> list[ItchioLink]:
    seen: set[str] = set()
    out: list[ItchioLink] = []
    for link in links:
        if link.target in seen:
            continue
        seen.add(link.target)
        out.append(link)
    return out


class _EmptyNode:
    def text(self, separator: str = " ") -> str:
        return ""
