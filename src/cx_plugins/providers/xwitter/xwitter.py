from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import timedelta
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote, urlencode, urlparse

from ..shared.progress import record_progress

_XWITTER_HOSTS = frozenset({"x.com", "twitter.com", "fixupx.com", "twittpr.com"})
_ALIAS_HOSTS = frozenset({"fixupx.com", "twittpr.com"})
_OEMBED_ENDPOINT = "https://publish.x.com/oembed"
_HTTP_TIMEOUT_SECONDS = 30
_TWEET_ID_RE = re.compile(r"^[0-9]+$")
_HTTP_HEADERS = {
    "User-Agent": "contextualize/xwitter",
    "Accept": "application/json,text/html;q=0.8,*/*;q=0.5",
}


def _log(message: str) -> None:
    from contextualize.runtime import get_verbose_logging

    if get_verbose_logging():
        print(message, file=sys.stderr, flush=True)


@dataclass(frozen=True)
class XwitterSettings:
    include_html: bool = False
    use_alias_fallback: bool = True
    resolve_tco_links: bool = True


@dataclass(frozen=True)
class XwitterTarget:
    kind: str
    original: str
    host: str
    tweet_id: str
    author: str | None
    canonical_url: str
    source_path: str


@dataclass(frozen=True)
class XwitterDocument:
    source_url: str
    kind: str
    canonical_url: str
    tweet_id: str
    author: str | None
    label: str
    trace_path: str
    context_subpath: str
    source_path: str
    rendered: str
    source_created: str | None = None
    source_modified: str | None = None


@dataclass(frozen=True)
class _TweetLink:
    href: str
    text: str | None
    resolved_url: str | None = None


@dataclass(frozen=True)
class _ResolvedTweet:
    canonical_url: str
    author_name: str | None
    author_url: str | None
    tweet_text: str | None
    posted_at_label: str | None
    html: str | None
    provider_name: str | None
    provider_url: str | None
    image_url: str | None
    links: tuple[_TweetLink, ...]
    fetched_via: str


def _parse_bool(value: str, *, default: bool) -> bool:
    cleaned = value.strip().lower()
    if not cleaned:
        return default
    return cleaned not in {"0", "false", "no", "off"}


def _normalize_bool_override(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _parse_bool(value, default=False)
    raise ValueError(f"{field} must be a boolean")


def _xwitter_settings_from_env() -> XwitterSettings:
    return XwitterSettings(
        include_html=_parse_bool(
            os.environ.get("XWITTER_INCLUDE_HTML", ""),
            default=False,
        ),
        use_alias_fallback=_parse_bool(
            os.environ.get("XWITTER_ALIAS_FALLBACK", ""),
            default=True,
        ),
        resolve_tco_links=_parse_bool(
            os.environ.get("XWITTER_RESOLVE_TCO_LINKS", ""),
            default=True,
        ),
    )


def build_xwitter_settings(overrides: dict[str, Any] | None = None) -> XwitterSettings:
    env = _xwitter_settings_from_env()
    if not overrides:
        return env
    include_html = env.include_html
    if "include_html" in overrides:
        include_html = _normalize_bool_override(
            overrides["include_html"],
            field="include_html",
        )
    use_alias_fallback = env.use_alias_fallback
    if "use_alias_fallback" in overrides:
        use_alias_fallback = _normalize_bool_override(
            overrides["use_alias_fallback"],
            field="use_alias_fallback",
        )
    resolve_tco_links = env.resolve_tco_links
    if "resolve_tco_links" in overrides:
        resolve_tco_links = _normalize_bool_override(
            overrides["resolve_tco_links"],
            field="resolve_tco_links",
        )
    return XwitterSettings(
        include_html=include_html,
        use_alias_fallback=use_alias_fallback,
        resolve_tco_links=resolve_tco_links,
    )


def xwitter_settings_cache_key(settings: XwitterSettings) -> tuple[Any, ...]:
    return (
        "v2",
        settings.include_html,
        settings.use_alias_fallback,
        settings.resolve_tco_links,
    )


def _normalize_host(hostname: str | None) -> str:
    host = (hostname or "").strip().lower().rstrip(".")
    for prefix in ("www.", "mobile."):
        if host.startswith(prefix):
            host = host.removeprefix(prefix)
    return host


def _canonical_url(author: str | None, tweet_id: str) -> str:
    if author:
        return f"https://x.com/{quote(author.strip('@'))}/status/{tweet_id}"
    return f"https://x.com/i/web/status/{tweet_id}"


def _source_path(author: str | None, tweet_id: str) -> str:
    if author:
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "-", author.strip("@").lower())
        cleaned = cleaned.strip("-") or "unknown"
        return f"{cleaned}/status/{tweet_id}"
    return f"status/{tweet_id}"


def _parse_tweet_path(path: str) -> tuple[str | None, str] | None:
    parts = [part for part in path.split("/") if part]
    if len(parts) >= 4 and parts[0] == "i" and parts[1] == "web":
        if parts[2] in {"status", "statuses"} and _TWEET_ID_RE.match(parts[3]):
            return None, parts[3]
    if len(parts) >= 3 and parts[0] == "i":
        if parts[1] in {"status", "statuses"} and _TWEET_ID_RE.match(parts[2]):
            return None, parts[2]
    if len(parts) >= 3 and parts[1] in {"status", "statuses"}:
        author = parts[0].strip("@")
        tweet_id = parts[2]
        if author and _TWEET_ID_RE.match(tweet_id):
            return author, tweet_id
    return None


def parse_xwitter_target(value: str) -> XwitterTarget | None:
    raw = value.strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    if parsed.scheme.lower() not in {"http", "https"}:
        return None
    host = _normalize_host(parsed.hostname)
    if host not in _XWITTER_HOSTS:
        return None
    tweet = _parse_tweet_path(parsed.path)
    if tweet is None:
        return None
    author, tweet_id = tweet
    return XwitterTarget(
        kind="tweet",
        original=value,
        host=host,
        tweet_id=tweet_id,
        author=author,
        canonical_url=_canonical_url(author, tweet_id),
        source_path=_source_path(author, tweet_id),
    )


def is_xwitter_url(value: str) -> bool:
    return parse_xwitter_target(value) is not None


def _documents_from_cached_payload(payload: Any) -> list[XwitterDocument] | None:
    if not isinstance(payload, list):
        return None
    documents: list[XwitterDocument] = []
    for item in payload:
        if not isinstance(item, dict):
            return None
        try:
            doc = XwitterDocument(**item)
        except TypeError:
            return None
        documents.append(doc)
    return documents


def _resolution_cache_identity(url: str, settings: XwitterSettings) -> str:
    payload = {
        "v": 1,
        "url": url,
        "settings": xwitter_settings_cache_key(settings),
    }
    return "xwitter-resolve:" + json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _http_get_json(url: str, *, params: dict[str, Any], timeout: int) -> dict[str, Any]:
    import requests

    response = requests.get(
        url,
        params=params,
        headers=_HTTP_HEADERS,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected JSON payload from {url}")
    return payload


def _http_get_text(url: str, *, timeout: int) -> str:
    import requests

    response = requests.get(url, headers=_HTTP_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def _clean_text(value: str) -> str:
    text = unescape(value)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)
    return text.strip()


class _OEmbedHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._p_depth = 0
        self._p_parts: list[str] = []
        self.paragraphs: list[str] = []
        self._anchor_stack: list[dict[str, Any]] = []
        self.links: list[_TweetLink] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key.lower(): value or "" for key, value in attrs}
        if tag.lower() == "p":
            if self._p_depth == 0:
                self._p_parts = []
            self._p_depth += 1
        elif tag.lower() == "br" and self._p_depth:
            self._p_parts.append("\n")
        elif tag.lower() == "a":
            self._anchor_stack.append(
                {"href": attrs_dict.get("href", ""), "parts": []}
            )

    def handle_data(self, data: str) -> None:
        if self._p_depth:
            self._p_parts.append(data)
        if self._anchor_stack:
            self._anchor_stack[-1]["parts"].append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "p" and self._p_depth:
            self._p_depth -= 1
            if self._p_depth == 0:
                text = _clean_text("".join(self._p_parts))
                if text:
                    self.paragraphs.append(text)
                self._p_parts = []
        elif tag == "a" and self._anchor_stack:
            anchor = self._anchor_stack.pop()
            href = str(anchor.get("href") or "").strip()
            text = _clean_text("".join(anchor.get("parts") or [])) or None
            if href:
                self.links.append(_TweetLink(href=href, text=text))


class _MetadataHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.links: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key.lower(): value or "" for key, value in attrs}
        if tag.lower() == "meta":
            key = attrs_dict.get("property") or attrs_dict.get("name")
            content = attrs_dict.get("content")
            if key and content:
                self.meta[key] = content
        elif tag.lower() == "link":
            self.links.append(attrs_dict)


def _parse_oembed_html(
    html: str,
) -> tuple[str | None, str | None, tuple[_TweetLink, ...]]:
    parser = _OEmbedHTMLParser()
    parser.feed(html)
    tweet_text = "\n\n".join(parser.paragraphs).strip() or None
    posted_at_label = None
    for link in reversed(parser.links):
        if link.text and "/status/" in link.href:
            posted_at_label = link.text
            break
    return tweet_text, posted_at_label, tuple(parser.links)


def _same_tweet_url(url: str, tweet_id: str) -> bool:
    parsed = parse_xwitter_target(url.split("?", 1)[0])
    return parsed is not None and parsed.tweet_id == tweet_id


def _filtered_links(
    links: tuple[_TweetLink, ...],
    *,
    tweet_id: str,
) -> tuple[_TweetLink, ...]:
    out: list[_TweetLink] = []
    seen: set[str] = set()
    for link in links:
        parsed = urlparse(link.href)
        cleaned = parsed._replace(query="", fragment="").geturl()
        if _same_tweet_url(cleaned, tweet_id):
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(link)
    return tuple(out)


def _is_tco_url(url: str) -> bool:
    parsed = urlparse(url)
    return _normalize_host(parsed.hostname) == "t.co"


def _tco_cache_identity(url: str) -> str:
    return "xwitter-tco:" + url.strip()


def _http_resolve_redirect(url: str, *, timeout: int) -> str | None:
    import requests

    response = requests.get(
        url,
        headers=_HTTP_HEADERS,
        timeout=timeout,
        allow_redirects=True,
        stream=True,
    )
    try:
        final_url = str(response.url or "").strip()
        if final_url and final_url != url:
            return final_url
        response.raise_for_status()
    finally:
        response.close()
    return None


def _resolve_tco_url(
    url: str,
    *,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
) -> str | None:
    if not _is_tco_url(url):
        return None
    from .cache import get_cached_api_json, store_api_json

    identity = _tco_cache_identity(url)
    if use_cache and not refresh_cache:
        cached = get_cached_api_json(identity, ttl=cache_ttl)
        if isinstance(cached, dict):
            final_url = cached.get("resolved_url")
            if isinstance(final_url, str) and final_url.strip():
                return final_url.strip()
    try:
        final_url = _http_resolve_redirect(url, timeout=_HTTP_TIMEOUT_SECONDS)
    except Exception as exc:
        _log(f"  xwitter t.co resolution failed for {url}: {exc}")
        return None
    if final_url and use_cache:
        store_api_json(identity, {"url": url, "resolved_url": final_url})
    return final_url


def _resolve_tco_links(
    links: tuple[_TweetLink, ...],
    *,
    settings: XwitterSettings,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
) -> tuple[_TweetLink, ...]:
    if not settings.resolve_tco_links:
        return links
    resolved: list[_TweetLink] = []
    for link in links:
        final_url = _resolve_tco_url(
            link.href,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
        )
        resolved.append(
            _TweetLink(
                href=link.href,
                text=link.text,
                resolved_url=final_url or link.resolved_url,
            )
        )
    return tuple(resolved)


def _tweet_text_with_resolved_links(
    tweet_text: str | None,
    links: tuple[_TweetLink, ...],
) -> str | None:
    if not tweet_text:
        return tweet_text
    rendered = tweet_text
    for link in links:
        if not link.resolved_url:
            continue
        candidates = [item for item in (link.text, link.href) if item]
        for candidate in candidates:
            if candidate in rendered:
                rendered = rendered.replace(candidate, link.resolved_url)
                break
    return rendered


def _fetch_oembed(
    target: XwitterTarget,
    *,
    settings: XwitterSettings,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
) -> _ResolvedTweet:
    payload = _http_get_json(
        _OEMBED_ENDPOINT,
        params={"url": target.canonical_url, "omit_script": "true"},
        timeout=_HTTP_TIMEOUT_SECONDS,
    )
    html = payload.get("html")
    html_text = html if isinstance(html, str) else ""
    tweet_text, posted_at_label, links = _parse_oembed_html(html_text)
    links = _resolve_tco_links(
        _filtered_links(links, tweet_id=target.tweet_id),
        settings=settings,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )
    canonical_url = payload.get("url")
    author_name = payload.get("author_name")
    author_url = payload.get("author_url")
    provider_name = payload.get("provider_name")
    provider_url = payload.get("provider_url")
    return _ResolvedTweet(
        canonical_url=canonical_url
        if isinstance(canonical_url, str)
        else target.canonical_url,
        author_name=author_name if isinstance(author_name, str) else target.author,
        author_url=author_url if isinstance(author_url, str) else None,
        tweet_text=_tweet_text_with_resolved_links(tweet_text, links),
        posted_at_label=posted_at_label,
        html=html_text or None,
        provider_name=provider_name if isinstance(provider_name, str) else None,
        provider_url=provider_url if isinstance(provider_url, str) else None,
        image_url=None,
        links=links,
        fetched_via="x-oembed",
    )


def _metadata_link(
    links: list[dict[str, str]],
    *,
    rel: str,
    type_: str | None = None,
) -> str | None:
    for link in links:
        rels = {part.lower() for part in link.get("rel", "").split()}
        if rel not in rels:
            continue
        if type_ is not None and link.get("type", "").lower() != type_.lower():
            continue
        href = link.get("href")
        if href:
            return href
    return None


def _fetch_alias_metadata(target: XwitterTarget) -> _ResolvedTweet | None:
    if target.host not in _ALIAS_HOSTS:
        return None
    html = _http_get_text(target.original, timeout=_HTTP_TIMEOUT_SECONDS)
    parser = _MetadataHTMLParser()
    parser.feed(html)
    meta = parser.meta
    canonical = (
        _metadata_link(parser.links, rel="canonical")
        or meta.get("og:url")
        or target.canonical_url
    )
    parsed = parse_xwitter_target(canonical)
    canonical_url = (
        parsed.canonical_url if parsed is not None else target.canonical_url
    )
    author = meta.get("twitter:creator") or target.author
    author = author.strip("@") if isinstance(author, str) else target.author
    author_url = f"https://x.com/{author}" if author else None
    alternate = _metadata_link(
        parser.links,
        rel="alternate",
        type_="application/json+oembed",
    )
    links: tuple[_TweetLink, ...] = (
        (_TweetLink(href=alternate, text="alias-oembed"),) if alternate else ()
    )
    return _ResolvedTweet(
        canonical_url=canonical_url,
        author_name=author,
        author_url=author_url,
        tweet_text=meta.get("og:description"),
        posted_at_label=None,
        html=None,
        provider_name=meta.get("og:site_name"),
        provider_url=f"https://{target.host}",
        image_url=meta.get("og:image"),
        links=links,
        fetched_via=f"{target.host}-open-graph",
    )


def _render_markdown_frontmatter(payload: dict[str, Any]) -> str:
    import yaml

    data = {key: value for key, value in payload.items() if value is not None}
    frontmatter = yaml.safe_dump(
        data,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    ).strip()
    return f"---\n{frontmatter}\n---"


def _render_links_section(links: tuple[_TweetLink, ...]) -> str | None:
    lines: list[str] = []
    for link in links:
        text = link.resolved_url or link.text
        if text:
            lines.append(f"- [{text}]({link.href})")
        else:
            lines.append(f"- {link.href}")
    return "\n".join(lines) or None


def _render_tweet_document(
    *,
    target: XwitterTarget,
    resolved: _ResolvedTweet,
    settings: XwitterSettings,
) -> str:
    metadata = {
        "url": resolved.canonical_url,
        "source_url": target.original,
        "tweet_id": target.tweet_id,
        "author": resolved.author_name or target.author,
        "author_url": resolved.author_url,
        "posted_at": resolved.posted_at_label,
        "fetched_via": resolved.fetched_via,
        "provider_name": resolved.provider_name,
        "provider_url": resolved.provider_url,
        "image_url": resolved.image_url,
    }
    sections: list[str] = []
    links_section = _render_links_section(resolved.links)
    if links_section:
        sections.append("## Links\n\n" + links_section)
    if settings.include_html and resolved.html:
        sections.append("## Embed HTML\n\n```html\n" + resolved.html.strip() + "\n```")
    lines = [
        _render_markdown_frontmatter(metadata),
        resolved.tweet_text or "(empty)",
    ]
    if sections:
        lines.extend(["***", "\n\n".join(sections)])
    return "\n\n".join(part.strip() for part in lines if part.strip())


def _label_for_target(target: XwitterTarget) -> str:
    author = target.author.strip("@").lower() if target.author else "status"
    author = re.sub(r"[^A-Za-z0-9_]+", "-", author).strip("-") or "status"
    return f"xwitter/{author}/status/{target.tweet_id}"


def _document_from_resolved(
    *,
    target: XwitterTarget,
    source_url: str,
    resolved: _ResolvedTweet,
    settings: XwitterSettings,
) -> XwitterDocument:
    label = _label_for_target(target)
    rendered = _render_tweet_document(
        target=target,
        resolved=resolved,
        settings=settings,
    )
    return XwitterDocument(
        source_url=source_url,
        kind="tweet",
        canonical_url=resolved.canonical_url,
        tweet_id=target.tweet_id,
        author=resolved.author_name or target.author,
        label=label,
        trace_path=label,
        context_subpath=f"{label}.md",
        source_path=target.source_path,
        rendered=rendered,
    )


def _fetch_resolved_tweet(
    target: XwitterTarget,
    *,
    settings: XwitterSettings,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
) -> _ResolvedTweet:
    try:
        return _fetch_oembed(
            target,
            settings=settings,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
        )
    except Exception as exc:
        if not settings.use_alias_fallback:
            raise
        alias = _fetch_alias_metadata(target)
        if alias is None:
            raise
        _log(f"  xwitter oEmbed failed; using alias metadata fallback: {exc}")
        return alias


def resolve_xwitter_url(
    url: str,
    *,
    settings: XwitterSettings | None = None,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
) -> list[XwitterDocument]:
    from .cache import get_cached_api_json, store_api_json

    target = parse_xwitter_target(url)
    if target is None:
        raise ValueError(f"Not an X/Twitter tweet URL: {url}")
    effective_settings = settings if settings is not None else _xwitter_settings_from_env()
    _log(f"Resolving X/Twitter target: {url}")
    cache_identity = _resolution_cache_identity(target.canonical_url, effective_settings)
    if use_cache and not refresh_cache:
        cached_docs = _documents_from_cached_payload(
            get_cached_api_json(cache_identity, ttl=cache_ttl)
        )
        if cached_docs is not None:
            record_progress(
                "xwitter",
                "resolution",
                "cache_hit",
                target=url,
                count=len(cached_docs),
            )
            return cached_docs
        record_progress("xwitter", "resolution", "cache_miss", target=url)

    resolved = _fetch_resolved_tweet(
        target,
        settings=effective_settings,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )
    documents = [
        _document_from_resolved(
            target=target,
            source_url=url,
            resolved=resolved,
            settings=effective_settings,
        )
    ]
    if use_cache:
        store_api_json(cache_identity, [asdict(document) for document in documents])
    record_progress(
        "xwitter",
        "resolution",
        "processed",
        target=url,
        count=len(documents),
    )
    return documents


def xwitter_oembed_url(target: str) -> str:
    parsed = parse_xwitter_target(target)
    if parsed is None:
        raise ValueError(f"Not an X/Twitter tweet URL: {target}")
    query = urlencode({"url": parsed.canonical_url, "omit_script": "true"})
    return f"{_OEMBED_ENDPOINT}?{query}"
