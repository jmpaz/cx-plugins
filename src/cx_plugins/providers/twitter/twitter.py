from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from html import escape as html_escape
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlparse

from ..shared.progress import record_progress

_TWITTER_HOSTS = frozenset(
    {"x.com", "twitter.com", "fixupx.com", "twittpr.com", "fxtwitter.com"}
)
_ALIAS_HOSTS = frozenset({"fixupx.com", "twittpr.com", "fxtwitter.com"})
_FX_API_HOST = "https://api.fxtwitter.com"
_OEMBED_ENDPOINT = "https://publish.x.com/oembed"
_HTTP_TIMEOUT_SECONDS = 30
_TWEET_ID_RE = re.compile(r"^[0-9]+$")
_VALID_MEDIA_MODES = frozenset({"describe", "transcribe"})
_HTTP_HEADERS = {
    "User-Agent": "contextualize/twitter",
    "Accept": "application/json,text/html;q=0.8,*/*;q=0.5",
}
_MEDIA_DOWNLOAD_HEADERS = {
    "User-Agent": _HTTP_HEADERS["User-Agent"],
    "Accept": "image/*,video/*,audio/*,*/*;q=0.5",
}


def _log(message: str) -> None:
    from contextualize.runtime import get_verbose_logging

    if get_verbose_logging():
        print(message, file=sys.stderr, flush=True)


@dataclass(frozen=True)
class TwitterSettings:
    include_html: bool = False
    use_fx_api: bool = True
    include_media_descriptions: bool = True
    media_mode: str = "describe"
    quote_depth: int = 2


@dataclass(frozen=True)
class TwitterTarget:
    kind: str
    original: str
    host: str
    tweet_id: str
    author: str | None
    canonical_url: str
    source_path: str


@dataclass(frozen=True)
class TwitterMedia:
    index: int
    kind: str
    url: str
    id: str | None = None
    width: int | None = None
    height: int | None = None
    alt: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class TwitterDocument:
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
    prose: str
    prose_authors: list[str]
    source_created: str | None = None
    source_modified: str | None = None


@dataclass(frozen=True)
class _TweetLink:
    href: str
    text: str | None
    resolved_url: str | None = None
    quote_target: TwitterTarget | None = None
    is_media: bool = False


@dataclass(frozen=True)
class _ResolvedTweet:
    canonical_url: str
    author_name: str | None
    author_handle: str | None
    author_url: str | None
    tweet_text: str | None
    posted_at_label: str | None
    source_created: str | None
    html: str | None
    provider_name: str | None
    provider_url: str | None
    links: tuple[_TweetLink, ...]
    media: tuple[TwitterMedia, ...]
    quote_target: TwitterTarget | None
    quote_payload: dict[str, Any] | None
    fetched_via: str
    resolution_error: str | None = None


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


def _parse_non_negative_int(value: str, *, default: int) -> int:
    cleaned = value.strip()
    if not cleaned:
        return default
    try:
        parsed = int(cleaned)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def _normalize_non_negative_int_override(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{field} must be >= 0")
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return 0
        try:
            parsed = int(cleaned)
        except ValueError as exc:
            raise ValueError(f"{field} must be a non-negative integer") from exc
        if parsed < 0:
            raise ValueError(f"{field} must be >= 0")
        return parsed
    raise ValueError(f"{field} must be a non-negative integer")


def _normalize_media_mode_override(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    cleaned = value.strip().lower()
    if cleaned not in _VALID_MEDIA_MODES:
        raise ValueError(f"{field} must be one of: describe, transcribe")
    return cleaned


def _twitter_settings_from_env() -> TwitterSettings:
    media_mode = os.environ.get("TWITTER_MEDIA_MODE", "").strip().lower()
    if media_mode not in _VALID_MEDIA_MODES:
        media_mode = "describe"
    return TwitterSettings(
        include_html=_parse_bool(
            os.environ.get("TWITTER_INCLUDE_HTML", ""),
            default=False,
        ),
        use_fx_api=_parse_bool(
            os.environ.get("TWITTER_USE_FX_API", ""),
            default=True,
        ),
        include_media_descriptions=_parse_bool(
            os.environ.get("TWITTER_MEDIA_DESCRIPTIONS", ""),
            default=True,
        ),
        media_mode=media_mode,
        quote_depth=_parse_non_negative_int(
            os.environ.get("TWITTER_QUOTE_DEPTH", ""),
            default=2,
        ),
    )


def build_twitter_settings(overrides: dict[str, Any] | None = None) -> TwitterSettings:
    env = _twitter_settings_from_env()
    if not overrides:
        return env
    include_html = env.include_html
    if "include_html" in overrides:
        include_html = _normalize_bool_override(
            overrides["include_html"],
            field="include_html",
        )
    use_fx_api = env.use_fx_api
    if "use_fx_api" in overrides:
        use_fx_api = _normalize_bool_override(
            overrides["use_fx_api"],
            field="use_fx_api",
        )
    include_media_descriptions = env.include_media_descriptions
    if "include_media_descriptions" in overrides:
        include_media_descriptions = _normalize_bool_override(
            overrides["include_media_descriptions"],
            field="include_media_descriptions",
        )
    media_mode = env.media_mode
    if "media_mode" in overrides:
        media_mode = _normalize_media_mode_override(
            overrides["media_mode"],
            field="media_mode",
        )
    quote_depth = env.quote_depth
    if "quote_depth" in overrides:
        quote_depth = _normalize_non_negative_int_override(
            overrides["quote_depth"],
            field="quote_depth",
        )
    return TwitterSettings(
        include_html=include_html,
        use_fx_api=use_fx_api,
        include_media_descriptions=include_media_descriptions,
        media_mode=media_mode,
        quote_depth=quote_depth,
    )


def twitter_settings_cache_key(settings: TwitterSettings) -> tuple[Any, ...]:
    return (
        "v1",
        settings.include_html,
        settings.use_fx_api,
        settings.include_media_descriptions,
        settings.media_mode,
        settings.quote_depth,
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


def parse_twitter_target(value: str) -> TwitterTarget | None:
    raw = value.strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    if parsed.scheme.lower() not in {"http", "https"}:
        return None
    host = _normalize_host(parsed.hostname)
    if host not in _TWITTER_HOSTS:
        return None
    tweet = _parse_tweet_path(parsed.path)
    if tweet is None:
        return None
    author, tweet_id = tweet
    return TwitterTarget(
        kind="tweet",
        original=value,
        host=host,
        tweet_id=tweet_id,
        author=author,
        canonical_url=_canonical_url(author, tweet_id),
        source_path=_source_path(author, tweet_id),
    )


def is_twitter_url(value: str) -> bool:
    return parse_twitter_target(value) is not None


def _target_from_parts(
    *,
    author: str | None,
    tweet_id: str,
    original: str | None = None,
) -> TwitterTarget:
    canonical = _canonical_url(author, tweet_id)
    return TwitterTarget(
        kind="tweet",
        original=original or canonical,
        host="x.com",
        tweet_id=tweet_id,
        author=author,
        canonical_url=canonical,
        source_path=_source_path(author, tweet_id),
    )


def _http_get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: int,
) -> dict[str, Any]:
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
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _created_at_iso(tweet: dict[str, Any]) -> str | None:
    timestamp = _int(tweet.get("created_timestamp") or tweet.get("date_epoch"))
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _fx_api_url(target: TwitterTarget) -> str:
    if target.author:
        author = quote(target.author.strip("@"))
        return f"{_FX_API_HOST}/{author}/status/{target.tweet_id}"
    return f"{_FX_API_HOST}/i/status/{target.tweet_id}"


def _fx_api_cache_identity(target: TwitterTarget) -> str:
    return "twitter-fx-api:" + json.dumps(
        {"author": target.author, "tweet_id": target.tweet_id},
        sort_keys=True,
        separators=(",", ":"),
    )


def _oembed_cache_identity(target: TwitterTarget) -> str:
    return "twitter-oembed:" + target.canonical_url


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
    parsed = parse_twitter_target(url.split("?", 1)[0])
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
    return "twitter-tco:" + url.strip()


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
        _log(f"  twitter t.co resolution failed for {url}: {exc}")
        return None
    if final_url and use_cache:
        store_api_json(identity, {"url": url, "resolved_url": final_url})
    return final_url


def _resolve_tco_links(
    links: tuple[_TweetLink, ...],
    *,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
) -> tuple[_TweetLink, ...]:
    resolved: list[_TweetLink] = []
    for link in links:
        final_url = _resolve_tco_url(
            link.href,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
        )
        resolved.append(replace(link, resolved_url=final_url or link.resolved_url))
    return tuple(resolved)


def _twitter_target_for_link(link: _TweetLink) -> TwitterTarget | None:
    for candidate in (link.resolved_url, link.href):
        if not candidate:
            continue
        parsed = parse_twitter_target(candidate)
        if parsed is not None:
            return parsed
    return None


def _is_photo_or_video_path(url: str | None) -> bool:
    if not url:
        return False
    parts = [part for part in urlparse(url).path.split("/") if part]
    return "photo" in parts or "video" in parts


def _is_media_placeholder(link: _TweetLink, *, current_tweet_id: str) -> bool:
    text = (link.text or "").strip().lower()
    if text.startswith(("pic.twitter.com/", "pic.x.com/")):
        return True
    target = _twitter_target_for_link(link)
    return (
        target is not None
        and target.tweet_id == current_tweet_id
        and _is_photo_or_video_path(link.resolved_url)
    )


def _annotate_oembed_links(
    links: tuple[_TweetLink, ...],
    *,
    current_tweet_id: str,
    classify_quotes: bool,
) -> tuple[_TweetLink, ...]:
    annotated: list[_TweetLink] = []
    for link in links:
        target = _twitter_target_for_link(link)
        quote_target = None
        is_media = _is_media_placeholder(link, current_tweet_id=current_tweet_id)
        if (
            classify_quotes
            and not is_media
            and target is not None
            and target.tweet_id != current_tweet_id
        ):
            quote_target = target
        resolved_url = target.canonical_url if target is not None else link.resolved_url
        annotated.append(
            replace(
                link,
                resolved_url=resolved_url,
                quote_target=quote_target,
                is_media=is_media,
            )
        )
    return tuple(annotated)


def _tweet_text_with_resolved_links(
    tweet_text: str | None,
    links: tuple[_TweetLink, ...],
) -> str | None:
    if not tweet_text:
        return tweet_text
    rendered = tweet_text
    for link in links:
        candidates = [item for item in (link.text, link.href) if item]
        if link.quote_target is not None or link.is_media:
            for candidate in candidates:
                if candidate in rendered:
                    rendered = rendered.replace(candidate, "")
                    break
            continue
        if not link.resolved_url:
            continue
        for candidate in candidates:
            if candidate in rendered:
                rendered = rendered.replace(candidate, link.resolved_url)
                break
    return _clean_text(rendered)


def _target_from_fx_tweet(
    tweet: dict[str, Any],
    *,
    fallback: TwitterTarget | None = None,
) -> TwitterTarget | None:
    tweet_id = _string(tweet.get("id") or tweet.get("tweetID"))
    if tweet_id is None:
        return fallback
    author = tweet.get("author")
    handle = None
    if isinstance(author, dict):
        handle = _string(author.get("screen_name") or author.get("screenName"))
    if handle is None:
        handle = _string(tweet.get("user_screen_name")) or (
            fallback.author if fallback is not None else None
        )
    original = _string(tweet.get("url") or tweet.get("tweetURL"))
    return _target_from_parts(author=handle, tweet_id=tweet_id, original=original)


def _text_from_fx_tweet(tweet: dict[str, Any]) -> str | None:
    raw_text = tweet.get("raw_text")
    if isinstance(raw_text, dict):
        text = _string(raw_text.get("text"))
        facets = raw_text.get("facets")
    else:
        text = None
        facets = None
    if text is None:
        text = _string(tweet.get("text"))
    if text is None:
        return None
    if isinstance(facets, list):
        for facet in facets:
            if not isinstance(facet, dict):
                continue
            original = _string(facet.get("original"))
            if original is None:
                continue
            facet_type = _string(facet.get("type")) or ""
            if facet_type == "media":
                text = text.replace(original, "")
                continue
            replacement = _string(facet.get("replacement")) or _string(
                facet.get("display")
            )
            if replacement:
                text = text.replace(original, replacement)
    return _clean_text(text)


def _media_kind(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"photo", "image"}:
        return "image"
    if raw in {"animated_gif", "gif"}:
        return "video"
    if raw in {"video", "audio"}:
        return raw
    return "file"


def _image_suffix(url: str) -> str:
    path = urlparse(url).path
    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".mp4", ".mov"}:
        return suffix
    return ".jpg"


def _media_render_cache_identity(
    *,
    media_cache_identity: str,
    prompt_append: str | None,
    suffix: str,
    mode: str,
    media_kind: str,
) -> str:
    return "twitter-media-render:" + json.dumps(
        {
            "media": media_cache_identity,
            "prompt": prompt_append,
            "suffix": suffix,
            "mode": mode,
            "kind": media_kind,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _should_refresh_kind(kind: str) -> bool:
    from contextualize.runtime import (
        get_refresh_audio,
        get_refresh_images,
        get_refresh_media,
        get_refresh_videos,
    )

    if get_refresh_media():
        return True
    if kind == "image":
        return get_refresh_images()
    if kind == "video":
        return get_refresh_videos()
    if kind == "audio":
        return get_refresh_audio()
    return False


def _normalize_media_description(markdown: str) -> str:
    text = markdown.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
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
    return "\n".join(lines).strip()


def _describe_media(
    media: TwitterMedia,
    *,
    settings: TwitterSettings,
    tweet_text: str | None,
    total_media: int,
) -> str | None:
    if not settings.include_media_descriptions:
        return None
    if media.alt:
        return media.alt

    from .cache import (
        get_cached_media_bytes,
        get_cached_rendered,
        store_media_bytes,
        store_rendered,
    )
    from ..shared.media import download_cached_media_to_temp

    suffix = _image_suffix(media.url)
    cache_identity = f"twitter:media:{media.id or media.url}"
    prompt_parts = [
        (
            f"Twitter {media.kind} {media.index} of {total_media}. "
            "Write concise alt text for this media item."
        )
    ]
    if tweet_text:
        prompt_parts.insert(0, f"Tweet text: {tweet_text}")
    prompt_append = "\n".join(prompt_parts)
    render_identity = _media_render_cache_identity(
        media_cache_identity=cache_identity,
        prompt_append=prompt_append,
        suffix=suffix,
        mode=settings.media_mode,
        media_kind=media.kind,
    )
    refresh_for_kind = _should_refresh_kind(media.kind)
    if not refresh_for_kind:
        cached = get_cached_rendered(render_identity)
        if isinstance(cached, str) and cached.strip():
            return cached.strip()

    tmp = download_cached_media_to_temp(
        media.url,
        suffix=suffix,
        headers=_MEDIA_DOWNLOAD_HEADERS,
        cache_identity=cache_identity,
        get_cached_media_bytes=get_cached_media_bytes,
        store_media_bytes=store_media_bytes,
        refresh_cache=refresh_for_kind,
    )
    if tmp is None:
        return None
    try:
        markdown = ""
        if settings.media_mode == "transcribe" and media.kind in {"audio", "video"}:
            from contextualize.transcription import transcribe_audio_file

            markdown = transcribe_audio_file(tmp)
        if not markdown:
            from contextualize.render.markitdown import convert_path_to_markdown

            result = convert_path_to_markdown(
                str(tmp),
                refresh_images=refresh_for_kind,
                prompt_append=prompt_append,
                source_url=media.url,
            )
            markdown = result.markdown
        description = _normalize_media_description(markdown or "")
        if description:
            store_rendered(render_identity, description)
            return description
    except Exception as exc:
        _log(f"  twitter media description failed for {media.url}: {exc}")
    finally:
        tmp.unlink(missing_ok=True)
    return None


def _media_from_fx_tweet(
    tweet: dict[str, Any],
    *,
    settings: TwitterSettings,
    tweet_text: str | None,
) -> tuple[TwitterMedia, ...]:
    media = tweet.get("media")
    if not isinstance(media, dict):
        return ()
    raw_items = media.get("all")
    if not isinstance(raw_items, list):
        raw_items = []
        seen_urls: set[str] = set()
        for key in ("photos", "videos", "animated_gifs"):
            value = media.get(key)
            if not isinstance(value, list):
                continue
            for item in value:
                if not isinstance(item, dict):
                    continue
                url = _string(item.get("url"))
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    raw_items.append(item)
    items: list[TwitterMedia] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        url = _string(raw_item.get("url") or raw_item.get("thumbnail_url"))
        if url is None:
            continue
        original_info = raw_item.get("original_info")
        size = raw_item.get("size")
        width = _int(raw_item.get("width"))
        height = _int(raw_item.get("height"))
        if isinstance(original_info, dict):
            width = width or _int(original_info.get("width"))
            height = height or _int(original_info.get("height"))
        if isinstance(size, dict):
            width = width or _int(size.get("width"))
            height = height or _int(size.get("height"))
        item = TwitterMedia(
            index=len(items) + 1,
            kind=_media_kind(raw_item.get("type")),
            url=url,
            id=_string(raw_item.get("id") or raw_item.get("id_str")),
            width=width,
            height=height,
            alt=_string(
                raw_item.get("alt")
                or raw_item.get("altText")
                or raw_item.get("ext_alt_text")
            ),
        )
        items.append(item)

    total = len(items)
    described: list[TwitterMedia] = []
    for item in items:
        try:
            description = _describe_media(
                item,
                settings=settings,
                tweet_text=tweet_text,
                total_media=total,
            )
        except Exception as exc:
            _log(f"  twitter media description failed for {item.url}: {exc}")
            description = None
        described.append(replace(item, description=description))
    return tuple(described)


def _author_name_from_fx(tweet: dict[str, Any]) -> str | None:
    author = tweet.get("author")
    if isinstance(author, dict):
        return _string(author.get("name") or author.get("screen_name"))
    return _string(tweet.get("user_name") or tweet.get("user_screen_name"))


def _author_handle_from_fx(tweet: dict[str, Any]) -> str | None:
    author = tweet.get("author")
    if isinstance(author, dict):
        return _string(author.get("screen_name") or author.get("screenName"))
    return _string(tweet.get("user_screen_name"))


def _author_url_from_fx(tweet: dict[str, Any], handle: str | None) -> str | None:
    author = tweet.get("author")
    if isinstance(author, dict):
        url = _string(author.get("url"))
        if url:
            return url
    if handle:
        return f"https://x.com/{quote(handle.strip('@'))}"
    return None


def _resolved_from_fx_tweet(
    tweet: dict[str, Any],
    *,
    target: TwitterTarget,
    settings: TwitterSettings,
) -> _ResolvedTweet:
    parsed_target = _target_from_fx_tweet(tweet, fallback=target) or target
    author_handle = _author_handle_from_fx(tweet) or parsed_target.author
    tweet_text = _text_from_fx_tweet(tweet)
    quote_payload = tweet.get("quote")
    if not isinstance(quote_payload, dict):
        quote_payload = None
    quote_target = (
        _target_from_fx_tweet(quote_payload) if quote_payload is not None else None
    )
    return _ResolvedTweet(
        canonical_url=parsed_target.canonical_url,
        author_name=_author_name_from_fx(tweet) or author_handle,
        author_handle=author_handle,
        author_url=_author_url_from_fx(tweet, author_handle),
        tweet_text=tweet_text,
        posted_at_label=_string(tweet.get("created_at") or tweet.get("date")),
        source_created=_created_at_iso(tweet),
        html=None,
        provider_name="FxTwitter",
        provider_url="https://api.fxtwitter.com",
        links=(),
        media=_media_from_fx_tweet(
            tweet,
            settings=settings,
            tweet_text=tweet_text,
        ),
        quote_target=quote_target,
        quote_payload=quote_payload,
        fetched_via="fxtwitter-api",
    )


def _fetch_fx_api(
    target: TwitterTarget,
    *,
    settings: TwitterSettings,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
) -> _ResolvedTweet:
    from .cache import get_cached_api_json, store_api_json

    identity = _fx_api_cache_identity(target)
    payload: Any | None = None
    if use_cache and not refresh_cache:
        payload = get_cached_api_json(identity, ttl=cache_ttl)
    if not isinstance(payload, dict):
        payload = _http_get_json(
            _fx_api_url(target),
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
        if use_cache:
            store_api_json(identity, payload)
    tweet = payload.get("tweet")
    if not isinstance(tweet, dict):
        raise ValueError(f"Unexpected FxTwitter payload for {target.canonical_url}")
    return _resolved_from_fx_tweet(tweet, target=target, settings=settings)


def _fetch_oembed(
    target: TwitterTarget,
    *,
    settings: TwitterSettings,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
    classify_quotes: bool,
    resolution_error: str | None = None,
) -> _ResolvedTweet:
    from .cache import get_cached_api_json, store_api_json

    identity = _oembed_cache_identity(target)
    payload: Any | None = None
    if use_cache and not refresh_cache:
        payload = get_cached_api_json(identity, ttl=cache_ttl)
    if not isinstance(payload, dict):
        payload = _http_get_json(
            _OEMBED_ENDPOINT,
            params={"url": target.canonical_url, "omit_script": "true"},
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
        if use_cache:
            store_api_json(identity, payload)
    html = payload.get("html")
    html_text = html if isinstance(html, str) else ""
    tweet_text, posted_at_label, links = _parse_oembed_html(html_text)
    links = _resolve_tco_links(
        _filtered_links(links, tweet_id=target.tweet_id),
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )
    links = _annotate_oembed_links(
        links,
        current_tweet_id=target.tweet_id,
        classify_quotes=classify_quotes,
    )
    canonical_url = payload.get("url")
    author_name = payload.get("author_name")
    author_url = payload.get("author_url")
    provider_name = payload.get("provider_name")
    provider_url = payload.get("provider_url")
    quote_targets = _quote_targets_from_links(links)
    return _ResolvedTweet(
        canonical_url=canonical_url
        if isinstance(canonical_url, str)
        else target.canonical_url,
        author_name=author_name if isinstance(author_name, str) else target.author,
        author_handle=target.author,
        author_url=author_url if isinstance(author_url, str) else None,
        tweet_text=_tweet_text_with_resolved_links(tweet_text, links),
        posted_at_label=posted_at_label,
        source_created=None,
        html=html_text or None,
        provider_name=provider_name if isinstance(provider_name, str) else None,
        provider_url=provider_url if isinstance(provider_url, str) else None,
        links=links,
        media=(),
        quote_target=quote_targets[0] if quote_targets else None,
        quote_payload=None,
        fetched_via="x-oembed",
        resolution_error=resolution_error,
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


def _fetch_alias_metadata(
    target: TwitterTarget,
    *,
    resolution_error: str | None = None,
) -> _ResolvedTweet | None:
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
    parsed = parse_twitter_target(canonical)
    canonical_url = parsed.canonical_url if parsed is not None else target.canonical_url
    author = meta.get("twitter:creator") or target.author
    author = author.strip("@") if isinstance(author, str) else target.author
    author_url = f"https://x.com/{author}" if author else None
    return _ResolvedTweet(
        canonical_url=canonical_url,
        author_name=author,
        author_handle=author,
        author_url=author_url,
        tweet_text=meta.get("og:description"),
        posted_at_label=None,
        source_created=None,
        html=None,
        provider_name=meta.get("og:site_name"),
        provider_url=f"https://{target.host}",
        links=(),
        media=(),
        quote_target=None,
        quote_payload=None,
        fetched_via=f"{target.host}-open-graph",
        resolution_error=resolution_error,
    )


def _fetch_resolved_tweet(
    target: TwitterTarget,
    *,
    settings: TwitterSettings,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
    classify_quotes: bool,
) -> _ResolvedTweet:
    fx_error = None
    if settings.use_fx_api:
        try:
            return _fetch_fx_api(
                target,
                settings=settings,
                use_cache=use_cache,
                cache_ttl=cache_ttl,
                refresh_cache=refresh_cache,
            )
        except Exception as exc:
            fx_error = f"FxTwitter API failed: {exc}"
            _log(f"  twitter FxTwitter API failed; falling back to oEmbed: {exc}")
    try:
        return _fetch_oembed(
            target,
            settings=settings,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
            classify_quotes=classify_quotes,
            resolution_error=fx_error,
        )
    except Exception as exc:
        alias = _fetch_alias_metadata(
            target,
            resolution_error=fx_error or f"X oEmbed failed: {exc}",
        )
        if alias is not None:
            _log(f"  twitter oEmbed failed; using alias metadata fallback: {exc}")
            return alias
        raise


def _quote_targets_from_links(links: tuple[_TweetLink, ...]) -> list[TwitterTarget]:
    targets: list[TwitterTarget] = []
    seen: set[str] = set()
    for link in links:
        target = link.quote_target
        if target is None or target.tweet_id in seen:
            continue
        seen.add(target.tweet_id)
        targets.append(target)
    return targets


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


def _append_literal_field(lines: list[str], key: str, value: str) -> None:
    cleaned = value.strip()
    if not cleaned:
        return
    if "\n" not in cleaned:
        lines.append(f"{key}: {cleaned}")
        return
    lines.append(f"{key}: |")
    lines.extend(f"  {line}" if line else "  " for line in cleaned.splitlines())


def _render_media_section(media_items: tuple[TwitterMedia, ...]) -> str | None:
    if not media_items:
        return None
    lines = ["## Media"]
    for media in media_items:
        attrs = [
            f'index="{media.index}"',
            f'type="{html_escape(media.kind, quote=True)}"',
            f'url="{html_escape(media.url, quote=True)}"',
        ]
        if media.id:
            attrs.append(f'id="{html_escape(media.id, quote=True)}"')
        if media.width is not None and media.height is not None:
            attrs.append(f'width="{media.width}"')
            attrs.append(f'height="{media.height}"')
        tag = "image" if media.kind == "image" else media.kind
        lines.extend(["", f"<{tag} {' '.join(attrs)}>"])
        if media.alt:
            _append_literal_field(lines, "native-alt", media.alt)
        if media.description and media.description != media.alt:
            _append_literal_field(lines, "description", media.description)
        if not media.alt and not media.description:
            lines.append("Description unavailable.")
        lines.append(f"</{tag}>")
    return "\n".join(lines)


def _render_tweet_document(
    *,
    target: TwitterTarget,
    resolved: _ResolvedTweet,
    settings: TwitterSettings,
) -> str:
    quoted_tweet = resolved.quote_target
    metadata = {
        "url": resolved.canonical_url,
        "source_url": target.original,
        "tweet_id": target.tweet_id,
        "author": resolved.author_name or resolved.author_handle or target.author,
        "author_url": resolved.author_url,
        "posted_at": resolved.posted_at_label,
        "source_created": resolved.source_created,
        "fetched_via": resolved.fetched_via,
        "provider_name": resolved.provider_name,
        "provider_url": resolved.provider_url,
        "media_count": len(resolved.media) if resolved.media else None,
        "quoted_tweet_url": quoted_tweet.canonical_url if quoted_tweet else None,
        "quoted_tweet_id": quoted_tweet.tweet_id if quoted_tweet else None,
        "resolution_error": resolved.resolution_error,
    }
    sections: list[str] = []
    media_section = _render_media_section(resolved.media)
    if media_section:
        sections.append(media_section)
    if settings.include_html and resolved.html:
        sections.append("## Embed HTML\n\n```html\n" + resolved.html.strip() + "\n```")
    lines = [
        _render_markdown_frontmatter(metadata),
        resolved.tweet_text or "(empty)",
    ]
    if sections:
        lines.extend(["***", "\n\n".join(sections)])
    return "\n\n".join(part.strip() for part in lines if part.strip())


def _label_for_target(target: TwitterTarget) -> str:
    author = target.author.strip("@").lower() if target.author else "status"
    author = re.sub(r"[^A-Za-z0-9_]+", "-", author).strip("-") or "status"
    return f"twitter/{author}/status/{target.tweet_id}"


def _document_from_resolved(
    *,
    target: TwitterTarget,
    source_url: str,
    resolved: _ResolvedTweet,
    settings: TwitterSettings,
) -> TwitterDocument:
    label = _label_for_target(target)
    rendered = _render_tweet_document(
        target=target,
        resolved=resolved,
        settings=settings,
    )
    author = resolved.author_name or resolved.author_handle or target.author
    return TwitterDocument(
        source_url=source_url,
        kind="tweet",
        canonical_url=resolved.canonical_url,
        tweet_id=target.tweet_id,
        author=author,
        label=label,
        trace_path=label,
        context_subpath=f"{label}.md",
        source_path=target.source_path,
        rendered=rendered,
        prose=resolved.tweet_text or "",
        prose_authors=[author] if author else [],
        source_created=resolved.source_created,
    )


def _collect_twitter_documents(
    *,
    target: TwitterTarget,
    source_url: str,
    settings: TwitterSettings,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
    remaining_quote_depth: int,
    emitted_tweet_ids: set[str],
    visiting_tweet_ids: set[str],
    preloaded_fx_payload: dict[str, Any] | None = None,
) -> list[TwitterDocument]:
    if target.tweet_id in emitted_tweet_ids or target.tweet_id in visiting_tweet_ids:
        return []
    visiting_tweet_ids.add(target.tweet_id)
    try:
        if preloaded_fx_payload is not None:
            resolved = _resolved_from_fx_tweet(
                preloaded_fx_payload,
                target=target,
                settings=settings,
            )
        else:
            resolved = _fetch_resolved_tweet(
                target,
                settings=settings,
                use_cache=use_cache,
                cache_ttl=cache_ttl,
                refresh_cache=refresh_cache,
                classify_quotes=remaining_quote_depth > 0,
            )
        nested_documents: list[TwitterDocument] = []
        if remaining_quote_depth > 0 and resolved.quote_target is not None:
            quote_target = resolved.quote_target
            if (
                quote_target.tweet_id not in emitted_tweet_ids
                and quote_target.tweet_id not in visiting_tweet_ids
            ):
                try:
                    nested_documents = _collect_twitter_documents(
                        target=quote_target,
                        source_url=quote_target.canonical_url,
                        settings=settings,
                        use_cache=use_cache,
                        cache_ttl=cache_ttl,
                        refresh_cache=refresh_cache,
                        remaining_quote_depth=remaining_quote_depth - 1,
                        emitted_tweet_ids=emitted_tweet_ids,
                        visiting_tweet_ids=visiting_tweet_ids,
                        preloaded_fx_payload=resolved.quote_payload,
                    )
                except Exception as exc:
                    _log(
                        "  twitter quote fetch failed for "
                        f"{quote_target.canonical_url}: {exc}"
                    )
        document = _document_from_resolved(
            target=target,
            source_url=source_url,
            resolved=resolved,
            settings=settings,
        )
        emitted_tweet_ids.add(target.tweet_id)
        return [document, *nested_documents]
    finally:
        visiting_tweet_ids.discard(target.tweet_id)


def resolve_twitter_url(
    url: str,
    *,
    settings: TwitterSettings | None = None,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
) -> list[TwitterDocument]:
    target = parse_twitter_target(url)
    if target is None:
        raise ValueError(f"Not an X/Twitter tweet URL: {url}")
    effective_settings = settings if settings is not None else _twitter_settings_from_env()
    _log(f"Resolving X/Twitter target: {url}")
    record_progress("twitter", "resolution", "cache_miss", target=url)
    documents = _collect_twitter_documents(
        target=target,
        source_url=url,
        settings=effective_settings,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
        remaining_quote_depth=effective_settings.quote_depth,
        emitted_tweet_ids=set(),
        visiting_tweet_ids=set(),
    )
    record_progress(
        "twitter",
        "resolution",
        "processed",
        target=url,
        count=len(documents),
    )
    return documents


def twitter_oembed_url(target: str) -> str:
    parsed = parse_twitter_target(target)
    if parsed is None:
        raise ValueError(f"Not an X/Twitter tweet URL: {target}")
    query = urlencode({"url": parsed.canonical_url, "omit_script": "true"})
    return f"{_OEMBED_ENDPOINT}?{query}"
