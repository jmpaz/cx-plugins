from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
import hashlib
import html
import os
import re
import sys
import time
from typing import Any
from urllib.parse import urljoin, urlparse

from selectolax.parser import HTMLParser

_DEFAULT_TIMEOUT_SECONDS = 20.0
_DEFAULT_MAX_COMMENTS = 500
_DEFAULT_COMMENT_BATCH_SIZE = 100
_DEFAULT_COMMENT_VIEW = "postCommentsOld"
_DEFAULT_USER_AGENT = (
    "contextualize-forum-magnum/0.1 "
    "(https://github.com/jmpaz/contextualize) requests"
)
_POST_RE = re.compile(r"^/posts/(?P<id>[A-Za-z0-9]+)(?:/|$)")
_SEQUENCE_POST_RE = re.compile(r"^/s/[A-Za-z0-9]+/p/(?P<id>[A-Za-z0-9]+)(?:/|$)")
_SHORT_POST_RE = re.compile(r"^/p/(?P<id>[A-Za-z0-9]+)(?:/|$)")
_SAFE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SKIP_RENDER_TAGS = {"script", "style", "noscript", "svg"}
_BLOCK_TAGS = {
    "article",
    "aside",
    "div",
    "figure",
    "figcaption",
    "li",
    "main",
    "p",
    "section",
}

POST_QUERY = """
query GetPost($id: String!) {
  post(input: {selector: {_id: $id}}) {
    result {
      _id
      title
      slug
      pageUrl
      postedAt
      baseScore
      voteCount
      commentCount
      contents { html }
      user { username displayName }
      tags { name }
    }
  }
}
"""

COMMENTS_QUERY = """
query GetComments($terms: JSON) {
  comments(input: {terms: $terms}) {
    results {
      _id
      postId
      parentCommentId
      pageUrl
      contents { html }
      baseScore
      postedAt
      user { username displayName }
    }
  }
}
"""


@dataclass(frozen=True)
class ForumSite:
    platform: str
    source_ref: str
    endpoint: str
    canonical_base_url: str


@dataclass(frozen=True)
class ParsedForumMagnumTarget:
    raw_target: str
    platform: str
    input_host: str
    source_ref: str
    endpoint: str
    canonical_base_url: str
    post_id: str
    source_path: str

    @property
    def canonical_id(self) -> str:
        return f"{self.platform}:post:{self.post_id}"


@dataclass(frozen=True)
class ForumMagnumSettings:
    include_comments: bool = True
    max_comments: int = _DEFAULT_MAX_COMMENTS
    comment_batch_size: int = _DEFAULT_COMMENT_BATCH_SIZE
    comment_view: str = _DEFAULT_COMMENT_VIEW


@dataclass(frozen=True)
class ForumMagnumResolvedDocument:
    label: str
    rendered: str
    prose: str
    prose_authors: tuple[str, ...]
    trace_path: str
    source_ref: str
    source_path: str
    context_subpath: str
    source_created: str | None
    source_modified: str | None
    kind: str
    platform: str
    canonical_id: str
    canonical_url: str
    post_id: str
    comment_id: str | None
    parent_comment_id: str | None
    author: str | None
    score: int | float | None
    dedupe_rank: int


_SITES: dict[str, ForumSite] = {
    "lesswrong.com": ForumSite(
        platform="lesswrong",
        source_ref="lesswrong.com",
        endpoint="https://www.lesswrong.com/graphql",
        canonical_base_url="https://www.lesswrong.com",
    ),
    "www.lesswrong.com": ForumSite(
        platform="lesswrong",
        source_ref="lesswrong.com",
        endpoint="https://www.lesswrong.com/graphql",
        canonical_base_url="https://www.lesswrong.com",
    ),
    "greaterwrong.com": ForumSite(
        platform="lesswrong",
        source_ref="lesswrong.com",
        endpoint="https://www.lesswrong.com/graphql",
        canonical_base_url="https://www.lesswrong.com",
    ),
    "www.greaterwrong.com": ForumSite(
        platform="lesswrong",
        source_ref="lesswrong.com",
        endpoint="https://www.lesswrong.com/graphql",
        canonical_base_url="https://www.lesswrong.com",
    ),
    "alignmentforum.org": ForumSite(
        platform="alignment_forum",
        source_ref="alignmentforum.org",
        endpoint="https://www.alignmentforum.org/graphql",
        canonical_base_url="https://www.alignmentforum.org",
    ),
    "www.alignmentforum.org": ForumSite(
        platform="alignment_forum",
        source_ref="alignmentforum.org",
        endpoint="https://www.alignmentforum.org/graphql",
        canonical_base_url="https://www.alignmentforum.org",
    ),
    "forum.effectivealtruism.org": ForumSite(
        platform="ea_forum",
        source_ref="forum.effectivealtruism.org",
        endpoint="https://forum.effectivealtruism.org/graphql",
        canonical_base_url="https://forum.effectivealtruism.org",
    ),
}

_last_request_at = 0.0


def build_forum_magnum_settings(
    raw: dict[str, Any] | None,
) -> ForumMagnumSettings:
    raw = raw or {}
    return ForumMagnumSettings(
        include_comments=_parse_bool(
            raw.get("include_comments"),
            default=True,
        ),
        max_comments=_parse_int(
            raw.get("max_comments"),
            default=_DEFAULT_MAX_COMMENTS,
            minimum=0,
        ),
        comment_batch_size=_parse_int(
            raw.get("comment_batch_size"),
            default=_DEFAULT_COMMENT_BATCH_SIZE,
            minimum=1,
        ),
        comment_view=_parse_comment_view(raw.get("comment_view")),
    )


def forum_magnum_settings_cache_key(settings: ForumMagnumSettings) -> tuple[Any, ...]:
    return (
        ("include_comments", settings.include_comments),
        ("max_comments", settings.max_comments),
        ("comment_batch_size", settings.comment_batch_size),
        ("comment_view", settings.comment_view),
    )


def parse_forum_magnum_target(target: str) -> ParsedForumMagnumTarget | None:
    cleaned = target.strip()
    if not cleaned:
        return None
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"}:
        return None
    host = parsed.netloc.lower()
    site = _SITES.get(host)
    if site is None:
        return None
    post_id = _post_id_from_path(parsed.path)
    if post_id is None:
        return None
    source_path = f"{site.platform}/posts/{post_id}"
    return ParsedForumMagnumTarget(
        raw_target=cleaned,
        platform=site.platform,
        input_host=host,
        source_ref=site.source_ref,
        endpoint=site.endpoint,
        canonical_base_url=site.canonical_base_url,
        post_id=post_id,
        source_path=source_path,
    )


def is_forum_magnum_target(target: str) -> bool:
    return parse_forum_magnum_target(target) is not None


def resolve_forum_magnum_url(
    target: str,
    *,
    settings: ForumMagnumSettings,
    use_cache: bool,
    cache_ttl: Any,
    refresh_cache: bool,
) -> list[ForumMagnumResolvedDocument]:
    parsed = parse_forum_magnum_target(target)
    if parsed is None:
        raise ValueError(f"Unsupported ForumMagnum target: {target}")

    cache_identity = _document_cache_identity(parsed, settings)
    if use_cache and not refresh_cache:
        from .cache import get_cached_document

        cached = get_cached_document(
            cache_identity,
            ttl=_cache_ttl_as_timedelta(cache_ttl),
        )
        documents = _documents_from_cached_payload(cached)
        if documents is not None:
            _log(f"  forum_magnum resolution cache hit: {target}")
            return documents

    post = _fetch_post(parsed)
    comments: list[dict[str, Any]] = []
    if settings.include_comments and settings.max_comments > 0:
        comment_count = _int_or_none(post.get("commentCount")) or 0
        if comment_count > 0:
            comments = _fetch_comments(parsed, settings=settings)

    documents = _render_documents(parsed, post, comments)
    if use_cache:
        from .cache import store_document

        store_document(cache_identity, [asdict(document) for document in documents])
    return documents


def _post_id_from_path(path: str) -> str | None:
    for pattern in (_SEQUENCE_POST_RE, _POST_RE, _SHORT_POST_RE):
        match = pattern.match(path)
        if match:
            return match.group("id")
    return None


def _fetch_post(parsed: ParsedForumMagnumTarget) -> dict[str, Any]:
    payload = _graphql(
        parsed.endpoint,
        POST_QUERY,
        {"id": parsed.post_id},
    )
    post = payload.get("data", {}).get("post", {}).get("result")
    if not isinstance(post, dict):
        raise RuntimeError(f"ForumMagnum post not found: {parsed.post_id}")
    return post


def _fetch_comments(
    parsed: ParsedForumMagnumTarget,
    *,
    settings: ForumMagnumSettings,
) -> list[dict[str, Any]]:
    comments: list[dict[str, Any]] = []
    offset = 0
    batch_size = min(settings.comment_batch_size, max(settings.max_comments, 1))
    while len(comments) < settings.max_comments:
        remaining = settings.max_comments - len(comments)
        limit = min(batch_size, remaining)
        payload = _graphql(
            parsed.endpoint,
            COMMENTS_QUERY,
            {
                "terms": {
                    "postId": parsed.post_id,
                    "limit": limit,
                    "offset": offset,
                    "view": settings.comment_view,
                }
            },
        )
        raw_batch = payload.get("data", {}).get("comments", {}).get("results")
        if not isinstance(raw_batch, list):
            return comments
        batch = [item for item in raw_batch if isinstance(item, dict)]
        comments.extend(batch)
        if len(batch) < limit:
            return comments
        offset += len(batch)
    return comments


def _graphql(endpoint: str, query: str, variables: dict[str, Any]) -> dict[str, Any]:
    import requests

    _pace_requests()
    response = requests.post(
        endpoint,
        headers={
            "User-Agent": _api_user_agent(),
            "Accept": "application/json",
        },
        json={"query": query, "variables": variables},
        timeout=_api_timeout_seconds(),
        allow_redirects=False,
    )
    if response.is_redirect:
        location = response.headers.get("Location") or "unknown redirect target"
        raise RuntimeError(f"ForumMagnum GraphQL endpoint redirected to {location}")
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("ForumMagnum GraphQL response was not a JSON object")
    errors = payload.get("errors")
    if isinstance(errors, list) and errors:
        first = errors[0]
        if isinstance(first, dict):
            message = first.get("message")
            raise RuntimeError(str(message or "ForumMagnum GraphQL error"))
        raise RuntimeError("ForumMagnum GraphQL error")
    return payload


def _render_documents(
    parsed: ParsedForumMagnumTarget,
    post: dict[str, Any],
    comments: list[dict[str, Any]],
) -> list[ForumMagnumResolvedDocument]:
    post_doc = _render_post_document(parsed, post, resolved_comment_count=len(comments))
    docs = [post_doc]
    title = _string_or_none(post.get("title")) or parsed.post_id
    canonical_url = _post_canonical_url(parsed, post)
    for index, comment in enumerate(comments, start=1):
        docs.append(
            _render_comment_document(
                parsed,
                comment,
                post_title=title,
                post_canonical_url=canonical_url,
                index=index,
            )
        )
    return docs


def _render_post_document(
    parsed: ParsedForumMagnumTarget,
    post: dict[str, Any],
    *,
    resolved_comment_count: int,
) -> ForumMagnumResolvedDocument:
    post_id = _string_or_none(post.get("_id")) or parsed.post_id
    title = _string_or_none(post.get("title")) or post_id
    author = _author(post)
    canonical_url = _post_canonical_url(parsed, post)
    html_body = _content_html(post)
    body = _html_to_markdown(html_body, canonical_url)
    tags = _tags(post)
    frontmatter = {
        "url": canonical_url,
        "platform": parsed.platform,
        "post_id": post_id,
        "author": author,
        "posted_at": _string_or_none(post.get("postedAt")),
        "score": post.get("baseScore"),
        "vote_count": post.get("voteCount"),
        "comment_count": post.get("commentCount"),
        "resolved_comment_count": resolved_comment_count,
        "tags": tags or None,
    }
    rendered = "\n".join(
        [
            _render_frontmatter(frontmatter),
            "",
            f"# {title}",
            "",
            body,
        ]
    ).strip()
    source_path = f"{parsed.platform}/posts/{post_id}"
    return ForumMagnumResolvedDocument(
        label=f"forum_magnum/{parsed.platform}/{post_id}",
        rendered=rendered,
        prose=body,
        prose_authors=(author,) if author else (),
        trace_path=source_path,
        source_ref=parsed.source_ref,
        source_path=source_path,
        context_subpath=f"forum-magnum/{parsed.platform}/{post_id}.md",
        source_created=_string_or_none(post.get("postedAt")),
        source_modified=None,
        kind="post",
        platform=parsed.platform,
        canonical_id=f"{parsed.platform}:post:{post_id}",
        canonical_url=canonical_url,
        post_id=post_id,
        comment_id=None,
        parent_comment_id=None,
        author=author,
        score=_number_or_none(post.get("baseScore")),
        dedupe_rank=0,
    )


def _render_comment_document(
    parsed: ParsedForumMagnumTarget,
    comment: dict[str, Any],
    *,
    post_title: str,
    post_canonical_url: str,
    index: int,
) -> ForumMagnumResolvedDocument:
    comment_id = _string_or_none(comment.get("_id")) or f"comment-{index}"
    post_id = _string_or_none(comment.get("postId")) or parsed.post_id
    parent_comment_id = _string_or_none(comment.get("parentCommentId"))
    author = _author(comment)
    canonical_url = (
        _string_or_none(comment.get("pageUrl"))
        or f"{post_canonical_url}#comment-{comment_id}"
    )
    body = _html_to_markdown(_content_html(comment), canonical_url)
    title = f"Comment by {author}" if author else "Comment"
    frontmatter = {
        "url": canonical_url,
        "platform": parsed.platform,
        "post_id": post_id,
        "post_title": post_title,
        "comment_id": comment_id,
        "parent_comment_id": parent_comment_id,
        "author": author,
        "posted_at": _string_or_none(comment.get("postedAt")),
        "score": comment.get("baseScore"),
    }
    rendered = "\n".join(
        [
            _render_frontmatter(frontmatter),
            "",
            f"# {title}",
            "",
            body,
        ]
    ).strip()
    source_path = f"{parsed.platform}/posts/{post_id}/comments/{comment_id}"
    return ForumMagnumResolvedDocument(
        label=f"forum_magnum/{parsed.platform}/{post_id}/comments/{comment_id}",
        rendered=rendered,
        prose=body,
        prose_authors=(author,) if author else (),
        trace_path=source_path,
        source_ref=parsed.source_ref,
        source_path=source_path,
        context_subpath=(
            f"forum-magnum/{parsed.platform}/{post_id}/comments/"
            f"{_safe_path_segment(comment_id, fallback='comment')}.md"
        ),
        source_created=_string_or_none(comment.get("postedAt")),
        source_modified=None,
        kind="comment",
        platform=parsed.platform,
        canonical_id=f"{parsed.platform}:comment:{comment_id}",
        canonical_url=canonical_url,
        post_id=post_id,
        comment_id=comment_id,
        parent_comment_id=parent_comment_id,
        author=author,
        score=_number_or_none(comment.get("baseScore")),
        dedupe_rank=index,
    )


def _html_to_markdown(html_text: str, base_url: str) -> str:
    if not html_text.strip():
        return ""
    tree = HTMLParser(f"<div>{html_text}</div>")
    root = tree.body or tree.root
    if root is None:
        return _clean_text(html_text, preserve_newlines=True)
    return _normalize_markdown(_render_children(root, base_url))


def _render_node(node: Any, base_url: str) -> str:
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
        return _wrap_inline(_render_children(node, base_url), "**")
    if tag in {"em", "i"}:
        return _wrap_inline(_render_children(node, base_url), "*")
    if tag == "code":
        text = _clean_inline(node.text(separator=" "))
        return f"`{text}`" if text else ""
    if tag == "pre":
        text = (node.text(separator="\n") or "").strip("\n")
        return f"\n\n```\n{text}\n```\n\n" if text else ""
    if tag == "a":
        text = _clean_inline(_render_children(node, base_url)) or _node_attr(
            node,
            "href",
        )
        href = _absolute_url(base_url, _node_attr(node, "href"))
        if text and href:
            return f"[{text}]({href})"
        return text or ""
    if tag == "img":
        src = _absolute_url(base_url, _node_attr(node, "src"))
        alt = _clean_inline(_node_attr(node, "alt") or _node_attr(node, "title"))
        if not src:
            return alt or ""
        return f"\n\n![{alt or 'image'}]({src})\n\n"
    if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        level = int(tag[1])
        text = _clean_inline(_render_children(node, base_url))
        return f"\n\n{'#' * level} {text}\n\n" if text else ""
    if tag == "blockquote":
        rendered = _normalize_markdown(_render_children(node, base_url))
        quoted = "\n".join(f"> {line}" if line else ">" for line in rendered.splitlines())
        return f"\n\n{quoted}\n\n" if quoted else ""
    if tag in {"ul", "ol"}:
        return _render_list(node, base_url, ordered=tag == "ol")
    if tag == "table":
        return _render_table(node, base_url)
    rendered = _render_children(node, base_url)
    if tag in _BLOCK_TAGS:
        return f"\n\n{rendered}\n\n"
    return rendered


def _render_children(node: Any, base_url: str) -> str:
    return "".join(_render_node(child, base_url) for child in _children(node))


def _render_list(node: Any, base_url: str, *, ordered: bool) -> str:
    lines: list[str] = []
    index = 1
    for child in _children(node):
        if _node_tag(child) != "li":
            continue
        rendered = _normalize_markdown(_render_children(child, base_url))
        if not rendered:
            continue
        prefix = f"{index}. " if ordered else "- "
        lines.append(prefix + rendered.replace("\n", "\n  "))
        index += 1
    return "\n\n" + "\n".join(lines) + "\n\n" if lines else ""


def _render_table(node: Any, base_url: str) -> str:
    rows: list[list[str]] = []
    for tr in node.css("tr"):
        cells = [
            _clean_inline(_render_children(cell, base_url))
            for cell in tr.css("td, th")
        ]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    header = padded[0]
    divider = ["---"] * width
    lines = [
        "| " + " | ".join(_escape_table_cell(cell) for cell in header) + " |",
        "| " + " | ".join(divider) + " |",
    ]
    for row in padded[1:]:
        lines.append("| " + " | ".join(_escape_table_cell(cell) for cell in row) + " |")
    return "\n\n" + "\n".join(lines) + "\n\n"


def _children(node: Any) -> list[Any]:
    children: list[Any] = []
    child = getattr(node, "child", None)
    while child is not None:
        children.append(child)
        child = getattr(child, "next", None)
    return children


def _render_frontmatter(payload: dict[str, Any]) -> str:
    import yaml

    data = {key: value for key, value in payload.items() if value is not None}
    frontmatter = yaml.safe_dump(
        data,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    ).strip()
    return f"---\n{frontmatter}\n---"


def _content_html(payload: dict[str, Any]) -> str:
    contents = payload.get("contents")
    if not isinstance(contents, dict):
        return ""
    html_text = contents.get("html")
    return html_text if isinstance(html_text, str) else ""


def _author(payload: dict[str, Any]) -> str | None:
    user = payload.get("user")
    if not isinstance(user, dict):
        return None
    return _string_or_none(user.get("displayName")) or _string_or_none(
        user.get("username"),
    )


def _tags(post: dict[str, Any]) -> list[str]:
    raw_tags = post.get("tags")
    if not isinstance(raw_tags, list):
        return []
    tags: list[str] = []
    for item in raw_tags:
        if not isinstance(item, dict):
            continue
        name = _string_or_none(item.get("name"))
        if name:
            tags.append(name)
    return tags


def _post_canonical_url(
    parsed: ParsedForumMagnumTarget,
    post: dict[str, Any],
) -> str:
    page_url = _string_or_none(post.get("pageUrl"))
    if page_url:
        return page_url
    slug = _string_or_none(post.get("slug"))
    if slug:
        return f"{parsed.canonical_base_url}/posts/{parsed.post_id}/{slug}"
    return f"{parsed.canonical_base_url}/posts/{parsed.post_id}"


def _document_cache_identity(
    parsed: ParsedForumMagnumTarget,
    settings: ForumMagnumSettings,
) -> str:
    payload = repr((parsed.canonical_id, forum_magnum_settings_cache_key(settings)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _documents_from_cached_payload(
    payload: Any,
) -> list[ForumMagnumResolvedDocument] | None:
    if not isinstance(payload, list):
        return None
    documents: list[ForumMagnumResolvedDocument] = []
    for item in payload:
        if not isinstance(item, dict):
            return None
        try:
            documents.append(ForumMagnumResolvedDocument(**item))
        except TypeError:
            return None
    return documents


def _api_user_agent() -> str:
    return _env_value("CONTEXTUALIZE_FORUM_MAGNUM_USER_AGENT") or _DEFAULT_USER_AGENT


def _api_timeout_seconds() -> float:
    return _env_float(
        "CONTEXTUALIZE_FORUM_MAGNUM_API_TIMEOUT",
        default=_DEFAULT_TIMEOUT_SECONDS,
        minimum=1.0,
    )


def _api_min_request_delay_seconds() -> float:
    return _env_float(
        "CONTEXTUALIZE_FORUM_MAGNUM_MIN_REQUEST_DELAY_SECONDS",
        default=0.2,
        minimum=0.0,
    )


def _pace_requests() -> None:
    global _last_request_at
    delay = _api_min_request_delay_seconds()
    if delay <= 0:
        return
    now = time.monotonic()
    wait = delay - (now - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def _env_value(name: str) -> str | None:
    try:
        from dotenv import find_dotenv, load_dotenv

        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=False)
    except Exception:
        pass
    value = os.environ.get(name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _env_float(name: str, *, default: float, minimum: float) -> float:
    raw = _env_value(name)
    if raw is None:
        return default
    try:
        return max(minimum, float(raw))
    except ValueError:
        return default


def _cache_ttl_as_timedelta(value: Any) -> timedelta | None:
    return value if isinstance(value, timedelta) else None


def _parse_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if not cleaned:
            return default
        return cleaned not in {"0", "false", "no", "off"}
    return default


def _parse_int(value: Any, *, default: int, minimum: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return max(minimum, value)
    if isinstance(value, str):
        try:
            return max(minimum, int(value.strip()))
        except ValueError:
            return default
    return default


def _parse_comment_view(value: Any) -> str:
    if not isinstance(value, str):
        return _DEFAULT_COMMENT_VIEW
    cleaned = value.strip()
    if cleaned in {"postCommentsOld", "postCommentsTop"}:
        return cleaned
    aliases = {
        "old": "postCommentsOld",
        "chronological": "postCommentsOld",
        "top": "postCommentsTop",
    }
    return aliases.get(cleaned.lower(), _DEFAULT_COMMENT_VIEW)


def _string_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _number_or_none(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def _safe_path_segment(value: str, *, fallback: str) -> str:
    cleaned = _SAFE_SEGMENT_RE.sub("_", value.strip()).strip("._")
    return cleaned or fallback


def _absolute_url(base_url: str, value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        return value
    return urljoin(base_url, value)


def _node_tag(node: Any) -> str:
    return str(getattr(node, "tag", "") or "").lower()


def _node_attr(node: Any, name: str) -> str | None:
    value = getattr(node, "attributes", {}).get(name)
    if value is None:
        return None
    cleaned = html.unescape(str(value)).strip()
    return cleaned or None


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


def _wrap_inline(value: str, marker: str) -> str:
    if not value:
        return ""
    normalized = value.replace("\n", " ")
    leading_match = re.match(r"^\s*", normalized)
    trailing_match = re.search(r"\s*$", normalized)
    leading = leading_match.group(0) if leading_match else ""
    trailing = trailing_match.group(0) if trailing_match else ""
    text = _clean_inline(normalized)
    return f"{leading}{marker}{text}{marker}{trailing}" if text else ""


def _normalize_markdown(value: str) -> str:
    value = html.unescape(value).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+$", "", line) for line in value.splitlines()]
    value = "\n".join(lines)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _log(msg: str) -> None:
    try:
        from contextualize.runtime import get_verbose_logging

        if get_verbose_logging():
            print(msg, file=sys.stderr, flush=True)
    except Exception:
        return
