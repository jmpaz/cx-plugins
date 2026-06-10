from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, quote, urlparse

from contextualize.references.helpers import parse_timestamp_or_duration
from contextualize.render.text import process_text
from contextualize.utils import count_tokens

from ..shared.progress import record_progress

_ARENA_CHANNEL_RE = re.compile(
    r"^https?://(?:www\.)?are\.na/"
    r"(?:channel/(?P<slug1>[^/?#]+)"
    r"|(?P<user>[^/?#]+)/(?P<slug2>[^/?#]+))$"
)
_ARENA_BLOCK_RE = re.compile(
    r"^https?://(?:www\.)?are\.na/block/(?P<id>\d+)(?:[?#].*)?$"
)
_ARENA_USER_TARGET_RE = re.compile(r"^arena:user:(?P<slug>[^/?#]+)$")
_ARENA_GROUP_TARGET_RE = re.compile(r"^arena:group:(?P<slug>[^/?#]+)$")
_ARENA_PROFILE_URL_RE = re.compile(r"^https?://(?:www\.)?are\.na/(?P<slug>[^/?#]+)$")
_ARENA_GROUP_URL_RE = re.compile(
    r"^https?://(?:www\.)?are\.na/groups?/(?P<slug>[^/?#]+)$"
)

_API_BASE = "https://api.are.na/v3"
_DEFAULT_GUEST_REQUESTS_PER_MINUTE = 30
_DEFAULT_AUTH_REQUESTS_PER_MINUTE = 120
_DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 60.0
_DEFAULT_RATE_LIMIT_SAFETY = 0.9
_DEFAULT_MIN_REQUEST_DELAY_SECONDS = 0.2


def _log(msg: str) -> None:
    from contextualize.runtime import get_verbose_logging

    if get_verbose_logging():
        print(msg, file=sys.stderr, flush=True)


_RESERVED_PATHS = frozenset(
    {
        "about",
        "explore",
        "search",
        "settings",
        "notifications",
        "feed",
        "blog",
        "pricing",
        "terms",
        "privacy",
        "sign_in",
        "sign_up",
        "log_in",
        "register",
        "block",
        "channel",
        "api",
        "developer",
        "developers",
        "group",
        "groups",
        "user",
        "users",
    }
)


def is_arena_url(url: str) -> bool:
    return (
        is_arena_channel_url(url)
        or is_arena_block_url(url)
        or is_arena_user_target(url)
        or is_arena_group_target(url)
    )


def is_arena_channel_url(url: str) -> bool:
    match = _ARENA_CHANNEL_RE.match(url)
    if not match:
        return False
    user = match.group("user")
    if user and user.lower() in _RESERVED_PATHS:
        return False
    return True


def is_arena_block_url(url: str) -> bool:
    return bool(_ARENA_BLOCK_RE.match(url))


def is_arena_user_target(target: str) -> bool:
    return extract_user_slug(target) is not None


def is_arena_group_target(target: str) -> bool:
    return extract_group_slug(target) is not None


@lru_cache(maxsize=1)
def warmup_arena_network_stack() -> None:
    try:
        import requests

        _ = requests.__version__
    except Exception:
        return


def extract_channel_slug(url: str) -> str | None:
    match = _ARENA_CHANNEL_RE.match(url)
    if not match:
        return None
    user = match.group("user")
    if user and user.lower() in _RESERVED_PATHS:
        return None
    return match.group("slug1") or match.group("slug2")


def extract_block_id(url: str) -> int | None:
    match = _ARENA_BLOCK_RE.match(url)
    if not match:
        return None
    return int(match.group("id"))


def _first_query_value(values: list[str] | None) -> str | None:
    if not values:
        return None
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def parse_arena_block_attachment_target(target: str) -> dict[str, Any] | None:
    block_id = extract_block_id(target)
    if block_id is None:
        return None
    query = parse_qs(urlparse(target).query)
    attachment_id = _first_query_value(query.get("attachment-id"))
    attachment_name = _first_query_value(query.get("attachment"))
    if not attachment_id and not attachment_name:
        return None
    parsed: dict[str, Any] = {"kind": "attachment", "block_id": block_id}
    if attachment_id:
        parsed["attachment_id"] = attachment_id
    if attachment_name:
        parsed["attachment_name"] = attachment_name
    return parsed


def is_arena_block_attachment_url(target: str) -> bool:
    return parse_arena_block_attachment_target(target) is not None


def extract_user_slug(target: str) -> str | None:
    match = _ARENA_USER_TARGET_RE.match(target)
    if match:
        return match.group("slug")
    return extract_profile_slug(target)


def extract_group_slug(target: str) -> str | None:
    match = _ARENA_GROUP_TARGET_RE.match(target)
    if match:
        return match.group("slug")
    match = _ARENA_GROUP_URL_RE.match(target)
    if not match:
        return None
    slug = match.group("slug")
    if slug.lower() in _RESERVED_PATHS:
        return None
    return slug


def extract_profile_slug(target: str) -> str | None:
    match = _ARENA_PROFILE_URL_RE.match(target)
    if not match:
        return None
    slug = match.group("slug")
    if slug.lower() in _RESERVED_PATHS:
        return None
    return slug


def _load_dotenv() -> None:
    try:
        from dotenv import find_dotenv, load_dotenv

        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=False)
    except Exception:
        pass


def _get_auth_headers() -> dict[str, str]:
    token = _resolve_arena_access_token()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _resolve_arena_access_token() -> str | None:
    from .cache import get_cached_user_access_token

    _load_dotenv()
    env_token = (os.environ.get("ARENA_ACCESS_TOKEN") or "").strip()
    if env_token:
        return env_token
    cached = get_cached_user_access_token(min_valid_seconds=60)
    if cached:
        return cached
    return None


def _auth_cache_partition() -> str:
    token = _resolve_arena_access_token()
    if not token:
        return "guest"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
    return f"auth:{digest}"


def _api_timeout_seconds() -> float:
    raw = (os.environ.get("ARENA_API_TIMEOUT") or "").strip()
    if not raw:
        return 30.0
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 30.0


def _api_max_attempts() -> int:
    raw = (os.environ.get("ARENA_API_MAX_ATTEMPTS") or "").strip()
    if not raw:
        return 6
    try:
        return max(1, int(raw))
    except ValueError:
        return 6


def _api_rate_limit_safety() -> float:
    raw = (os.environ.get("ARENA_API_RATE_LIMIT_SAFETY") or "").strip()
    if not raw:
        return _DEFAULT_RATE_LIMIT_SAFETY
    try:
        return min(1.0, max(0.1, float(raw)))
    except ValueError:
        return _DEFAULT_RATE_LIMIT_SAFETY


def _api_min_request_delay_seconds() -> float:
    raw = (os.environ.get("ARENA_API_MIN_REQUEST_DELAY_SECONDS") or "").strip()
    if not raw:
        return _DEFAULT_MIN_REQUEST_DELAY_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        return _DEFAULT_MIN_REQUEST_DELAY_SECONDS


def _retry_delay_seconds(attempt: int) -> float:
    import random

    base = min(30.0, 1.0 * (2 ** max(0, attempt - 1)))
    return base + random.uniform(0.0, 0.25)


def _server_error_retry_delay_seconds(attempt: int) -> float:
    import random

    base = min(30.0, 5.0 * (2 ** max(0, attempt - 1)))
    return base + random.uniform(0.0, 0.25)


def _retry_after_seconds(resp: object) -> float | None:
    import time

    headers = getattr(resp, "headers", None) or {}
    reset = headers.get("X-RateLimit-Reset")
    if reset:
        try:
            value = float(reset)
            return max(0.0, value - time.time())
        except ValueError:
            pass
    retry_after = headers.get("Retry-After")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    return None


def _positive_float_header(headers: object, name: str) -> float | None:
    if not hasattr(headers, "get"):
        return None
    raw = headers.get(name)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return value


class _ArenaApiRateLimiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._limit_per_window: float | None = None
        self._window_seconds = _DEFAULT_RATE_LIMIT_WINDOW_SECONDS
        self._next_request_at = 0.0

    def wait_for_slot(self, *, authenticated: bool) -> None:
        while True:
            with self._lock:
                now = time.time()
                if self._next_request_at <= now:
                    self._next_request_at = now + self._request_interval(authenticated)
                    return
                wait = self._next_request_at - now
            time.sleep(wait)

    def update_from_response(self, resp: object, *, authenticated: bool) -> None:
        headers = getattr(resp, "headers", None) or {}
        limit = _positive_float_header(headers, "X-RateLimit-Limit")
        window = _positive_float_header(headers, "X-RateLimit-Window")
        with self._lock:
            if limit is not None:
                self._limit_per_window = limit
            elif self._limit_per_window is None:
                self._limit_per_window = self._default_limit(authenticated)
            if window is not None:
                self._window_seconds = window

    def defer_for(self, seconds: float) -> None:
        if seconds <= 0:
            return
        until = time.time() + seconds
        with self._lock:
            self._next_request_at = max(self._next_request_at, until)

    def reset(self) -> None:
        with self._lock:
            self._limit_per_window = None
            self._window_seconds = _DEFAULT_RATE_LIMIT_WINDOW_SECONDS
            self._next_request_at = 0.0

    def _request_interval(self, authenticated: bool) -> float:
        limit = self._limit_per_window or self._default_limit(authenticated)
        safety = _api_rate_limit_safety()
        interval = self._window_seconds / max(1.0, limit * safety)
        return max(_api_min_request_delay_seconds(), interval)

    @staticmethod
    def _default_limit(authenticated: bool) -> float:
        if authenticated:
            return float(_DEFAULT_AUTH_REQUESTS_PER_MINUTE)
        return float(_DEFAULT_GUEST_REQUESTS_PER_MINUTE)


_ARENA_API_RATE_LIMITER = _ArenaApiRateLimiter()


def _requests_exception_type(requests_module: object) -> type[Exception]:
    namespace = getattr(requests_module, "exceptions", None)
    request_exception = getattr(namespace, "RequestException", None)
    if isinstance(request_exception, type) and issubclass(request_exception, Exception):
        return request_exception
    return Exception


def _api_get(path: str, params: dict | None = None) -> dict:
    import requests

    url = f"{_API_BASE}{path}"
    headers = {**_get_auth_headers(), "Accept": "application/json"}
    authenticated = "Authorization" in headers
    timeout = _api_timeout_seconds()
    max_attempts = _api_max_attempts()
    transient_statuses = {429, 500, 502, 503, 504}
    request_exception_type = _requests_exception_type(requests)

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            _ARENA_API_RATE_LIMITER.wait_for_slot(authenticated=authenticated)
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        except request_exception_type as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break
            wait = _retry_delay_seconds(attempt)
            _log(
                f"  Are.na request failed ({type(exc).__name__}); retrying in {wait:.1f}s "
                f"(attempt {attempt}/{max_attempts})"
            )
            time.sleep(wait)
            continue

        _ARENA_API_RATE_LIMITER.update_from_response(
            resp,
            authenticated=authenticated,
        )
        if resp.status_code == 404:
            raise ValueError(f"Are.na resource not found: {path}")

        if resp.status_code in transient_statuses and attempt < max_attempts:
            if resp.status_code == 429:
                wait = _retry_after_seconds(resp)
                if wait is None:
                    wait = _retry_delay_seconds(attempt)
                _ARENA_API_RATE_LIMITER.defer_for(wait)
                message = (
                    "  Are.na API returned 429; waiting "
                    f"{wait:.1f}s for rate-limit reset "
                    f"(attempt {attempt}/{max_attempts})"
                )
            else:
                wait = _server_error_retry_delay_seconds(attempt)
                message = (
                    f"  Are.na API returned {resp.status_code}; retrying in "
                    f"{wait:.1f}s (attempt {attempt}/{max_attempts})"
                )
            _log(message)
            time.sleep(wait)
            continue

        resp.raise_for_status()
        return resp.json()

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Are.na request failed unexpectedly for {path}")


def _fetch_channel(slug: str) -> dict:
    return _api_get(f"/channels/{slug}")


def _fetch_channel_page(slug: str, page: int, per: int = 100) -> dict:
    return _api_get(f"/channels/{slug}/contents", {"page": page, "per": per})


def _owner_contents_path(kind: Literal["user", "group"], slug: str) -> str:
    if kind == "user":
        return f"/users/{slug}/contents"
    return f"/groups/{slug}/contents"


def _owner_profile_path(kind: Literal["user", "group"], slug: str) -> str:
    if kind == "user":
        return f"/users/{slug}"
    return f"/groups/{slug}"


def _fetch_owner_profile(kind: Literal["user", "group"], slug: str) -> dict:
    return _api_get(_owner_profile_path(kind, slug))


def _fetch_owner_channel_page(
    kind: Literal["user", "group"],
    slug: str,
    page: int,
    *,
    per: int = 100,
    sort: str = "created_at_asc",
) -> dict:
    return _api_get(
        _owner_contents_path(kind, slug),
        {"page": page, "per": per, "type": "Channel", "sort": sort},
    )


def _fetch_user_groups_page(
    slug: str,
    page: int,
    *,
    per: int = 100,
    sort: str = "name_asc",
) -> dict:
    return _api_get(
        f"/users/{slug}/groups",
        {"page": page, "per": per, "sort": sort},
    )


def _get_max_depth() -> int:
    raw = (
        os.environ.get("ARENA_RECURSE_DEPTH")
        or os.environ.get("ARENA_MAX_DEPTH")
        or "1"
    )
    try:
        return max(0, int(raw))
    except ValueError:
        return 1


def _get_include_descriptions() -> bool:
    raw = os.environ.get("ARENA_BLOCK_DESCRIPTION", "1").lower()
    return raw not in ("0", "false", "no")


def _get_include_comments() -> bool:
    raw = os.environ.get("ARENA_BLOCK_COMMENTS", "1").lower()
    return raw not in ("0", "false", "no")


def _get_include_connections() -> bool:
    raw = os.environ.get("ARENA_BLOCK_CONNECTIONS", "1").lower()
    return raw not in ("0", "false", "no")


def _get_connections_max_items() -> int | None:
    raw = (
        os.environ.get("ARENA_BLOCK_CONNECTIONS_MAX_ITEMS")
        or os.environ.get("ARENA_CONNECTIONS_MAX_ITEMS")
        or "30"
    )
    value = raw.strip().lower()
    if value in {"0", "all", "none", "unlimited"}:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return 30
    if parsed <= 0:
        return None
    return parsed


def _get_include_link_image_descriptions() -> bool:
    raw = os.environ.get("ARENA_BLOCK_LINK_IMAGE_DESC", "0").lower()
    return raw not in ("0", "false", "no")


def _get_include_pdf_content() -> bool:
    raw = os.environ.get("ARENA_BLOCK_PDF_CONTENT", "0").lower()
    return raw not in ("0", "false", "no")


def _get_include_media_descriptions() -> bool:
    raw = os.environ.get("ARENA_BLOCK_MEDIA_DESC", "1").lower()
    return raw not in ("0", "false", "no")


def _parse_cache_ttl(raw: str) -> timedelta | None:
    raw = raw.strip().lower()
    if not raw:
        return None
    if raw == "0":
        return timedelta(0)
    match = re.fullmatch(r"(\d+)([smhdw])", raw)
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2)
    if unit == "s":
        return timedelta(seconds=amount)
    if unit == "m":
        return timedelta(minutes=amount)
    if unit == "h":
        return timedelta(hours=amount)
    if unit == "d":
        return timedelta(days=amount)
    return timedelta(weeks=amount)


def _get_comments_cache_ttl() -> timedelta | None:
    return _parse_cache_ttl(os.environ.get("ARENA_COMMENTS_CACHE_TTL") or "")


def _get_connections_cache_ttl() -> timedelta | None:
    raw = (
        os.environ.get("ARENA_BLOCK_CONNECTIONS_CACHE_TTL")
        or os.environ.get("ARENA_CONNECTIONS_CACHE_TTL")
        or ""
    )
    if raw:
        return _parse_cache_ttl(raw)
    return _get_comments_cache_ttl()


def _get_sort_order() -> str:
    return (
        (os.environ.get("ARENA_BLOCK_SORT") or os.environ.get("ARENA_SORT") or "desc")
        .lower()
        .strip()
    )


def _get_recurse_users() -> set[str] | None:
    raw = os.environ.get("ARENA_RECURSE_USERS", "").strip()
    if not raw:
        return {"self"}
    if raw.lower() == "all":
        return None
    return {s.strip().lower() for s in raw.split(",") if s.strip()}


def _get_max_blocks_per_channel() -> int | None:
    raw = (os.environ.get("ARENA_MAX_BLOCKS_PER_CHANNEL") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


@dataclass(frozen=True)
class ArenaRecurseBlockLimit:
    kind: Literal["count", "ratio"]
    value: int | float

    def cache_key(self) -> str:
        if self.kind == "count":
            return f"count:{self.value}"
        return f"ratio:{self.value:g}"


def parse_arena_recurse_blocks(raw: Any) -> ArenaRecurseBlockLimit | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, ArenaRecurseBlockLimit):
        return raw
    if isinstance(raw, bool):
        raise ValueError("Are.na recurse blocks must be a positive integer or ratio")
    if isinstance(raw, int):
        if raw <= 0:
            raise ValueError("Are.na recurse blocks must be greater than 0")
        return ArenaRecurseBlockLimit("count", raw)
    if isinstance(raw, float):
        if raw <= 0:
            raise ValueError("Are.na recurse blocks ratio must be greater than 0")
        if raw > 1:
            raise ValueError("Are.na recurse blocks ratio must be <= 1")
        return ArenaRecurseBlockLimit("ratio", raw)
    if not isinstance(raw, str):
        raise ValueError("Are.na recurse blocks must be a positive integer or ratio")

    value = raw.strip()
    if not value:
        return None
    if value.endswith("%"):
        percent_raw = value[:-1].strip()
        try:
            percent = float(percent_raw)
        except ValueError as exc:
            raise ValueError("Are.na recurse blocks percent must be numeric") from exc
        if percent <= 0 or percent > 100:
            raise ValueError("Are.na recurse blocks percent must be > 0 and <= 100")
        return ArenaRecurseBlockLimit("ratio", percent / 100)
    if re.fullmatch(r"\d+", value):
        count = int(value)
        if count <= 0:
            raise ValueError("Are.na recurse blocks must be greater than 0")
        return ArenaRecurseBlockLimit("count", count)

    try:
        ratio = float(value)
    except ValueError as exc:
        raise ValueError(
            "Are.na recurse blocks must be a positive integer, ratio, or percent"
        ) from exc
    if ratio <= 0 or ratio > 1:
        raise ValueError("Are.na recurse blocks ratio must be > 0 and <= 1")
    return ArenaRecurseBlockLimit("ratio", ratio)


def _get_recurse_blocks() -> ArenaRecurseBlockLimit | None:
    raw = (os.environ.get("ARENA_RECURSE_BLOCKS") or "").strip()
    return parse_arena_recurse_blocks(raw)


@dataclass(frozen=True)
class ArenaChannelExclusions:
    ids: frozenset[int] = field(default_factory=frozenset)
    slugs: frozenset[str] = field(default_factory=frozenset)

    def matches(self, channel: dict) -> bool:
        channel_id = channel.get("id")
        if isinstance(channel_id, int) and channel_id in self.ids:
            return True
        slug = channel.get("slug")
        return isinstance(slug, str) and slug.lower() in self.slugs

    def cache_key(self) -> str:
        ids = ",".join(str(value) for value in sorted(self.ids))
        slugs = ",".join(sorted(self.slugs))
        return f"ids={ids}:slugs={slugs}"


def _split_channel_selector_values(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _channel_selector_from_value(value: object) -> tuple[int | None, str | None]:
    if isinstance(value, int) and not isinstance(value, bool):
        return value, None
    if not isinstance(value, str):
        raise ValueError("Are.na excluded channels must be strings or integers")
    text = value.strip()
    if not text:
        return None, None
    if text.isdigit():
        return int(text), None
    slug = extract_channel_slug(text)
    if slug is not None:
        return None, slug.lower()
    return None, text.lower()


def parse_arena_channel_exclusions(raw: Any) -> ArenaChannelExclusions:
    ids: set[int] = set()
    slugs: set[str] = set()
    if isinstance(raw, ArenaChannelExclusions):
        return raw
    if raw is None or raw == "":
        return ArenaChannelExclusions()
    if isinstance(raw, str):
        values: list[object] = _split_channel_selector_values(raw)
    elif isinstance(raw, (list, tuple, set, frozenset)):
        values = list(raw)
    else:
        values = [raw]

    for value in values:
        channel_id, slug = _channel_selector_from_value(value)
        if channel_id is not None:
            ids.add(channel_id)
        if slug is not None:
            slugs.add(slug)
    return ArenaChannelExclusions(frozenset(ids), frozenset(slugs))


def _get_exclude_channels() -> ArenaChannelExclusions:
    return parse_arena_channel_exclusions(os.environ.get("ARENA_EXCLUDE_CHANNELS"))


def _parse_iso_datetime(value: str) -> datetime | None:
    return parse_timestamp_or_duration(value)


def _get_window_bound(name: str) -> datetime | None:
    return _parse_iso_datetime(os.environ.get(name, ""))


def _normalize_optional_datetime_override(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        parsed = _parse_iso_datetime(value)
        if parsed is None:
            raise ValueError(
                "Are.na time window value must be a valid timestamp (ISO, epoch, or relative duration)"
            )
        return parsed
    raise ValueError(
        "Are.na time window value must be a timestamp string (ISO, epoch, or relative duration)"
    )


def _normalize_connections_max_items_override(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Are.na block connections max-items must be a positive integer")
    if isinstance(value, int):
        if value <= 0:
            return None
        return value
    if not isinstance(value, str):
        raise ValueError("Are.na block connections max-items must be a positive integer")
    text = value.strip().lower()
    if not text:
        return None
    if text in {"0", "all", "none", "unlimited"}:
        return None
    try:
        parsed = int(text)
    except ValueError as exc:
        raise ValueError(
            "Are.na block connections max-items must be a positive integer"
        ) from exc
    if parsed <= 0:
        return None
    return parsed


def _validate_time_window(settings: ArenaSettings) -> None:
    has_connected = (
        settings.connected_after is not None or settings.connected_before is not None
    )
    has_created = (
        settings.created_after is not None or settings.created_before is not None
    )
    if has_connected and has_created:
        raise ValueError("Are.na time window cannot mix connected-* and created-*")
    if (
        settings.connected_after is not None
        and settings.connected_before is not None
        and settings.connected_after > settings.connected_before
    ):
        raise ValueError("ARENA_CONNECTED_AFTER must be <= ARENA_CONNECTED_BEFORE")
    if (
        settings.created_after is not None
        and settings.created_before is not None
        and settings.created_after > settings.created_before
    ):
        raise ValueError("ARENA_CREATED_AFTER must be <= ARENA_CREATED_BEFORE")


VALID_SORT_ORDERS = frozenset(
    {
        "asc",
        "desc",
        "date-asc",
        "date-desc",
        "random",
        "position-asc",
        "position-desc",
    }
)


@dataclass(frozen=True)
class ArenaSettings:
    max_depth: int = 1
    sort_order: str = "desc"
    max_blocks_per_channel: int | None = None
    recurse_blocks: ArenaRecurseBlockLimit | None = None
    include_descriptions: bool = True
    include_comments: bool = True
    include_connections: bool = True
    connections_max_items: int | None = 30
    include_link_image_descriptions: bool = False
    include_pdf_content: bool = False
    include_media_descriptions: bool = True
    recurse_users: set[str] | None = field(default_factory=lambda: {"self"})
    exclude_channels: ArenaChannelExclusions = field(
        default_factory=ArenaChannelExclusions
    )
    connected_after: datetime | None = None
    connected_before: datetime | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None


def _arena_settings_from_env() -> ArenaSettings:
    _load_dotenv()
    settings = ArenaSettings(
        max_depth=_get_max_depth(),
        sort_order=_get_sort_order(),
        max_blocks_per_channel=_get_max_blocks_per_channel(),
        recurse_blocks=_get_recurse_blocks(),
        include_descriptions=_get_include_descriptions(),
        include_comments=_get_include_comments(),
        include_connections=_get_include_connections(),
        connections_max_items=_get_connections_max_items(),
        include_link_image_descriptions=_get_include_link_image_descriptions(),
        include_pdf_content=_get_include_pdf_content(),
        include_media_descriptions=_get_include_media_descriptions(),
        recurse_users=_get_recurse_users(),
        exclude_channels=_get_exclude_channels(),
        connected_after=_get_window_bound("ARENA_CONNECTED_AFTER"),
        connected_before=_get_window_bound("ARENA_CONNECTED_BEFORE"),
        created_after=_get_window_bound("ARENA_CREATED_AFTER"),
        created_before=_get_window_bound("ARENA_CREATED_BEFORE"),
    )
    _validate_time_window(settings)
    return settings


def _has_env_recurse_depth_override() -> bool:
    raw_recurse_depth = os.environ.get("ARENA_RECURSE_DEPTH")
    if raw_recurse_depth is not None and raw_recurse_depth.strip():
        return True
    raw_max_depth = os.environ.get("ARENA_MAX_DEPTH")
    return raw_max_depth is not None and raw_max_depth.strip() != ""


def build_arena_settings(overrides: dict | None = None) -> ArenaSettings:
    env = _arena_settings_from_env()
    if not overrides:
        return env

    if _has_env_recurse_depth_override():
        max_depth = env.max_depth
    else:
        max_depth = overrides.get("max_depth", env.max_depth)
    sort_order = overrides.get("sort_order", env.sort_order)
    max_blocks_per_channel = overrides.get(
        "max_blocks_per_channel", env.max_blocks_per_channel
    )
    if max_blocks_per_channel is not None:
        try:
            max_blocks_per_channel = int(max_blocks_per_channel)
        except (TypeError, ValueError):
            max_blocks_per_channel = env.max_blocks_per_channel
        if max_blocks_per_channel is not None and max_blocks_per_channel <= 0:
            max_blocks_per_channel = env.max_blocks_per_channel
    recurse_blocks = parse_arena_recurse_blocks(
        overrides.get("recurse_blocks", env.recurse_blocks)
    )
    include_descriptions = overrides.get(
        "include_descriptions", env.include_descriptions
    )
    include_comments = overrides.get("include_comments", env.include_comments)
    include_connections = overrides.get("include_connections", env.include_connections)
    connections_max_items = _normalize_connections_max_items_override(
        overrides.get("connections_max_items", env.connections_max_items)
    )
    include_link_image_descriptions = overrides.get(
        "include_link_image_descriptions", env.include_link_image_descriptions
    )
    include_pdf_content = overrides.get("include_pdf_content", env.include_pdf_content)
    include_media_descriptions = overrides.get(
        "include_media_descriptions", env.include_media_descriptions
    )
    recurse_users = overrides.get("recurse_users", env.recurse_users)
    exclude_channels = parse_arena_channel_exclusions(
        overrides.get("exclude_channels", env.exclude_channels)
    )
    connected_after = _normalize_optional_datetime_override(
        overrides.get("connected_after", env.connected_after)
    )
    connected_before = _normalize_optional_datetime_override(
        overrides.get("connected_before", env.connected_before)
    )
    created_after = _normalize_optional_datetime_override(
        overrides.get("created_after", env.created_after)
    )
    created_before = _normalize_optional_datetime_override(
        overrides.get("created_before", env.created_before)
    )

    settings = ArenaSettings(
        max_depth=max_depth,
        sort_order=sort_order,
        max_blocks_per_channel=max_blocks_per_channel,
        recurse_blocks=recurse_blocks,
        include_descriptions=include_descriptions,
        include_comments=include_comments,
        include_connections=include_connections,
        connections_max_items=connections_max_items,
        include_link_image_descriptions=include_link_image_descriptions,
        include_pdf_content=include_pdf_content,
        include_media_descriptions=include_media_descriptions,
        recurse_users=recurse_users,
        exclude_channels=exclude_channels,
        connected_after=connected_after,
        connected_before=connected_before,
        created_after=created_after,
        created_before=created_before,
    )
    _validate_time_window(settings)
    return settings


def _owner_slug(obj: dict) -> str:
    owner = obj.get("owner") or obj.get("user") or {}
    return (owner.get("slug") or "").lower()


def _owner_id(obj: dict) -> int | None:
    owner = obj.get("owner") or obj.get("user") or {}
    return owner.get("id")


def _should_recurse(
    item: dict,
    recurse_users: set[str] | None,
    root_owner_id: int | None,
) -> bool:
    if recurse_users is None:
        return True
    if recurse_users & {"self", "author", "owner"}:
        return _owner_id(item) == root_owner_id
    return _owner_slug(item) in recurse_users


def _channel_page_contents(page_data: dict) -> list[dict]:
    return list(page_data.get("data", page_data.get("contents", [])))


def _is_channel_item(item: dict) -> bool:
    return item.get("base_type") == "Channel" or item.get("type") == "Channel"


def _dedupe_channels(channels: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[str] = set()
    for channel in channels:
        identity = _block_identity(channel)
        if identity is not None:
            if identity in seen:
                continue
            seen.add(identity)
        deduped.append(channel)
    return deduped


def fetch_owner_profile(
    kind: Literal["user", "group"],
    slug: str,
    *,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
) -> dict:
    cache_key = f"v1:owner-profile:{kind}:{slug}:auth={_auth_cache_partition()}"
    if use_cache and not refresh_cache:
        from .cache import get_cached_channel

        cached = get_cached_channel(cache_key, cache_ttl)
        if cached is not None:
            payload = json.loads(cached)
            if isinstance(payload, dict):
                return payload

    profile = _fetch_owner_profile(kind, slug)

    if use_cache:
        from .cache import store_channel

        store_channel(
            cache_key,
            json.dumps(profile, ensure_ascii=False),
            block_count=1,
        )
    return profile


def fetch_user_groups(
    slug: str,
    *,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
    sort: str = "name_asc",
) -> list[dict]:
    cache_key = f"v1:user-groups:{slug}:sort={sort}:auth={_auth_cache_partition()}"
    if use_cache and not refresh_cache:
        from .cache import get_cached_channel

        cached = get_cached_channel(cache_key, cache_ttl)
        if cached is not None:
            payload = json.loads(cached)
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]

    first_page = _fetch_user_groups_page(slug, 1, sort=sort)
    groups = [
        item for item in _channel_page_contents(first_page) if isinstance(item, dict)
    ]
    meta = first_page.get("meta", {})
    total_pages = meta.get("total_pages", 1)
    for page in range(2, total_pages + 1):
        _log(f"  fetching user groups: {slug} page {page}/{total_pages}")
        page_data = _fetch_user_groups_page(slug, page, sort=sort)
        groups.extend(
            item for item in _channel_page_contents(page_data) if isinstance(item, dict)
        )

    if use_cache:
        from .cache import store_channel

        store_channel(
            cache_key,
            json.dumps(groups, ensure_ascii=False),
            block_count=len(groups),
        )
    return groups


def fetch_owner_channels(
    kind: Literal["user", "group"],
    slug: str,
    *,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
    settings: ArenaSettings | None = None,
    sort: str = "created_at_asc",
) -> list[dict]:
    if settings is None:
        settings = _arena_settings_from_env()
    cache_key = (
        f"v1:owner-channels:{kind}:{slug}:sort={sort}:"
        f"auth={_auth_cache_partition()}:exclude={settings.exclude_channels.cache_key()}"
    )
    if use_cache and not refresh_cache:
        from .cache import get_cached_channel

        cached = get_cached_channel(cache_key, cache_ttl)
        if cached is not None:
            payload = json.loads(cached)
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]

    first_page = _fetch_owner_channel_page(kind, slug, 1, sort=sort)
    channels = [
        item
        for item in _channel_page_contents(first_page)
        if _is_channel_item(item) and not settings.exclude_channels.matches(item)
    ]
    meta = first_page.get("meta", {})
    total_pages = meta.get("total_pages", 1)
    for page in range(2, total_pages + 1):
        _log(f"  fetching {kind} channels: {slug} page {page}/{total_pages}")
        page_data = _fetch_owner_channel_page(kind, slug, page, sort=sort)
        channels.extend(
            item
            for item in _channel_page_contents(page_data)
            if _is_channel_item(item) and not settings.exclude_channels.matches(item)
        )
    channels = _dedupe_channels(channels)

    if use_cache:
        from .cache import store_channel

        store_channel(
            cache_key,
            json.dumps(channels, ensure_ascii=False),
            block_count=len(channels),
        )
    return channels


def _has_time_window(settings: ArenaSettings | None) -> bool:
    return settings is not None and (
        settings.connected_after is not None
        or settings.connected_before is not None
        or settings.created_after is not None
        or settings.created_before is not None
    )


def _can_short_circuit_channel_paging(
    *,
    sort_order: str,
    max_blocks_per_channel: int | None,
    has_time_window: bool,
) -> bool:
    return (
        max_blocks_per_channel is not None
        and max_blocks_per_channel > 0
        and not has_time_window
        and sort_order in {"asc", "position-asc", "desc", "position-desc"}
    )


def _channel_content_count(metadata: dict, fallback: int | None = None) -> int | None:
    counts = metadata.get("counts")
    if not isinstance(counts, dict):
        return fallback
    value = counts.get("contents")
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return fallback


def _resolve_recurse_block_count(
    recurse_blocks: ArenaRecurseBlockLimit | None,
    root_content_count: int | None,
) -> int | None:
    if recurse_blocks is None:
        return None
    if recurse_blocks.kind == "count":
        return int(recurse_blocks.value)
    if root_content_count is None:
        raise ValueError(
            "Are.na recurse blocks ratio requires a root channel content count"
        )
    return max(1, math.ceil(root_content_count * float(recurse_blocks.value)))


def _stricter_limit(*limits: int | None) -> int | None:
    values = [value for value in limits if value is not None]
    if not values:
        return None
    return min(values)


def _fetch_all_channel_contents(
    slug: str,
    *,
    max_depth: int | None = None,
    sort_order: str | None = None,
    max_blocks_per_channel: int | None = None,
    recurse_blocks: ArenaRecurseBlockLimit | None = None,
    _time_window_settings: ArenaSettings | None = None,
    _depth: int = 0,
    _visited: set[int] | None = None,
    _root_owner_id: int | None = None,
    _root_content_count: int | None = None,
    _recurse_users: set[str] | None = ...,
) -> tuple[dict, list[dict]]:
    if max_depth is None:
        max_depth = _get_max_depth()
    if sort_order is None:
        sort_order = _get_sort_order()
    if max_blocks_per_channel is None:
        max_blocks_per_channel = _get_max_blocks_per_channel()
    if _recurse_users is ...:
        _recurse_users = _get_recurse_users()
    if _visited is None:
        _visited = set()

    metadata = _fetch_channel(slug)
    channel_id = metadata.get("id")
    channel_title = metadata.get("title") or slug
    if channel_id:
        _visited.add(channel_id)

    if _root_owner_id is None:
        _root_owner_id = _owner_id(metadata)
    if _depth == 0 and _root_content_count is None:
        _root_content_count = _channel_content_count(metadata)

    recurse_block_count = None
    if _depth > 0:
        recurse_block_count = _resolve_recurse_block_count(
            recurse_blocks, _root_content_count
        )
    effective_max_blocks = _stricter_limit(max_blocks_per_channel, recurse_block_count)

    indent = "  " * (_depth + 1)
    _log(f"{indent}fetching channel: {channel_title}")

    all_contents: list[dict] = []
    first_page = _fetch_channel_page(slug, 1)
    first_page_contents = _channel_page_contents(first_page)

    meta = first_page.get("meta", {})
    total_pages = meta.get("total_pages", 1)
    has_time_window = _has_time_window(_time_window_settings)
    can_short_circuit = _can_short_circuit_channel_paging(
        sort_order=sort_order,
        max_blocks_per_channel=effective_max_blocks,
        has_time_window=has_time_window,
    )

    if can_short_circuit and sort_order in {"asc", "position-asc"}:
        all_contents.extend(first_page_contents)
        for page in range(2, total_pages + 1):
            if (
                effective_max_blocks is not None
                and len(all_contents) >= effective_max_blocks
            ):
                break
            _log(f"{indent}  page {page}/{total_pages}")
            page_data = _fetch_channel_page(slug, page)
            all_contents.extend(_channel_page_contents(page_data))
    elif can_short_circuit:
        needed = effective_max_blocks or 0
        newest_first: list[dict] = []
        for page in range(total_pages, 0, -1):
            if len(newest_first) >= needed:
                break
            if page == 1:
                page_data = first_page
                page_contents = first_page_contents
            else:
                _log(f"{indent}  page {page}/{total_pages}")
                page_data = _fetch_channel_page(slug, page)
                page_contents = _channel_page_contents(page_data)
            page_contents.reverse()
            remaining = needed - len(newest_first)
            newest_first.extend(page_contents[:remaining])
        all_contents = list(reversed(newest_first))
    else:
        all_contents.extend(first_page_contents)
        for page in range(2, total_pages + 1):
            _log(f"{indent}  page {page}/{total_pages}")
            page_data = _fetch_channel_page(slug, page)
            all_contents.extend(_channel_page_contents(page_data))

    all_contents = _sort_channel_contents(all_contents, sort_order)

    if _depth == 0 and _root_content_count is None:
        _root_content_count = len(all_contents)

    if has_time_window:
        all_contents = [
            item
            for item in all_contents
            if (
                item.get("base_type") == "Channel"
                or item.get("type") == "Channel"
                or _passes_block_time_window(item, _time_window_settings)
            )
        ]

    if has_time_window:
        count_before_limit = len(all_contents)
    else:
        count_before_limit = _channel_content_count(
            metadata, fallback=len(all_contents)
        )
    if effective_max_blocks is not None:
        all_contents = all_contents[:effective_max_blocks]

    if (
        _depth > 0
        and recurse_blocks is not None
        and effective_max_blocks is not None
        and count_before_limit > effective_max_blocks
    ):
        metadata["_contextualize_sampled_by_recurse_blocks"] = True

    if _time_window_settings is not None:
        all_contents = [
            item
            for item in all_contents
            if not (
                _is_channel_item(item)
                and _time_window_settings.exclude_channels.matches(item)
            )
        ]

    if _depth < max_depth:
        expanded: list[dict] = []
        for item in all_contents:
            if item.get("base_type") == "Channel" or item.get("type") == "Channel":
                nested_id = item.get("id")
                nested_slug = item.get("slug")
                if nested_id and nested_id in _visited:
                    expanded.append(item)
                    continue
                if nested_slug and _should_recurse(
                    item, _recurse_users, _root_owner_id
                ):
                    _visited.add(nested_id)
                    nested_meta, nested_contents = _fetch_all_channel_contents(
                        nested_slug,
                        max_depth=max_depth,
                        sort_order=sort_order,
                        max_blocks_per_channel=max_blocks_per_channel,
                        recurse_blocks=recurse_blocks,
                        _time_window_settings=_time_window_settings,
                        _depth=_depth + 1,
                        _visited=_visited,
                        _root_owner_id=_root_owner_id,
                        _root_content_count=_root_content_count,
                        _recurse_users=_recurse_users,
                    )
                    item["_nested_metadata"] = nested_meta
                    item["_nested_contents"] = nested_contents
                    item["_nested_sampled_by_recurse_blocks"] = bool(
                        nested_meta.get("_contextualize_sampled_by_recurse_blocks")
                    )
            expanded.append(item)
        all_contents = expanded

    return metadata, all_contents


def _fetch_block(block_id: int) -> dict:
    return _api_get(f"/blocks/{block_id}")


def _fetch_block_comments_page(block_id: int, page: int, per: int = 100) -> dict:
    return _api_get(f"/blocks/{block_id}/comments", {"page": page, "per": per})


def _fetch_block_connections_page(block_id: int, page: int, per: int = 100) -> dict:
    return _api_get(
        f"/blocks/{block_id}/connections",
        {"page": page, "per": per, "sort": "created_at_desc"},
    )


def _fetch_channel_connections_page(
    channel_id: int, page: int, per: int = 100
) -> dict:
    return _api_get(
        f"/channels/{channel_id}/connections",
        {"page": page, "per": per, "sort": "created_at_desc"},
    )


def _fetch_all_block_comments(block_id: int) -> list[dict]:
    first_page = _fetch_block_comments_page(block_id, 1)
    comments = list(first_page.get("data", first_page.get("comments", [])))
    meta = first_page.get("meta", {})
    total_pages = meta.get("total_pages", 1)
    for page in range(2, total_pages + 1):
        page_data = _fetch_block_comments_page(block_id, page)
        comments.extend(page_data.get("data", page_data.get("comments", [])))
    return comments


def _block_source_channel_context(block: dict) -> dict | None:
    context = block.get("_contextualize_channel_context")
    return context if isinstance(context, dict) else None


def _channel_matches_source_context(channel: dict, source_context: dict | None) -> bool:
    if not source_context:
        return False
    source_id = source_context.get("id")
    channel_id = channel.get("id")
    if isinstance(source_id, int) and isinstance(channel_id, int):
        return source_id == channel_id
    source_slug = source_context.get("slug")
    channel_slug = channel.get("slug")
    if isinstance(source_slug, str) and isinstance(channel_slug, str):
        return source_slug.lower() == channel_slug.lower()
    return False


def _fetch_block_connections(
    block_id: int,
    *,
    max_items: int | None,
    source_context: dict | None,
) -> tuple[list[dict], bool]:
    requested_count = None if max_items is None else max_items + 1
    per_page = 100 if requested_count is None else max(1, min(100, requested_count))
    channels: list[dict] = []
    page = 1

    while True:
        page_data = _fetch_block_connections_page(block_id, page, per=per_page)
        page_channels = list(page_data.get("data", page_data.get("channels", [])))
        for channel in page_channels:
            if _channel_matches_source_context(channel, source_context):
                continue
            channels.append(channel)
            if requested_count is not None and len(channels) >= requested_count:
                return channels[:max_items], True

        meta = page_data.get("meta") or {}
        next_page = meta.get("next_page")
        if isinstance(next_page, int):
            page = next_page
            continue
        total_pages = meta.get("total_pages")
        if isinstance(total_pages, int) and page < total_pages:
            page += 1
            continue
        if meta.get("has_more_pages"):
            page += 1
            continue
        break

    if max_items is None:
        return channels, False
    return channels[:max_items], len(channels) > max_items


def _fetch_channel_connections(
    channel_id: int,
    *,
    max_items: int | None,
    source_context: dict | None,
) -> tuple[list[dict], bool]:
    requested_count = None if max_items is None else max_items + 1
    per_page = 100 if requested_count is None else max(1, min(100, requested_count))
    channels: list[dict] = []
    page = 1

    while True:
        page_data = _fetch_channel_connections_page(channel_id, page, per=per_page)
        page_channels = list(page_data.get("data", page_data.get("channels", [])))
        for channel in page_channels:
            if _channel_matches_source_context(channel, source_context):
                continue
            channels.append(channel)
            if requested_count is not None and len(channels) >= requested_count:
                return channels[:max_items], True

        meta = page_data.get("meta") or {}
        next_page = meta.get("next_page")
        if isinstance(next_page, int):
            page = next_page
            continue
        total_pages = meta.get("total_pages")
        if isinstance(total_pages, int) and page < total_pages:
            page += 1
            continue
        if meta.get("has_more_pages"):
            page += 1
            continue
        break

    if max_items is None:
        return channels, False
    return channels[:max_items], len(channels) > max_items


def _block_image_urls(block: dict) -> list[str]:
    image = block.get("image") or {}
    return list(
        dict.fromkeys(
            [
                u
                for u in (
                    image.get("src"),
                    (image.get("large") or {}).get("src"),
                    (image.get("original") or {}).get("url"),
                    image.get("url"),
                )
                if u
            ]
        )
    )


_DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; contextualize/1.0)",
    "Accept": "image/*,application/*;q=0.9,*/*;q=0.8",
    "Referer": "https://www.are.na/",
}


def _download_to_temp(
    url: str, suffix: str = "", *, media_cache_identity: str | None = None
) -> Path | None:
    from .cache import get_cached_media_bytes, store_media_bytes
    from ..shared.media import download_cached_media_to_temp
    from contextualize.runtime import get_refresh_media

    cache_identity = media_cache_identity or url
    return download_cached_media_to_temp(
        url,
        suffix=suffix,
        headers=_DOWNLOAD_HEADERS,
        cache_identity=cache_identity,
        get_cached_media_bytes=get_cached_media_bytes,
        store_media_bytes=store_media_bytes,
        refresh_cache=get_refresh_media(),
    )


def _render_block_binary(
    url: str,
    suffix: str,
    *,
    media_cache_identity: str | None = None,
    send_label: str | None = None,
) -> str:
    from contextualize.render.markitdown import (
        MarkItDownConversionError,
        convert_path_to_markdown,
    )

    tmp = _download_to_temp(
        url, suffix=suffix, media_cache_identity=media_cache_identity
    )
    if tmp is None:
        return ""
    try:
        from contextualize.runtime import get_refresh_images

        label = send_label or "arena-media"
        _log(f"  processing media: {label} ({url})")
        refresh_images = get_refresh_images()
        result = convert_path_to_markdown(str(tmp), refresh_images=refresh_images)
        return result.markdown
    except MarkItDownConversionError as exc:
        _log(f"  image conversion failed for {url}: {exc}")
        return ""
    finally:
        tmp.unlink(missing_ok=True)


def _desc_separator(description: str) -> str:
    max_stars = 0
    for line in description.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"\*{3,}", stripped):
            max_stars = max(max_stars, len(stripped))
    return "*" * (max_stars + 2) if max_stars >= 3 else "***"


def _block_creator(block: dict) -> dict | None:
    for key in ("created_by", "user", "owner"):
        value = block.get(key)
        if isinstance(value, dict) and value:
            return value
    return None


def _block_connection(block: dict) -> dict:
    connection = block.get("connection")
    return connection if isinstance(connection, dict) else {}


def _block_added_by(block: dict) -> dict | None:
    connection = _block_connection(block)
    value = connection.get("connected_by") or block.get("connected_by")
    return value if isinstance(value, dict) and value else None


def _format_event_line(label: str, timestamp: str, actor: dict | None) -> str:
    line = f"{label:<7} {timestamp}"
    if actor:
        line += f" by {_entity_name(actor)}"
    return line


def _format_metadata_line(label: str, value: str) -> str:
    return f"{label:<7} {value}"


def _normalize_source_title(value: str) -> str:
    return " ".join(value.split()).casefold()


def _format_source_line(block: dict, *, block_title: str, block_type: str) -> str:
    if block_type == "Link":
        return ""
    source = block.get("source")
    if not isinstance(source, dict):
        return ""
    source_url = source.get("url")
    if not isinstance(source_url, str) or not source_url.strip():
        return ""
    source_value = source_url.strip()
    source_title = source.get("title")
    if isinstance(source_title, str) and source_title.strip():
        clean_title = " ".join(source_title.split())
        if _normalize_source_title(clean_title) != _normalize_source_title(
            block_title
        ):
            source_value = f"{clean_title} <{source_value}>"
    return _format_metadata_line("source", source_value)


def _format_date_line(
    block: dict, *, block_title: str = "", block_type: str = ""
) -> str:
    lines: list[str] = []
    source_line = _format_source_line(
        block, block_title=block_title, block_type=block_type
    )
    if source_line:
        lines.append(source_line)

    created_at = _format_metadata_timestamp(block.get("created_at"))
    if created_at:
        lines.append(_format_event_line("created", created_at, _block_creator(block)))

    if _block_source_channel_context(block):
        connection = _block_connection(block)
        added_at = _format_metadata_timestamp(
            connection.get("connected_at") or block.get("connected_at")
        )
        if added_at:
            lines.append(_format_event_line("added", added_at, _block_added_by(block)))

    return "\n".join(lines)


_DESCRIPTION_HEADING_RE = re.compile(
    r"(?m)^#{1,6}\s+Description(?: \(auto-generated\))?:?\s*$"
)
_IMAGE_SIZE_RE = re.compile(r"(?i)^image\s*size\s*:\s*([0-9]+\s*[x×]\s*[0-9]+)\s*$")
_STAR_LINE_RE = re.compile(r"^\*{3,}$")


def _extract_markdown_like_text(raw: object) -> str:
    if isinstance(raw, dict):
        return raw.get("markdown") or raw.get("plain") or ""
    if isinstance(raw, str):
        return raw
    return ""


def _format_metadata_timestamp(raw: object) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return ""
    return _format_comment_timestamp(raw.strip())


def _format_channel_description(description: str) -> str:
    if "\n" in description:
        return f"Description:\n{description}"
    return f"Description: {description}"


def _format_info_section(info: str) -> str:
    if "\n" in info:
        return f"Info:\n{info}"
    return f"Info: {info}"


def _format_comment_timestamp(iso_timestamp: str) -> str:
    if not iso_timestamp:
        return ""
    try:
        parsed = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
    except ValueError:
        return iso_timestamp
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.strftime("%Y-%m-%dT%H:%MZ")


def _comment_author(comment: dict) -> str:
    user = comment.get("user") or comment.get("owner") or {}
    for key in ("name", "username", "slug"):
        value = user.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Unknown"


def _comment_body(comment: dict) -> str:
    raw = (
        comment.get("content")
        or comment.get("comment")
        or comment.get("body")
        or comment.get("text")
        or ""
    )
    text = _extract_markdown_like_text(raw)
    return " ".join(text.split())


def _render_comments_section(comments: list[dict]) -> str:
    if not comments:
        return ""
    sorted_comments = sorted(
        comments,
        key=lambda c: (c.get("created_at") or "", c.get("id") or 0),
        reverse=True,
    )
    lines: list[str] = []
    for comment in sorted_comments:
        author = _comment_author(comment)
        body = _comment_body(comment)
        timestamp = _format_comment_timestamp(comment.get("created_at") or "")
        prefix = author
        if timestamp:
            prefix += f" @ {timestamp}"
        if body:
            lines.append(f"{prefix}: {body}")
        else:
            lines.append(f"{prefix}:")
    return "## Comments\n\n" + "\n\n".join(lines)


def _source_channel_cache_key(source_context: dict | None) -> str:
    if not source_context:
        return "direct"
    channel_id = source_context.get("id")
    if isinstance(channel_id, int):
        return f"id:{channel_id}"
    slug = source_context.get("slug")
    if isinstance(slug, str) and slug.strip():
        return f"slug:{slug.strip().lower()}"
    return "unknown"


def _block_connection_cache_identity(
    block: dict,
    *,
    max_items: int | None,
    source_context: dict | None,
) -> str | None:
    block_id = block.get("id")
    if not isinstance(block_id, int):
        return None
    connection = block.get("connection")
    connection_key = "none"
    if isinstance(connection, dict):
        connection_id = connection.get("id")
        connected_at = connection.get("connected_at")
        if isinstance(connection_id, int):
            connection_key = f"id:{connection_id}"
        elif isinstance(connected_at, str) and connected_at:
            connection_key = f"at:{connected_at}"
    limit_key = "all" if max_items is None else str(max_items)
    return (
        f"v2:block:{block_id}:source={_source_channel_cache_key(source_context)}:"
        f"connection={connection_key}:limit={limit_key}:auth={_auth_cache_partition()}"
    )


def _render_connected_channel_line(channel: dict) -> str:
    title = channel.get("title") or channel.get("slug") or "Untitled"
    details: list[str] = []
    owner = channel.get("owner") or channel.get("user")
    owner_name = ""
    if isinstance(owner, dict) and owner:
        owner_name = _entity_name(owner)
    visibility = channel.get("visibility")
    owner_visibility = owner_name
    if isinstance(visibility, str) and visibility.strip():
        owner_visibility = (
            f"{owner_visibility}, {visibility.strip()}"
            if owner_visibility
            else visibility.strip()
        )
    if owner_visibility:
        details.append(owner_visibility)
    counts = channel.get("counts")
    if isinstance(counts, dict):
        block_count = counts.get("blocks")
        if block_count is None or block_count == "":
            block_count = counts.get("contents")
        if block_count is not None and block_count != "":
            suffix = "block" if block_count == 1 else "blocks"
            details.append(f"{block_count} {suffix}")
    slug = channel.get("slug")
    if isinstance(slug, str) and slug:
        details.append(slug)
    if details:
        return f"- {title} ({'; '.join(details)})"
    return f"- {title}"


def _render_connected_channels_section(
    channels: list[dict],
    *,
    cap_hit: bool,
    source_context: dict | None,
) -> str:
    if not channels:
        return ""
    heading = "## Other channels" if source_context else "## Channels"
    lines = [heading, ""]
    if cap_hit:
        lines.extend(
            [
                (
                    f"Showing first {len(channels)} channels; "
                    "more omitted by limit."
                ),
                "",
            ]
        )
    lines.extend(_render_connected_channel_line(channel) for channel in channels)
    return "\n".join(lines)


def _block_comment_count_hint(block: dict) -> int | None:
    counts = block.get("counts")
    if isinstance(counts, dict):
        raw = counts.get("comments")
        if isinstance(raw, int):
            return raw
    for key in ("comments_count", "comment_count"):
        raw = block.get(key)
        if isinstance(raw, int):
            return raw
    return None


def _strip_description_heading(markdown: str) -> str:
    stripped = markdown.strip()
    return _DESCRIPTION_HEADING_RE.sub("", stripped, count=1).strip()


def _normalize_image_size(raw: str) -> str:
    return raw.replace(" ", "").replace("×", "x").lower()


def _normalize_image_description_markdown(markdown: str) -> str:
    body = _strip_description_heading(markdown)
    if not body:
        return ""
    lines = body.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)

    dimensions = ""
    if lines:
        match = _IMAGE_SIZE_RE.match(lines[0].strip())
        if match:
            dimensions = _normalize_image_size(match.group(1))
            lines.pop(0)
            while lines and not lines[0].strip():
                lines.pop(0)

    normalized_body = "\n".join(lines).strip()
    heading = "## Block image description (auto-generated"
    if dimensions:
        heading += f", dimensions: {dimensions}"
    heading += ")"
    if not normalized_body:
        return heading
    return f"{heading}\n\n{normalized_body}"


def _render_link_image_description(
    block: dict, *, block_id: object, updated_at: str, title: str
) -> str:
    image_urls = _block_image_urls(block)
    for image_url in image_urls:
        suffix = Path(image_url.split("?")[0]).suffix or ".jpg"
        media_cache_identity = (
            f"arena:block:{block_id}:{updated_at}:link-image:{image_url}"
            if block_id and updated_at
            else image_url
        )
        send_label = f"link-image:{block_id or 'unknown'}:{(title or 'untitled')[:80]}"
        converted = _render_block_binary(
            image_url,
            suffix,
            media_cache_identity=media_cache_identity,
            send_label=send_label,
        )
        if not converted:
            continue
        normalized = _normalize_image_description_markdown(converted)
        if not normalized:
            continue
        return normalized
    return ""


def _render_attachment_preview_description(
    block: dict, *, block_id: object, updated_at: str, title: str
) -> str:
    image_urls = _block_image_urls(block)
    for image_url in image_urls:
        suffix = Path(image_url.split("?")[0]).suffix or ".jpg"
        media_cache_identity = (
            f"arena:block:{block_id}:{updated_at}:attachment-preview:{image_url}"
            if block_id and updated_at
            else image_url
        )
        send_label = (
            f"attachment-preview:{block_id or 'unknown'}:"
            f"{(title or 'untitled')[:80]}"
        )
        converted = _render_block_binary(
            image_url,
            suffix,
            media_cache_identity=media_cache_identity,
            send_label=send_label,
        )
        if not converted:
            continue
        normalized = _normalize_image_description_markdown(converted)
        if not normalized:
            continue
        return normalized
    return ""


def _comments_separator(rendered: str) -> str:
    best = ""
    for line in rendered.splitlines():
        stripped = line.strip()
        if _STAR_LINE_RE.fullmatch(stripped):
            if len(stripped) > len(best):
                best = stripped
    return best or "***"


def _append_block_detail_sections(
    rendered: str | None, *sections: str
) -> str | None:
    if not rendered:
        return rendered
    output = rendered
    for section in sections:
        if not section:
            continue
        separator = _comments_separator(output)
        output = f"{output}\n\n{separator}\n\n{section}"
    return output


def _append_comments_section(rendered: str | None, comments_section: str) -> str | None:
    return _append_block_detail_sections(rendered, comments_section)


def _block_comments_output(block: dict, *, include_comments: bool) -> str:
    from .cache import (
        get_cached_block_comments,
        store_block_comments,
    )
    from contextualize.runtime import get_refresh_cache

    if not include_comments:
        return ""
    block_id = block.get("id")
    if not isinstance(block_id, int):
        return ""
    hint = _block_comment_count_hint(block)
    if hint == 0:
        return ""

    if not get_refresh_cache():
        cached = get_cached_block_comments(block_id, ttl=_get_comments_cache_ttl())
        if cached is not None:
            return cached

    try:
        comments = _fetch_all_block_comments(block_id)
    except Exception as exc:
        _log(f"  failed to fetch comments for block {block_id}: {type(exc).__name__}")
        return ""
    rendered = _render_comments_section(comments)
    store_block_comments(block_id, rendered)
    return rendered


def _block_connections_output(
    block: dict,
    *,
    include_connections: bool,
    max_items: int | None,
) -> str:
    from .cache import (
        get_cached_block_connections,
        store_block_connections,
    )

    from contextualize.runtime import get_refresh_cache

    if not include_connections:
        return ""
    block_id = block.get("id")
    if not isinstance(block_id, int):
        return ""

    source_context = _block_source_channel_context(block)
    cache_identity = _block_connection_cache_identity(
        block,
        max_items=max_items,
        source_context=source_context,
    )
    if cache_identity is not None and not get_refresh_cache():
        cached = get_cached_block_connections(
            cache_identity, ttl=_get_connections_cache_ttl()
        )
        if cached is not None:
            return cached

    connected_section = ""
    try:
        channels, cap_hit = _fetch_block_connections(
            block_id,
            max_items=max_items,
            source_context=source_context,
        )
        connected_section = _render_connected_channels_section(
            channels,
            cap_hit=cap_hit,
            source_context=source_context,
        )
    except Exception as exc:
        _log(
            f"  failed to fetch connected channels for block {block_id}: {type(exc).__name__}"
        )

    rendered = connected_section
    if cache_identity is not None:
        store_block_connections(cache_identity, rendered)
    return rendered


def _attachment_filename(attachment: dict[str, Any]) -> str:
    filename = attachment.get("filename")
    if isinstance(filename, str) and filename.strip():
        return filename.strip()
    url = attachment.get("url")
    if isinstance(url, str) and url.strip():
        name = Path(urlparse(url).path).name
        if name:
            return name
    return "attachment"


def _attachment_content_type(attachment: dict[str, Any]) -> str:
    content_type = attachment.get("content_type")
    return content_type.strip() if isinstance(content_type, str) else ""


def _attachment_extension(attachment: dict[str, Any], filename: str) -> str:
    extension = attachment.get("file_extension")
    if isinstance(extension, str) and extension.strip():
        return extension.strip().lstrip(".")
    return Path(filename).suffix.lstrip(".")


def _attachment_size_bytes(attachment: dict[str, Any]) -> int | None:
    for key in ("size", "file_size", "byte_size", "bytes"):
        raw = attachment.get(key)
        if isinstance(raw, int) and raw >= 0:
            return raw
        if isinstance(raw, str) and raw.isdigit():
            return int(raw)
    return None


def _attachment_fallback_lines(
    block: dict[str, Any],
    attachment: dict[str, Any],
    *,
    filename: str,
    content_type: str,
    att_url: str,
) -> list[str]:
    lines = [f"[Attachment: {filename or block.get('title') or block.get('id')}]"]
    if content_type:
        lines.append(f"Type: {content_type}")
    size = _attachment_size_bytes(attachment)
    if size is not None:
        lines.append(f"Size: {size} bytes")
    if att_url:
        lines.append(f"URL: {att_url}")
    return lines


def _attachment_media_kind(
    *, filename: str, extension: str, content_type: str
) -> str | None:
    ctype = content_type.lower().strip()
    if ctype == "application/pdf":
        return "pdf"
    if ctype.startswith("image/"):
        return "image"
    if ctype.startswith("video/"):
        return "video"
    if ctype.startswith("audio/"):
        return "audio"

    suffix = (
        f".{extension.lstrip('.').lower()}"
        if extension
        else Path(filename).suffix.lower()
    )
    if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".heif", ".avif"}:
        return "image"
    if suffix in {".mp4", ".mov", ".webm", ".mkv", ".avi", ".mpeg", ".mpg", ".m4v"}:
        return "video"
    if suffix in {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".aiff"}:
        return "audio"
    if suffix == ".pdf":
        return "pdf"
    return None


def _block_attachment(block: dict[str, Any]) -> dict[str, Any] | None:
    attachment = block.get("attachment")
    if isinstance(attachment, dict) and attachment:
        return attachment
    return None


def build_arena_block_attachment_target(
    block_id: int, attachment: dict[str, Any]
) -> str:
    base = f"https://www.are.na/block/{block_id}"
    for key in ("id", "uuid"):
        value = attachment.get(key)
        if value is not None and str(value).strip():
            return f"{base}?attachment-id={quote(str(value).strip(), safe='')}"
    return f"{base}?attachment={quote(_attachment_filename(attachment), safe='')}"


def _attachment_media_cache_identity(
    block: dict[str, Any], attachment: dict[str, Any]
) -> str:
    att_url = str(attachment.get("url") or "")
    block_id = block.get("id")
    updated_at = block.get("updated_at") or ""
    if block_id and updated_at and att_url:
        return f"arena:block:{block_id}:{updated_at}:attachment:{att_url}"
    return att_url


def _select_attachment_for_target(
    block: dict[str, Any], parsed: dict[str, Any]
) -> dict[str, Any]:
    attachment = _block_attachment(block)
    if attachment is None:
        raise ValueError(f"Are.na block {parsed['block_id']} does not have an attachment")

    attachment_id = parsed.get("attachment_id")
    if attachment_id:
        candidates = [
            str(value).strip()
            for value in (attachment.get("id"), attachment.get("uuid"))
            if value is not None and str(value).strip()
        ]
        if candidates and str(attachment_id) not in candidates:
            raise ValueError(
                f"Are.na block {parsed['block_id']} does not have attachment id {attachment_id}"
            )
        return attachment

    attachment_name = parsed.get("attachment_name")
    if attachment_name:
        filename = _attachment_filename(attachment)
        if filename != str(attachment_name):
            raise ValueError(
                f"Are.na block {parsed['block_id']} does not have attachment {attachment_name}"
            )
    return attachment


def _target_label(value: object, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _append_structured_target(
    items: list[dict[str, Any]],
    seen: set[str],
    target: str,
    *,
    label: str,
    kind: str,
    metadata: dict[str, Any],
) -> None:
    if not target or target in seen:
        return
    seen.add(target)
    items.append(
        {
            "target": target,
            "label": label,
            "kind": kind,
            "metadata": metadata,
        }
    )


def list_arena_block_targets(
    block: dict[str, Any], *, source_target: str
) -> list[dict[str, Any]]:
    block_id = block.get("id")
    items: list[dict[str, Any]] = []
    seen: set[str] = set()

    attachment = _block_attachment(block)
    if isinstance(block_id, int) and attachment is not None:
        att_url = str(attachment.get("url") or "")
        if att_url:
            filename = _attachment_filename(attachment)
            content_type = _attachment_content_type(attachment)
            extension = _attachment_extension(attachment, filename)
            media_kind = _attachment_media_kind(
                filename=filename,
                extension=extension,
                content_type=content_type,
            )
            child_target = build_arena_block_attachment_target(block_id, attachment)
            _append_structured_target(
                items,
                seen,
                child_target,
                label=filename,
                kind=f"attachment:{media_kind or 'file'}",
                metadata={
                    "block_id": block_id,
                    "filename": filename,
                    "content_type": content_type or None,
                    "url": att_url,
                    "bytes": _attachment_size_bytes(attachment),
                    "source_block": source_target,
                },
            )

    source = block.get("source")
    if isinstance(source, dict):
        source_url = source.get("url")
        if isinstance(source_url, str) and source_url.strip():
            _append_structured_target(
                items,
                seen,
                source_url.strip(),
                label=_target_label(source.get("title"), source_url.strip()),
                kind="link",
                metadata={"source": "source", "block_id": block_id},
            )

    embed = block.get("embed")
    if isinstance(embed, dict):
        embed_url = embed.get("url")
        if isinstance(embed_url, str) and embed_url.strip():
            _append_structured_target(
                items,
                seen,
                embed_url.strip(),
                label=_target_label(embed.get("title"), embed_url.strip()),
                kind="embed",
                metadata={"source": "embed", "block_id": block_id},
            )

    for image_index, image_url in enumerate(_block_image_urls(block)):
        _append_structured_target(
            items,
            seen,
            image_url,
            label=Path(urlparse(image_url).path).name or image_url,
            kind="image",
            metadata={
                "source": "image",
                "image_index": image_index,
                "block_id": block_id,
            },
        )
    return items


def _should_refresh_attachment_media(
    *, filename: str, extension: str, content_type: str
) -> bool:
    from contextualize.runtime import (
        get_refresh_audio,
        get_refresh_images,
        get_refresh_media,
        get_refresh_videos,
    )

    if get_refresh_media():
        return True

    media_kind = _attachment_media_kind(
        filename=filename, extension=extension, content_type=content_type
    )
    if media_kind == "image":
        return get_refresh_images()
    if media_kind == "video":
        return get_refresh_videos()
    if media_kind == "audio":
        return get_refresh_audio()
    return False


def _format_block_output(
    title: str, description: str, content: str, date: str = ""
) -> str | None:
    if not title and not description and not content and not date:
        return None

    parts: list[str] = []
    if title and date:
        parts.append(f"{_format_metadata_line('title', title)}\n{date}\n---")
    elif date:
        parts.append(f"{date}\n---")
    elif title:
        parts.append(f"{_format_metadata_line('title', title)}\n---")

    if description and content:
        sep = _desc_separator(description)
        parts.append(f"{description}\n\n{sep}")
        parts.append(content)
    elif description:
        parts.append(description)
    elif content:
        parts.append(content)

    return "\n\n".join(parts)


def _format_media_fallback_output(
    title: str, description: str, content: str, *, date: str
) -> str:
    if date or description:
        return _format_block_output(title, description, content, date=date) or content
    return content


def _render_block(
    block: dict,
    *,
    include_descriptions: bool | None = None,
    include_comments: bool | None = None,
    include_connections: bool | None = None,
    connections_max_items: int | None = None,
    include_link_image_descriptions: bool | None = None,
    include_pdf_content: bool | None = None,
    include_media_descriptions: bool | None = None,
) -> str | None:
    from .cache import get_cached_block_render, store_block_render

    block_type = block.get("class") or block.get("type", "")
    state = block.get("state")
    if state == "processing" or block_type == "PendingBlock":
        return None

    block_id = block.get("id")
    updated_at = block.get("updated_at") or ""
    from contextualize.runtime import (
        get_refresh_images,
        get_refresh_media,
        get_refresh_videos,
    )

    title = block.get("title") or ""
    if include_descriptions is None:
        include_descriptions = _get_include_descriptions()
    if include_descriptions:
        description = _extract_markdown_like_text(block.get("description") or "")
    else:
        description = ""
    if include_comments is None:
        include_comments = _get_include_comments()
    if include_connections is None:
        include_connections = _get_include_connections()
    if connections_max_items is None:
        connections_max_items = _get_connections_max_items()
    if include_link_image_descriptions is None:
        include_link_image_descriptions = _get_include_link_image_descriptions()
    if include_pdf_content is None:
        include_pdf_content = _get_include_pdf_content()
    if include_media_descriptions is None:
        include_media_descriptions = _get_include_media_descriptions()

    render_variant_version = "v3" if block_type == "Link" else "v4"
    render_variant = (
        f"{render_variant_version}:type={str(block_type).lower()}"
        f":desc={int(bool(include_descriptions))}"
        f":linkimg={int(bool(include_link_image_descriptions))}"
        f":pdf={int(bool(include_pdf_content))}"
        f":media={int(bool(include_media_descriptions))}"
    )
    if block_type == "Attachment":
        render_variant += ":attachment-fallback=2"

    date = _format_date_line(block, block_title=title, block_type=str(block_type))
    core_output: str | None = None

    if block_type == "Text":
        content = _extract_markdown_like_text(block.get("content") or "")
        if description == content:
            description = ""
        core_output = _format_block_output(title, description, content, date=date)

    elif block_type == "Image":
        refresh_image = get_refresh_images() or get_refresh_media()
        if block_id and updated_at and not refresh_image:
            cached = get_cached_block_render(
                block_id, updated_at, render_variant=render_variant
            )
            if cached is not None:
                core_output = cached
        if core_output is None and include_media_descriptions:
            image_urls = _block_image_urls(block)
            for image_url in image_urls:
                suffix = Path(image_url.split("?")[0]).suffix or ".jpg"
                media_cache_identity = (
                    f"arena:block:{block_id}:{updated_at}:image:{image_url}"
                    if block_id and updated_at
                    else image_url
                )
                send_label = (
                    f"image:{block_id or 'unknown'}:{(title or 'untitled')[:80]}"
                )
                converted = _render_block_binary(
                    image_url,
                    suffix,
                    media_cache_identity=media_cache_identity,
                    send_label=send_label,
                )
                if converted:
                    converted = _normalize_image_description_markdown(converted)
                    if not converted:
                        continue
                    core_output = _format_block_output(
                        title, description, converted, date=date
                    )
                    break
            if core_output is None:
                fallback_url = image_urls[0] if image_urls else ""
                fallback = f"[Image: {title or block.get('id')}]"
                if fallback_url:
                    fallback += f"\nURL: {fallback_url}"
                core_output = _format_media_fallback_output(
                    title, description, fallback, date=date
                )
            if core_output and block_id and updated_at:
                store_block_render(
                    block_id,
                    updated_at,
                    core_output,
                    render_variant=render_variant,
                )
        elif core_output is None:
            image_urls = _block_image_urls(block)
            fallback_url = image_urls[0] if image_urls else ""
            fallback = f"[Image: {title or block.get('id')}]"
            if fallback_url:
                fallback += f"\nURL: {fallback_url}"
            core_output = _format_media_fallback_output(
                title, description, fallback, date=date
            )

    elif block_type == "Link":
        refresh_link = get_refresh_images() or get_refresh_media()
        if block_id and updated_at and not refresh_link:
            cached = get_cached_block_render(
                block_id, updated_at, render_variant=render_variant
            )
            if cached is not None:
                core_output = cached
        if core_output is not None:
            connections_section = _block_connections_output(
                block,
                include_connections=bool(include_connections),
                max_items=connections_max_items,
            )
            comments_section = _block_comments_output(
                block, include_comments=bool(include_comments)
            )
            return _append_block_detail_sections(
                core_output, connections_section, comments_section
            )

        source = block.get("source") or {}
        source_url = source.get("url") or ""
        content = _extract_markdown_like_text(block.get("content") or "")
        link_parts = []
        link_header = source_url or content
        if link_header:
            link_parts.append(link_header)
        if include_link_image_descriptions and include_media_descriptions:
            link_image_description = _render_link_image_description(
                block,
                block_id=block_id,
                updated_at=updated_at,
                title=title,
            )
            if link_image_description:
                link_parts.append(link_image_description)
        core_output = _format_block_output(
            title, description, "\n\n".join(link_parts), date=date
        )
        if core_output and block_id and updated_at:
            store_block_render(
                block_id,
                updated_at,
                core_output,
                render_variant=render_variant,
            )

    elif block_type == "Attachment":
        attachment = block.get("attachment") or {}
        att_url = attachment.get("url") or ""
        filename = attachment.get("filename") or ""
        content_type = attachment.get("content_type") or ""
        extension = attachment.get("file_extension") or ""
        attachment_media_kind = _attachment_media_kind(
            filename=filename,
            extension=extension,
            content_type=content_type,
        )
        should_skip_pdf_content = (
            attachment_media_kind == "pdf" and not include_pdf_content
        )
        should_skip_media_description = (
            attachment_media_kind in {"image", "video", "audio"}
            and not include_media_descriptions
        )
        uses_pdf_preview_description = (
            attachment_media_kind == "pdf"
            and bool(include_media_descriptions)
            and should_skip_pdf_content
        )
        refresh_attachment = _should_refresh_attachment_media(
            filename=filename,
            extension=extension,
            content_type=content_type,
        ) or (
            uses_pdf_preview_description
            and (get_refresh_images() or get_refresh_media())
        )
        if (
            block_id
            and updated_at
            and not refresh_attachment
            and (
                uses_pdf_preview_description
                or (not should_skip_pdf_content and not should_skip_media_description)
            )
        ):
            cached = get_cached_block_render(
                block_id, updated_at, render_variant=render_variant
            )
            if cached is not None:
                core_output = cached
        if core_output is None:
            if (
                att_url
                and not should_skip_pdf_content
                and not should_skip_media_description
            ):
                suffix = f".{extension}" if extension else Path(filename).suffix or ""
                media_cache_identity = (
                    f"arena:block:{block_id}:{updated_at}:attachment:{att_url}"
                    if block_id and updated_at
                    else att_url
                )
                send_label = f"attachment:{block_id or 'unknown'}:{(filename or title or 'untitled')[:80]}"
                converted = _render_block_binary(
                    att_url,
                    suffix,
                    media_cache_identity=media_cache_identity,
                    send_label=send_label,
                )
                if converted:
                    if attachment_media_kind == "image":
                        converted = _normalize_image_description_markdown(converted)
                    if converted:
                        att_title = title if title != filename else ""
                        core_output = _format_block_output(
                            att_title, description, converted, date=date
                        )
            if core_output is None:
                fallback_lines = _attachment_fallback_lines(
                    block,
                    attachment,
                    filename=filename,
                    content_type=content_type,
                    att_url=att_url,
                )
                if uses_pdf_preview_description:
                    preview_description = _render_attachment_preview_description(
                        block,
                        block_id=block_id,
                        updated_at=updated_at,
                        title=title,
                    )
                    if preview_description:
                        fallback_lines.extend(["", preview_description])
                att_title = title if title != filename else ""
                att_description = description
                if att_title and description.strip() == att_title.strip():
                    att_description = ""
                core_output = _format_block_output(
                    att_title, att_description, "\n".join(fallback_lines), date=date
                )
            if core_output and block_id and updated_at:
                store_block_render(
                    block_id,
                    updated_at,
                    core_output,
                    render_variant=render_variant,
                )

    elif block_type == "Embed":
        refresh_embed = (
            get_refresh_images() or get_refresh_media() or get_refresh_videos()
        )
        if block_id and updated_at and not refresh_embed:
            cached = get_cached_block_render(
                block_id, updated_at, render_variant=render_variant
            )
            if cached is not None:
                core_output = cached
        if core_output is None:
            embed = block.get("embed") or {}
            embed_url = embed.get("url") or ""
            embed_type = embed.get("type") or ""
            embed_parts = []
            if embed_type == "video" and include_media_descriptions:
                image_urls = _block_image_urls(block)
                for image_url in image_urls:
                    suffix = Path(image_url.split("?")[0]).suffix or ".jpg"
                    media_cache_identity = (
                        f"arena:block:{block_id}:{updated_at}:embed-image:{image_url}"
                        if block_id and updated_at
                        else image_url
                    )
                    send_label = f"embed-image:{block_id or 'unknown'}:{(title or 'untitled')[:80]}"
                    converted = _render_block_binary(
                        image_url,
                        suffix,
                        media_cache_identity=media_cache_identity,
                        send_label=send_label,
                    )
                    if converted:
                        converted = _normalize_image_description_markdown(converted)
                        if not converted:
                            continue
                        embed_parts.append(converted)
                        break
            if embed_url:
                embed_parts.append(embed_url)
            if embed_type:
                embed_parts.append(f"Type: {embed_type}")
            core_output = _format_block_output(
                title, description, "\n\n".join(embed_parts), date=date
            )
            if core_output and block_id and updated_at:
                store_block_render(
                    block_id,
                    updated_at,
                    core_output,
                    render_variant=render_variant,
                )

    else:
        core_output = _format_block_output(title, description, "", date=date)

    connections_section = _block_connections_output(
        block,
        include_connections=bool(include_connections),
        max_items=connections_max_items,
    )
    comments_section = _block_comments_output(
        block, include_comments=bool(include_comments)
    )
    return _append_block_detail_sections(core_output, connections_section, comments_section)


def _render_channel_stub(item: dict) -> str:
    ch_title = item.get("title") or item.get("slug") or "Untitled"
    ch_owner = (item.get("owner") or {}).get("name") or ""
    ch_slug = item.get("slug") or ""
    ch_counts = item.get("counts") or {}
    ch_blocks = ch_counts.get("contents")
    if ch_blocks is None or ch_blocks == "":
        ch_blocks = "?"
    description = _extract_markdown_like_text(item.get("description") or "").strip()
    created_at = _format_metadata_timestamp(item.get("created_at"))
    updated_at = _format_metadata_timestamp(item.get("updated_at"))
    parts = [f"[Channel: {ch_title}]"]
    if ch_owner:
        parts.append(f"Owner: {ch_owner}")
    parts.append(f"Blocks: {ch_blocks}")
    if created_at:
        parts.append(f"Started: {created_at}")
    if updated_at:
        parts.append(f"Modified: {updated_at}")
    if ch_slug:
        parts.append(f"https://www.are.na/channel/{ch_slug}")
    if description:
        parts.append(_format_channel_description(description))
    return "\n".join(parts)


def materialize_arena_attachment_target(
    target: str,
    parsed: dict[str, Any],
    *,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
    cache_only: bool = False,
    block: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    block_id = parsed["block_id"]
    if block is None:
        block = _fetch_block(block_id)
    attachment = _select_attachment_for_target(block, parsed)
    att_url = str(attachment.get("url") or "")
    if not att_url:
        raise ValueError(f"Are.na attachment target {target} did not include a URL")

    filename = _attachment_filename(attachment)
    content_type = _attachment_content_type(attachment)
    extension = _attachment_extension(attachment, filename)
    suffix = f".{extension}" if extension else Path(urlparse(att_url).path).suffix
    media_cache_identity = _attachment_media_cache_identity(block, attachment)

    from .cache import get_cached_media_bytes, store_media_bytes
    from contextualize.runtime import get_refresh_media
    from ..shared.media import download_cached_media_to_temp

    content: bytes | None = None
    if cache_only:
        content = get_cached_media_bytes(media_cache_identity)
    else:
        tmp = download_cached_media_to_temp(
            att_url,
            suffix=suffix,
            headers=_DOWNLOAD_HEADERS,
            cache_identity=media_cache_identity,
            get_cached_media_bytes=(
                get_cached_media_bytes if use_cache else lambda _key: None
            ),
            store_media_bytes=(
                store_media_bytes if use_cache else lambda _key, _content: None
            ),
            refresh_cache=refresh_cache or get_refresh_media(),
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
            "label": filename,
            "filename": filename,
            "content": content,
            "content_type": content_type or None,
            "metadata": {
                "provider": "arena",
                "kind": "attachment",
                "sourceBlockUrl": f"https://www.are.na/block/{block_id}",
                "attachmentUrl": att_url,
                "attachmentName": filename,
                "blockId": block_id,
                "bytes": len(content),
            },
        }
    ]


def resolve_arena_attachment_target(
    target: str,
    parsed: dict[str, Any],
    *,
    settings_key: tuple[Any, ...],
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
    plugin_overrides: dict[str, Any] | None = None,
    block: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    from contextualize.references.url import URLReference

    block_id = parsed["block_id"]
    if block is None:
        block = _fetch_block(block_id)
    attachment = _select_attachment_for_target(block, parsed)
    att_url = str(attachment.get("url") or "")
    if not att_url:
        raise ValueError(f"Are.na attachment target {target} did not include a URL")

    filename = _attachment_filename(attachment)
    attachment_key = str(
        parsed.get("attachment_id") or parsed.get("attachment_name") or filename
    )
    suffix = Path(filename).suffix or ".txt"
    reference = URLReference(
        att_url,
        format="raw",
        label="name",
        filename_override=filename,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
        plugin_overrides=plugin_overrides,
    )
    created_at = block.get("connected_at") or block.get("created_at")
    updated_at = block.get("updated_at") or created_at
    return [
        {
            "source": target,
            "label": f"arena:block/{block_id}/attachment/{filename}",
            "content": reference.read(),
            "metadata": {
                "trace_path": target,
                "provider": "arena",
                "source_ref": "are.na",
                "source_path": f"{block_id}/attachment/{attachment_key}",
                "context_subpath": f"arena-block-{block_id}/attachment-{attachment_key}{suffix}",
                "source_created": created_at,
                "source_modified": updated_at,
                "kind": "attachment",
                "settings_key": settings_key,
                "blockId": block_id,
                "attachmentName": filename,
                "attachmentUrl": att_url,
            },
        }
    ]


def _entity_display(entity: dict) -> str:
    name = _entity_name(entity)
    slug = entity.get("slug")
    if isinstance(slug, str) and slug.strip():
        return f"{name} (@{slug.strip()})"
    return name


def _entity_name(entity: dict) -> str:
    name = entity.get("name") or entity.get("username") or entity.get("slug")
    if not isinstance(name, str) or not name.strip():
        name = "Unknown"
    return name.strip()


def _append_count_line(parts: list[str], label: str, counts: dict, key: str) -> None:
    value = counts.get(key)
    if value is not None and value != "":
        parts.append(f"{label}: {value}")


def _group_summary_line(group: dict) -> str:
    parts = [_entity_display(group)]
    owner = group.get("user") or group.get("owner")
    if isinstance(owner, dict) and owner:
        parts.append(f"Owner: {_entity_display(owner)}")
    counts = group.get("counts") or {}
    if isinstance(counts, dict):
        channels = counts.get("channels")
        users = counts.get("users")
        if channels is not None and channels != "":
            parts.append(f"Channels: {channels}")
        if users is not None and users != "":
            parts.append(f"Members: {users}")
    return "- " + ", ".join(parts)


def render_owner_profile(
    kind: Literal["user", "group"],
    profile: dict,
    *,
    user_groups: list[dict] | None = None,
) -> str:
    name = profile.get("name") or profile.get("slug") or "Untitled"
    slug = profile.get("slug") or ""
    bio = _extract_markdown_like_text(profile.get("bio") or "").strip()
    created_at = _format_metadata_timestamp(profile.get("created_at"))
    updated_at = _format_metadata_timestamp(profile.get("updated_at"))
    counts = profile.get("counts") or {}
    if not isinstance(counts, dict):
        counts = {}

    if kind == "user":
        parts = [f"[User: {name}]"]
        if created_at:
            parts.append(f"Joined: {created_at}")
        if updated_at:
            parts.append(f"Modified: {updated_at}")
        _append_count_line(parts, "Channels", counts, "channels")
        _append_count_line(parts, "Followers", counts, "followers")
        _append_count_line(parts, "Following", counts, "following")
    else:
        parts = [f"[Group: {name}]"]
        owner = profile.get("user") or profile.get("owner")
        if isinstance(owner, dict) and owner:
            parts.append(f"Owner: {_entity_display(owner)}")
        _append_count_line(parts, "Channels", counts, "channels")
        _append_count_line(parts, "Members", counts, "users")
        if created_at:
            parts.append(f"Created: {created_at}")
        if updated_at:
            parts.append(f"Modified: {updated_at}")

    if isinstance(slug, str) and slug.strip():
        parts.append(f"https://www.are.na/{slug.strip()}")
    if bio:
        parts.append(_format_info_section(bio))
    if kind == "user" and user_groups:
        parts.append("Groups:")
        parts.extend(_group_summary_line(group) for group in user_groups)
    return "\n".join(parts)


def _is_missing_metadata_value(value: object) -> bool:
    return value is None or value == "" or value == {} or value == []


def _merge_channel_metadata(item: dict, metadata: object) -> dict:
    block = dict(item)
    if not isinstance(metadata, dict):
        return block
    for key, value in metadata.items():
        if key.startswith("_nested_"):
            continue
        if _is_missing_metadata_value(block.get(key)):
            block[key] = value
    return block


def _channel_context_from_metadata(metadata: dict, fallback_slug: str = "") -> dict:
    context: dict[str, Any] = {}
    channel_id = metadata.get("id")
    if isinstance(channel_id, int):
        context["id"] = channel_id
    slug = metadata.get("slug") or fallback_slug
    if isinstance(slug, str) and slug:
        context["slug"] = slug
    title = metadata.get("title")
    if isinstance(title, str) and title:
        context["title"] = title
    owner = metadata.get("owner") or metadata.get("user")
    if isinstance(owner, dict) and owner:
        context["owner"] = owner
    return context


def _attach_channel_context(block: dict, source_context: dict | None) -> dict:
    if source_context:
        block["_contextualize_channel_context"] = source_context
    connection = block.get("connection")
    if isinstance(connection, dict):
        connected_at = connection.get("connected_at")
        connected_by = connection.get("connected_by")
        if connected_at and not block.get("connected_at"):
            block["connected_at"] = connected_at
        if isinstance(connected_by, dict) and not block.get("connected_by"):
            block["connected_by"] = connected_by
    return block


def _block_label(block: dict, channel_slug: str, channel_path: str = "") -> str:
    block_id = block.get("id", "unknown")
    prefix = channel_path or channel_slug or "are.na/block"
    return f"{prefix}/{block_id}"


def _flatten_channel_blocks(
    contents: list[dict],
    channel_slug: str,
    channel_path: str = "",
    channel_slug_path: tuple[str, ...] | None = None,
    channel_context: dict | None = None,
) -> list[tuple[str, dict]]:
    result: list[tuple[str, dict]] = []
    path = channel_path or channel_slug
    current_slug_path = channel_slug_path
    if current_slug_path is None:
        current_slug_path = (channel_slug,) if channel_slug else ()

    for item in contents:
        if item.get("base_type") == "Channel" or item.get("type") == "Channel":
            nested_contents = item.get("_nested_contents")
            if nested_contents is not None:
                nested_meta = item.get("_nested_metadata", item)
                nested_slug = nested_meta.get("slug") or item.get("slug") or ""
                nested_title = (
                    nested_meta.get("title") or item.get("title") or "channel"
                )
                safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", nested_title)[:60].strip(
                    "-"
                )
                sub_path = f"{path}/{safe_name or nested_slug}"
                nested_slug_path = current_slug_path + (
                    (nested_slug,) if nested_slug else ()
                )
                nested_context = _channel_context_from_metadata(
                    nested_meta, nested_slug
                )
                if item.get("_nested_sampled_by_recurse_blocks"):
                    block = _merge_channel_metadata(item, nested_meta)
                    block.pop("_nested_metadata", None)
                    block.pop("_nested_contents", None)
                    block = _attach_channel_context(block, channel_context)
                    block["_contextualize_keep_duplicate"] = True
                    if current_slug_path:
                        block["_channel_slug_path"] = list(current_slug_path)
                    result.append((path, block))
                result.extend(
                    _flatten_channel_blocks(
                        nested_contents,
                        nested_slug,
                        sub_path,
                        nested_slug_path,
                        nested_context,
                    )
                )
            else:
                block = dict(item)
                block = _attach_channel_context(block, channel_context)
                if current_slug_path:
                    block["_channel_slug_path"] = list(current_slug_path)
                result.append((path, block))
        else:
            block = dict(item)
            block = _attach_channel_context(block, channel_context)
            if current_slug_path:
                block["_channel_slug_path"] = list(current_slug_path)
            result.append((path, block))

    return result


def _block_identity(block: dict) -> str | None:
    block_id = block.get("id")
    if block_id is not None:
        return f"id:{block_id}"
    slug = block.get("slug")
    if slug:
        return f"slug:{slug}"
    return None


def _dedupe_flat_blocks(flat: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    deduped: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for path, block in flat:
        identity = _block_identity(block)
        if block.get("_contextualize_keep_duplicate"):
            deduped.append((path, block))
            continue
        if identity is not None:
            if identity in seen:
                continue
            seen.add(identity)
        deduped.append((path, block))
    return deduped


def _sort_blocks(flat: list[tuple[str, dict]], order: str) -> list[tuple[str, dict]]:
    if order == "position-asc" or order == "asc":
        return flat
    if order == "position-desc" or order == "desc":
        return list(reversed(flat))
    if order == "random":
        import random

        shuffled = list(flat)
        random.shuffle(shuffled)
        return shuffled
    use_reverse = order == "date-desc"
    return sorted(
        flat,
        key=lambda pair: _block_chrono_value(pair[1]),
        reverse=use_reverse,
    )


def _sort_channel_contents(contents: list[dict], order: str) -> list[dict]:
    if order == "position-asc" or order == "asc":
        return contents
    if order == "position-desc" or order == "desc":
        return list(reversed(contents))
    if order == "random":
        import random

        shuffled = list(contents)
        random.shuffle(shuffled)
        return shuffled
    use_reverse = order == "date-desc"
    return sorted(contents, key=_block_chrono_value, reverse=use_reverse)


def _block_chrono_value(block: dict) -> str:
    return _block_connected_at(block) or block.get("created_at") or ""


def _block_connected_at(block: dict) -> str | None:
    connected_at = block.get("connected_at")
    if isinstance(connected_at, str) and connected_at:
        return connected_at
    connection = block.get("connection")
    if isinstance(connection, dict):
        connected_at = connection.get("connected_at")
        if isinstance(connected_at, str) and connected_at:
            return connected_at
    return None


def _parse_block_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return _parse_iso_datetime(value)


def _passes_block_time_window(block: dict, settings: ArenaSettings) -> bool:
    timestamp: datetime | None = None
    after: datetime | None = None
    before: datetime | None = None

    if settings.connected_after is not None or settings.connected_before is not None:
        timestamp = _parse_block_timestamp(_block_connected_at(block))
        after = settings.connected_after
        before = settings.connected_before
    elif settings.created_after is not None or settings.created_before is not None:
        timestamp = _parse_block_timestamp(block.get("created_at"))
        after = settings.created_after
        before = settings.created_before

    if timestamp is None:
        return True
    if after is not None and timestamp < after:
        return False
    if before is not None and timestamp > before:
        return False
    return True


def _filter_flat_blocks(
    flat: list[tuple[str, dict]], settings: ArenaSettings
) -> list[tuple[str, dict]]:
    return [
        (path, block)
        for path, block in flat
        if _passes_block_time_window(block, settings)
    ]


def _limit_flat_blocks(
    flat: list[tuple[str, dict]], settings: ArenaSettings
) -> list[tuple[str, dict]]:
    max_blocks = settings.max_blocks_per_channel
    if max_blocks is None:
        return flat
    limited: list[tuple[str, dict]] = []
    counted = 0
    for item in flat:
        _path, block = item
        if block.get("_contextualize_keep_duplicate"):
            limited.append(item)
            continue
        if counted >= max_blocks:
            continue
        limited.append(item)
        counted += 1
    return limited


def resolve_channel(
    slug: str,
    *,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
    settings: ArenaSettings | None = None,
) -> tuple[dict, list[tuple[str, dict]]]:
    if settings is None:
        settings = _arena_settings_from_env()
    max_depth = settings.max_depth
    recurse_users = settings.recurse_users
    sort_order = settings.sort_order
    max_blocks_per_channel = settings.max_blocks_per_channel
    recurse_blocks = settings.recurse_blocks
    ru_key = ",".join(sorted(recurse_users)) if recurse_users else "all"
    mb_key = (
        str(max_blocks_per_channel) if max_blocks_per_channel is not None else "all"
    )
    rb_key = recurse_blocks.cache_key() if recurse_blocks is not None else "none"
    auth_key = _auth_cache_partition()
    exclude_key = settings.exclude_channels.cache_key()
    ca_key = (
        settings.connected_after.isoformat() if settings.connected_after else "none"
    )
    cb_key = (
        settings.connected_before.isoformat() if settings.connected_before else "none"
    )
    cra_key = settings.created_after.isoformat() if settings.created_after else "none"
    crb_key = settings.created_before.isoformat() if settings.created_before else "none"
    cache_key = (
        f"v4:{slug}:d={max_depth}:u={ru_key}:s={sort_order}:m={mb_key}:rb={rb_key}:"
        f"ca={ca_key}:cb={cb_key}:cra={cra_key}:crb={crb_key}:"
        f"auth={auth_key}:exclude={exclude_key}"
    )

    if use_cache and not refresh_cache:
        from .cache import get_cached_channel

        cached = get_cached_channel(cache_key, cache_ttl)
        if cached is not None:
            data = json.loads(cached)
            metadata = data["metadata"]
            flat_unsorted = [(path, block) for path, block in data["blocks"]]
            flat = _sort_blocks(flat_unsorted, sort_order)
            flat = _filter_flat_blocks(flat, settings)
            flat = _limit_flat_blocks(flat, settings)
            channel_title = metadata.get("title") or slug
            _log(f"  using cached channel: {channel_title}")
            record_progress(
                "arena",
                "channel",
                "cache_hit",
                target=str(channel_title),
                count=len(flat),
            )
            return metadata, flat
        record_progress("arena", "channel", "cache_miss", target=slug)

    metadata, contents = _fetch_all_channel_contents(
        slug,
        max_depth=max_depth,
        sort_order=sort_order,
        max_blocks_per_channel=max_blocks_per_channel,
        recurse_blocks=recurse_blocks,
        _time_window_settings=settings,
        _recurse_users=recurse_users,
    )
    flat_unsorted = _flatten_channel_blocks(
        contents,
        slug,
        channel_context=_channel_context_from_metadata(metadata, slug),
    )
    flat = _sort_blocks(flat_unsorted, sort_order)
    flat = _filter_flat_blocks(flat, settings)
    flat = _limit_flat_blocks(flat, settings)

    if use_cache:
        from .cache import store_channel

        data = json.dumps(
            {"metadata": metadata, "blocks": flat_unsorted}, ensure_ascii=False
        )
        block_count = len([b for _, b in flat_unsorted if b.get("type") != "Channel"])
        store_channel(cache_key, data, block_count)
    record_progress(
        "arena",
        "channel",
        "processed",
        target=str(metadata.get("title") or slug),
        count=len(flat),
    )

    return metadata, flat


@dataclass
class ArenaReference:
    url: str
    block: dict
    channel_path: str = ""
    format: str = "md"
    label: str = "relative"
    token_target: str = "cl100k_base"
    include_token_count: bool = False
    label_suffix: str | None = None
    inject: bool = False
    depth: int = 5
    trace_collector: list = None
    include_descriptions: bool | None = None
    include_comments: bool | None = None
    include_connections: bool | None = None
    connections_max_items: int | None = None
    include_link_image_descriptions: bool | None = None
    include_pdf_content: bool | None = None
    include_media_descriptions: bool | None = None

    def __post_init__(self) -> None:
        self.file_content = ""
        self.original_file_content = ""
        self.output = self._get_contents()

    @property
    def path(self) -> str:
        return self.url

    @property
    def trace_path(self) -> str:
        block_type = self.block.get("type", "")
        is_channel = block_type == "Channel" or self.block.get("base_type") == "Channel"
        if is_channel:
            prefix = self.channel_path or self.block.get("slug") or "are.na/channel"
            block_id = self.block.get("id", "unknown")
            return f"{prefix}/{block_id}"
        return _block_label(self.block, "are.na/block", self.channel_path)

    def read(self) -> str:
        return self.original_file_content

    def exists(self) -> bool:
        return True

    def token_count(self, encoding: str = "cl100k_base") -> int:
        return count_tokens(self.original_file_content, target=encoding)["count"]

    def get_label(self) -> str:
        block_type = self.block.get("type", "")
        is_channel = block_type == "Channel" or self.block.get("base_type") == "Channel"

        if is_channel:
            slug = self.block.get("slug") or ""
            name = self.block.get("title") or slug or "channel"
        else:
            name = _block_label(self.block, "", self.channel_path)

        if self.label == "relative":
            return name
        if self.label == "name":
            return name.rsplit("/", 1)[-1] if "/" in name else name
        if self.label == "ext":
            return ""
        return self.label

    def _get_contents(self) -> str:
        block_type = self.block.get("type", "")
        is_channel = block_type == "Channel" or self.block.get("base_type") == "Channel"

        if is_channel:
            text = _render_channel_stub(self.block)
        else:
            if block_type in ("Image", "Attachment"):
                block_title = self.block.get("title") or f"block-{self.block.get('id')}"
                _log(f"  resolving {block_type.lower()}: {block_title[:60]}")
            text = (
                _render_block(
                    self.block,
                    include_descriptions=self.include_descriptions,
                    include_comments=self.include_comments,
                    include_connections=self.include_connections,
                    connections_max_items=self.connections_max_items,
                    include_link_image_descriptions=self.include_link_image_descriptions,
                    include_pdf_content=self.include_pdf_content,
                    include_media_descriptions=self.include_media_descriptions,
                )
                or ""
            )

        self.original_file_content = text
        self.file_content = text

        if self.inject and text:
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
