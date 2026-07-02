from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from cx_plugins.providers.wikipedia import cache as wikipedia_cache
from cx_plugins.providers.wikipedia import plugin as wikipedia_plugin
from cx_plugins.providers.wikipedia import wikipedia


class _FakeWikipediaResponse:
    def __init__(
        self,
        status_code: int,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")


class _FakeWikipediaRequests:
    class exceptions:
        class RequestException(Exception):
            pass

    def __init__(self, responses: list[_FakeWikipediaResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> _FakeWikipediaResponse:
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


def _install_fake_wikipedia_clock(monkeypatch) -> list[float]:
    now = [1_000.0]
    sleeps: list[float] = []

    def _sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(wikipedia.time, "time", lambda: now[0])
    monkeypatch.setattr(wikipedia.time, "sleep", _sleep)
    return sleeps


def _use_wikipedia_rate_limiter(monkeypatch, tmp_path: Path):
    limiter = wikipedia._WikimediaApiRateLimiter(  # noqa: SLF001
        tmp_path / "wikimedia-rate-limit.sqlite"
    )
    monkeypatch.setattr(wikipedia, "_WIKIMEDIA_API_RATE_LIMITER", limiter)
    return limiter


def _set_fast_wikipedia_rate_env(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXTUALIZE_WIKIMEDIA_REQUESTS_PER_MINUTE", "200")
    monkeypatch.setenv("CONTEXTUALIZE_WIKIMEDIA_RATE_LIMIT_SAFETY", "1")
    monkeypatch.setenv("CONTEXTUALIZE_WIKIMEDIA_MIN_REQUEST_DELAY_SECONDS", "0")
    monkeypatch.setenv("CONTEXTUALIZE_WIKIMEDIA_API_MAX_ATTEMPTS", "2")


def test_parse_wikipedia_target_supports_urls_and_schemes() -> None:
    parsed_url = wikipedia.parse_wikipedia_target(
        "https://en.wikipedia.org/wiki/Alan_Turing?oldid=12345"
    )
    assert parsed_url is not None
    assert parsed_url.language == "en"
    assert parsed_url.title == "Alan Turing"
    assert parsed_url.revision_id == 12345

    parsed_scheme = wikipedia.parse_wikipedia_target("wikipedia://fr/Jean_Piaget")
    assert parsed_scheme is not None
    assert parsed_scheme.language == "fr"
    assert parsed_scheme.title == "Jean Piaget"

    parsed_short = wikipedia.parse_wikipedia_target("wiki:es/Teor%C3%ADa_de_juegos")
    assert parsed_short is not None
    assert parsed_short.language == "es"
    assert parsed_short.title.lower().startswith("teor")
    assert parsed_short.title.lower().endswith("juegos")


def test_wikipedia_request_headers_use_contactable_user_agent_and_auth(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "CONTEXTUALIZE_WIKIMEDIA_USER_AGENT",
        "cx-test/1.0 (https://example.com/contact)",
    )
    monkeypatch.setenv("CONTEXTUALIZE_WIKIMEDIA_ACCESS_TOKEN", "test-token")

    headers = wikipedia._request_headers({"Accept": "application/json"})  # noqa: SLF001

    assert headers["User-Agent"] == "cx-test/1.0 (https://example.com/contact)"
    assert headers["Authorization"] == "Bearer test-token"
    assert headers["Accept"] == "application/json"


def test_http_get_paces_wikimedia_requests(monkeypatch, tmp_path: Path) -> None:
    limiter = _use_wikipedia_rate_limiter(monkeypatch, tmp_path)
    sleeps = _install_fake_wikipedia_clock(monkeypatch)
    _set_fast_wikipedia_rate_env(monkeypatch)
    fake_requests = _FakeWikipediaRequests(
        [
            _FakeWikipediaResponse(200, {"ok": 1}),
            _FakeWikipediaResponse(200, {"ok": 2}),
        ]
    )
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    try:
        assert (
            wikipedia._http_get(  # noqa: SLF001
                "https://en.wikipedia.org/w/api.php",
                timeout=1,
            ).json()
            == {"ok": 1}
        )
        assert (
            wikipedia._http_get(  # noqa: SLF001
                "https://en.wikipedia.org/w/api.php",
                timeout=1,
            ).json()
            == {"ok": 2}
        )
    finally:
        limiter.reset()

    assert len(fake_requests.calls) == 2
    assert fake_requests.calls[0]["headers"]["User-Agent"].startswith(
        "contextualize-wikipedia/"
    )
    assert sleeps == [pytest.approx(60.0 / 200.0)]


def test_http_get_defers_after_retry_after_429(
    monkeypatch,
    tmp_path: Path,
) -> None:
    limiter = _use_wikipedia_rate_limiter(monkeypatch, tmp_path)
    sleeps = _install_fake_wikipedia_clock(monkeypatch)
    _set_fast_wikipedia_rate_env(monkeypatch)
    fake_requests = _FakeWikipediaRequests(
        [
            _FakeWikipediaResponse(
                429,
                {"error": "too many requests"},
                {"Retry-After": "7"},
            ),
            _FakeWikipediaResponse(200, {"ok": True}),
        ]
    )
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    try:
        assert (
            wikipedia._http_get(  # noqa: SLF001
                "https://en.wikipedia.org/w/api.php",
                timeout=1,
            ).json()
            == {"ok": True}
        )
    finally:
        limiter.reset()

    assert len(fake_requests.calls) == 2
    assert sleeps == [pytest.approx(7.0)]


def test_wikipedia_rate_limiter_shares_state_between_instances(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sleeps = _install_fake_wikipedia_clock(monkeypatch)
    _set_fast_wikipedia_rate_env(monkeypatch)
    store_path = tmp_path / "wikimedia-rate-limit.sqlite"
    first = wikipedia._WikimediaApiRateLimiter(store_path)  # noqa: SLF001
    second = wikipedia._WikimediaApiRateLimiter(store_path)  # noqa: SLF001

    first.wait_for_slot(key="wikimedia:guest", authenticated=False)
    second.wait_for_slot(key="wikimedia:guest", authenticated=False)

    assert sleeps == [pytest.approx(60.0 / 200.0)]


def test_can_resolve_uses_default_lang_override() -> None:
    context = {"overrides": {"wikipedia": {"default-lang": "de"}}}
    assert wikipedia_plugin.can_resolve("wiki:Konrad_Zuse", context) is True

    classified = wikipedia_plugin.classify_target("wiki:Konrad_Zuse", context)
    assert classified is not None
    assert classified["provider"] == "wikipedia"
    assert classified["kind"] == "article"


def test_extract_article_data_keeps_inline_wiki_links_and_strips_citations() -> None:
    html = (
        "<p>See <a href='/wiki/Ficciones'>Ficciones</a> and "
        "<a href='https://example.com'>Example</a>"
        "<sup class='reference'><a href='#cite_note-1'>[1]</a></sup>.</p>"
    )
    extracted = wikipedia.extract_article_data(html)
    assert extracted.sections
    content = extracted.sections[0].content
    assert "[[Ficciones]]" in content
    assert "[https://example.com Example]" in content
    assert "[1]" not in content
    assert "citation needed" not in content.lower()


def test_resolve_wikipedia_article_renders_frontmatter_and_media_tags(
    monkeypatch,
) -> None:
    target = "https://en.wikipedia.org/wiki/Alan_Turing"

    monkeypatch.setattr(
        wikipedia,
        "_resolve_parse_payload",
        lambda _parsed, timeout_seconds: {
            "title": "Alan Turing",
            "displaytitle": "Alan <i>Turing</i>",
            "text": {"*": "<p>unused</p>"},
            "categories": [{"*": "Mathematicians"}],
        },
    )
    monkeypatch.setattr(
        wikipedia,
        "_resolve_summary",
        lambda _parsed, timeout_seconds: wikipedia.WikipediaSummary(
            description="Argentine writer (1899-1986)",
            extract="Fallback intro.",
        ),
    )
    monkeypatch.setattr(
        wikipedia,
        "extract_article_data",
        lambda _html: wikipedia._ExtractedArticle(  # noqa: SLF001
            sections=(
                wikipedia.WikipediaSection(
                    index=0,
                    level=1,
                    title="Introduction",
                    content="Intro body.",
                ),
                wikipedia.WikipediaSection(
                    index=1,
                    level=2,
                    title="Works",
                    content="See [[Labyrinth]] and [https://example.com Example].",
                ),
                wikipedia.WikipediaSection(
                    index=2,
                    level=2,
                    title="External links",
                    content="Should be skipped.",
                ),
            ),
            references=("Reference A",),
            external_links=(("External", "https://example.com"),),
        ),
    )
    monkeypatch.setattr(
        wikipedia,
        "_resolve_media_list",
        lambda _parsed, timeout_seconds: (
            wikipedia.WikipediaMedia(
                kind="image",
                url="https://upload.wikimedia.org/image.jpg",
                filename="Jorge_Luis_Borges.jpg",
                caption="Portrait",
                description=None,
                width=640,
                height=480,
                section_index=0,
            ),
            wikipedia.WikipediaMedia(
                kind="video",
                url="https://upload.wikimedia.org/interview.ogv",
                filename="Interview.ogv",
                caption="Interview clip",
                description=None,
                width=320,
                height=200,
                section_index=1,
            ),
        ),
    )
    monkeypatch.setattr(
        wikipedia,
        "_describe_media_items",
        lambda media, enabled: (
            wikipedia.WikipediaMedia(
                kind="image",
                url="https://upload.wikimedia.org/image.jpg",
                filename="Jorge_Luis_Borges.jpg",
                caption="Portrait",
                description="LLM alt text.",
                width=640,
                height=480,
                section_index=0,
            ),
            wikipedia.WikipediaMedia(
                kind="video",
                url="https://upload.wikimedia.org/interview.ogv",
                filename="Interview.ogv",
                caption="Interview clip",
                description=None,
                width=320,
                height=200,
                section_index=1,
            ),
        ),
    )

    document = wikipedia.resolve_wikipedia_article(
        target,
        settings=wikipedia.build_wikipedia_settings(None),
        use_cache=False,
        cache_ttl=None,
        refresh_cache=False,
    )

    rendered = document.rendered
    assert rendered.startswith("---\nurl: https://en.wikipedia.org/wiki/Alan_Turing\n")
    assert "description: Argentine writer (1899-1986)" in rendered
    assert "\n# Alan Turing\n" in rendered
    assert "- URL:" not in rendered
    assert "- Language:" not in rendered
    assert "## Works" in rendered
    assert "# Media" not in rendered
    assert "# External Links" not in rendered
    assert "<attachment" not in rendered
    assert (
        '<image filename="Jorge_Luis_Borges.jpg" caption="Portrait">\n'
        "LLM alt text.\n"
        "</image>"
    ) in rendered
    assert '<video filename="Interview.ogv" caption="Interview clip" />' in rendered
    assert "[Example](https://example.com)" not in rendered
    assert "[[Labyrinth]]" in rendered
    assert "# References" in rendered
    assert "# Categories" in rendered


def test_resolve_wikipedia_article_reuses_cached_document(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(wikipedia_cache, "WIKIPEDIA_CACHE_ROOT", tmp_path)
    monkeypatch.setattr(wikipedia_cache, "DOCUMENT_CACHE_ROOT", tmp_path / "documents")
    calls = {"parse": 0}

    def _fake_parse_payload(_parsed, timeout_seconds):
        calls["parse"] += 1
        return {
            "title": "Alan Turing",
            "displaytitle": "Alan Turing",
            "text": {"*": "<p>unused</p>"},
            "categories": [],
        }

    monkeypatch.setattr(wikipedia, "_resolve_parse_payload", _fake_parse_payload)
    monkeypatch.setattr(
        wikipedia,
        "_resolve_summary",
        lambda _parsed, timeout_seconds: wikipedia.WikipediaSummary(
            description=None,
            extract="Fallback intro.",
        ),
    )
    monkeypatch.setattr(
        wikipedia,
        "extract_article_data",
        lambda _html: wikipedia._ExtractedArticle(  # noqa: SLF001
            sections=(
                wikipedia.WikipediaSection(
                    index=0,
                    level=1,
                    title="Introduction",
                    content="Intro body.",
                ),
            ),
            references=(),
            external_links=(),
        ),
    )

    settings = wikipedia.build_wikipedia_settings({"include_media": False})
    first = wikipedia.resolve_wikipedia_article(
        "https://en.wikipedia.org/wiki/Alan_Turing",
        settings=settings,
        use_cache=True,
        cache_ttl=None,
        refresh_cache=False,
    )
    monkeypatch.setattr(
        wikipedia,
        "_resolve_parse_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cache was not used")
        ),
    )

    second = wikipedia.resolve_wikipedia_article(
        "https://en.wikipedia.org/wiki/Alan_Turing",
        settings=settings,
        use_cache=True,
        cache_ttl=None,
        refresh_cache=False,
    )

    assert first == second
    assert calls["parse"] == 1


def test_resolve_wikipedia_article_self_closes_media_when_descriptions_disabled(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        wikipedia,
        "_resolve_parse_payload",
        lambda _parsed, timeout_seconds: {
            "title": "Alan Turing",
            "displaytitle": "Alan Turing",
            "text": {"*": "<p>unused</p>"},
            "categories": [],
        },
    )
    monkeypatch.setattr(
        wikipedia,
        "_resolve_summary",
        lambda _parsed, timeout_seconds: wikipedia.WikipediaSummary(
            description=None,
            extract=None,
        ),
    )
    monkeypatch.setattr(
        wikipedia,
        "extract_article_data",
        lambda _html: wikipedia._ExtractedArticle(  # noqa: SLF001
            sections=(
                wikipedia.WikipediaSection(
                    index=0,
                    level=1,
                    title="Introduction",
                    content="Intro body.",
                ),
            ),
            references=(),
            external_links=(),
        ),
    )
    monkeypatch.setattr(
        wikipedia,
        "_resolve_media_list",
        lambda _parsed, timeout_seconds: (
            wikipedia.WikipediaMedia(
                kind="image",
                url="https://upload.wikimedia.org/image.jpg",
                filename="Jorge_Luis_Borges.jpg",
                caption="Borges in 1967",
                description=None,
                width=640,
                height=480,
                section_index=0,
            ),
        ),
    )

    document = wikipedia.resolve_wikipedia_article(
        "https://en.wikipedia.org/wiki/Alan_Turing",
        settings=wikipedia.build_wikipedia_settings(
            {"include_media_descriptions": False}
        ),
        use_cache=False,
        cache_ttl=None,
        refresh_cache=False,
    )

    rendered = document.rendered
    assert (
        '<image filename="Jorge_Luis_Borges.jpg" caption="Borges in 1967" />'
    ) in rendered
    assert "</image>" not in rendered


def test_runtime_overrides_parse_manifest_style_aliases() -> None:
    parsed = wikipedia_plugin._wikipedia_runtime_overrides(  # noqa: SLF001
        {
            "default-lang": "fr",
            "include-references": False,
            "media": {
                "enabled": True,
                "describe": False,
            },
        }
    )
    assert parsed is not None
    assert parsed["default_lang"] == "fr"
    assert parsed["include_references"] is False
    assert parsed["include_media"] is True
    assert parsed["include_media_descriptions"] is False


def test_plugin_resolve_emits_metadata_and_dedupe(monkeypatch) -> None:
    monkeypatch.setattr(
        wikipedia,
        "resolve_wikipedia_article",
        lambda *_args, **_kwargs: wikipedia.WikipediaResolvedDocument(
            label="wikipedia/en/Alan_Turing",
            rendered="# Alan Turing",
            prose="Alan Turing was a mathematician.",
            source_ref="en.wikipedia.org",
            source_path="en/Alan_Turing",
            context_subpath="wikipedia/en/Alan_Turing.md",
            kind="article",
            canonical_id="en:Alan_Turing",
        ),
    )

    docs = wikipedia_plugin.resolve(
        "https://en.wikipedia.org/wiki/Alan_Turing",
        {"overrides": {"wikipedia": {"default-lang": "en"}}},
    )

    assert len(docs) == 1
    assert docs[0]["prose"] == "Alan Turing was a mathematician."
    metadata = docs[0]["metadata"]
    assert metadata["provider"] == "wikipedia"
    assert metadata["source_ref"] == "en.wikipedia.org"
    assert metadata["source_path"] == "en/Alan_Turing"
    assert metadata["context_subpath"] == "wikipedia/en/Alan_Turing.md"
    assert metadata["kind"] == "article"
    assert "wikipedia-article:en:Alan_Turing" in metadata["hydrate_dedupe"]["key"]
