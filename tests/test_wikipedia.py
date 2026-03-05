from __future__ import annotations

from cx_plugins.providers.wikipedia import plugin as wikipedia_plugin
from cx_plugins.providers.wikipedia import wikipedia


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
        use_cache=True,
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
        use_cache=True,
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
    metadata = docs[0]["metadata"]
    assert metadata["provider"] == "wikipedia"
    assert metadata["source_ref"] == "en.wikipedia.org"
    assert metadata["source_path"] == "en/Alan_Turing"
    assert metadata["context_subpath"] == "wikipedia/en/Alan_Turing.md"
    assert metadata["kind"] == "article"
    assert "wikipedia-article:en:Alan_Turing" in metadata["hydrate_dedupe"]["key"]
