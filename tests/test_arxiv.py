from __future__ import annotations

from datetime import timedelta
import types

from cx_plugins.providers.arxiv import arxiv
from cx_plugins.providers.arxiv import plugin as arxiv_plugin

_BASE_ID = "1111.00001"
_VERSIONED_ID = f"{_BASE_ID}v2"
_UPDATED_ID = f"{_BASE_ID}v4"


def _api_feed_xml(*, identifier: str = f"{_BASE_ID}v3") -> str:
    return f"""<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns='http://www.w3.org/2005/Atom' xmlns:arxiv='http://arxiv.org/schemas/atom'>
  <entry>
    <id>https://arxiv.org/abs/{identifier}</id>
    <updated>2024-01-02T00:00:00Z</updated>
    <published>2023-06-06T00:00:00Z</published>
    <title>Synthetic Title</title>
    <summary>Synthetic summary content.</summary>
    <author><name>Author One</name></author>
    <author><name>Author Two</name></author>
    <arxiv:primary_category term='cs.LG' />
    <category term='cs.LG' />
    <link rel='related' type='application/pdf' href='https://arxiv.org/pdf/{identifier}.pdf' title='pdf' />
  </entry>
</feed>
"""


def test_arxiv_target_parsing_supports_ids_and_paper_urls() -> None:
    parsed = arxiv.parse_arxiv_paper_target(_BASE_ID)
    assert parsed is not None
    assert parsed.canonical_id == _BASE_ID

    parsed_abs = arxiv.parse_arxiv_paper_target(
        f"https://arxiv.org/abs/{_VERSIONED_ID}"
    )
    assert parsed_abs is not None
    assert parsed_abs.canonical_id == _VERSIONED_ID

    parsed_pdf = arxiv.parse_arxiv_paper_target(f"https://arxiv.org/pdf/{_BASE_ID}.pdf")
    assert parsed_pdf is not None
    assert parsed_pdf.canonical_id == _BASE_ID

    parsed_scheme = arxiv.parse_arxiv_paper_target(f"arxiv:{_BASE_ID}v1")
    assert parsed_scheme is not None
    assert parsed_scheme.canonical_id == f"{_BASE_ID}v1"


def test_arxiv_plugin_rejects_search_targets() -> None:
    target = "https://arxiv.org/search/?query=placeholder+query"
    assert arxiv.is_arxiv_paper_target(target) is False
    assert arxiv_plugin.can_resolve(target, {}) is False
    assert arxiv_plugin.classify_target(target, {}) is None


def test_resolve_arxiv_paper_prefers_source_text_and_emits_sidecars(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        arxiv,
        "_http_get",
        lambda _url, timeout: types.SimpleNamespace(
            text=_api_feed_xml(identifier=_UPDATED_ID)
        ),
    )
    source_bundle = arxiv._SourceBundle(
        main_path="main.tex",
        main_text="\\documentclass{article}\\begin{document}hello\\end{document}",
        sidecars=(
            arxiv._SourceFile(path="main.tex", text="\\documentclass{article}"),
            arxiv._SourceFile(path="sections/intro.tex", text="intro"),
        ),
    )
    monkeypatch.setattr(
        arxiv,
        "_fetch_source_bundle",
        lambda _parsed, max_tex_sidecars: source_bundle,
    )
    monkeypatch.setattr(
        arxiv, "_fetch_pdf_text", lambda *args, **kwargs: "pdf fallback"
    )
    monkeypatch.setattr(
        arxiv,
        "_convert_latex_source_to_markdown",
        lambda source: "converted markdown body",
    )

    docs = arxiv.resolve_arxiv_paper(
        _BASE_ID,
        settings=arxiv.ArxivSettings(
            format="md", include_tex_sidecars=True, max_tex_sidecars=6
        ),
        use_cache=True,
        cache_ttl=timedelta(days=1),
        refresh_cache=False,
    )

    assert docs[0].kind == "paper"
    assert docs[0].label == f"arXiv/{_BASE_ID}"
    assert docs[0].context_subpath == f"arxiv/{_BASE_ID}/paper.md"
    assert docs[0].rendered.startswith("title: Synthetic Title\n")
    assert f"url: https://arxiv.org/abs/{_BASE_ID}" in docs[0].rendered
    assert "arxiv_id:" not in docs[0].rendered
    assert "text_source:" not in docs[0].rendered
    assert "primary_category:" not in docs[0].rendered
    assert "## Abstract" not in docs[0].rendered
    assert "## Paper" not in docs[0].rendered
    assert "§ ABSTRACT" in docs[0].rendered
    assert "converted markdown body" in docs[0].rendered
    source_docs = [doc for doc in docs if doc.kind == "source_tex"]
    assert len(source_docs) == 2
    assert source_docs[0].context_subpath == f"arxiv/{_BASE_ID}/source/main.tex"


def test_resolve_arxiv_paper_falls_back_to_pdf_when_source_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        arxiv,
        "_http_get",
        lambda _url, timeout: types.SimpleNamespace(
            text=_api_feed_xml(identifier=_VERSIONED_ID)
        ),
    )
    monkeypatch.setattr(arxiv, "_fetch_source_bundle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(arxiv, "_fetch_pdf_text", lambda *args, **kwargs: "pdf content")

    docs = arxiv.resolve_arxiv_paper(
        f"https://arxiv.org/abs/{_VERSIONED_ID}",
        settings=arxiv.ArxivSettings(
            format="md", include_tex_sidecars=False, max_tex_sidecars=4
        ),
        use_cache=True,
        cache_ttl=None,
        refresh_cache=False,
    )

    assert len(docs) == 1
    assert docs[0].kind == "paper"
    assert docs[0].label == f"arXiv/{_VERSIONED_ID}"
    assert docs[0].context_subpath == f"arxiv/{_VERSIONED_ID}/paper.md"
    assert docs[0].rendered.startswith("title: Synthetic Title\n")
    assert f"url: https://arxiv.org/abs/{_VERSIONED_ID}" in docs[0].rendered
    assert "text_source:" not in docs[0].rendered


def test_arxiv_settings_support_format_modes() -> None:
    defaults = arxiv.build_arxiv_settings(None)
    assert defaults.format == "md"

    parsed_tex = arxiv.build_arxiv_settings({"format": "tex"})
    assert parsed_tex.format == "tex"

    invalid = arxiv.build_arxiv_settings({"format": "unknown"})
    assert invalid.format == "md"


def test_arxiv_plugin_runtime_overrides_accept_format_keys() -> None:
    parsed = arxiv_plugin._arxiv_runtime_overrides(  # noqa: SLF001
        {"paper-format": "tex", "paper": {"format": "md"}}
    )
    assert parsed is not None
    assert parsed["format"] == "md"


def test_expand_tex_includes_replaces_nested_inputs_and_includes() -> None:
    files = {
        "main.tex": (
            "\\section{Intro}\n\\input{sections/intro}\n\\include{sections/method}\n"
        ),
        "sections/intro.tex": "Intro body.\n\\input{nested/details}\n",
        "sections/nested/details.tex": "Nested details.\n",
        "sections/method.tex": "Method body.\n",
    }
    expanded = arxiv._expand_tex_includes("main.tex", files)  # noqa: SLF001
    assert "Intro body." in expanded
    assert "Nested details." in expanded
    assert "Method body." in expanded
    assert "\\input{sections/intro}" not in expanded
    assert "\\include{sections/method}" not in expanded


def test_expand_tex_includes_ignores_comments_and_cycles() -> None:
    files = {
        "main.tex": "% \\input{sections/hidden}\n\\input{sections/loop}\n",
        "sections/hidden.tex": "Hidden content.\n",
        "sections/loop.tex": "\\input{../main}\nLoop body.\n",
    }
    expanded = arxiv._expand_tex_includes("main.tex", files)  # noqa: SLF001
    assert "Hidden content." not in expanded
    assert "Loop body." in expanded
    assert "\\input{../main}" in expanded


def test_resolve_arxiv_paper_uses_raw_tex_when_format_is_tex(monkeypatch) -> None:
    monkeypatch.setattr(
        arxiv,
        "_http_get",
        lambda _url, timeout: types.SimpleNamespace(
            text=_api_feed_xml(identifier=_VERSIONED_ID)
        ),
    )
    source_bundle = arxiv._SourceBundle(
        main_path="main.tex",
        main_text="\\documentclass{article}\\begin{document}raw\\end{document}",
        sidecars=(arxiv._SourceFile(path="main.tex", text="\\documentclass{article}"),),
    )
    monkeypatch.setattr(
        arxiv,
        "_fetch_source_bundle",
        lambda _parsed, max_tex_sidecars: source_bundle,
    )

    def _should_not_convert(_source: str) -> str:
        raise AssertionError("conversion should be bypassed in tex mode")

    monkeypatch.setattr(arxiv, "_convert_latex_source_to_markdown", _should_not_convert)

    docs = arxiv.resolve_arxiv_paper(
        _BASE_ID,
        settings=arxiv.ArxivSettings(format="tex", include_tex_sidecars=False),
        use_cache=True,
        cache_ttl=None,
        refresh_cache=False,
    )
    assert "\\documentclass{article}" in docs[0].rendered


def test_resolve_arxiv_paper_falls_back_to_raw_source_on_bad_conversion(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        arxiv,
        "_http_get",
        lambda _url, timeout: types.SimpleNamespace(
            text=_api_feed_xml(identifier=_VERSIONED_ID)
        ),
    )
    source_bundle = arxiv._SourceBundle(
        main_path="main.tex",
        main_text="\\documentclass{article}\\begin{document}raw\\end{document}",
        sidecars=(arxiv._SourceFile(path="main.tex", text="\\documentclass{article}"),),
    )
    monkeypatch.setattr(
        arxiv,
        "_fetch_source_bundle",
        lambda _parsed, max_tex_sidecars: source_bundle,
    )
    monkeypatch.setattr(
        arxiv, "_convert_latex_source_to_markdown", lambda _source: None
    )

    docs = arxiv.resolve_arxiv_paper(
        _BASE_ID,
        settings=arxiv.ArxivSettings(format="md", include_tex_sidecars=False),
        use_cache=True,
        cache_ttl=None,
        refresh_cache=False,
    )
    assert "\\documentclass{article}" in docs[0].rendered


def test_conversion_pipeline_strips_preamble_before_converter(monkeypatch) -> None:
    seen: dict[str, str] = {}

    def _capture_converter(source: str) -> str:
        seen["value"] = source
        return "converted body"

    monkeypatch.setattr(arxiv, "_latex_to_text", _capture_converter)
    latex = (
        "\\documentclass{article}\n"
        "\\usepackage{amsmath}\n"
        "\\begin{document}\n"
        "Body text\n"
        "\\end{document}\n"
    )
    converted = arxiv._convert_latex_source_to_markdown(latex)  # noqa: SLF001
    assert converted is not None
    assert "converted body" in converted
    assert "\\documentclass" not in seen["value"]
    assert "\\usepackage" not in seen["value"]


def test_conversion_pipeline_replaces_graphics_markers_inline(monkeypatch) -> None:
    monkeypatch.setattr(
        arxiv,
        "_latex_to_text",
        lambda _source: "Intro\n\n    < g r a p h i c s >\n\nOutro",
    )
    latex = (
        "\\begin{document}\n"
        "Intro\n"
        "\\begin{figure}\n"
        "\\caption{Example figure}\n"
        "\\end{figure}\n"
        "\\end{document}\n"
    )
    converted = arxiv._convert_latex_source_to_markdown(latex)  # noqa: SLF001
    assert converted is not None
    assert "< g r a p h i c s >" not in converted
    assert "Intro" in converted
    assert "Outro" in converted
    assert "```tex" in converted
    assert "\\begin{figure}" in converted
    assert "\n    ```tex\n" not in converted
    assert "## Figures (LaTeX)" not in converted


def test_extract_figure_blocks_ignores_commented_endfigure_tokens() -> None:
    latex = "\\begin{figure}\n% \\end{figure}\n\\caption{Visible}\n\\end{figure}\n"
    blocks = arxiv._extract_figure_blocks(latex)  # noqa: SLF001
    assert len(blocks) == 1
    assert "\\caption{Visible}" in blocks[0]
    assert blocks[0].rstrip().endswith("\\end{figure}")


def test_conversion_pipeline_appends_unmatched_figures_to_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        arxiv,
        "_latex_to_text",
        lambda _source: "One marker\n\n< g r a p h i c s >\n\nEnd",
    )
    latex = (
        "\\begin{document}\n"
        "\\begin{figure}\n"
        "\\caption{First}\n"
        "\\end{figure}\n"
        "\\begin{figure}\n"
        "\\caption{Second}\n"
        "\\end{figure}\n"
        "\\end{document}\n"
    )
    converted = arxiv._convert_latex_source_to_markdown(latex)  # noqa: SLF001
    assert converted is not None
    assert converted.count("```tex") == 2
    assert "\\caption{First}" in converted
    assert "\\caption{Second}" in converted
    assert "## Figures (LaTeX)" in converted


def test_conversion_pipeline_leaves_extra_graphics_markers_unchanged(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        arxiv,
        "_latex_to_text",
        lambda _source: "< g r a p h i c s >\n\n< g r a p h i c s >",
    )
    latex = (
        "\\begin{document}\n"
        "\\begin{figure}\n"
        "\\caption{Only}\n"
        "\\end{figure}\n"
        "\\end{document}\n"
    )
    converted = arxiv._convert_latex_source_to_markdown(latex)  # noqa: SLF001
    assert converted is not None
    assert "\\caption{Only}" in converted
    assert converted.count("< g r a p h i c s >") == 1


def test_conversion_pipeline_dedupes_caption_after_inserted_figure(monkeypatch) -> None:
    monkeypatch.setattr(
        arxiv,
        "_latex_to_text",
        lambda _source: (
            "Before\n\n< g r a p h i c s >\n\n"
            "A sample caption line with placeholder wording\n\nAfter"
        ),
    )
    latex = (
        "\\begin{document}\n"
        "\\begin{figure}\n"
        "\\caption{A sample caption line with placeholder wording}\n"
        "\\end{figure}\n"
        "\\end{document}\n"
    )
    converted = arxiv._convert_latex_source_to_markdown(latex)  # noqa: SLF001
    assert converted is not None
    assert converted.count("```tex") == 1
    assert converted.count("A sample caption line with placeholder wording") == 1
    assert "After" in converted


def test_conversion_pipeline_dedupes_caption_with_ref_tokens(monkeypatch) -> None:
    monkeypatch.setattr(
        arxiv,
        "_latex_to_text",
        lambda _source: (
            "Before\n\n< g r a p h i c s >\n\n"
            "A sample figure description with reference marker <ref>.\n\nAfter"
        ),
    )
    latex = (
        "\\begin{document}\n"
        "\\begin{figure}\n"
        "\\caption{A sample figure description with reference marker \\ref{fig:tree}.}\n"
        "\\label{fig:tree}\n"
        "\\end{figure}\n"
        "\\end{document}\n"
    )
    converted = arxiv._convert_latex_source_to_markdown(latex)  # noqa: SLF001
    assert converted is not None
    assert converted.count("```tex") == 1
    assert converted.count("A sample figure description with reference marker") == 1
    assert "After" in converted


def test_conversion_pipeline_strips_maketitle_artifacts_and_dedents_prose(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        arxiv,
        "_latex_to_text",
        lambda _source: (
            "    [NO \\title GIVEN]\n"
            "    [NO \\author GIVEN]\n"
            "    January 1, 2000\n"
            "======================\n\n"
            "    Paragraph text"
        ),
    )
    latex = "\\begin{document}\n\\maketitle\nBody\n\\end{document}\n"
    converted = arxiv._convert_latex_source_to_markdown(latex)  # noqa: SLF001
    assert converted is not None
    assert "[NO \\title GIVEN]" not in converted
    assert "[NO \\author GIVEN]" not in converted
    assert "January 1, 2000" not in converted
    assert "Paragraph text" in converted
    assert "\n    Paragraph text" not in converted


def test_format_main_document_prefers_api_summary_and_drops_presection_noise() -> None:
    entry = arxiv.ArxivEntry(
        entry_id="https://arxiv.org/abs/1111.99999",
        title="Synthetic Title For Formatting",
        summary="API abstract summary.",
        authors=("Author One",),
        published="2026-02-03T15:47:32Z",
        updated="2026-02-05T03:55:49Z",
        primary_category="cs.AI",
        categories=("cs.AI", "cs.LG"),
        pdf_url="https://arxiv.org/pdf/1111.99999.pdf",
    )
    paper_text = (
        "[\nNoise author block\n]\n\nWrong abstract block\n\n"
        "§ INTRODUCTION\n\nBody starts here."
    )

    rendered = arxiv._format_main_document(  # noqa: SLF001
        entry=entry,
        canonical_id="1111.99999",
        paper_text=paper_text,
    )

    assert "§ ABSTRACT\n\nAPI abstract summary." in rendered
    assert "Noise author block" not in rendered
    assert "Wrong abstract block" not in rendered
    assert "§ INTRODUCTION" in rendered
    assert "Body starts here." in rendered


def test_arxiv_plugin_resolve_sets_metadata_and_dedupe(monkeypatch) -> None:
    monkeypatch.setattr(
        arxiv,
        "resolve_arxiv_paper",
        lambda *args, **kwargs: [
            arxiv.ArxivResolvedDocument(
                label=f"arXiv/{_BASE_ID}",
                rendered="paper",
                source_path=_BASE_ID,
                context_subpath=f"arxiv/{_BASE_ID}/paper.md",
                canonical_id=_BASE_ID,
                kind="paper",
                dedupe_rank=0,
                source_created="2023-06-06T00:00:00Z",
                source_modified="2024-01-02T00:00:00Z",
            )
        ],
    )

    result = arxiv_plugin.resolve(
        _BASE_ID,
        {
            "overrides": {},
            "use_cache": True,
            "refresh_cache": False,
            "cache_ttl": None,
        },
    )

    assert len(result) == 1
    metadata = result[0]["metadata"]
    assert metadata["provider"] == "arxiv"
    assert metadata["context_subpath"] == f"arxiv/{_BASE_ID}/paper.md"
    assert metadata["kind"] == "paper"
    assert metadata["hydrate_dedupe"]["mode"] == "canonical_symlink"
    assert metadata["hydrate_dedupe"]["rank"] == 0
    assert f"arxiv-paper:{_BASE_ID}" in metadata["hydrate_dedupe"]["key"]
