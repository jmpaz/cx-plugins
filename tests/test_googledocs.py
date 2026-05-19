from __future__ import annotations

import io
import types
import zipfile

import cx_plugins.providers.googledocs.plugin as googledocs_plugin
from cx_plugins.providers.googledocs import googledocs


DOC_ID = "178iNWfAt7d_A15tffPoPKLOSZGQf08ow22g2380jIh8"


def test_parse_google_doc_target_supports_urls_and_schemes() -> None:
    parsed = googledocs.parse_google_doc_target(
        f"https://docs.google.com/document/d/{DOC_ID}/edit?tab=t.123"
    )
    assert parsed is not None
    assert parsed.doc_id == DOC_ID
    assert parsed.tab == "t.123"
    assert parsed.canonical_id == f"{DOC_ID}:tab:t.123"

    scheme = googledocs.parse_google_doc_target(f"gdoc:{DOC_ID}")
    assert scheme is not None
    assert scheme.doc_id == DOC_ID

    published = googledocs.parse_google_doc_target(
        "https://docs.google.com/document/d/e/2PACX-1vTpub/pub"
    )
    assert published is not None
    assert published.published is True

    assert googledocs.parse_google_doc_target("https://drive.google.com/file/d/abc") is None


def test_plugin_classifies_google_doc_urls() -> None:
    target = f"https://docs.google.com/document/d/{DOC_ID}/edit"
    assert googledocs_plugin.can_resolve(target, {}) is True
    classified = googledocs_plugin.classify_target(target, {})
    assert classified == {
        "provider": "googledocs",
        "kind": "document",
        "is_external": True,
        "group_key": "document",
    }


def test_runtime_overrides_parse_manifest_style_aliases() -> None:
    parsed = googledocs_plugin._googledocs_runtime_overrides(  # noqa: SLF001
        {
            "include-media": True,
            "media": {"describe": False},
            "timeout": 42,
        }
    )
    assert parsed is not None
    assert parsed["include_media"] is True
    assert parsed["include_media_descriptions"] is False
    assert parsed["timeout_seconds"] == 42


def test_resolve_uses_public_markdown_export(monkeypatch) -> None:
    calls: list[tuple[str, float]] = []

    def fake_get(url: str, *, timeout: float):
        calls.append((url, timeout))
        return types.SimpleNamespace(
            headers={"Content-Type": "text/x-markdown"},
            content=b"# Public Doc\n\nHello from anonymous export.\n",
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr("requests.get", fake_get)

    docs = googledocs_plugin.resolve(
        f"https://docs.google.com/document/d/{DOC_ID}/edit",
        {"use_cache": False},
    )

    assert calls == [(f"https://docs.google.com/document/d/{DOC_ID}/export?format=md", 30.0)]
    assert len(docs) == 1
    doc = docs[0]
    assert doc["label"] == "googledocs/Public_Doc"
    assert "# Public Doc" in doc["content"]
    assert "Hello from anonymous export." in doc["content"]
    metadata = doc["metadata"]
    assert metadata["provider"] == "googledocs"
    assert metadata["source_ref"] == "docs.google.com"
    assert metadata["source_path"] == DOC_ID
    assert metadata["context_subpath"] == "googledocs/Public_Doc.md"
    assert metadata["canonical_id"] == DOC_ID
    assert metadata["hydrate_dedupe"]["mode"] == "canonical_symlink"

def test_public_export_failure_returns_error_document(monkeypatch) -> None:
    def fake_get(url: str, *, timeout: float):
        return types.SimpleNamespace(
            headers={"Content-Type": "text/html"},
            content=b"<html>not public</html>",
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr("requests.get", fake_get)

    docs = googledocs_plugin.resolve(
        f"https://docs.google.com/document/d/{DOC_ID}/edit",
        {"use_cache": False},
    )

    assert len(docs) == 1
    doc = docs[0]
    assert doc["label"] == f"googledocs/{DOC_ID}-error"
    assert "generic webpage fallback was suppressed" in doc["content"]
    metadata = doc["metadata"]
    assert metadata["provider"] == "googledocs"
    assert metadata["canonical_id"] == DOC_ID
    assert "resolution_error" in metadata



def test_zip_markdown_export_rewrites_local_media(monkeypatch) -> None:
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("doc.md", "# Illustrated\n\n![Diagram](images/diagram.png)\n")
        archive.writestr("images/diagram.png", b"fake image bytes")

    def fake_convert_path_to_markdown(path: str, *, prompt_append: str = ""):
        assert path.endswith(".png")
        assert "Diagram" in prompt_append
        return types.SimpleNamespace(markdown="# Description (auto-generated):\n\nA diagram.")

    monkeypatch.setattr(
        "contextualize.render.markitdown.convert_path_to_markdown",
        fake_convert_path_to_markdown,
    )


    def fake_fetch_public_markdown(
        _parsed: googledocs.ParsedGoogleDocTarget,
        _settings: googledocs.GoogleDocsSettings,
    ) -> bytes:
        return archive_bytes.getvalue()

    monkeypatch.setattr(googledocs, "_fetch_public_markdown", fake_fetch_public_markdown)
    document = googledocs.resolve_google_doc(
        f"https://docs.google.com/document/d/{DOC_ID}/edit",
        settings=googledocs.GoogleDocsSettings(),
        use_cache=False,
        cache_ttl=None,
        refresh_cache=False,
    )

    assert '<image filename="diagram.png" caption="Diagram">' in document.rendered
    assert "A diagram." in document.rendered
    assert "</image>" in document.rendered
    assert document.media_count == 1


def test_reference_style_data_uri_media_becomes_contextualize_tags() -> None:
    rendered = googledocs._render_export(  # noqa: SLF001
        (
            "# Public Doc\n\n"
            "Text before ![][image1] and after.\n\n"
            "[image1]: <data:image/png;base64,ZmFrZQ==>\n"
        ).encode(),
        settings=googledocs.GoogleDocsSettings(include_media_descriptions=False),
    )

    assert '<image filename="image1.png" />' in rendered.markdown
    assert "[image1]:" not in rendered.markdown
    assert rendered.media_count == 1


def test_include_media_false_strips_reference_images() -> None:
    rendered = googledocs._render_export(  # noqa: SLF001
        (
            "# Public Doc\n\n"
            "Alt only: ![Diagram][image1].\n\n"
            "[image1]: <data:image/png;base64,ZmFrZQ==>\n"
        ).encode(),
        settings=googledocs.GoogleDocsSettings(include_media=False),
    )

    assert "Alt only: Diagram." in rendered.markdown
    assert "[image1]:" not in rendered.markdown
    assert rendered.media_count == 0
