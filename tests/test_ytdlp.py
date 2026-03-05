from __future__ import annotations

from cx_plugins.providers.atproto import plugin as atproto_plugin
from cx_plugins.providers.soundcloud import plugin as soundcloud_plugin
from cx_plugins.providers.ytdlp import plugin as ytdlp_plugin
from cx_plugins.providers.ytdlp import ytdlp


def test_ytdlp_priority_is_below_specialized_providers() -> None:
    assert ytdlp_plugin.PLUGIN_PRIORITY < soundcloud_plugin.PLUGIN_PRIORITY
    assert ytdlp_plugin.PLUGIN_PRIORITY < atproto_plugin.PLUGIN_PRIORITY


def test_can_resolve_uses_probe_with_fast_timeout(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []

    def _probe(url: str, *, timeout_seconds: int = 10) -> bool:
        calls.append((url, timeout_seconds))
        return url.endswith("/ok")

    monkeypatch.setattr(ytdlp, "probe_ytdlp_url", _probe)

    assert ytdlp_plugin.can_resolve("https://example.com/ok", {}) is True
    assert ytdlp_plugin.can_resolve("https://example.com/nope", {}) is False
    assert calls == [
        ("https://example.com/ok", 5),
        ("https://example.com/nope", 5),
    ]


def test_classify_target_uses_probe_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        ytdlp,
        "probe_ytdlp_metadata",
        lambda _url, timeout_seconds=10: {"duration": 90},
    )
    classified = ytdlp_plugin.classify_target("https://example.com/video", {})
    assert classified is not None
    assert classified["provider"] == "ytdlp"
    assert classified["kind"] == "video"

    monkeypatch.setattr(
        ytdlp,
        "probe_ytdlp_metadata",
        lambda _url, timeout_seconds=10: {"duration": 0},
    )
    classified = ytdlp_plugin.classify_target("https://example.com/resource", {})
    assert classified is not None
    assert classified["kind"] == "resource"

    monkeypatch.setattr(
        ytdlp,
        "probe_ytdlp_metadata",
        lambda _url, timeout_seconds=10: None,
    )
    assert ytdlp_plugin.classify_target("https://example.com/none", {}) is None


def test_resolve_uses_generalized_reference_metadata(monkeypatch) -> None:
    class _DummyReference:
        def __init__(
            self,
            url: str,
            *,
            format: str,
            label: str,
            inject: bool,
            use_cache: bool,
            cache_ttl,
            refresh_cache: bool,
        ) -> None:
            assert url == "https://example.com/watch"
            assert format == "raw"
            assert label == "relative"
            assert inject is False
            assert use_cache is False
            assert cache_ttl is None
            assert refresh_cache is True

        def get_label(self) -> str:
            return "https://example.com/watch"

        def read(self) -> str:
            return "transcript"

        def source_ref(self) -> str:
            return "example.com"

        def source_path(self) -> str:
            return "vimeo:abc123"

        def context_subpath(self) -> str:
            return "ytdlp-vimeo-abc123.md"

        def get_kind(self) -> str:
            return "video"

    monkeypatch.setattr(ytdlp, "YtDlpReference", _DummyReference)

    docs = ytdlp_plugin.resolve(
        "https://example.com/watch",
        {"use_cache": False, "refresh_cache": True},
    )
    assert len(docs) == 1
    metadata = docs[0]["metadata"]
    assert metadata["provider"] == "ytdlp"
    assert metadata["source_ref"] == "example.com"
    assert metadata["source_path"] == "vimeo:abc123"
    assert metadata["context_subpath"] == "ytdlp-vimeo-abc123.md"
    assert metadata["kind"] == "video"


def test_build_identity_uses_extractor_and_id_or_url_hash() -> None:
    with_id = ytdlp._build_identity(  # noqa: SLF001
        "https://example.com/media",
        {"extractor_key": "YouTube", "id": "abc123"},
    )
    assert with_id.cache_identity == "youtube:abc123"
    assert with_id.display_name == "youtube:abc123"
    assert with_id.slug == "youtube-abc123"

    without_id = ytdlp._build_identity(  # noqa: SLF001
        "https://example.com/media#fragment",
        {},
    )
    without_id_dupe = ytdlp._build_identity(  # noqa: SLF001
        "https://example.com/media",
        {},
    )
    assert without_id.cache_identity.startswith("url:")
    assert without_id.display_name.startswith("url:")
    assert without_id.slug.startswith("url-")
    assert without_id.cache_identity == without_id_dupe.cache_identity
