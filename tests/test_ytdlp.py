from __future__ import annotations

from pathlib import Path

from cx_plugins.providers.atproto import plugin as atproto_plugin
from cx_plugins.providers.soundcloud import plugin as soundcloud_plugin
from cx_plugins.providers.ytdlp import plugin as ytdlp_plugin
from cx_plugins.providers.ytdlp import ytdlp


def test_ytdlp_priority_is_below_specialized_providers() -> None:
    assert ytdlp_plugin.PLUGIN_PRIORITY < soundcloud_plugin.PLUGIN_PRIORITY
    assert ytdlp_plugin.PLUGIN_PRIORITY < atproto_plugin.PLUGIN_PRIORITY


def test_can_resolve_uses_extractor_matching_without_probe(monkeypatch) -> None:
    class _MatchExtractor:
        IE_NAME = "youtube"

        @staticmethod
        def suitable(url: str) -> bool:
            return url.endswith("/ok")

    class _GenericExtractor:
        IE_NAME = "generic"

        @staticmethod
        def suitable(url: str) -> bool:
            return True

    monkeypatch.setattr(
        ytdlp,
        "_ytdlp_extractors",
        lambda: (_MatchExtractor(), _GenericExtractor()),
    )
    ytdlp.matching_ytdlp_extractors.cache_clear()
    ytdlp.looks_like_ytdlp_url.cache_clear()

    assert ytdlp_plugin.can_resolve("https://example.com/ok", {}) is True
    assert ytdlp_plugin.can_resolve("https://example.com/nope", {}) is False


def test_can_resolve_probes_substack_before_claiming(monkeypatch) -> None:
    class _SubstackExtractor:
        IE_NAME = "Substack"

        @staticmethod
        def suitable(url: str) -> bool:
            return url == "https://regroup.substack.com/p/on-fluid-objects-static-self"

    monkeypatch.setattr(ytdlp, "_ytdlp_extractors", lambda: (_SubstackExtractor(),))
    ytdlp.matching_ytdlp_extractors.cache_clear()
    ytdlp.looks_like_ytdlp_url.cache_clear()
    monkeypatch.setattr(ytdlp, "probe_ytdlp_metadata", lambda *_args, **_kwargs: None)

    assert (
        ytdlp_plugin.can_resolve(
            "https://regroup.substack.com/p/on-fluid-objects-static-self",
            {},
        )
        is False
    )

    monkeypatch.setattr(
        ytdlp,
        "probe_ytdlp_metadata",
        lambda *_args, **_kwargs: {"duration": 120},
    )

    assert (
        ytdlp_plugin.can_resolve(
            "https://regroup.substack.com/p/on-fluid-objects-static-self",
            {},
        )
        is True
    )


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
    monkeypatch.setattr(ytdlp, "looks_like_ytdlp_url", lambda _url: True)
    classified = ytdlp_plugin.classify_target("https://example.com/none", {})
    assert classified is not None
    assert classified["kind"] == "video"


def test_classify_target_skips_probe_required_url_without_metadata(monkeypatch) -> None:
    monkeypatch.setattr(ytdlp, "looks_like_ytdlp_url", lambda _url: True)
    monkeypatch.setattr(ytdlp, "requires_ytdlp_probe_for_claim", lambda _url: True)
    monkeypatch.setattr(ytdlp, "probe_ytdlp_metadata", lambda *_args, **_kwargs: None)

    assert (
        ytdlp_plugin.classify_target(
            "https://regroup.substack.com/p/on-fluid-objects-static-self",
            {},
        )
        is None
    )


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
            plugin_overrides,
        ) -> None:
            assert url == "https://example.com/watch"
            assert format == "raw"
            assert label == "relative"
            assert inject is False
            assert use_cache is False
            assert cache_ttl is None
            assert refresh_cache is True
            assert plugin_overrides == {"transcribe": {"provider": "mistral"}}

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
        {
            "use_cache": False,
            "refresh_cache": True,
            "overrides": {"transcribe": {"provider": "mistral"}},
        },
    )
    assert len(docs) == 1
    metadata = docs[0]["metadata"]
    assert metadata["provider"] == "ytdlp"
    assert metadata["source_ref"] == "example.com"
    assert metadata["source_path"] == "vimeo:abc123"
    assert metadata["context_subpath"] == "ytdlp-vimeo-abc123.md"
    assert metadata["kind"] == "video"


def test_resolve_returns_explicit_failure_doc_when_claimed_media_breaks(
    monkeypatch,
) -> None:
    class _BrokenReference:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(ytdlp, "looks_like_ytdlp_url", lambda _url: True)
    monkeypatch.setattr(ytdlp, "YtDlpReference", _BrokenReference)

    docs = ytdlp_plugin.resolve("https://youtu.be/example", {})

    assert len(docs) == 1
    assert docs[0]["metadata"]["provider"] == "ytdlp"
    assert docs[0]["metadata"]["kind"] == "video"
    assert "failed to resolve" in docs[0]["content"]


def test_resolve_reraises_unexpected_reference_bugs(monkeypatch) -> None:
    class _BuggyReference:
        def __init__(self, *args, **kwargs) -> None:
            raise TypeError("unexpected bug")

    monkeypatch.setattr(ytdlp, "looks_like_ytdlp_url", lambda _url: True)
    monkeypatch.setattr(ytdlp, "YtDlpReference", _BuggyReference)

    try:
        ytdlp_plugin.resolve("https://youtu.be/example", {})
    except TypeError as exc:
        assert str(exc) == "unexpected bug"
    else:
        raise AssertionError("expected unexpected reference bugs to be reraised")


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


def test_get_transcript_passes_transcription_cache_flags(
    tmp_path: Path, monkeypatch
) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio_path = audio_dir / "clip.mp3"
    audio_path.write_bytes(b"audio")

    ref = object.__new__(ytdlp.YtDlpReference)
    ref.use_cache = False
    ref.refresh_cache = True
    ref.plugin_overrides = {"transcribe": {"provider": "mistral"}}

    def _extract_audio(self: ytdlp.YtDlpReference) -> Path:
        return audio_path

    captured: dict[str, object] = {}

    def _transcribe(
        path: Path,
        *,
        use_cache: bool = True,
        refresh_cache: bool | None = None,
        timeout: float = 600,
        plugin_overrides=None,
    ) -> str:
        captured["path"] = path
        captured["use_cache"] = use_cache
        captured["refresh_cache"] = refresh_cache
        captured["timeout"] = timeout
        captured["plugin_overrides"] = plugin_overrides
        return "transcript"

    monkeypatch.setattr(ytdlp.YtDlpReference, "_extract_audio", _extract_audio)
    monkeypatch.setattr(
        "contextualize.references.audio_transcription.transcribe_media_file",
        _transcribe,
    )

    transcript, source = ytdlp.YtDlpReference._get_transcript(ref, 0)

    assert transcript == "transcript"
    assert source == "whisper"
    assert captured == {
        "path": audio_path,
        "use_cache": False,
        "refresh_cache": True,
        "timeout": 600,
        "plugin_overrides": {"transcribe": {"provider": "mistral"}},
    }
    assert not audio_dir.exists()


def test_extract_audio_uses_cached_media_bytes(tmp_path: Path, monkeypatch) -> None:
    ref = object.__new__(ytdlp.YtDlpReference)
    ref.url = "https://example.com/watch"
    ref.use_cache = True
    ref.refresh_cache = False
    ref.plugin_overrides = None
    ref._metadata = {"extractor_key": "YouTube", "id": "abc123"}
    ref._identity = ytdlp._build_identity(ref.url, ref._metadata)  # noqa: SLF001

    monkeypatch.setattr(
        "contextualize.cache.youtube.get_cached_media_bytes",
        lambda identity: (
            b"cached-audio" if identity == "audio:youtube:abc123" else None
        ),
    )
    monkeypatch.setattr(
        "contextualize.runtime.get_refresh_audio",
        lambda: False,
    )

    calls: list[list[str]] = []

    def _run(*args, **kwargs):
        calls.append(args[0])

        class _Result:
            returncode = 0
            stderr = ""
            stdout = ""

        return _Result()

    monkeypatch.setattr(ytdlp, "_run_ytdlp", _run)

    audio_path = ytdlp.YtDlpReference._extract_audio(ref)

    assert audio_path.read_bytes() == b"cached-audio"
    assert calls == []
    audio_path.unlink(missing_ok=True)
