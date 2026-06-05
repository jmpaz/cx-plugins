from __future__ import annotations

from pathlib import Path
import sys

from contextualize import transcription
from contextualize.runtime import reset_verbose_logging, set_verbose_logging
from cx_plugins.providers.atproto import plugin as atproto_plugin
from cx_plugins.providers.soundcloud import plugin as soundcloud_plugin
from cx_plugins.providers.ytdlp import plugin as ytdlp_plugin
from cx_plugins.providers.ytdlp import ytdlp


SYNTHETIC_TIKTOK_ITEM_ID = "7000000000000000001"
SYNTHETIC_IMAGE_SOUND_ID = "7000000000000000002"
SYNTHETIC_VIDEO_SOUND_ID = "7000000000000000003"
SYNTHETIC_TIKTOK_HOST = "www.tiktok.com"
SYNTHETIC_TIKTOK_USER = "synthetic-user"
SYNTHETIC_TIKTOK_SHORT_CODE = "SYNTHETIC"
SYNTHETIC_SUBSTACK_URL = (
    "https://synthetic-substack.substack.com/p/synthetic-post"
)


def _synthetic_tiktok_url(kind: str) -> str:
    return (
        f"https://{SYNTHETIC_TIKTOK_HOST}/@{SYNTHETIC_TIKTOK_USER}/"
        f"{kind}/{SYNTHETIC_TIKTOK_ITEM_ID}"
    )


def _synthetic_tiktok_short_url() -> str:
    return f"https://{SYNTHETIC_TIKTOK_HOST}/t/{SYNTHETIC_TIKTOK_SHORT_CODE}/"


def test_ytdlp_priority_is_below_specialized_providers() -> None:
    assert ytdlp_plugin.PLUGIN_PRIORITY < soundcloud_plugin.PLUGIN_PRIORITY
    assert ytdlp_plugin.PLUGIN_PRIORITY < atproto_plugin.PLUGIN_PRIORITY


def _render_ref(url: str):
    ref = object.__new__(ytdlp.YtDlpReference)
    ref.url = url
    ref.format = "raw"
    ref.label = "relative"
    ref.token_target = "cl100k_base"
    ref.include_token_count = False
    ref.label_suffix = None
    ref.inject = False
    ref.depth = 5
    ref.trace_collector = None
    ref.use_cache = True
    ref.cache_ttl = None
    ref.refresh_cache = False
    ref.plugin_overrides = {"transcribe": {"diarize": True, "speakers": 2}}
    ref._metadata = None
    ref._identity = None
    ref.file_content = ""
    ref.original_file_content = ""
    return ref


def test_render_cache_identity_varies_by_transcription_overrides(
    monkeypatch,
) -> None:
    base = "youtube:abc123"
    monkeypatch.setattr(
        transcription,
        "transcription_routing_identity",
        lambda **_kwargs: {"model": "cohere"},
    )

    es_identity = ytdlp._render_cache_identity(
        base,
        {"transcribe": {"language": "es"}},
    )
    en_identity = ytdlp._render_cache_identity(
        base,
        {"transcribe": {"language": "en"}},
    )

    assert ytdlp._render_cache_identity(base, None) != base
    assert es_identity != base
    assert en_identity != base
    assert es_identity != en_identity
    assert (
        ytdlp._render_cache_identity(base, {"transcribe": {"language": "es"}})
        == es_identity
    )


def test_render_cache_identity_varies_by_effective_transcription_routing(
    monkeypatch,
) -> None:
    base = "youtube:abc123"
    routing = {"model": "cohere"}

    def _routing_identity(**_kwargs):
        return dict(routing)

    monkeypatch.setattr(
        transcription, "transcription_routing_identity", _routing_identity
    )

    cohere_identity = ytdlp._render_cache_identity(
        base,
        {"transcribe": {"diarize": True, "speakers": 2}},
    )
    routing["model"] = "mistral"
    mistral_identity = ytdlp._render_cache_identity(
        base,
        {"transcribe": {"diarize": True, "speakers": 2}},
    )

    assert cohere_identity != mistral_identity


def test_render_cache_hit_records_transcription_routing_without_changing_output(
    monkeypatch,
) -> None:
    cached_text = "---\ntitle: Cached\n---\n\ncached transcript"
    ref = object.__new__(ytdlp.YtDlpReference)
    ref.url = "https://example.com/watch"
    ref.format = "raw"
    ref.label = "relative"
    ref.token_target = "cl100k_base"
    ref.include_token_count = False
    ref.label_suffix = None
    ref.inject = False
    ref.depth = 5
    ref.trace_collector = None
    ref.use_cache = True
    ref.cache_ttl = None
    ref.refresh_cache = False
    ref.plugin_overrides = {"transcribe": {"diarize": True, "speakers": 2}}
    ref._metadata = {"extractor_key": "YouTube", "id": "abc123", "title": "Cached"}
    ref._identity = ytdlp._build_identity(ref.url, ref._metadata)  # noqa: SLF001
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "cx_plugins.providers.ytdlp.cache.get_cached_transcript",
        lambda *_args, **_kwargs: cached_text,
    )
    monkeypatch.setattr(
        transcription,
        "record_transcription_routing_summary",
        lambda **kwargs: calls.append(kwargs),
    )

    output = ytdlp.YtDlpReference._get_contents(ref)

    assert output == cached_text
    assert calls == [
        {
            "filename": "media.mp3",
            "content_type": "audio/mpeg",
            "plugin_overrides": {"transcribe": {"diarize": True, "speakers": 2}},
            "source": "render-cache",
        }
    ]


def test_youtube_render_cache_hit_does_not_fetch_metadata(monkeypatch) -> None:
    cached_text = "---\ntitle: Cached\n---\n\ncached transcript"
    ref = _render_ref("https://youtu.be/abc123?si=share")
    checked_identities: list[str] = []

    def _get_cached(identity: str, *_args, **_kwargs) -> str | None:
        checked_identities.append(identity)
        if identity.startswith("youtube:abc123:transcribe-fast:"):
            return cached_text
        return None

    def _fetch_metadata(_self):
        raise AssertionError("metadata should not be fetched on cache hit")

    monkeypatch.setattr(
        "cx_plugins.providers.ytdlp.cache.get_cached_transcript",
        _get_cached,
    )
    monkeypatch.setattr(ytdlp.YtDlpReference, "_fetch_metadata", _fetch_metadata)

    output = ytdlp.YtDlpReference._get_contents(ref)

    assert output == cached_text
    assert checked_identities[0].startswith("youtube:abc123:transcribe-fast:")
    assert ref.source_ref() == "youtu.be"
    assert ref.source_path() == "youtube:abc123"
    assert ref.context_subpath() == "ytdlp-youtube-abc123.md"
    assert ref.get_kind() == "video"


def test_legacy_metadata_cache_hit_backfills_url_cache(monkeypatch) -> None:
    cached_text = "---\ntitle: Cached\n---\n\ncached transcript"
    ref = _render_ref("https://vimeo.com/abc123")
    ref._metadata = {"extractor_key": "Vimeo", "id": "abc123", "title": "Cached"}
    checked_identities: list[str] = []
    stored: list[tuple[str, str, str]] = []

    def _get_cached(identity: str, *_args, **_kwargs) -> str | None:
        checked_identities.append(identity)
        if identity.startswith("vimeo:abc123:transcribe:"):
            return cached_text
        return None

    def _store(identity: str, content: str, source: str = "unknown") -> None:
        stored.append((identity, content, source))

    monkeypatch.setattr(
        "cx_plugins.providers.ytdlp.cache.get_cached_transcript",
        _get_cached,
    )
    monkeypatch.setattr("cx_plugins.providers.ytdlp.cache.store_transcript", _store)

    output = ytdlp.YtDlpReference._get_contents(ref)

    assert output == cached_text
    assert any(
        identity.startswith("vimeo:abc123:transcribe:")
        for identity in checked_identities
    )
    stored_identities = [identity for identity, _content, _source in stored]
    assert any(
        identity.startswith("vimeo:abc123:transcribe-fast:")
        for identity in stored_identities
    )
    assert any(identity.startswith("url:") for identity in stored_identities)
    assert all(content == cached_text for _identity, content, _source in stored)
    assert all(source == "render-cache" for _identity, _content, source in stored)


def test_providers_do_not_import_private_transcription_helpers() -> None:
    provider_root = Path(__file__).parents[1] / "src" / "cx_plugins" / "providers"
    offenders = []
    for path in provider_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "contextualize.references.audio_transcription import _" in text:
            offenders.append(path.relative_to(provider_root).as_posix())

    assert offenders == []


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


def test_twitter_urls_are_excluded_without_probe(monkeypatch) -> None:
    def _probe(*_args, **_kwargs):
        raise AssertionError("twitter urls should not be probed by ytdlp")

    monkeypatch.setattr(ytdlp, "probe_ytdlp_metadata", _probe)

    for target in (
        "https://x.com/synthetic_user/status/2000000000000000001?s=46",
        "https://twitter.com/synthetic_user/status/2000000000000000002",
        "https://mobile.twitter.com/synthetic_user/status/1",
    ):
        assert ytdlp.is_excluded_ytdlp_url(target) is True
        assert ytdlp_plugin.can_resolve(target, {}) is False
        assert ytdlp_plugin.classify_target(target, {}) is None


def test_tiktok_photo_urls_are_claimed_without_ytdlp_probe(monkeypatch) -> None:
    target = _synthetic_tiktok_url("photo")

    def _probe(*_args, **_kwargs):
        raise AssertionError("photo urls should not need a ytdlp probe")

    monkeypatch.setattr(ytdlp, "probe_ytdlp_metadata", _probe)

    assert ytdlp.is_tiktok_photo_url(target) is True
    assert ytdlp.looks_like_ytdlp_url(target) is True
    assert ytdlp_plugin.can_resolve(target, {}) is True
    classified = ytdlp_plugin.classify_target(target, {})
    assert classified is not None
    assert classified["kind"] == "image"
    assert classified["group_key"] == "image"


def test_tiktok_short_url_resolution_uses_tiktok_redirect(monkeypatch) -> None:
    captured: dict[str, object] = {}
    target = _synthetic_tiktok_short_url()
    resolved = _synthetic_tiktok_url("video")

    class _Response:
        headers = {"Location": resolved}

    class _Requests:
        @staticmethod
        def get(url: str, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return _Response()

    monkeypatch.setitem(sys.modules, "requests", _Requests)

    assert ytdlp._resolve_tiktok_short_url(target, timeout_seconds=30) == resolved
    assert captured["url"] == target
    assert captured["kwargs"]["allow_redirects"] is False


def test_probe_ytdlp_metadata_uses_resolved_tiktok_short_url(monkeypatch) -> None:
    captured: dict[str, object] = {}
    target = _synthetic_tiktok_short_url()
    resolved = _synthetic_tiktok_url("video")

    monkeypatch.setattr(ytdlp, "_check_ytdlp", lambda: None)
    monkeypatch.setattr(
        ytdlp,
        "_resolve_tiktok_short_url",
        lambda _url, *, timeout_seconds: resolved,
    )

    def _run(args: list[str], **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return ytdlp.subprocess.CompletedProcess(
            args,
            0,
            stdout=ytdlp.json.dumps(
                {
                    "extractor_key": "TikTok",
                    "id": SYNTHETIC_TIKTOK_ITEM_ID,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(ytdlp, "_run_ytdlp", _run)

    metadata = ytdlp.probe_ytdlp_metadata(target, timeout_seconds=12)

    assert metadata == {
        "extractor_key": "TikTok",
        "id": SYNTHETIC_TIKTOK_ITEM_ID,
    }
    assert captured["args"][-1] == resolved
    assert captured["kwargs"]["timeout_seconds"] == 12


def test_ytdlp_command_prefers_bundled_python_module(monkeypatch) -> None:
    monkeypatch.setattr(ytdlp, "_yt_dlp_module_available", lambda: True)
    monkeypatch.setattr(ytdlp.shutil, "which", lambda _name: "/usr/bin/yt-dlp")

    assert ytdlp._yt_dlp_command() == [sys.executable, "-m", "yt_dlp"]


def test_ytdlp_command_falls_back_to_path_executable(monkeypatch) -> None:
    monkeypatch.setattr(ytdlp, "_yt_dlp_module_available", lambda: False)
    monkeypatch.setattr(ytdlp.shutil, "which", lambda name: f"/bin/{name}")

    assert ytdlp._yt_dlp_command() == ["/bin/yt-dlp"]


def test_ytdlp_command_reports_missing_runtime(monkeypatch) -> None:
    monkeypatch.setattr(ytdlp, "_yt_dlp_module_available", lambda: False)
    monkeypatch.setattr(ytdlp.shutil, "which", lambda _name: None)

    try:
        ytdlp._yt_dlp_command()
    except RuntimeError as exc:
        assert "yt_dlp Python package or yt-dlp executable" in str(exc)
    else:
        raise AssertionError("expected missing yt-dlp runtime to raise")


def test_run_ytdlp_uses_resolved_command(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(ytdlp, "_yt_dlp_command", lambda: ["python", "-m", "yt_dlp"])

    def _run(command: list[str], **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs

        class _Result:
            returncode = 0
            stderr = ""
            stdout = ""

        return _Result()

    monkeypatch.setattr(ytdlp.subprocess, "run", _run)

    ytdlp._run_ytdlp(["--version"], timeout_seconds=3)

    assert captured["command"] == ["python", "-m", "yt_dlp", "--version"]
    assert captured["kwargs"]["timeout"] == 3


def test_run_ytdlp_streams_progress_to_verbose_stderr(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        ytdlp,
        "_yt_dlp_command",
        lambda: [
            sys.executable,
            "-c",
            "import sys; print('download phase', flush=True)",
        ],
    )
    token = set_verbose_logging(True)
    try:
        result = ytdlp._run_ytdlp(
            ["--ignored"],
            timeout_seconds=5,
            idle_timeout_seconds=5,
            stream_progress=True,
        )
    finally:
        reset_verbose_logging(token)

    captured = capsys.readouterr()
    assert result.returncode == 0
    assert "download phase" in result.stderr
    assert captured.out == ""
    assert "[ytdlp] yt-dlp: download phase" in captured.err


def test_can_resolve_probes_substack_before_claiming(monkeypatch) -> None:
    class _SubstackExtractor:
        IE_NAME = "Substack"

        @staticmethod
        def suitable(url: str) -> bool:
            return url == SYNTHETIC_SUBSTACK_URL

    monkeypatch.setattr(ytdlp, "_ytdlp_extractors", lambda: (_SubstackExtractor(),))
    ytdlp.matching_ytdlp_extractors.cache_clear()
    ytdlp.looks_like_ytdlp_url.cache_clear()
    monkeypatch.setattr(ytdlp, "probe_ytdlp_metadata", lambda *_args, **_kwargs: None)

    assert (
        ytdlp_plugin.can_resolve(
            SYNTHETIC_SUBSTACK_URL,
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
            SYNTHETIC_SUBSTACK_URL,
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


def test_classify_target_marks_tiktok_photomode_metadata_as_image(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        ytdlp,
        "probe_ytdlp_metadata",
        lambda _url, timeout_seconds=10: {
            "extractor_key": "TikTok",
            "id": SYNTHETIC_TIKTOK_ITEM_ID,
            "duration": 278,
            "thumbnail": "https://example.com/tplv-photomode-image.jpeg",
        },
    )

    classified = ytdlp_plugin.classify_target(
        _synthetic_tiktok_url("video"),
        {},
    )

    assert classified is not None
    assert classified["kind"] == "image"
    assert classified["group_key"] == "image"


def test_classify_target_skips_probe_required_url_without_metadata(monkeypatch) -> None:
    monkeypatch.setattr(ytdlp, "looks_like_ytdlp_url", lambda _url: True)
    monkeypatch.setattr(ytdlp, "requires_ytdlp_probe_for_claim", lambda _url: True)
    monkeypatch.setattr(ytdlp, "probe_ytdlp_metadata", lambda *_args, **_kwargs: None)

    assert (
        ytdlp_plugin.classify_target(
            SYNTHETIC_SUBSTACK_URL,
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


def test_resolve_returns_explicit_failure_doc_when_read_breaks(
    monkeypatch,
) -> None:
    class _BrokenReference:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def read(self) -> str:
            raise RuntimeError("download timed out")

    monkeypatch.setattr(ytdlp, "looks_like_ytdlp_url", lambda _url: True)
    monkeypatch.setattr(ytdlp, "YtDlpReference", _BrokenReference)

    docs = ytdlp_plugin.resolve("https://youtu.be/example", {})

    assert len(docs) == 1
    assert docs[0]["metadata"]["provider"] == "ytdlp"
    assert docs[0]["metadata"]["resolution_error"] == "download timed out"
    assert "download timed out" in docs[0]["content"]


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


def test_resolve_reraises_unexpected_read_bugs(monkeypatch) -> None:
    class _BuggyReference:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def read(self) -> str:
            raise TypeError("unexpected read bug")

    monkeypatch.setattr(ytdlp, "looks_like_ytdlp_url", lambda _url: True)
    monkeypatch.setattr(ytdlp, "YtDlpReference", _BuggyReference)

    try:
        ytdlp_plugin.resolve("https://youtu.be/example", {})
    except TypeError as exc:
        assert str(exc) == "unexpected read bug"
    else:
        raise AssertionError("expected unexpected read bugs to be reraised")


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


def test_tiktok_image_post_render_describes_each_image(monkeypatch) -> None:
    ref = _render_ref(_synthetic_tiktok_url("photo"))
    ref._metadata = {
        "extractor_key": "TikTok",
        "id": SYNTHETIC_TIKTOK_ITEM_ID,
        "title": "Synthetic image title",
        "description": "Synthetic image caption",
        "channel": "Synthetic image channel",
        "uploader": SYNTHETIC_TIKTOK_USER,
        "thumbnail": "https://example.com/tplv-photomode-image.jpeg",
    }
    ref._identity = ytdlp._build_identity(ref.url, ref._metadata)  # noqa: SLF001
    image_post = {
        "images": [
            {
                "imageWidth": 2160,
                "imageHeight": 3240,
                "imageURL": {"urlList": ["https://cdn.example/first.jpeg"]},
            },
            {
                "imageWidth": 1080,
                "imageHeight": 1920,
                "imageURL": {"urlList": ["https://cdn.example/second.jpeg"]},
            },
        ]
    }
    tiktok_item = {
        "imagePost": image_post,
        "music": {
            "id": SYNTHETIC_IMAGE_SOUND_ID,
            "title": "Synthetic image sound",
            "authorName": "Synthetic image artist",
            "duration": 47,
            "original": True,
        },
    }
    stored: list[tuple[str, str, str]] = []

    def _get_transcript(*_args, **_kwargs):
        raise AssertionError("image posts should not be transcribed")

    def _describe(self, image, *, metadata, total_images):
        assert metadata is ref._metadata
        assert total_images == 2
        return f"Alt text {image['index']}"

    monkeypatch.setattr(ytdlp, "_fetch_tiktok_item", lambda _metadata: tiktok_item)
    monkeypatch.setattr(ytdlp, "_should_refresh_tiktok_images", lambda: False)
    monkeypatch.setattr(ytdlp.YtDlpReference, "_get_transcript", _get_transcript)
    monkeypatch.setattr(ytdlp.YtDlpReference, "_describe_tiktok_image", _describe)
    monkeypatch.setattr(
        "cx_plugins.providers.ytdlp.cache.get_cached_transcript",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "cx_plugins.providers.ytdlp.cache.store_transcript",
        lambda identity, content, source="unknown": stored.append(
            (identity, content, source)
        ),
    )

    output = ytdlp.YtDlpReference._get_contents(ref)

    assert 'kind: image' in output
    assert 'image_count: 2' in output
    assert "## Sound" in output
    assert "- title: Synthetic image sound" in output
    assert "- artist: Synthetic image artist" in output
    assert "- duration_seconds: 47" in output
    image_sound_url = ytdlp._tiktok_music_url(
        "Synthetic image sound", SYNTHETIC_IMAGE_SOUND_ID
    )
    assert f"- url: {image_sound_url}" in output
    assert "\n- id:" not in output
    assert "- original: true" in output
    assert '<image index="1" width="2160" height="3240">' in output
    assert "Alt text 1" in output
    assert '<image index="2" width="1080" height="1920">' in output
    assert "Alt text 2" in output
    assert stored
    assert stored[0][0] == f"tiktok:{SYNTHETIC_TIKTOK_ITEM_ID}:image-post:v2"
    assert stored[0][2] == "image-post"


def test_tiktok_video_render_includes_sound_metadata(monkeypatch) -> None:
    ref = _render_ref(_synthetic_tiktok_url("video"))
    ref._metadata = {
        "extractor_key": "TikTok",
        "id": SYNTHETIC_TIKTOK_ITEM_ID,
        "title": "Synthetic video title",
        "description": "Synthetic video caption",
        "channel": "Synthetic video channel",
        "uploader": SYNTHETIC_TIKTOK_USER,
        "duration": 12,
        "track": "Synthetic fallback sound",
        "artists": ["Synthetic fallback artist"],
    }
    ref._identity = ytdlp._build_identity(ref.url, ref._metadata)  # noqa: SLF001
    tiktok_item = {
        "music": {
            "id": SYNTHETIC_VIDEO_SOUND_ID,
            "title": "Synthetic video sound",
            "authorName": "Synthetic video artist",
            "duration": 31,
            "original": False,
        }
    }
    stored: list[tuple[str, str, str]] = []

    monkeypatch.setattr(ytdlp, "_fetch_tiktok_item", lambda _metadata: tiktok_item)
    monkeypatch.setattr(
        "cx_plugins.providers.ytdlp.cache.get_cached_transcript",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "cx_plugins.providers.ytdlp.cache.store_transcript",
        lambda identity, content, source="unknown": stored.append(
            (identity, content, source)
        ),
    )
    monkeypatch.setattr(
        ytdlp.YtDlpReference,
        "_get_transcript",
        lambda _self, _duration: ("Transcript body.", "transcription"),
    )

    output = ytdlp.YtDlpReference._get_contents(ref)

    assert "## Sound" in output
    assert "- title: Synthetic video sound" in output
    assert "- artist: Synthetic video artist" in output
    assert "- duration_seconds: 31" in output
    video_sound_url = ytdlp._tiktok_music_url(
        "Synthetic video sound", SYNTHETIC_VIDEO_SOUND_ID
    )
    assert f"- url: {video_sound_url}" in output
    assert "\n- id:" not in output
    assert "- original: false" in output
    assert "Transcript body." in output
    assert stored
    assert stored[0][0].startswith(
        f"tiktok:{SYNTHETIC_TIKTOK_ITEM_ID}:tiktok-sound:v1:transcribe:"
    )
    assert stored[0][2] == "transcription"


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
        timeout: float | None = None,
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
        "contextualize.transcription.transcribe_media_file",
        _transcribe,
    )

    transcript, source = ytdlp.YtDlpReference._get_transcript(ref, 0)

    assert transcript == "transcript"
    assert source == "transcription"
    assert captured == {
        "path": audio_path,
        "use_cache": False,
        "refresh_cache": True,
        "timeout": None,
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
        "cx_plugins.providers.ytdlp.cache.get_cached_media_bytes",
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


def test_extract_audio_requests_audio_only_format(monkeypatch) -> None:
    ref = object.__new__(ytdlp.YtDlpReference)
    ref.url = "https://example.com/watch"
    ref.use_cache = False
    ref.refresh_cache = False
    ref.plugin_overrides = None
    ref._metadata = {"extractor_key": "YouTube", "id": "abc123"}
    ref._identity = ytdlp._build_identity(ref.url, ref._metadata)  # noqa: SLF001

    monkeypatch.setattr(
        "contextualize.runtime.get_refresh_audio",
        lambda: False,
    )
    calls: list[list[str]] = []

    def _run(args, **kwargs):
        calls.append(args)
        output_template = args[args.index("-o") + 1]
        output_path = Path(output_template.replace("%(ext)s", "mp3"))
        output_path.write_bytes(b"audio")

        class _Result:
            returncode = 0
            stderr = ""
            stdout = ""

        return _Result()

    monkeypatch.setattr(ytdlp, "_run_ytdlp", _run)

    audio_path = ytdlp.YtDlpReference._extract_audio(ref)

    assert audio_path.read_bytes() == b"audio"
    assert calls
    assert calls[0][0:2] == ["-f", "bestaudio/best"]
    assert calls[0][2:4] == ["--concurrent-fragments", "16"]
