from __future__ import annotations

from pathlib import Path

from contextualize import transcription
from cx_plugins.providers.snipd import plugin as snipd_plugin
from cx_plugins.providers.snipd import snipd

CLIP_ID = "11111111-2222-3333-4444-555555555555"
PUBLIC_CLIP_ID = "66666666-7777-8888-9999-aaaaaaaaaaaa"
EPISODE_ID = "bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
SHOW_ID = "12345678-1234-1234-1234-123456789abc"
SNIPD_URL = f"https://share.snipd.com/snip/{CLIP_ID}"


def _state_html(
    *,
    clip_id: str = CLIP_ID,
    note_md: str = "# Synthetic Note\n\n- detail",
    generated_title: str | None = None,
    user_title: str | None = None,
) -> str:
    clip: dict[str, object] = {
        "__typename": "public_clips",
        "clip_id": clip_id,
        "public_clip_id": PUBLIC_CLIP_ID,
        "generated_title": generated_title,
        "user_title": user_title,
        "user_start_time_seconds": 12.345,
        "user_end_time_seconds": 45.678,
        "note_md": note_md,
        "episode": {"__ref": f"episodes:{EPISODE_ID}"},
    }
    state = {
        "apollo.state": {
            f"shows:{SHOW_ID}": {
                "__typename": "shows",
                "id": SHOW_ID,
                "title": "Synthetic Show",
            },
            f"episodes:{EPISODE_ID}": {
                "__typename": "episodes",
                "id": EPISODE_ID,
                "title": "Synthetic Episode",
                "audio_url": "https://media.example.test/episode.mp3",
                "show": {"__ref": f"shows:{SHOW_ID}"},
            },
            "ROOT_QUERY": {
                f'public_clips({{"where":{{"clip_id":{{"_eq":"{clip_id}"}}}}}})': [
                    clip
                ],
            },
        },
    }
    import json

    return (
        "<html><head><title>Synthetic Page Title</title></head><body>"
        f'<script id="serverApp-state" type="application/json">{json.dumps(state)}</script>'
        "</body></html>"
    )


def _render_ref(url: str = SNIPD_URL) -> snipd.SnipdReference:
    ref = object.__new__(snipd.SnipdReference)
    ref.url = url
    ref.use_cache = True
    ref.cache_ttl = None
    ref.refresh_cache = False
    ref.plugin_overrides = {"transcribe": {"diarize": True, "speakers": 2}}
    ref._clip = None
    ref.file_content = ""
    ref.original_file_content = ""
    return ref


def test_parse_snipd_target_accepts_public_snip_urls_only() -> None:
    parsed = snipd.parse_snipd_target(SNIPD_URL)

    assert parsed is not None
    assert parsed.clip_id == CLIP_ID
    assert snipd_plugin.can_resolve(SNIPD_URL, {}) is True
    assert snipd_plugin.can_resolve("https://share.snipd.com/episode/example", {}) is False
    assert snipd_plugin.can_resolve("https://example.test/snip/" + CLIP_ID, {}) is False


def test_classify_target_marks_snipd_as_audio() -> None:
    classified = snipd_plugin.classify_target(SNIPD_URL, {})

    assert classified == {
        "provider": "snipd",
        "kind": "audio",
        "is_external": True,
        "group_key": "audio",
    }


def test_resolve_snipd_clip_extracts_embedded_apollo_state(monkeypatch) -> None:
    monkeypatch.setattr(snipd, "_fetch_html", lambda _url: _state_html())

    clip = snipd.resolve_snipd_clip(SNIPD_URL)

    assert clip.clip_id == CLIP_ID
    assert clip.public_clip_id == PUBLIC_CLIP_ID
    assert clip.episode_id == EPISODE_ID
    assert clip.title == "Synthetic Note"
    assert clip.episode_title == "Synthetic Episode"
    assert clip.show_title == "Synthetic Show"
    assert clip.audio_url == "https://media.example.test/episode.mp3"
    assert clip.start_seconds == 12.345
    assert clip.end_seconds == 45.678
    assert round(clip.duration_seconds, 3) == 33.333


def test_title_prefers_generated_and_user_fields(monkeypatch) -> None:
    monkeypatch.setattr(
        snipd,
        "_fetch_html",
        lambda _url: _state_html(
            generated_title="Generated Title",
            user_title="User Title",
        ),
    )

    clip = snipd.resolve_snipd_clip(SNIPD_URL)

    assert clip.title == "Generated Title"


def test_format_output_omits_clip_id_from_text() -> None:
    clip = snipd.SnipdClip(
        clip_id=CLIP_ID,
        public_clip_id=PUBLIC_CLIP_ID,
        episode_id=EPISODE_ID,
        title="Synthetic",
        episode_title="Episode",
        show_title="Show",
        audio_url="https://media.example.test/episode.mp3",
        start_seconds=10.0,
        end_seconds=20.0,
    )

    output = snipd._format_output(clip, "transcript")

    assert "clip_id:" not in output
    assert CLIP_ID not in output
    assert "start_seconds: 10.000" in output


def test_render_cache_identity_varies_by_transcription_overrides(monkeypatch) -> None:
    monkeypatch.setattr(
        transcription,
        "transcription_routing_identity",
        lambda **_kwargs: {"model": "cohere"},
    )

    es_identity = snipd._render_cache_identity(
        f"snipd:{CLIP_ID}",
        {"transcribe": {"language": "es"}},
    )
    en_identity = snipd._render_cache_identity(
        f"snipd:{CLIP_ID}",
        {"transcribe": {"language": "en"}},
    )

    assert es_identity.startswith(f"snipd:{CLIP_ID}:transcribe:")
    assert en_identity.startswith(f"snipd:{CLIP_ID}:transcribe:")
    assert es_identity != en_identity


def test_render_cache_identity_varies_by_output_schema(monkeypatch) -> None:
    monkeypatch.setattr(
        transcription,
        "transcription_routing_identity",
        lambda **_kwargs: {"model": "cohere"},
    )

    current_identity = snipd._render_cache_identity(f"snipd:{CLIP_ID}", {})
    monkeypatch.setattr(snipd, "_RENDER_CACHE_SCHEMA", snipd._RENDER_CACHE_SCHEMA + 1)
    next_identity = snipd._render_cache_identity(f"snipd:{CLIP_ID}", {})

    assert current_identity != next_identity


def test_render_cache_hit_does_not_fetch_snipd_metadata(monkeypatch) -> None:
    cached_text = "---\ntitle: Cached\n---\n\ncached transcript"
    ref = _render_ref()
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "cx_plugins.providers.snipd.cache.get_cached_transcript",
        lambda *_args, **_kwargs: cached_text,
    )
    monkeypatch.setattr(
        snipd,
        "resolve_snipd_clip",
        lambda _url: (_ for _ in ()).throw(AssertionError("should not fetch metadata")),
    )
    monkeypatch.setattr(
        transcription,
        "record_transcription_routing_summary",
        lambda **kwargs: calls.append(kwargs),
    )

    output = snipd.SnipdReference._get_contents(ref)

    assert output == cached_text
    assert ref.loaded_clip() is None
    assert calls == [
        {
            "filename": "media.mp3",
            "content_type": "audio/mpeg",
            "plugin_overrides": {"transcribe": {"diarize": True, "speakers": 2}},
            "source": "render-cache",
        }
    ]


def test_extract_audio_uses_cached_media_bytes(monkeypatch) -> None:
    ref = _render_ref()
    ref._clip = snipd.SnipdClip(
        clip_id=CLIP_ID,
        public_clip_id=PUBLIC_CLIP_ID,
        episode_id=EPISODE_ID,
        title="Synthetic",
        episode_title="Episode",
        show_title=None,
        audio_url="https://media.example.test/episode.mp3",
        start_seconds=10.0,
        end_seconds=20.0,
    )

    monkeypatch.setattr(
        "cx_plugins.providers.snipd.cache.get_cached_media_bytes",
        lambda identity: b"cached-audio"
        if identity == f"audio:snipd:{CLIP_ID}"
        else None,
    )
    monkeypatch.setattr("contextualize.runtime.get_refresh_audio", lambda: False)
    monkeypatch.setattr(snipd.subprocess, "run", lambda *_args, **_kwargs: None)

    audio_path = snipd.SnipdReference._extract_audio(ref)

    assert audio_path.read_bytes() == b"cached-audio"
    assert audio_path.parent.name.startswith("snipd-audio-")
    audio_path.unlink(missing_ok=True)
    audio_path.parent.rmdir()


def test_extract_audio_cuts_snipd_range_with_ffmpeg(tmp_path: Path, monkeypatch) -> None:
    ref = _render_ref()
    ref.use_cache = True
    ref._clip = snipd.SnipdClip(
        clip_id=CLIP_ID,
        public_clip_id=PUBLIC_CLIP_ID,
        episode_id=EPISODE_ID,
        title="Synthetic",
        episode_title="Episode",
        show_title=None,
        audio_url="https://media.example.test/episode.mp3",
        start_seconds=10.125,
        end_seconds=20.625,
    )
    calls: list[list[str]] = []
    stored: list[tuple[str, bytes]] = []

    monkeypatch.setattr(
        "cx_plugins.providers.snipd.cache.get_cached_media_bytes",
        lambda _identity: None,
    )
    monkeypatch.setattr(
        "cx_plugins.providers.snipd.cache.store_media_bytes",
        lambda identity, content: stored.append((identity, content)),
    )
    monkeypatch.setattr("contextualize.runtime.get_refresh_audio", lambda: False)
    monkeypatch.setattr(snipd.shutil, "which", lambda _name: "/bin/ffmpeg")

    def _run(command: list[str], **_kwargs):
        calls.append(command)
        output_path = Path(command[-1])
        output_path.write_bytes(b"cut-audio")

        class _Result:
            returncode = 0
            stderr = ""

        return _Result()

    monkeypatch.setattr(snipd.subprocess, "run", _run)

    audio_path = snipd.SnipdReference._extract_audio(ref)

    assert audio_path.read_bytes() == b"cut-audio"
    assert calls[0][0] == "/bin/ffmpeg"
    assert calls[0][calls[0].index("-ss") + 1] == "10.125"
    assert calls[0][calls[0].index("-i") + 1] == "https://media.example.test/episode.mp3"
    assert calls[0][calls[0].index("-t") + 1] == "10.500"
    assert stored == [(f"audio:snipd:{CLIP_ID}", b"cut-audio")]
    audio_path.unlink(missing_ok=True)


def test_get_transcript_passes_transcription_cache_flags(
    tmp_path: Path, monkeypatch
) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio_path = audio_dir / "clip.mp3"
    audio_path.write_bytes(b"audio")

    ref = _render_ref()
    ref.use_cache = False
    ref.refresh_cache = True
    ref.plugin_overrides = {"transcribe": {"provider": "mistral"}}
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        snipd.SnipdReference,
        "_extract_audio",
        lambda _self: audio_path,
    )

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

    monkeypatch.setattr(
        "contextualize.transcription.transcribe_media_file",
        _transcribe,
    )

    transcript, source = snipd.SnipdReference._get_transcript(ref)

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


def test_plugin_resolve_returns_metadata_without_refetching_on_cached_content(
    monkeypatch,
) -> None:
    cached_text = "---\ntitle: Cached\n---\n\ncached transcript"

    monkeypatch.setattr(
        "cx_plugins.providers.snipd.cache.get_cached_transcript",
        lambda *_args, **_kwargs: cached_text,
    )
    monkeypatch.setattr(
        snipd,
        "resolve_snipd_clip",
        lambda _url: (_ for _ in ()).throw(AssertionError("should not fetch metadata")),
    )
    monkeypatch.setattr(
        transcription,
        "record_transcription_routing_summary",
        lambda **_kwargs: None,
    )

    docs = snipd_plugin.resolve(SNIPD_URL, {"overrides": {"transcribe": {}}})

    assert len(docs) == 1
    assert docs[0]["content"] == cached_text
    assert docs[0]["metadata"]["provider"] == "snipd"
    assert docs[0]["metadata"]["source_path"] == f"snipd:{CLIP_ID}"
    assert docs[0]["metadata"]["context_subpath"] == f"snipd-{CLIP_ID}.md"
    assert docs[0]["metadata"]["clip_id"] == CLIP_ID
    assert docs[0]["metadata"]["clip_start_seconds"] is None


def test_plugin_resolve_failure_doc_keeps_clip_identity(monkeypatch) -> None:
    class _BrokenReference:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("transcription unavailable")

    monkeypatch.setattr(snipd, "SnipdReference", _BrokenReference)

    docs = snipd_plugin.resolve(SNIPD_URL, {})

    assert len(docs) == 1
    assert docs[0]["metadata"]["provider"] == "snipd"
    assert docs[0]["metadata"]["source_path"] == f"snipd:{CLIP_ID}"
    assert docs[0]["metadata"]["clip_id"] == CLIP_ID
    assert docs[0]["metadata"]["resolution_error"] == "transcription unavailable"
