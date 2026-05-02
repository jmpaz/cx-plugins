from __future__ import annotations

import pytest

from contextualize.plugins.api import (
    TranscriptionProviderError,
    TranscriptionRequest,
)
from cx_plugins.providers.transcribe import openai


class _Response:
    def __init__(
        self,
        status_code: int,
        *,
        text: str = "",
        payload: dict[str, object] | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self._payload = payload

    def json(self) -> dict[str, object]:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


@pytest.fixture
def openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "OPENAI_TRANSCRIPTION_URL", "http://localhost:9001/v1/audio/transcriptions"
    )
    monkeypatch.delenv("OPENAI_TRANSCRIPTION_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_TRANSCRIPTION_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_TRANSCRIPTION_MODEL", raising=False)


def test_openai_provider_refreshes_stale_speaches_model(
    openai_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_url = "http://localhost:9001/v1/models/CohereLabs/cohere-transcribe-03-2026"
    stale_model_error = (
        '{"detail":"The model repository does not contain a valid model card. '
        "You should try to delete the model and re-download it using "
        "`DELETE /v1/models/CohereLabs/cohere-transcribe-03-2026` and then "
        '`POST /v1/models`."}'
    )
    post_calls: list[str] = []
    delete_calls: list[str] = []

    def _post(url: str, **kwargs: object) -> _Response:
        post_calls.append(url)
        if len(post_calls) == 1:
            return _Response(500, text=stale_model_error)
        if len(post_calls) == 2:
            return _Response(200, text="{}")
        return _Response(200, payload={"text": "transcribed"})

    def _delete(url: str, **kwargs: object) -> _Response:
        delete_calls.append(url)
        return _Response(200, text="{}")

    monkeypatch.setattr(openai, "_requests_post", _post)
    monkeypatch.setattr(openai, "_requests_delete", _delete)

    result = openai.build_openai_provider().transcribe(
        TranscriptionRequest(
            data=b"audio",
            filename="clip.mp3",
            content_type="audio/mpeg",
            timeout=30,
            language="en",
            model="CohereLabs/cohere-transcribe-03-2026",
            prompt="",
            bias_terms=(),
            diarize=False,
            speaker_count=None,
        )
    )

    assert result.text == "transcribed"
    assert delete_calls == [model_url]
    assert post_calls == [
        "http://localhost:9001/v1/audio/transcriptions",
        model_url,
        "http://localhost:9001/v1/audio/transcriptions",
    ]


def test_openai_provider_keeps_regular_500s_on_chunk_fallback_path(
    openai_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _post(url: str, **kwargs: object) -> _Response:
        return _Response(500, text="ordinary server error")

    def _delete(url: str, **kwargs: object) -> _Response:
        raise AssertionError("regular 500 should not refresh the model")

    monkeypatch.setattr(openai, "_requests_post", _post)
    monkeypatch.setattr(openai, "_requests_delete", _delete)
    monkeypatch.setattr(openai.shutil, "which", lambda _name: None)

    with pytest.raises(TranscriptionProviderError, match="ordinary server error"):
        openai.build_openai_provider().transcribe(
            TranscriptionRequest(
                data=b"audio",
                filename="clip.mp3",
                content_type="audio/mpeg",
                timeout=30,
                language="en",
                prompt="",
                bias_terms=(),
                diarize=False,
                speaker_count=None,
            )
        )


def test_openai_provider_reports_failed_model_refresh_without_chunking(
    openai_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_model_error = (
        '{"detail":"The model repository does not contain a valid model card. '
        "You should try to delete the model and re-download it using "
        '`DELETE /v1/models/CohereLabs/cohere-transcribe-03-2026`."}'
    )

    def _post(url: str, **kwargs: object) -> _Response:
        return _Response(500, text=stale_model_error)

    def _delete(url: str, **kwargs: object) -> _Response:
        return _Response(500, text="delete failed")

    def _run_ffmpeg(*args: object, **kwargs: object) -> object:
        raise AssertionError("failed model refresh should not chunk audio")

    monkeypatch.setattr(openai, "_requests_post", _post)
    monkeypatch.setattr(openai, "_requests_delete", _delete)
    monkeypatch.setattr(openai.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(openai.subprocess, "run", _run_ffmpeg)

    with pytest.raises(TranscriptionProviderError, match="model refresh failed"):
        openai.build_openai_provider().transcribe(
            TranscriptionRequest(
                data=b"audio",
                filename="clip.mp3",
                content_type="audio/mpeg",
                timeout=30,
                language="en",
                model="CohereLabs/cohere-transcribe-03-2026",
                prompt="",
                bias_terms=(),
                diarize=False,
                speaker_count=None,
            )
        )


def test_openai_provider_sends_language(
    openai_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _post(url: str, **kwargs: object) -> _Response:
        captured["data"] = kwargs.get("data")
        return _Response(200, payload={"text": "transcribed"})

    monkeypatch.setattr(openai, "_requests_post", _post)

    result = openai.build_openai_provider().transcribe(
        TranscriptionRequest(
            data=b"audio",
            filename="clip.mp3",
            content_type="audio/mpeg",
            timeout=30,
            language="es",
            prompt="",
            bias_terms=(),
            diarize=False,
            speaker_count=None,
        )
    )

    assert result.text == "transcribed"
    assert captured["data"] == {
        "response_format": "verbose_json",
        "language": "es",
    }


def test_openai_provider_omits_language_when_unspecified(
    openai_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _post(url: str, **kwargs: object) -> _Response:
        captured["data"] = kwargs.get("data")
        return _Response(200, payload={"text": "transcribed"})

    monkeypatch.setattr(openai, "_requests_post", _post)

    result = openai.build_openai_provider().transcribe(
        TranscriptionRequest(
            data=b"audio",
            filename="clip.mp3",
            content_type="audio/mpeg",
            timeout=30,
            language=None,
            prompt="",
            bias_terms=(),
            diarize=False,
            speaker_count=None,
        )
    )

    assert result.text == "transcribed"
    assert captured["data"] == {
        "response_format": "verbose_json",
    }


def test_openai_provider_sends_explicit_model(
    openai_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _post(url: str, **kwargs: object) -> _Response:
        captured["data"] = kwargs.get("data")
        return _Response(200, payload={"text": "transcribed"})

    monkeypatch.setattr(openai, "_requests_post", _post)

    result = openai.build_openai_provider().transcribe(
        TranscriptionRequest(
            data=b"audio",
            filename="clip.mp3",
            content_type="audio/mpeg",
            timeout=30,
            language=None,
            model="cohere",
            prompt="",
            bias_terms=(),
            diarize=False,
            speaker_count=None,
        )
    )

    assert result.text == "transcribed"
    assert result.model == "cohere"
    assert captured["data"] == {
        "model": "cohere",
        "response_format": "verbose_json",
    }


def test_openai_provider_sends_env_model_when_request_omits_model(
    openai_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_TRANSCRIPTION_MODEL", "distilwhisper")
    captured: dict[str, object] = {}

    def _post(url: str, **kwargs: object) -> _Response:
        captured["data"] = kwargs.get("data")
        return _Response(200, payload={"text": "transcribed"})

    monkeypatch.setattr(openai, "_requests_post", _post)

    result = openai.build_openai_provider().transcribe(
        TranscriptionRequest(
            data=b"audio",
            filename="clip.mp3",
            content_type="audio/mpeg",
            timeout=30,
            language=None,
            prompt="",
            bias_terms=(),
            diarize=False,
            speaker_count=None,
        )
    )

    assert result.text == "transcribed"
    assert result.model == "distilwhisper"
    assert captured["data"] == {
        "model": "distilwhisper",
        "response_format": "verbose_json",
    }


def test_openai_provider_lists_configured_server_models(
    openai_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _get(url: str, **kwargs: object) -> _Response:
        captured["url"] = url
        return _Response(
            200,
            payload={
                "object": "list",
                "data": [
                    {
                        "id": "cohere",
                        "model_id": "CohereLabs/cohere-transcribe-03-2026",
                        "aliases": ["cohere-transcribe"],
                    },
                    {
                        "id": "distilwhisper",
                        "model_id": "distil-whisper/distil-large-v3",
                        "aliases": ["distil-whisper/distil-large-v3"],
                    },
                ],
            },
        )

    monkeypatch.setattr(openai, "_requests_get", _get)

    assert openai.list_openai_model_options("distil") == [
        "distil-whisper/distil-large-v3",
        "distilwhisper",
    ]
    assert captured["url"] == "http://localhost:9001/v1/models"


def test_openai_cache_identity_varies_by_model(openai_env: None) -> None:
    provider = openai.build_openai_provider()
    first = TranscriptionRequest(
        data=b"audio",
        filename="clip.mp3",
        content_type="audio/mpeg",
        timeout=30,
        language=None,
        model="distilwhisper",
        prompt="",
        bias_terms=(),
        diarize=False,
        speaker_count=None,
    )
    second = TranscriptionRequest(
        data=b"audio",
        filename="clip.mp3",
        content_type="audio/mpeg",
        timeout=30,
        language=None,
        model="cohere",
        prompt="",
        bias_terms=(),
        diarize=False,
        speaker_count=None,
    )

    assert provider.cache_identity(first)["model"] == "distilwhisper"
    assert provider.cache_identity(second)["model"] == "cohere"
