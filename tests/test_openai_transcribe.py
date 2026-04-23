from __future__ import annotations

import pytest

from contextualize.plugins.api import TranscriptionProviderError, TranscriptionRequest
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
    monkeypatch.setenv("WHISPER_URL", "http://localhost:9001/v1/audio/transcriptions")
    monkeypatch.setenv("WHISPER_MODEL", "CohereLabs/cohere-transcribe-03-2026")
    monkeypatch.delenv("WHISPER_API_BASE", raising=False)
    monkeypatch.delenv("WHISPER_API_KEY", raising=False)


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
                prompt="",
                bias_terms=(),
                diarize=False,
                speaker_count=None,
            )
        )
