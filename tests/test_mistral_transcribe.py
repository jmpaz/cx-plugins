from __future__ import annotations

from contextualize.plugins.api import TranscriptionRequest
from cx_plugins.providers.transcribe import mistral


class _Response:
    status_code = 200
    text = ""

    def json(self) -> dict[str, object]:
        return {"text": "transcribed"}


def test_mistral_provider_sends_language_without_timestamps(monkeypatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "key")
    captured: dict[str, object] = {}

    def _post(*args: object, **kwargs: object) -> _Response:
        captured["data"] = kwargs.get("data")
        return _Response()

    monkeypatch.setattr(mistral, "_requests_post", _post)

    result = mistral.build_mistral_provider().transcribe(
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
        "model": "voxtral-mini-latest",
        "language": "es",
    }


def test_mistral_provider_keeps_timestamps_when_language_is_auto(monkeypatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "key")
    captured: dict[str, object] = {}

    def _post(*args: object, **kwargs: object) -> _Response:
        captured["data"] = kwargs.get("data")
        return _Response()

    monkeypatch.setattr(mistral, "_requests_post", _post)

    mistral.build_mistral_provider().transcribe(
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

    assert captured["data"] == {
        "model": "voxtral-mini-latest",
        "timestamp_granularities[]": "segment",
    }


def test_mistral_provider_uses_explicit_model_override(monkeypatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "key")
    monkeypatch.setenv("MISTRAL_MODEL", "voxtral-mini-latest")
    captured: dict[str, object] = {}

    def _post(*args: object, **kwargs: object) -> _Response:
        captured["data"] = kwargs.get("data")
        return _Response()

    monkeypatch.setattr(mistral, "_requests_post", _post)

    result = mistral.build_mistral_provider().transcribe(
        TranscriptionRequest(
            data=b"audio",
            filename="clip.mp3",
            content_type="audio/mpeg",
            timeout=30,
            language=None,
            model="voxtral-large-latest",
            prompt="",
            bias_terms=(),
            diarize=False,
            speaker_count=None,
        )
    )

    assert result.text == "transcribed"
    assert result.model == "voxtral-large-latest"
    assert captured["data"] == {
        "model": "voxtral-large-latest",
        "timestamp_granularities[]": "segment",
    }
