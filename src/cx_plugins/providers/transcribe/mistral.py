from __future__ import annotations

import hashlib
import os

from contextualize.auth.common import load_dotenv_optional
from contextualize.plugins.api import (
    TranscriptionProvider,
    TranscriptionProviderAuthError,
    TranscriptionProviderError,
    TranscriptionProviderUnavailableError,
    TranscriptionRequest,
    TranscriptionResult,
)

_MISTRAL_AUTH_CODES = {401, 402, 403}


def _requests_post(*args: object, **kwargs: object):
    import requests

    return requests.post(*args, **kwargs)


def build_mistral_provider() -> TranscriptionProvider:
    return TranscriptionProvider(
        name="mistral",
        priority=100,
        transcribe=_transcribe_mistral,
        cache_identity=_mistral_cache_identity,
        is_available=_is_mistral_available,
    )


def _is_mistral_available() -> bool:
    load_dotenv_optional()
    return bool((os.environ.get("MISTRAL_API_KEY") or "").strip())


def _mistral_endpoint() -> str:
    load_dotenv_optional()
    value = (os.environ.get("MISTRAL_URL") or "").strip()
    return value or "https://api.mistral.ai/v1/audio/transcriptions"


def _mistral_model(request: TranscriptionRequest | None = None) -> str:
    if request is not None and request.model:
        return request.model
    load_dotenv_optional()
    value = (os.environ.get("MISTRAL_MODEL") or "").strip()
    return value or "voxtral-mini-latest"


def _normalize_segment(text: str) -> str:
    return " ".join(text.split())


def _extract_segments(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return []
    segments: list[str] = []
    raw_segments = payload.get("segments")
    if isinstance(raw_segments, list):
        for segment in raw_segments:
            if not isinstance(segment, dict):
                continue
            text = segment.get("text") or segment.get("delta")
            if not isinstance(text, str):
                continue
            normalized = _normalize_segment(text)
            if normalized:
                segments.append(normalized)
        if segments:
            return segments

    text = payload.get("text") or payload.get("delta")
    if isinstance(text, str):
        normalized = _normalize_segment(text)
        return [normalized] if normalized else []
    return segments


def _format_diarized_segments(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        return ""

    groups: list[tuple[str, str]] = []
    speaker_labels: dict[str, str] = {}
    next_speaker = 1
    for segment in raw_segments:
        if not isinstance(segment, dict):
            continue
        speaker = segment.get("speaker_id")
        if not isinstance(speaker, str):
            return ""
        text = segment.get("text")
        if not isinstance(text, str):
            continue
        normalized = _normalize_segment(text)
        if not normalized:
            continue
        label = speaker_labels.get(speaker)
        if label is None:
            label = f"Speaker {next_speaker}"
            speaker_labels[speaker] = label
            next_speaker += 1
        if groups and groups[-1][0] == label:
            groups[-1] = (label, groups[-1][1] + " " + normalized)
        else:
            groups.append((label, normalized))

    if not groups:
        return ""
    return "\n\n".join(f"[{speaker}] {text}" for speaker, text in groups)


def _transcribe_mistral(request: TranscriptionRequest) -> TranscriptionResult:
    load_dotenv_optional()
    api_key = (os.environ.get("MISTRAL_API_KEY") or "").strip()
    if not api_key:
        raise TranscriptionProviderUnavailableError(
            "Mistral transcription requires MISTRAL_API_KEY"
        )

    model = _mistral_model(request)
    data = {"model": model}
    if request.language:
        data["language"] = request.language
    else:
        data["timestamp_granularities[]"] = "segment"
    if request.diarize:
        data["diarize"] = "true"
    if request.bias_terms:
        data["context_bias"] = ", ".join(request.bias_terms)

    response = _requests_post(
        _mistral_endpoint(),
        headers={"Authorization": f"Bearer {api_key}"},
        files={
            "file": (
                request.filename,
                request.data,
                request.content_type or "application/octet-stream",
            )
        },
        data=data,
        timeout=request.timeout,
    )
    if response.status_code in _MISTRAL_AUTH_CODES:
        raise TranscriptionProviderAuthError(
            f"Mistral transcription failed: {response.status_code} {response.text}",
            status_code=response.status_code,
        )
    if response.status_code >= 400:
        raise TranscriptionProviderError(
            f"Mistral transcription failed: {response.status_code} {response.text}",
            retryable=response.status_code >= 500 or response.status_code == 429,
            status_code=response.status_code,
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise TranscriptionProviderError(
            "Mistral transcription response was not valid JSON"
        ) from exc

    if request.diarize:
        diarized = _format_diarized_segments(payload)
        if diarized:
            return TranscriptionResult(
                text=diarized,
                model=model,
                provider="mistral",
            )

    segments = _extract_segments(payload)
    text = "\n\n".join(segments) if segments else ""
    if not text:
        raise TranscriptionProviderError("Mistral transcription returned no text")
    return TranscriptionResult(text=text, model=model, provider="mistral")


def _mistral_cache_identity(request: TranscriptionRequest) -> dict[str, object]:
    prompt_hash = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()
    bias_hash = hashlib.sha256(",".join(request.bias_terms).encode("utf-8")).hexdigest()
    return {
        "provider": "mistral",
        "endpoint": _mistral_endpoint(),
        "model": _mistral_model(request),
        "language": request.language,
        "prompt_hash": prompt_hash,
        "bias_hash": bias_hash,
        "diarize": request.diarize,
        "speaker_count": request.speaker_count,
    }
