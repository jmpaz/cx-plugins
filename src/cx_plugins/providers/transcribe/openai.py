from __future__ import annotations

import hashlib
import mimetypes
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.parse import quote, urlsplit, urlunsplit

from contextualize.auth.common import load_dotenv_optional
from contextualize.plugins.api import (
    TranscriptionProvider,
    TranscriptionProviderAuthError,
    TranscriptionProviderError,
    TranscriptionProviderUnavailableError,
    TranscriptionRequest,
    TranscriptionResult,
)

_AUDIO_SUFFIX_TO_MIME: dict[str, str] = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".aiff": "audio/aiff",
}


def _requests_post(*args: object, **kwargs: object):
    import requests

    return requests.post(*args, **kwargs)


def _requests_delete(*args: object, **kwargs: object):
    import requests

    return requests.delete(*args, **kwargs)


def build_openai_provider() -> TranscriptionProvider:
    return TranscriptionProvider(
        name="openai",
        priority=200,
        transcribe=_transcribe_openai,
        cache_identity=_openai_cache_identity,
        is_available=_is_openai_available,
    )


def _is_openai_available() -> bool:
    load_dotenv_optional()
    return bool(
        (os.environ.get("OPENAI_TRANSCRIPTION_API_BASE") or "").strip()
        or (os.environ.get("OPENAI_TRANSCRIPTION_URL") or "").strip()
        or (os.environ.get("OPENAI_TRANSCRIPTION_API_KEY") or "").strip()
    )


def _openai_endpoint() -> str:
    load_dotenv_optional()
    direct = (os.environ.get("OPENAI_TRANSCRIPTION_URL") or "").strip()
    if direct:
        return direct
    api_base = (os.environ.get("OPENAI_TRANSCRIPTION_API_BASE") or "").strip()
    if not api_base:
        return "https://api.openai.com/v1/audio/transcriptions"
    return f"{api_base.rstrip('/')}/audio/transcriptions"


def _openai_model(request: TranscriptionRequest | None = None) -> str | None:
    if request is not None and request.model:
        return request.model
    load_dotenv_optional()
    value = (os.environ.get("OPENAI_TRANSCRIPTION_MODEL") or "").strip()
    return value or None


def _openai_auth_headers() -> dict[str, str]:
    load_dotenv_optional()
    headers: dict[str, str] = {}
    api_key = (os.environ.get("OPENAI_TRANSCRIPTION_API_KEY") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _openai_model_endpoint(model: str) -> str | None:
    endpoint = _openai_endpoint()
    split = urlsplit(endpoint)
    path = split.path.rstrip("/")
    for suffix in ("/audio/transcriptions", "/audio/translations"):
        if path.endswith(suffix):
            base_path = path[: -len(suffix)]
            model_path = quote(model.strip().lstrip("/"), safe="/:@._-")
            return urlunsplit(
                split._replace(
                    path=f"{base_path}/models/{model_path}",
                    query="",
                    fragment="",
                )
            )
    return None


def _chunk_seconds_cache_component() -> str:
    load_dotenv_optional()
    value = (os.environ.get("OPENAI_TRANSCRIPTION_CHUNK_SECONDS") or "").strip()
    return value or "25"


def _guess_audio_content_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in _AUDIO_SUFFIX_TO_MIME:
        return _AUDIO_SUFFIX_TO_MIME[suffix]
    guessed, _ = mimetypes.guess_type(filename)
    if guessed and guessed.startswith("audio/"):
        return guessed
    return "audio/mpeg"


def _extract_transcription_text(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""

    raw_segments = payload.get("segments")
    if isinstance(raw_segments, list):
        extracted: list[str] = []
        for segment in raw_segments:
            if not isinstance(segment, dict):
                continue
            text = segment.get("text")
            if not isinstance(text, str):
                continue
            normalized = " ".join(text.split())
            if normalized:
                extracted.append(normalized)
        if extracted:
            return "\n\n".join(extracted)

    text = payload.get("text")
    if isinstance(text, str):
        return text.strip()
    return ""


def _merge_prompt(request_prompt: str) -> str:
    load_dotenv_optional()
    env_prompt = (os.environ.get("OPENAI_TRANSCRIPTION_PROMPT") or "").strip()
    parts = [part for part in (env_prompt, request_prompt.strip()) if part]
    return "\n\n".join(parts)


def _transcribe_openai(request: TranscriptionRequest) -> TranscriptionResult:
    if not _is_openai_available():
        raise TranscriptionProviderUnavailableError(
            "OpenAI-compatible transcription requires OPENAI_TRANSCRIPTION_API_BASE, "
            "OPENAI_TRANSCRIPTION_URL, or OPENAI_TRANSCRIPTION_API_KEY"
        )

    content_type = request.content_type or _guess_audio_content_type(request.filename)
    model = _openai_model(request)
    merged_prompt = _merge_prompt(request.prompt)
    try:
        text = _transcribe_openai_once_with_model_repair(
            request.data,
            filename=request.filename,
            content_type=content_type,
            timeout=request.timeout,
            model=model,
            language=request.language,
            prompt=merged_prompt,
        )
    except TranscriptionProviderError as exc:
        if not _should_retry_chunked_transcription(exc):
            raise
        text = _transcribe_audio_in_chunks(
            request.data,
            filename=request.filename,
            timeout=request.timeout,
            model=model,
            language=request.language,
            prompt=merged_prompt,
        )
    return TranscriptionResult(
        text=text,
        model=model or "server-default",
        provider="openai",
    )


def _transcribe_openai_once_with_model_repair(
    data: bytes,
    *,
    filename: str,
    content_type: str,
    timeout: float,
    model: str | None,
    language: str | None,
    prompt: str,
) -> str:
    try:
        return _transcribe_openai_once(
            data,
            filename=filename,
            content_type=content_type,
            timeout=timeout,
            model=model,
            language=language,
            prompt=prompt,
        )
    except TranscriptionProviderError as exc:
        if not _should_recreate_model_after_error(exc):
            raise
        if model is None:
            raise
        try:
            _recreate_openai_model(model, timeout=timeout)
        except TranscriptionProviderError as refresh_exc:
            raise TranscriptionProviderError(
                "OpenAI-compatible transcription hit a stale local model cache "
                f"for {model!r}, but automatic model refresh failed: {refresh_exc}",
                retryable=False,
            ) from refresh_exc
        return _transcribe_openai_once(
            data,
            filename=filename,
            content_type=content_type,
            timeout=timeout,
            model=model,
            language=language,
            prompt=prompt,
        )


def _transcribe_openai_once(
    data: bytes,
    *,
    filename: str,
    content_type: str,
    timeout: float,
    model: str | None,
    language: str | None,
    prompt: str,
) -> str:
    load_dotenv_optional()
    endpoint = _openai_endpoint()
    headers = _openai_auth_headers()

    form_data: dict[str, str] = {"response_format": "verbose_json"}
    if model:
        form_data["model"] = model
    if language:
        form_data["language"] = language
    if prompt:
        form_data["prompt"] = prompt

    response = _requests_post(
        endpoint,
        headers=headers,
        files={"file": (filename, data, content_type)},
        data=form_data,
        timeout=timeout,
    )
    if response.status_code in {401, 402, 403}:
        raise TranscriptionProviderAuthError(
            f"OpenAI-compatible transcription failed: {response.status_code} {response.text}",
            status_code=response.status_code,
        )
    if response.status_code >= 400:
        raise TranscriptionProviderError(
            f"OpenAI-compatible transcription failed: {response.status_code} {response.text}",
            retryable=response.status_code >= 500 or response.status_code == 429,
            status_code=response.status_code,
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise TranscriptionProviderError(
            "OpenAI-compatible transcription response was not valid JSON"
        ) from exc
    text = _extract_transcription_text(payload)
    if not text:
        raise TranscriptionProviderError(
            "OpenAI-compatible transcription returned no transcription text"
        )
    return text


def _should_recreate_model_after_error(exc: TranscriptionProviderError) -> bool:
    if exc.status_code != 500:
        return False
    normalized = str(exc).lower()
    return "valid model card" in normalized and "delete /v1/models" in normalized


def _recreate_openai_model(model: str, *, timeout: float) -> None:
    model_endpoint = _openai_model_endpoint(model)
    if model_endpoint is None:
        raise TranscriptionProviderError(
            "could not derive a /models endpoint from OPENAI_TRANSCRIPTION_URL or "
            "OPENAI_TRANSCRIPTION_API_BASE"
        )

    headers = _openai_auth_headers()
    delete_response = _requests_delete(
        model_endpoint,
        headers=headers,
        timeout=max(10.0, min(timeout, 60.0)),
    )
    if delete_response.status_code >= 400 and delete_response.status_code != 404:
        raise TranscriptionProviderError(
            "model delete failed: "
            f"{delete_response.status_code} {delete_response.text}",
            retryable=delete_response.status_code >= 500,
            status_code=delete_response.status_code,
        )

    create_response = _requests_post(
        model_endpoint,
        headers=headers,
        timeout=max(timeout, 60.0),
    )
    if create_response.status_code >= 400:
        raise TranscriptionProviderError(
            "model download failed: "
            f"{create_response.status_code} {create_response.text}",
            retryable=create_response.status_code >= 500,
            status_code=create_response.status_code,
        )


def _should_retry_chunked_transcription(exc: TranscriptionProviderError) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    return bool(exc.status_code and exc.status_code >= 500)


def _transcribe_audio_in_chunks(
    data: bytes,
    *,
    filename: str,
    timeout: float,
    model: str | None,
    language: str | None,
    prompt: str,
) -> str:
    chunk_seconds = _get_chunk_seconds()
    suffix = Path(filename).suffix.lower() or ".wav"
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / f"source{suffix}"
        source_path.write_bytes(data)
        pattern = str(Path(tmpdir) / "chunk_%04d.wav")
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "segment",
                "-segment_time",
                str(chunk_seconds),
                "-c:a",
                "pcm_s16le",
                pattern,
            ],
            capture_output=True,
            text=True,
            timeout=max(timeout, 120),
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise TranscriptionProviderError(f"ffmpeg audio chunking failed: {detail}")

        chunk_paths = sorted(Path(tmpdir).glob("chunk_*.wav"))
        if not chunk_paths:
            raise TranscriptionProviderError("ffmpeg audio chunking produced no chunks")

        parts: list[str] = []
        for chunk_path in chunk_paths:
            text = _transcribe_openai_once_with_model_repair(
                chunk_path.read_bytes(),
                filename=chunk_path.name,
                content_type="audio/wav",
                timeout=timeout,
                model=model,
                language=language,
                prompt=prompt,
            ).strip()
            if text:
                parts.append(text)
        return "\n\n".join(parts)


def _get_chunk_seconds() -> int:
    raw = os.environ.get("OPENAI_TRANSCRIPTION_CHUNK_SECONDS", "25")
    try:
        value = int(raw)
    except ValueError:
        raise TranscriptionProviderError(
            f"Invalid OPENAI_TRANSCRIPTION_CHUNK_SECONDS value: {raw!r}. "
            "Expected a positive integer."
        ) from None
    if value <= 0:
        raise TranscriptionProviderError(
            f"Invalid OPENAI_TRANSCRIPTION_CHUNK_SECONDS value: {raw!r}. "
            "Expected a positive integer."
        )
    return value


def _openai_cache_identity(request: TranscriptionRequest) -> dict[str, object]:
    prompt_hash = hashlib.sha256(
        _merge_prompt(request.prompt).encode("utf-8")
    ).hexdigest()
    return {
        "provider": "openai",
        "endpoint": _openai_endpoint(),
        "model": _openai_model(request),
        "chunk_seconds": _chunk_seconds_cache_component(),
        "language": request.language,
        "prompt_hash": prompt_hash,
        "diarize": request.diarize,
        "speaker_count": request.speaker_count,
    }
