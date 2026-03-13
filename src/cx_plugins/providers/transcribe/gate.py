from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any
import sys

from contextualize.plugins.api import TranscriptionGate, TranscriptionGateDecision

_MODEL_ID = "pyannote/speaker-diarization-community-1"
_SCRIPT_PATH = Path(__file__).with_name("pyannote_script.py")


def build_pyannote_gate() -> TranscriptionGate:
    return TranscriptionGate(
        name="pyannote-community-1",
        analyze=_analyze_with_pyannote,
        cache_identity=_gate_cache_identity,
    )


def _log(message: str) -> None:
    try:
        from contextualize.runtime import get_verbose_logging

        if get_verbose_logging():
            print(f"[transcribe-gate] {message}", file=sys.stderr, flush=True)
    except Exception:
        return


def _analyze_with_pyannote(
    data: bytes,
    filename: str,
    content_type: str | None,
    timeout: float,
    config: dict[str, Any],
) -> TranscriptionGateDecision:
    del content_type
    with tempfile.TemporaryDirectory(prefix="cx-pyannote-gate-") as tmpdir:
        source_path = Path(tmpdir) / (Path(filename).name or "audio.bin")
        source_path.write_bytes(data)
        payload_path = Path(tmpdir) / "config.json"
        payload_path.write_text(json.dumps(config), encoding="utf-8")
        _log(
            f"running pyannote gate for {filename} "
            f"(auto_provider={config.get('auto_diarize_provider')})"
        )
        result = subprocess.run(
            [
                "uv",
                "--quiet",
                "run",
                "--no-project",
                str(_SCRIPT_PATH),
                str(source_path),
                str(payload_path),
            ],
            capture_output=True,
            text=True,
            env={
                **dict(os.environ),
                "CONTEXTUALIZE_VERBOSE_GATE": "1",
            },
            timeout=max(timeout, 120),
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise RuntimeError(f"pyannote gate failed: {detail}")
        if result.stderr.strip():
            _log(result.stderr.strip())
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"pyannote gate returned invalid JSON: {exc}") from exc

    _log(
        f"gate decision for {filename}: "
        f"needs_diarization={bool(payload.get('needs_diarization'))}, "
        f"speaker_count={payload.get('speaker_count')}"
    )
    return TranscriptionGateDecision(
        needs_diarization=bool(payload.get("needs_diarization")),
        speaker_count=_as_positive_int(payload.get("speaker_count")),
        confidence=_as_float(payload.get("confidence")),
        metadata=dict(payload.get("metadata") or {}),
    )


def _gate_cache_identity(config: dict[str, Any]) -> dict[str, Any]:
    prompt_hash = hashlib.sha256(
        json.dumps(
            {
                "provider": config.get("provider"),
                "priorities": config.get("priorities"),
                "prompt_parts": config.get("prompt_parts"),
                "prompt_files": config.get("prompt_files"),
                "auto_diarize_provider": config.get("auto_diarize_provider"),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "gate": "pyannote-community-1",
        "model": _MODEL_ID,
        "prompt_hash": prompt_hash,
    }


def _as_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
