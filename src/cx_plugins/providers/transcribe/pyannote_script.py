# /// script
# dependencies = [
#   "pyannote.audio>=4.0.0",
#   "python-dotenv>=1.0.0",
# ]
# ///

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import warnings


MODEL_ID = "pyannote/speaker-diarization-community-1"


def load_dotenv_optional() -> None:
    try:
        from dotenv import find_dotenv, load_dotenv

        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=False)
    except Exception:
        return


def hf_token() -> str | None:
    load_dotenv_optional()
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    return None


def log(message: str) -> None:
    if (os.environ.get("CONTEXTUALIZE_VERBOSE_GATE") or "").strip() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    print(f"[pyannote-gate] {message}", file=sys.stderr, flush=True)


def configure_warning_filters() -> None:
    # These warnings are useful when developing pyannote itself, but they drown
    # out the operator-facing gate logs during normal use.
    warnings.filterwarnings(
        "ignore",
        message=r".*TensorFloat-32 \(TF32\) has been disabled.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*std\(\): degrees of freedom is <= 0.*",
    )


def device_name() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _speaker_annotation(diarization):
    annotation = getattr(diarization, "speaker_diarization", None)
    return annotation if annotation is not None else diarization


def _exclusive_annotation(diarization):
    annotation = getattr(diarization, "exclusive_speaker_diarization", None)
    return annotation if annotation is not None else _speaker_annotation(diarization)


def _media_duration_seconds(path: Path) -> float | None:
    ffprobe = shutil_which("ffprobe")
    if ffprobe is None:
        return None
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return None
    return duration if duration > 0 else None


def shutil_which(name: str) -> str | None:
    from shutil import which

    return which(name)


def _sample_windows(duration: float | None) -> list[tuple[float, float]]:
    if duration is None or duration <= 45.0:
        return [(0.0, -1.0)]

    window = 20.0
    windows: list[tuple[float, float]] = [(0.0, window)]
    if duration > 90.0:
        middle_start = max(0.0, (duration / 2.0) - (window / 2.0))
        windows.append((middle_start, window))
    tail_start = max(0.0, duration - window)
    if all(abs(start - tail_start) > 1.0 for start, _ in windows):
        windows.append((tail_start, window))
    return windows


def _prepare_analysis_paths(audio_path: Path) -> list[Path]:
    duration = _media_duration_seconds(audio_path)
    windows = _sample_windows(duration)
    ffmpeg = shutil_which("ffmpeg")
    if ffmpeg is None or windows == [(0.0, -1.0)]:
        if duration is not None:
            log(f"analyzing full file ({duration:.1f}s)")
        else:
            log("analyzing full file (duration unknown)")
        return [audio_path]

    prepared: list[Path] = []
    tmpdir = tempfile.mkdtemp(prefix="pyannote-slices-")
    for index, (start, length) in enumerate(windows, start=1):
        out = Path(tmpdir) / f"sample-{index:02d}.wav"
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            "16000",
        ]
        if length > 0:
            cmd.extend(["-t", f"{length:.3f}"])
        cmd.extend(["-c:a", "pcm_s16le", str(out)])
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not out.exists():
            log(
                f"failed to build sample window start={start:.1f}s length={length:.1f}s; "
                "falling back to full file"
            )
            return [audio_path]
        log(
            f"prepared sample window {index}/{len(windows)} start={start:.1f}s "
            + (f"length={length:.1f}s" if length > 0 else "full")
        )
        prepared.append(out)
    return prepared


def main() -> int:
    configure_warning_filters()
    if len(sys.argv) != 3:
        print("expected audio path and config path", file=sys.stderr)
        return 2
    audio_path = Path(sys.argv[1])
    config_path = Path(sys.argv[2])
    config = json.loads(config_path.read_text(encoding="utf-8"))

    from pyannote.audio import Pipeline

    token = hf_token()
    if not token:
        print(
            "pyannote auto-diarization requires HF_TOKEN or HUGGINGFACE_HUB_TOKEN",
            file=sys.stderr,
        )
        return 1

    log(f"loading pipeline {MODEL_ID}")
    pipeline = Pipeline.from_pretrained(MODEL_ID, token=token)
    try:
        import torch

        pipeline.to(torch.device(device_name()))
    except Exception:
        pass

    analysis_paths = _prepare_analysis_paths(audio_path)
    labels: set[str] = set()
    speaker_count = 0
    turn_count = 0
    exclusive_turn_count = 0
    for index, candidate in enumerate(analysis_paths, start=1):
        log(
            f"running diarization on sample {index}/{len(analysis_paths)} "
            f"({candidate.name}) using device={device_name()}"
        )
        diarization = pipeline(str(candidate))
        speaker_annotation = _speaker_annotation(diarization)
        exclusive_annotation = _exclusive_annotation(diarization)
        current_labels = {str(label) for label in speaker_annotation.labels()}
        labels.update(current_labels)
        speaker_count = max(speaker_count, len(current_labels))
        turn_count = max(
            turn_count, sum(1 for _ in speaker_annotation.itertracks(yield_label=True))
        )
        exclusive_turn_count = max(
            exclusive_turn_count,
            sum(1 for _ in exclusive_annotation.itertracks(yield_label=True)),
        )
        if speaker_count > 1:
            log("multi-speaker detected; stopping early")
            break
    payload = {
        "needs_diarization": speaker_count > 1,
        "speaker_count": speaker_count if speaker_count > 1 else None,
        "confidence": None,
        "metadata": {
            "model": MODEL_ID,
            "device": device_name(),
            "turn_count": turn_count,
            "exclusive_turn_count": exclusive_turn_count,
            "labels": sorted(labels),
            "samples_analyzed": len(analysis_paths),
            "auto_diarize_provider": config.get("auto_diarize_provider"),
        },
    }
    log(
        f"decision needs_diarization={payload['needs_diarization']} "
        f"speaker_count={payload['speaker_count']}"
    )
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
