from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

import click

from .gate import build_pyannote_gate
from .mistral import build_mistral_provider
from .openai import build_openai_provider

PLUGIN_API_VERSION = "1"
PLUGIN_NAME = "transcribe"
PLUGIN_PRIORITY = 250
PLUGIN_KIND = "processor"
_PROVIDER_ALIASES = {
    "auto": "auto",
    "mistral": "mistral",
    "openai": "openai",
}


def can_resolve(_target: str, _context: dict[str, Any]) -> bool:
    return False


def resolve(_target: str, _context: dict[str, Any]) -> list[dict[str, Any]]:
    return []


def _normalize_provider_name(raw: Any, *, allow_auto: bool = True) -> str | None:
    if not isinstance(raw, str):
        return None
    normalized = _PROVIDER_ALIASES.get(raw.strip().lower())
    if normalized == "auto" and not allow_auto:
        return None
    return normalized


def _coerce_model(raw: Any) -> str | None:
    if raw in (None, ""):
        return None
    if not isinstance(raw, str):
        raise ValueError("model must be a string")
    model = raw.strip()
    return model or None


def _coerce_prompt_parts(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray, str)):
        values: list[str] = []
        for item in raw:
            if not isinstance(item, str):
                raise ValueError("prompt entries must be strings")
            text = item.strip()
            if text:
                values.append(text)
        return values
    raise ValueError("prompt must be a string or list of strings")


def _coerce_prompt_files(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray, str)):
        values: list[str] = []
        for item in raw:
            if not isinstance(item, str):
                raise ValueError("prompt_files entries must be strings")
            text = item.strip()
            if text:
                values.append(text)
        return values
    raise ValueError("prompt_files must be a string or list of strings")


def _coerce_priorities(raw: Any) -> dict[str, int]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("priorities must be a mapping")
    priorities: dict[str, int] = {}
    for name, value in raw.items():
        normalized_name = _normalize_provider_name(name, allow_auto=False)
        if normalized_name is None:
            raise ValueError(f"unsupported transcription provider {name!r}")
        try:
            priorities[normalized_name] = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("priority values must be integers") from exc
    return priorities


def _coerce_language(raw: Any) -> str | None:
    if raw in (None, ""):
        return None
    if not isinstance(raw, str):
        raise ValueError("language must be a string")
    language = raw.strip().lower()
    if language in {"", "auto"}:
        return None
    if not re.fullmatch(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})?", language):
        raise ValueError("language must be a BCP-47 or ISO language code")
    return language


def _coerce_diarize(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError("diarize must be a boolean")


def _coerce_speakers(raw: Any) -> int | None:
    if raw in (None, ""):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("speakers must be a positive integer") from exc
    if value <= 0:
        raise ValueError("speakers must be a positive integer")
    return value


def normalize_manifest_config(
    raw_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if raw_config is None:
        return None
    if not isinstance(raw_config, dict):
        raise ValueError("transcribe config must be a mapping")

    normalized: dict[str, Any] = {}
    raw_provider = raw_config.get("provider")
    if isinstance(raw_provider, str) and not raw_provider.strip():
        raw_provider = None
    provider = _normalize_provider_name(raw_provider)
    if raw_provider not in (None, "") and provider is None:
        raise ValueError(f"unsupported transcription provider {raw_provider!r}")
    if provider:
        normalized["provider"] = provider

    model = _coerce_model(raw_config.get("model"))
    if model:
        normalized["model"] = model

    language = _coerce_language(raw_config.get("language"))
    if language:
        normalized["language"] = language

    priorities = _coerce_priorities(raw_config.get("priorities"))
    if priorities:
        normalized["priorities"] = priorities

    prompt_parts = _coerce_prompt_parts(
        raw_config.get("prompt_parts", raw_config.get("prompt"))
    )
    if prompt_parts:
        normalized["prompt_parts"] = prompt_parts

    prompt_files = _coerce_prompt_files(
        raw_config.get("prompt_files", raw_config.get("prompt_file"))
    )
    if prompt_files:
        normalized["prompt_files"] = prompt_files

    if "diarize" in raw_config:
        normalized["diarize"] = _coerce_diarize(raw_config.get("diarize"))

    speakers = _coerce_speakers(raw_config.get("speakers"))
    if speakers is not None:
        normalized["speakers"] = speakers

    if "auto_diarize" in raw_config:
        normalized["auto_diarize"] = _coerce_diarize(raw_config.get("auto_diarize"))

    auto_provider = raw_config.get("auto_diarize_provider")
    if isinstance(auto_provider, str) and auto_provider.strip():
        provider = auto_provider.strip().lower()
        if provider not in {"mistral", "local"}:
            raise ValueError("auto_diarize_provider must be 'mistral' or 'local'")
        normalized["auto_diarize_provider"] = provider

    return normalized or None


def _option_already_registered(command: click.Command, *, name: str) -> bool:
    return any(getattr(param, "name", None) == name for param in command.params)


def _append_option(command: click.Command, option: click.Option) -> None:
    if _option_already_registered(command, name=option.name):
        return
    command.params.append(option)


def register_cli_options(command_name: str, command: click.Command) -> None:
    if command_name not in {"cat", "hydrate"}:
        return
    _append_option(
        command,
        click.Option(
            ["--transcribe-provider"],
            type=click.Choice(["auto", "openai", "mistral"]),
            default="auto",
            help="Choose the transcription provider for audio/video inputs.",
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--transcribe-model"],
            default=None,
            help="Set the transcription model; omit to use the provider or server default.",
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--transcribe-priority"],
            multiple=True,
            help="Override provider priority with NAME=INTEGER (repeatable).",
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--transcribe-language"],
            default=None,
            help="Set the spoken language code for transcription, e.g. es or en.",
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--transcribe-prompt"],
            multiple=True,
            help="Append transcription prompt or dictionary text (repeatable).",
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--transcribe-prompt-file"],
            multiple=True,
            type=click.Path(exists=True, dir_okay=False),
            help="Read transcription prompt or dictionary text from a file (repeatable).",
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--transcribe-diarize/--no-transcribe-diarize"],
            default=False,
            help="Enable or disable diarization for supported providers.",
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--transcribe-speakers"],
            type=int,
            default=None,
            help="Hint the expected number of speakers for multi-speaker dialogue.",
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--transcribe-auto-diarize"],
            is_flag=True,
            default=False,
            help="Run a local diarization gate and only escalate files that appear multi-speaker.",
        ),
    )
    _append_option(
        command,
        click.Option(
            ["--transcribe-auto-diarize-provider"],
            type=click.Choice(["mistral", "local"]),
            default="mistral",
            help="Provider to use when auto-diarization decides a file needs diarization.",
        ),
    )


def _parse_priority_items(values: Sequence[str]) -> dict[str, int]:
    priorities: dict[str, int] = {}
    for raw in values:
        item = raw.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("--transcribe-priority expects NAME=INTEGER entries")
        name, priority_raw = item.split("=", 1)
        normalized_name = _normalize_provider_name(name, allow_auto=False)
        if normalized_name is None:
            raise ValueError(f"unsupported transcription provider {name!r}")
        try:
            priority = int(priority_raw)
        except ValueError as exc:
            raise ValueError("--transcribe-priority values must be integers") from exc
        priorities[normalized_name] = priority
    return priorities


def collect_cli_overrides(
    command_name: str,
    params: dict[str, Any],
) -> dict[str, Any] | None:
    if command_name not in {"cat", "hydrate"}:
        return None
    provider = _normalize_provider_name(params.get("transcribe_provider"))
    model = _coerce_model(params.get("transcribe_model"))
    language = _coerce_language(params.get("transcribe_language"))
    priorities = _parse_priority_items(params.get("transcribe_priority") or ())
    prompt_parts = [
        item.strip()
        for item in params.get("transcribe_prompt") or ()
        if isinstance(item, str) and item.strip()
    ]
    prompt_files = [
        item.strip()
        for item in params.get("transcribe_prompt_file") or ()
        if isinstance(item, str) and item.strip()
    ]
    diarize = bool(params.get("transcribe_diarize", False))
    speakers = params.get("transcribe_speakers")
    auto_diarize = bool(params.get("transcribe_auto_diarize", False))
    auto_diarize_provider = params.get("transcribe_auto_diarize_provider")

    raw_mapping: dict[str, Any] = {}
    if provider:
        raw_mapping["provider"] = provider
    if model:
        raw_mapping["model"] = model
    if language:
        raw_mapping["language"] = language
    if priorities:
        raw_mapping["priorities"] = priorities
    if prompt_parts:
        raw_mapping["prompt_parts"] = prompt_parts
    if prompt_files:
        raw_mapping["prompt_files"] = prompt_files
    if diarize:
        raw_mapping["diarize"] = True
    if speakers is not None:
        raw_mapping["speakers"] = speakers
    if auto_diarize:
        raw_mapping["auto_diarize"] = True
    if auto_diarize_provider is not None:
        raw_mapping["auto_diarize_provider"] = auto_diarize_provider
    return normalize_manifest_config(raw_mapping)


def get_transcription_providers() -> tuple[object, ...]:
    return (
        build_openai_provider(),
        build_mistral_provider(),
    )


def get_transcription_gates() -> tuple[object, ...]:
    return (build_pyannote_gate(),)
