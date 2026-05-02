from __future__ import annotations

import click
import pytest

from cx_plugins.providers.transcribe.plugin import (
    _complete_transcribe_model,
    collect_cli_overrides,
    get_transcription_providers,
    normalize_manifest_config,
)


def test_normalize_manifest_config_supports_transcription_model() -> None:
    normalized = normalize_manifest_config(
        {
            "provider": "openai",
            "model": "cohere",
            "language": "ES",
            "priorities": {"mistral": 300},
            "prompt": ["names", "places"],
            "prompt_files": ["notes.yaml"],
            "diarize": True,
            "speakers": 2,
            "auto_diarize": True,
            "auto_diarize_provider": "mistral",
        }
    )

    assert normalized == {
        "provider": "openai",
        "model": "cohere",
        "language": "es",
        "priorities": {"mistral": 300},
        "prompt_parts": ["names", "places"],
        "prompt_files": ["notes.yaml"],
        "diarize": True,
        "speakers": 2,
        "auto_diarize": True,
        "auto_diarize_provider": "mistral",
    }


def test_normalize_manifest_config_rejects_whisper_provider() -> None:
    with pytest.raises(ValueError, match="unsupported transcription provider"):
        normalize_manifest_config({"provider": "whisper"})


def test_collect_cli_overrides_builds_transcribe_mapping() -> None:
    overrides = collect_cli_overrides(
        "cat",
        {
            "transcribe_provider": "mistral",
            "transcribe_model": "cohere",
            "transcribe_language": "es",
            "transcribe_priority": ("openai=10", "mistral=500"),
            "transcribe_prompt": ("names",),
            "transcribe_prompt_file": ("notes.yaml",),
            "transcribe_diarize": True,
            "transcribe_speakers": 2,
            "transcribe_auto_diarize": True,
            "transcribe_auto_diarize_provider": "mistral",
        },
    )

    assert overrides == {
        "provider": "mistral",
        "model": "cohere",
        "language": "es",
        "priorities": {"openai": 10, "mistral": 500},
        "prompt_parts": ["names"],
        "prompt_files": ["notes.yaml"],
        "diarize": True,
        "speakers": 2,
        "auto_diarize": True,
        "auto_diarize_provider": "mistral",
    }


def test_transcribe_model_completion_uses_openai_compatible_models(monkeypatch) -> None:
    from cx_plugins.providers.transcribe import openai

    monkeypatch.setattr(
        openai,
        "list_openai_model_options",
        lambda incomplete: ["cohere", f"{incomplete}-model"],
    )

    completions = _complete_transcribe_model(None, None, "distil")

    assert [completion.value for completion in completions] == [
        "cohere",
        "distil-model",
    ]


def test_transcribe_model_help_shows_openai_compatible_models(monkeypatch) -> None:
    from cx_plugins.providers.transcribe import openai

    monkeypatch.setattr(
        openai,
        "list_openai_model_options",
        lambda incomplete="": ["cohere", "distilwhisper"],
    )

    command = click.Command("cat")
    from cx_plugins.providers.transcribe.plugin import register_cli_options

    register_cli_options("cat", command)
    ctx = click.Context(command)
    help_text = command.get_help(ctx)

    assert "--transcribe-model [cohere|distilwhisper]" in help_text


def test_get_transcription_providers_returns_openai_and_mistral() -> None:
    providers = get_transcription_providers()

    assert [provider.name for provider in providers] == ["openai", "mistral"]


def test_get_transcription_gates_returns_pyannote_gate() -> None:
    from cx_plugins.providers.transcribe.plugin import get_transcription_gates

    gates = get_transcription_gates()

    assert [gate.name for gate in gates] == ["pyannote-community-1"]
