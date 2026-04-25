from __future__ import annotations

from cx_plugins.providers.transcribe.plugin import (
    collect_cli_overrides,
    get_transcription_providers,
    normalize_manifest_config,
)


def test_normalize_manifest_config_supports_provider_aliases() -> None:
    normalized = normalize_manifest_config(
        {
            "provider": "whisper",
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
        "language": "es",
        "priorities": {"mistral": 300},
        "prompt_parts": ["names", "places"],
        "prompt_files": ["notes.yaml"],
        "diarize": True,
        "speakers": 2,
        "auto_diarize": True,
        "auto_diarize_provider": "mistral",
    }


def test_collect_cli_overrides_builds_transcribe_mapping() -> None:
    overrides = collect_cli_overrides(
        "cat",
        {
            "transcribe_provider": "mistral",
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
        "language": "es",
        "priorities": {"openai": 10, "mistral": 500},
        "prompt_parts": ["names"],
        "prompt_files": ["notes.yaml"],
        "diarize": True,
        "speakers": 2,
        "auto_diarize": True,
        "auto_diarize_provider": "mistral",
    }


def test_get_transcription_providers_returns_openai_and_mistral() -> None:
    providers = get_transcription_providers()

    assert [provider.name for provider in providers] == ["openai", "mistral"]


def test_get_transcription_gates_returns_pyannote_gate() -> None:
    from cx_plugins.providers.transcribe.plugin import get_transcription_gates

    gates = get_transcription_gates()

    assert [gate.name for gate in gates] == ["pyannote-community-1"]
