from __future__ import annotations

from cx_plugins.providers.discord import discord
from cx_plugins.providers.discord import plugin as discord_plugin


def test_parse_discord_url_supports_attachment_selectors() -> None:
    parsed = discord.parse_discord_url(
        "https://discord.com/channels/1/2/3?attachment-id=444"
    )
    assert parsed is not None
    assert parsed["kind"] == "attachment"
    assert parsed["attachment_id"] == "444"

    parsed = discord.parse_discord_url(
        "https://discord.com/channels/1/2/3?attachment=notes.json"
    )
    assert parsed is not None
    assert parsed["kind"] == "attachment"
    assert parsed["attachment_name"] == "notes.json"


def test_collect_cli_overrides_emits_media_controls() -> None:
    overrides = discord_plugin.collect_cli_overrides(
        "cat",
        {
            "discord_file_content": False,
            "discord_skip_file_content_message": (
                "https://discord.com/channels/1/2/3",
                "4",
            ),
            "discord_max_file_content_size": "50kb",
        },
    )

    assert overrides == {
        "media": {
            "file-content": False,
            "skip-file-content-messages": [
                "https://discord.com/channels/1/2/3",
                "4",
            ],
            "max-file-content-size": "50kb",
        }
    }


def test_parse_discord_config_mapping_accepts_new_media_controls() -> None:
    parsed = discord.parse_discord_config_mapping(
        {
            "media": {
                "file-content": False,
                "skip-file-content-messages": [
                    "https://discord.com/channels/1/2/3",
                    "4",
                ],
                "max-file-content-size": "50K",
            }
        },
        prefix="discord",
    )

    assert parsed is not None
    assert parsed["include_file_content"] is False
    assert parsed["skip_file_content_messages"] == frozenset({"3", "4"})
    assert parsed["max_file_content_size_bytes"] == 50 * 1024


def test_build_discord_settings_uses_skip_message_ids_and_size_limit(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        discord, "_discord_settings_from_env", lambda: discord.DiscordSettings()
    )

    settings = discord.build_discord_settings(
        {
            "skip_file_content_messages": [
                "https://discord.com/channels/1/2/3",
                "4",
            ],
            "max_file_content_size_bytes": "50kb",
        }
    )

    assert settings.skip_file_content_messages == frozenset({"3", "4"})
    assert settings.max_file_content_size_bytes == 50 * 1024


def test_normalize_attachment_nodes_skips_large_or_selected_files(
    monkeypatch,
) -> None:
    extract_calls: list[str] = []
    describe_calls: list[str] = []

    monkeypatch.setattr(
        discord,
        "_extract_utf8_remote_media",
        lambda url, **_kwargs: extract_calls.append(url) or "hello",
    )
    monkeypatch.setattr(
        discord,
        "_describe_remote_media",
        lambda url, **_kwargs: describe_calls.append(url) or "desc",
    )

    message = {
        "id": "33",
        "attachments": [
            {
                "id": "a1",
                "filename": "big.json",
                "content_type": "application/json",
                "url": "https://cdn.example/big.json",
                "size": 60 * 1024,
            },
            {
                "id": "a2",
                "filename": "small.json",
                "content_type": "application/json",
                "url": "https://cdn.example/small.json",
                "size": 2 * 1024,
            },
        ],
    }

    nodes = discord._normalize_attachment_nodes(  # noqa: SLF001
        message,
        guild_id="1",
        channel_id="2",
        include_media_descriptions=True,
        include_embed_media_descriptions=True,
        include_file_content=True,
        media_mode="describe",
        skip_file_content_messages=frozenset({"33"}),
        max_file_content_size_bytes=50 * 1024,
    )

    assert extract_calls == []
    assert describe_calls == []
    assert nodes[0]["bytes"] == 60 * 1024
    assert nodes[0]["size"] == "60K"
    assert nodes[0]["ref"] == "https://discord.com/channels/1/2/33?attachment-id=a1"
    assert "text_content" not in nodes[0]
    assert "text_content" not in nodes[1]


def test_normalize_message_passes_skip_controls_without_settings_scope(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        discord,
        "_extract_message_content",
        lambda _message: "hello",
    )
    monkeypatch.setattr(discord, "_reply_to_id", lambda _message: None)
    monkeypatch.setattr(
        discord,
        "_reply_to_content",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        discord,
        "_forward_metadata",
        lambda *_args, **_kwargs: None,
    )

    captured: dict[str, object] = {}

    def _normalize_attachment_nodes(message, **kwargs):
        captured["message_id"] = message["id"]
        captured["skip_file_content_messages"] = kwargs.get(
            "skip_file_content_messages"
        )
        captured["max_file_content_size_bytes"] = kwargs.get(
            "max_file_content_size_bytes"
        )
        return []

    monkeypatch.setattr(discord, "_normalize_attachment_nodes", _normalize_attachment_nodes)

    normalized = discord._normalize_message(  # noqa: SLF001
        {
            "id": "33",
            "timestamp": "2026-03-30T12:00:00Z",
            "content": "hello",
            "type": 0,
        },
        guild_id="1",
        channel_id="2",
        thread_id=None,
        include_media_descriptions=True,
        include_embed_media_descriptions=True,
        include_file_content=True,
        media_mode="describe",
        skip_file_content_messages=frozenset({"33"}),
        max_file_content_size_bytes=50 * 1024,
    )

    assert normalized is not None
    assert captured["message_id"] == "33"
    assert captured["skip_file_content_messages"] == frozenset({"33"})
    assert captured["max_file_content_size_bytes"] == 50 * 1024


def test_format_attachment_idiomatic_includes_ref_and_size_attrs() -> None:
    lines = discord._format_attachment_idiomatic(  # noqa: SLF001
        {
            "type": "file",
            "filename": "notes.json",
            "url": "https://cdn.example/notes.json",
            "ref": "https://discord.com/channels/1/2/3?attachment-id=a1",
            "size": "26K",
            "bytes": 26624,
        }
    )

    assert lines == [
        '<attachment type="file" url="https://cdn.example/notes.json" filename="notes.json" ref="https://discord.com/channels/1/2/3?attachment-id=a1" size="26K" bytes="26624" />'
    ]


def test_resolve_discord_attachment_target_uses_attachment_url_reference(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        discord,
        "_fetch_message",
        lambda *_args, **_kwargs: {
            "id": "3",
            "timestamp": "2026-03-30T12:00:00Z",
            "attachments": [
                {
                    "id": "a1",
                    "filename": "notes.json",
                    "url": "https://cdn.example/notes.json",
                }
            ],
        },
    )

    class _DummyReference:
        def __init__(
            self,
            url: str,
            *,
            format: str,
            label: str,
            filename_override: str | None,
            use_cache: bool,
            cache_ttl,
            refresh_cache: bool,
            plugin_overrides,
        ) -> None:
            assert url == "https://cdn.example/notes.json"
            assert format == "raw"
            assert label == "name"
            assert filename_override == "notes.json"
            assert use_cache is False
            assert refresh_cache is True
            assert plugin_overrides == {"transcribe": {"provider": "mistral"}}

        def read(self) -> str:
            return '{"ok":true}'

    monkeypatch.setattr("contextualize.references.url.URLReference", _DummyReference)

    docs = discord.resolve_discord_attachment_target(
        "https://discord.com/channels/1/2/3?attachment-id=a1",
        {
            "kind": "attachment",
            "guild_id": "1",
            "channel_id": "2",
            "message_id": "3",
            "attachment_id": "a1",
        },
        settings=discord.DiscordSettings(),
        settings_key=("settings",),
        use_cache=False,
        cache_ttl=None,
        refresh_cache=True,
        plugin_overrides={"transcribe": {"provider": "mistral"}},
    )

    assert len(docs) == 1
    assert docs[0]["content"] == '{"ok":true}'
    assert docs[0]["metadata"]["kind"] == "attachment"
    assert docs[0]["metadata"]["attachmentId"] == "a1"
    assert (
        docs[0]["metadata"]["sourceMessageUrl"] == "https://discord.com/channels/1/2/3"
    )
