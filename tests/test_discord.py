from __future__ import annotations

import zipfile
from pathlib import Path

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


def test_discord_inspect_classifies_thread_and_container_relations(monkeypatch) -> None:
    channels = {
        "20": {"id": "20", "type": 11, "name": "in-game music", "parent_id": "10"},
        "10": {"id": "10", "type": 0, "name": "music", "parent_id": "5"},
        "5": {"id": "5", "type": 4, "name": "Projects"},
    }
    monkeypatch.setattr(
        discord,
        "_fetch_channel",
        lambda channel_id, **_kwargs: channels[channel_id],
    )
    monkeypatch.setattr(
        discord,
        "_fetch_guild",
        lambda guild_id, **_kwargs: {"id": guild_id, "name": "deep"},
    )

    descriptor = discord_plugin.classify_target(
        "https://discord.com/channels/1/20",
        {"use_cache": False},
    )

    assert descriptor is not None
    assert descriptor["kind"] == "thread"
    assert descriptor["metadata"]["channel_name"] == "in-game music"
    assert [item["metadata"]["relation"] for item in descriptor["relations"]] == [
        "contained_by",
        "contained_by",
        "contained_by",
    ]
    assert descriptor["relations"][0]["kind"] == "guild"
    assert descriptor["relations"][0]["label"] == "deep"
    assert descriptor["relations"][1]["target"] == "https://discord.com/channels/1/10"
    assert descriptor["relations"][1]["label"] == "#music"
    assert descriptor["relations"][2]["kind"] == "category"
    assert descriptor["relations"][2]["label"] == "Projects"


def test_discord_inspect_message_exposes_started_thread(monkeypatch) -> None:
    monkeypatch.setattr(
        discord,
        "_fetch_channel",
        lambda channel_id, **_kwargs: {
            "id": channel_id,
            "type": 0,
            "name": "projects",
        },
    )
    monkeypatch.setattr(
        discord,
        "_fetch_guild",
        lambda guild_id, **_kwargs: {"id": guild_id, "name": "deep"},
    )
    monkeypatch.setattr(
        discord,
        "_fetch_message",
        lambda *_args, **_kwargs: {
            "id": "3",
            "thread": {"id": "30", "name": "untitled godot game(s)"},
        },
    )

    descriptor = discord_plugin.classify_target(
        "https://discord.com/channels/1/2/3",
        {"use_cache": False},
    )

    assert descriptor is not None
    started = [
        item
        for item in descriptor["relations"]
        if item["metadata"]["relation"] == "starts_thread"
    ]
    assert started == [
        {
            "target": "https://discord.com/channels/1/30",
            "label": "untitled godot game(s)",
            "kind": "thread",
            "traverse": True,
            "metadata": {
                "relation": "starts_thread",
                "source_target": "https://discord.com/channels/1/2/3",
                "guild_id": "1",
                "channel_id": "30",
                "parent_channel_id": "2",
                "thread_name": "untitled godot game(s)",
            },
        }
    ]


def test_discord_guild_listing_exposes_categories_and_channels(monkeypatch) -> None:
    monkeypatch.setattr(
        discord,
        "_fetch_guild_channels",
        lambda guild_id, **_kwargs: [
            {"id": "5", "type": 4, "name": "Projects", "position": 1},
            {
                "id": "10",
                "type": 0,
                "name": "music",
                "parent_id": "5",
                "position": 2,
            },
            {"id": "11", "type": 15, "name": "ideas", "position": 3},
            {"id": "12", "type": 2, "name": "voice", "position": 4},
        ],
    )

    listing = discord_plugin.list_targets(
        "https://discord.com/channels/1",
        {"use_cache": False},
    )

    assert listing["targets"] == [
        {
            "target": "https://discord.com/channels/1/5",
            "label": "Projects",
            "kind": "category",
            "traverse": False,
            "metadata": {
                "relation": "contains",
                "source_target": "https://discord.com/channels/1",
                "guild_id": "1",
                "category_id": "5",
                "category_name": "Projects",
                "position": 1,
            },
        },
        {
            "target": "https://discord.com/channels/1/10",
            "label": "#music",
            "kind": "channel",
            "traverse": False,
            "metadata": {
                "relation": "contains",
                "source_target": "https://discord.com/channels/1",
                "guild_id": "1",
                "channel_id": "10",
                "channel_name": "music",
                "channel_type": 0,
                "parent_channel_id": "5",
                "position": 2,
                "category_id": "5",
                "category_name": "Projects",
            },
        },
        {
            "target": "https://discord.com/channels/1/11",
            "label": "#ideas",
            "kind": "forum",
            "traverse": False,
            "metadata": {
                "relation": "contains",
                "source_target": "https://discord.com/channels/1",
                "guild_id": "1",
                "channel_id": "11",
                "channel_name": "ideas",
                "channel_type": 15,
                "position": 3,
                "category_id": None,
                "category_name": None,
            },
        },
    ]
    assert listing["summary"]["categoryCount"] == 1
    assert listing["summary"]["channelCount"] == 2


def test_discord_message_listing_exposes_attachments_and_links(monkeypatch) -> None:
    monkeypatch.setattr(
        discord,
        "_fetch_message",
        lambda *_args, **_kwargs: {
            "id": "3",
            "content": "see https://youtu.be/example.",
            "attachments": [
                {
                    "id": "a1",
                    "filename": "WhatsApp Chat.zip",
                    "content_type": "application/zip",
                    "url": "https://cdn.example/chat.zip",
                    "size": 1024,
                }
            ],
            "embeds": [{"url": "https://example.com/post", "title": "Post"}],
        },
    )

    listing = discord_plugin.list_targets(
        "https://discord.com/channels/1/2/3",
        {"use_cache": False, "refresh_cache": True},
    )
    assert isinstance(listing, dict)
    items = listing["targets"]

    assert items == [
        {
            "target": "https://discord.com/channels/1/2/3?attachment-id=a1",
            "label": "WhatsApp Chat.zip",
            "kind": "attachment:file",
            "metadata": {
                "attachment_id": "a1",
                "attachment_index": 0,
                "filename": "WhatsApp Chat.zip",
                "content_type": "application/zip",
                "url": "https://cdn.example/chat.zip",
                "bytes": 1024,
                "source_message": "https://discord.com/channels/1/2/3",
            },
        },
        {
            "target": "https://youtu.be/example",
            "label": "https://youtu.be/example",
            "kind": "link",
            "metadata": {"source": "message_content"},
        },
        {
            "target": "https://example.com/post",
            "label": "Post",
            "kind": "link",
            "metadata": {"source": "embed", "embed_index": 0},
        },
    ]
    assert listing["summary"] == {
        "message": {
            "kind": "message",
            "guild_id": "1",
            "channel_id": "2",
            "message_id": "3",
        },
        "targetCount": 3,
    }
    assert listing["pagination"] == {"returned": 3, "totalCount": 3, "hasMore": False}
    assert listing["metadata"]["provider"] == "discord"


def test_discord_attachment_materialize_downloads_bytes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        discord,
        "_fetch_message",
        lambda *_args, **_kwargs: {
            "id": "3",
            "attachments": [
                {
                    "id": "a1",
                    "filename": "WhatsApp Chat.zip",
                    "content_type": "application/zip",
                    "url": "https://cdn.example/chat.zip",
                }
            ],
        },
    )

    payload_path = tmp_path / "chat.zip"
    payload_path.write_bytes(b"zip-bytes")

    def _download(url: str, **kwargs) -> Path:
        assert url == "https://cdn.example/chat.zip"
        assert kwargs["cache_identity"] == "discord:message:3:attachment:a1"
        return payload_path

    monkeypatch.setattr(
        "cx_plugins.providers.shared.media.download_cached_media_to_temp",
        _download,
    )

    files = discord_plugin.materialize(
        "https://discord.com/channels/1/2/3?attachment-id=a1",
        {"use_cache": False, "refresh_cache": True},
    )

    assert files == [
        {
            "source": "https://discord.com/channels/1/2/3?attachment-id=a1",
            "label": "WhatsApp Chat.zip",
            "filename": "WhatsApp Chat.zip",
            "content": b"zip-bytes",
            "content_type": "application/zip",
            "metadata": {
                "provider": "discord",
                "kind": "attachment",
                "sourceMessageUrl": "https://discord.com/channels/1/2/3",
                "attachmentUrl": "https://cdn.example/chat.zip",
                "attachmentId": "a1",
                "attachmentName": "WhatsApp Chat.zip",
                "channelId": "2",
                "messageId": "3",
                "bytes": 9,
            },
        }
    ]


def test_discord_whatsapp_zip_attachment_resolves_as_whatsapp(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        discord,
        "_fetch_message",
        lambda *_args, **_kwargs: {
            "id": "3",
            "timestamp": "2026-05-13T12:00:00.000000+00:00",
            "attachments": [
                {
                    "id": "a1",
                    "filename": "WhatsApp_Chat_-_Manu.zip",
                    "content_type": "application/zip",
                    "url": "https://cdn.example/chat.zip",
                }
            ],
        },
    )

    payload_path = tmp_path / "chat.zip"
    with zipfile.ZipFile(payload_path, "w") as archive:
        archive.writestr(
            "_chat.txt",
            "[5/12/26, 5:47:29 PM] Josh: like whether and to what extent",
        )

    def _download(url: str, **kwargs) -> Path:
        assert url == "https://cdn.example/chat.zip"
        return payload_path

    monkeypatch.setattr(
        "cx_plugins.providers.shared.media.download_cached_media_to_temp",
        _download,
    )

    docs = discord_plugin.resolve(
        "https://discord.com/channels/1/2/3?attachment-id=a1",
        {"use_cache": False, "refresh_cache": True},
    )

    assert len(docs) == 1
    assert docs[0]["source"] == "https://discord.com/channels/1/2/3?attachment-id=a1"
    assert "like whether and to what extent" in docs[0]["content"]
    assert "/tmp/discord-whatsapp-" not in docs[0]["content"]
    assert (
        "url: https://discord.com/channels/1/2/3?attachment-id=a1"
        in docs[0]["content"]
    )
    metadata = docs[0]["metadata"]
    assert metadata["provider"] == "whatsapp"
    assert metadata["source_ref"] == "whatsapp"
    assert metadata["hostProvider"] == "discord"
    assert metadata["attachmentName"] == "WhatsApp_Chat_-_Manu.zip"
    assert metadata["sourceMessageUrl"] == "https://discord.com/channels/1/2/3"


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

    monkeypatch.setattr(
        discord, "_normalize_attachment_nodes", _normalize_attachment_nodes
    )

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
