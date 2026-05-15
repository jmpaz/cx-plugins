from __future__ import annotations

import types
import zipfile
from datetime import datetime
from pathlib import Path

import pytest

from cx_plugins.providers.whatsapp import plugin as whatsapp_plugin
from cx_plugins.providers.whatsapp import whatsapp


def _write_archive(tmp_path: Path, text: str, media: dict[str, bytes] | None = None) -> Path:
    path = tmp_path / "WhatsApp Chat - Manu.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("_chat.txt", text)
        for name, data in (media or {}).items():
            zf.writestr(name, data)
    return path


def test_archive_parser_handles_bidi_prefixed_media_and_empty_captions(
    tmp_path: Path,
) -> None:
    archive = _write_archive(
        tmp_path,
        "\n".join(
            [
                "[5/12/26, 5:47:29\u202fPM] Josh: like whether and to what extent",
                "",
                "continuation",
                "[5/12/26, 9:31:21\u202fPM] Josh: would-have-pinned",
                "\u200e[5/13/26, 10:33:28\u202fAM] Josh: \u200e<attached: photo.jpg>",
                "\u200e[5/13/26, 10:36:51\u202fAM] Josh: caption \u200e<attached: audio.opus>",
            ]
        ),
        {
            "photo.jpg": b"\xff\xd8\xfffake",
            "audio.opus": b"OggSfake",
        },
    )

    source = whatsapp.WhatsAppArchiveSource(archive)
    messages = source.iter_messages("manu")

    assert [message.id for message in messages] == [
        "m000001",
        "m000002",
        "m000003",
        "m000004",
    ]
    assert messages[0].content == "like whether and to what extent\n\ncontinuation"
    assert messages[2].content == ""
    assert messages[2].attachments[0].filename == "photo.jpg"
    assert messages[3].content == "caption"
    assert messages[3].attachments[0].filename == "audio.opus"


def test_archive_media_identity_uses_media_content_hash(tmp_path: Path) -> None:
    archive = _write_archive(
        tmp_path,
        "\u200e[5/13/26, 10:33:28\u202fAM] Josh: \u200e<attached: photo.jpg>",
        {"photo.jpg": b"\xff\xd8\xfffake"},
    )
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    archive_with_different_chat_text = _write_archive(
        other_dir,
        "\u200e[5/13/26, 10:33:28\u202fAM] Josh: different words \u200e<attached: photo.jpg>",
        {"photo.jpg": b"\xff\xd8\xfffake"},
    )

    first_source = whatsapp.WhatsAppArchiveSource(archive)
    second_source = whatsapp.WhatsAppArchiveSource(archive_with_different_chat_text)
    first_attachment = first_source.iter_messages("manu")[0].attachments[0]
    second_attachment = second_source.iter_messages("manu")[0].attachments[0]

    identity = first_source.media_identity(first_attachment)
    assert identity.startswith("whatsapp:media:sha256:")
    assert identity == second_source.media_identity(second_attachment)


def test_media_render_cache_reuses_matching_media_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache: dict[str, str] = {}
    convert_calls: list[bytes] = []

    def fake_convert_path_to_markdown(path: str, *, refresh_images: bool = False):
        convert_calls.append(Path(path).read_bytes())
        return types.SimpleNamespace(markdown="cached description")

    monkeypatch.setattr(whatsapp, "_get_cached_rendered", cache.get)
    monkeypatch.setattr(
        whatsapp,
        "_store_rendered",
        lambda identity, content: cache.setdefault(identity, content),
    )
    monkeypatch.setattr(
        "contextualize.render.markitdown.convert_path_to_markdown",
        fake_convert_path_to_markdown,
    )

    first = whatsapp._describe_media_bytes(
        b"\xff\xd8\xffsame",
        filename="first.jpg",
        kind="image",
        content_type="image/jpeg",
        mode="describe",
    )
    second = whatsapp._describe_media_bytes(
        b"\xff\xd8\xffsame",
        filename="second.jpg",
        kind="image",
        content_type="image/jpeg",
        mode="describe",
    )

    assert first == "cached description"
    assert second == "cached description"
    assert convert_calls == [b"\xff\xd8\xffsame"]


def test_quote_like_content_is_not_promoted_to_reply_metadata(tmp_path: Path) -> None:
    archive = _write_archive(
        tmp_path,
        "\n".join(
            [
                "[5/12/26, 5:47:29\u202fPM] Josh: like whether and to what extent",
                "[5/12/26, 9:24:07\u202fPM] Josh: > like whether and to what extent",
                "manual second layer",
                "[5/12/26, 9:31:21\u202fPM] Josh: would-have-pinned",
            ]
        ),
    )

    docs = whatsapp.resolve_whatsapp_target(
        str(archive),
        settings=whatsapp.WhatsAppSettings(include_media_descriptions=False),
    )

    messages = docs[0].messages
    quoted = messages[1]
    would_have_pinned = messages[2]
    assert quoted["content"].startswith("> like whether")
    assert "reply_to_id" not in quoted
    assert "reply_to_id" not in would_have_pinned


def test_pin_service_rows_render_as_system_events(tmp_path: Path) -> None:
    archive = _write_archive(
        tmp_path,
        "\n".join(
            [
                "[5/12/26, 9:29:32\u202fPM] Manu: \u200eYou pinned a message",
                "[5/12/26, 9:29:38\u202fPM] Josh: pinned start of thread",
                "[5/12/26, 9:31:21\u202fPM] Josh: would-have-pinned",
                "[5/12/26, 9:40:33\u202fPM] Manu: \u200eManu pinned a message",
            ]
        ),
    )

    docs = whatsapp.resolve_whatsapp_target(
        str(archive),
        settings=whatsapp.WhatsAppSettings(include_media_descriptions=False),
    )

    messages = docs[0].messages
    assert messages[0]["sender"] == "system"
    assert messages[0]["is_system"] is True
    assert messages[0]["pins"] == [
        {"type": "message_pin", "actor": "You", "archive_sender": "Manu"}
    ]
    assert messages[1]["sender"] == "Josh"
    assert messages[1]["is_system"] is False
    assert messages[2]["sender"] == "Josh"
    assert messages[3]["sender"] == "system"
    assert messages[3]["pins"] == [
        {"type": "message_pin", "actor": "Manu", "archive_sender": "Manu"}
    ]
    assert "[system]\nYou pinned a message" in docs[0].rendered
    assert "[Josh]\npinned start of thread" in docs[0].rendered


def test_author_name_mode_first_keeps_archive_sender_metadata(tmp_path: Path) -> None:
    archive = _write_archive(
        tmp_path,
        "[5/12/26, 9:31:21\u202fPM] Josh Pazmino: full name in archive",
    )

    docs = whatsapp.resolve_whatsapp_target(
        str(archive),
        settings=whatsapp.WhatsAppSettings(
            include_media_descriptions=False,
            author_name_mode="first",
        ),
    )

    message = docs[0].messages[0]
    assert message["sender"] == "Josh"
    assert message["archive_sender"] == "Josh Pazmino"
    assert "[Josh]\nfull name in archive" in docs[0].rendered


def test_window_filtering_and_message_context(tmp_path: Path) -> None:
    archive = _write_archive(
        tmp_path,
        "\n".join(
            [
                "[5/12/26, 5:47:29\u202fPM] Josh: older",
                "[5/12/26, 9:24:07\u202fPM] Josh: target",
                "[5/12/26, 9:31:21\u202fPM] Josh: later",
            ]
        ),
    )

    window_docs = whatsapp.resolve_whatsapp_target(
        str(archive),
        settings=whatsapp.WhatsAppSettings(
            include_media_descriptions=False,
            start=datetime(2026, 5, 12, 21, 0, 0),
            end=datetime(2026, 5, 12, 21, 30, 0),
        ),
    )
    assert [message["content"] for message in window_docs[0].messages] == ["target"]

    target = f"whatsapp:zip:{archive}?message=m000002"
    context_docs = whatsapp.resolve_whatsapp_target(
        target,
        settings=whatsapp.WhatsAppSettings(
            include_media_descriptions=False,
            before_messages=1,
            after_messages=1,
        ),
    )
    assert [message["content"] for message in context_docs[0].messages] == [
        "older",
        "target",
        "later",
    ]


def test_parse_whatsapp_config_mapping_accepts_window_and_media_controls() -> None:
    parsed = whatsapp.parse_whatsapp_config_mapping(
        {
            "author-name": "first",
            "window": {
                "start": "2026-05-12T21:00:00",
                "end": "2026-05-12T22:00:00",
                "message-context": 4,
            },
            "media": {
                "describe": False,
                "file-content": False,
                "skip-file-content-messages": ["m000002", "2"],
                "max-file-content-size": "50K",
                "mode": "transcribe",
            },
        },
        prefix="whatsapp",
    )

    assert parsed is not None
    assert parsed["author_name_mode"] == "first"
    assert parsed["include_media_descriptions"] is False
    assert parsed["include_file_content"] is False
    assert parsed["message_context"] == 4
    assert parsed["skip_file_content_messages"] == frozenset({"m000002"})
    assert parsed["max_file_content_size_bytes"] == 50 * 1024
    assert parsed["media_mode"] == "transcribe"


def test_plugin_listing_and_cli_overrides(tmp_path: Path) -> None:
    archive = _write_archive(
        tmp_path,
        "[5/12/26, 9:31:21\u202fPM] Josh: would-have-pinned",
    )

    items = whatsapp_plugin.list_targets(str(archive), {})
    assert items == [
        {
            "target": f"whatsapp:zip:{str(archive).replace(' ', '%20')}?chat=manu",
            "label": "Manu",
            "kind": "chat",
            "metadata": {
                "provider": "whatsapp",
                "chatId": "manu",
                "sourcePath": str(archive),
            },
        }
    ]

    overrides = whatsapp_plugin.collect_cli_overrides(
        "cat",
        {
            "whatsapp_file_content": False,
            "whatsapp_skip_file_content_message": ("m000002",),
            "whatsapp_max_file_content_size": "50kb",
        },
    )
    assert overrides == {
        "media": {
            "file-content": False,
            "skip-file-content-messages": ["m000002"],
            "max-file-content-size": "50kb",
        }
    }
