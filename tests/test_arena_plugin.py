from datetime import datetime, timezone

import click
import pytest

from cx_plugins.providers.arena import arena
from cx_plugins.providers.arena.arena import (
    ArenaSettings,
    parse_arena_recurse_blocks,
    resolve_channel,
)
from cx_plugins.providers.arena.plugin import (
    DEFAULT_RICH_MEDIA_CHANNEL_MAX_BLOCKS,
    _apply_channel_safety_defaults,
    _arena_runtime_overrides,
    collect_cli_overrides,
    register_cli_options,
)


def test_apply_channel_safety_defaults_caps_explicit_rich_media_channel() -> None:
    overrides = _arena_runtime_overrides({"block": {"pdf-content": True}})
    settings = ArenaSettings(include_pdf_content=True)

    capped = _apply_channel_safety_defaults(settings, overrides)

    assert capped.max_blocks_per_channel == DEFAULT_RICH_MEDIA_CHANNEL_MAX_BLOCKS
    assert settings.max_blocks_per_channel is None


def test_apply_channel_safety_defaults_leaves_bounded_or_windowed_channels() -> None:
    overrides = _arena_runtime_overrides({"block": {"media-desc": True}})

    bounded = _apply_channel_safety_defaults(
        ArenaSettings(max_blocks_per_channel=25, include_media_descriptions=True),
        overrides,
    )
    assert bounded.max_blocks_per_channel == 25

    windowed = _apply_channel_safety_defaults(
        ArenaSettings(
            include_media_descriptions=True,
            connected_after=datetime(2026, 4, 17, tzinfo=timezone.utc),
        ),
        overrides,
    )
    assert windowed.max_blocks_per_channel is None

    sampled = _apply_channel_safety_defaults(
        ArenaSettings(
            include_media_descriptions=True,
            recurse_blocks=parse_arena_recurse_blocks("0.8"),
        ),
        overrides,
    )
    assert sampled.max_blocks_per_channel is None


def test_apply_channel_safety_defaults_ignores_implicit_media_default() -> None:
    settings = ArenaSettings(include_media_descriptions=True)

    unchanged = _apply_channel_safety_defaults(settings, None)

    assert unchanged is settings
    assert unchanged.max_blocks_per_channel is None


def test_register_cli_options_exposes_arena_controls() -> None:
    command = click.Command("cat")

    register_cli_options("cat", command)

    option_names = {
        opt for param in command.params for opt in getattr(param, "opts", ())
    }
    assert "--arena-recurse-depth" in option_names
    assert "--arena-recurse-user" in option_names
    assert "--arena-recurse-users" in option_names
    assert "--arena-recurse-blocks" in option_names
    assert "--arena-link-image-descriptions" in option_names
    assert "--arena-media-descriptions" in option_names
    assert "--arena-connected-after" in option_names


def test_collect_cli_overrides_builds_arena_mapping() -> None:
    overrides = collect_cli_overrides(
        "cat",
        {
            "arena_recurse_depth": 2,
            "arena_recurse_user": ("self,friend",),
            "arena_max_blocks_per_channel": 25,
            "arena_recurse_blocks": "5%",
            "arena_block_sort": "DATE-DESC",
            "arena_connected_after": "2026-04-01",
            "arena_connected_before": "",
            "arena_created_after": None,
            "arena_created_before": None,
            "arena_block_descriptions": False,
            "arena_block_comments": True,
            "arena_link_image_descriptions": True,
            "arena_pdf_content": True,
            "arena_media_descriptions": False,
        },
    )

    assert overrides == {
        "recurse-depth": 2,
        "recurse-users": ["self", "friend"],
        "max-blocks-per-channel": 25,
        "recurse-blocks": "5%",
        "block-sort": "date-desc",
        "connected-after": "2026-04-01",
        "block": {
            "description": False,
            "comments": True,
            "link-image-desc": True,
            "pdf-content": True,
            "media-desc": False,
        },
    }


def test_collect_cli_overrides_accepts_arena_recurse_all() -> None:
    overrides = collect_cli_overrides(
        "hydrate",
        {
            "arena_recurse_depth": None,
            "arena_recurse_user": ("all",),
            "arena_max_blocks_per_channel": None,
            "arena_recurse_blocks": None,
            "arena_block_sort": None,
            "arena_connected_after": None,
            "arena_connected_before": None,
            "arena_created_after": None,
            "arena_created_before": None,
            "arena_block_descriptions": None,
            "arena_block_comments": None,
            "arena_link_image_descriptions": None,
            "arena_pdf_content": None,
            "arena_media_descriptions": None,
        },
    )

    assert overrides == {"recurse-users": "all"}


def test_collect_cli_overrides_rejects_mixed_recurse_all() -> None:
    try:
        collect_cli_overrides(
            "cat",
            {
                "arena_recurse_depth": None,
                "arena_recurse_user": ("all", "friend"),
                "arena_max_blocks_per_channel": None,
                "arena_recurse_blocks": None,
                "arena_block_sort": None,
                "arena_connected_after": None,
                "arena_connected_before": None,
                "arena_created_after": None,
                "arena_created_before": None,
                "arena_block_descriptions": None,
                "arena_block_comments": None,
                "arena_link_image_descriptions": None,
                "arena_pdf_content": None,
                "arena_media_descriptions": None,
            },
        )
    except ValueError as exc:
        assert "--arena-recurse-user all cannot be combined" in str(exc)
    else:
        raise AssertionError("expected mixed all/slugs to fail")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("50", ("count", 50)),
        (50, ("count", 50)),
        ("0.05", ("ratio", 0.05)),
        (0.05, ("ratio", 0.05)),
        ("5%", ("ratio", 0.05)),
    ],
)
def test_parse_arena_recurse_blocks_accepts_counts_and_ratios(raw, expected) -> None:
    parsed = parse_arena_recurse_blocks(raw)

    assert parsed is not None
    assert (parsed.kind, parsed.value) == expected


@pytest.mark.parametrize("raw", ["0", 0, "-1", "1.5", 1.5, "150%", "nope"])
def test_parse_arena_recurse_blocks_rejects_invalid_values(raw) -> None:
    with pytest.raises(ValueError):
        parse_arena_recurse_blocks(raw)


def _channel(slug: str, id_: int, title: str, count: int) -> dict:
    return {
        "id": id_,
        "slug": slug,
        "title": title,
        "type": "Channel",
        "base_type": "Channel",
        "counts": {"contents": count},
        "owner": {"id": 1, "slug": "owner", "name": "Owner"},
    }


def _text_block(id_: int) -> dict:
    return {
        "id": id_,
        "type": "Text",
        "base_type": "Block",
        "title": f"Block {id_}",
        "content": f"body {id_}",
    }


def test_recurse_blocks_ratio_caps_nested_channels_and_keeps_stub(monkeypatch) -> None:
    child_blocks = [_text_block(i) for i in range(1000, 1100)]

    monkeypatch.setattr(
        arena,
        "_fetch_channel",
        lambda slug: {
            "root": _channel("root", 1, "Root", 1000),
            "child": _channel("child", 2, "Child", 100),
        }[slug],
    )
    monkeypatch.setattr(
        arena,
        "_fetch_channel_page",
        lambda slug, page, per=100: {
            "root": {
                "contents": [_channel("child", 2, "Child", 100)],
                "meta": {"total_pages": 1},
            },
            "child": {
                "contents": child_blocks,
                "meta": {"total_pages": 1},
            },
        }[slug],
    )

    _metadata, flat = resolve_channel(
        "root",
        use_cache=False,
        settings=ArenaSettings(
            max_depth=1,
            sort_order="asc",
            recurse_users=None,
            recurse_blocks=parse_arena_recurse_blocks("0.05"),
        ),
    )

    assert len(flat) == 51
    assert flat[0][1]["type"] == "Channel"
    assert flat[0][1]["slug"] == "child"
    assert [block["id"] for _, block in flat[1:]] == list(range(1000, 1050))


def test_recurse_blocks_omits_stub_when_nested_channel_is_not_sampled(
    monkeypatch,
) -> None:
    child_blocks = [_text_block(1000), _text_block(1001)]

    monkeypatch.setattr(
        arena,
        "_fetch_channel",
        lambda slug: {
            "root": _channel("root", 1, "Root", 1000),
            "child": _channel("child", 2, "Child", 2),
        }[slug],
    )
    monkeypatch.setattr(
        arena,
        "_fetch_channel_page",
        lambda slug, page, per=100: {
            "root": {
                "contents": [_channel("child", 2, "Child", 2)],
                "meta": {"total_pages": 1},
            },
            "child": {
                "contents": child_blocks,
                "meta": {"total_pages": 1},
            },
        }[slug],
    )

    _metadata, flat = resolve_channel(
        "root",
        use_cache=False,
        settings=ArenaSettings(
            max_depth=1,
            sort_order="asc",
            recurse_users=None,
            recurse_blocks=parse_arena_recurse_blocks("0.05"),
        ),
    )

    assert [block["id"] for _, block in flat] == [1000, 1001]


def test_recurse_blocks_uses_stricter_global_channel_limit(monkeypatch) -> None:
    child_blocks = [_text_block(i) for i in range(1000, 1100)]

    monkeypatch.setattr(
        arena,
        "_fetch_channel",
        lambda slug: {
            "root": _channel("root", 1, "Root", 1000),
            "child": _channel("child", 2, "Child", 100),
        }[slug],
    )
    monkeypatch.setattr(
        arena,
        "_fetch_channel_page",
        lambda slug, page, per=100: {
            "root": {
                "contents": [_channel("child", 2, "Child", 100)],
                "meta": {"total_pages": 1},
            },
            "child": {
                "contents": child_blocks,
                "meta": {"total_pages": 1},
            },
        }[slug],
    )

    _metadata, flat = resolve_channel(
        "root",
        use_cache=False,
        settings=ArenaSettings(
            max_depth=1,
            sort_order="asc",
            max_blocks_per_channel=25,
            recurse_users=None,
            recurse_blocks=parse_arena_recurse_blocks("50"),
        ),
    )

    assert len(flat) == 26
    assert flat[0][1]["type"] == "Channel"
    assert [block["id"] for _, block in flat[1:]] == list(range(1000, 1025))
