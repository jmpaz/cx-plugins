from datetime import datetime, timezone

import click

from cx_plugins.providers.arena.arena import ArenaSettings
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
