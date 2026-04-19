from datetime import datetime, timezone

from cx_plugins.providers.arena.arena import ArenaSettings
from cx_plugins.providers.arena.plugin import (
    DEFAULT_RICH_MEDIA_CHANNEL_MAX_BLOCKS,
    _apply_channel_safety_defaults,
    _arena_runtime_overrides,
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
