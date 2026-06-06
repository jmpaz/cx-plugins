from datetime import datetime, timezone
from pathlib import Path

import click
import pytest

from cx_plugins.providers.arena import arena
from cx_plugins.providers.arena import plugin as arena_plugin
from cx_plugins.providers.arena.arena import (
    ArenaSettings,
    parse_arena_channel_exclusions,
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
    command = click.Command("payload")

    register_cli_options("payload", command)

    option_names = {
        opt for param in command.params for opt in getattr(param, "opts", ())
    }
    assert "--arena-recurse-depth" in option_names
    assert "--arena-recurse-user" in option_names
    assert "--arena-recurse-users" in option_names
    assert "--arena-recurse-blocks" in option_names
    assert "--arena-link-image-descriptions" in option_names
    assert "--arena-media-descriptions" in option_names
    assert "--arena-block-connections" in option_names
    assert "--arena-block-connections-max-items" in option_names
    assert "--arena-connected-after" in option_names
    assert "--arena-exclude-channel" in option_names


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
            "arena_block_connections": True,
            "arena_block_connections_max_items": 30,
            "arena_link_image_descriptions": True,
            "arena_pdf_content": True,
            "arena_media_descriptions": False,
            "arena_exclude_channel": (
                "skip-one,123",
                "https://www.are.na/channel/skip-two",
            ),
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
            "connections": True,
            "connections-max-items": 30,
            "link-image-desc": True,
            "pdf-content": True,
            "media-desc": False,
        },
        "exclude-channels": [
            "skip-one",
            "123",
            "https://www.are.na/channel/skip-two",
        ],
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
            "arena_exclude_channel": (),
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
                "arena_exclude_channel": (),
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


def test_arena_user_and_group_targets_are_recognized() -> None:
    assert arena.is_arena_user_target("arena:user:alice")
    assert arena.is_arena_user_target("https://www.are.na/alice")
    assert arena.is_arena_group_target("arena:group:studio")
    assert arena.is_arena_group_target("https://www.are.na/groups/studio")
    assert not arena.is_arena_user_target("https://www.are.na/channel/root")


def test_parse_arena_channel_exclusions_accepts_ids_slugs_and_urls() -> None:
    exclusions = parse_arena_channel_exclusions(
        ["123", "Root", "https://www.are.na/channel/skip-me"]
    )

    assert exclusions.ids == frozenset({123})
    assert exclusions.slugs == frozenset({"root", "skip-me"})


def test_parse_arena_channel_exclusions_accepts_existing_value() -> None:
    exclusions = parse_arena_channel_exclusions(["skip-me"])

    assert parse_arena_channel_exclusions(exclusions) is exclusions


def _channel(
    slug: str,
    id_: int,
    title: str,
    count: int,
    *,
    description: object | None = None,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> dict:
    channel = {
        "id": id_,
        "slug": slug,
        "title": title,
        "type": "Channel",
        "base_type": "Channel",
        "counts": {"contents": count},
        "owner": {"id": 1, "slug": "owner", "name": "Owner"},
    }
    if description is not None:
        channel["description"] = description
    if created_at is not None:
        channel["created_at"] = created_at
    if updated_at is not None:
        channel["updated_at"] = updated_at
    return channel


def _text_block(id_: int) -> dict:
    return {
        "id": id_,
        "type": "Text",
        "base_type": "Block",
        "title": f"Block {id_}",
        "content": f"body {id_}",
    }


def _user_profile(slug: str = "alice", name: str = "Alice") -> dict:
    return {
        "id": 20,
        "type": "User",
        "name": name,
        "slug": slug,
        "bio": {"markdown": "Building a garden."},
        "created_at": "2024-01-02T03:04:05Z",
        "updated_at": "2024-02-03T04:05:06Z",
        "counts": {"channels": 2, "followers": 3, "following": 4},
    }


def _group_profile(slug: str = "studio", name: str = "Studio") -> dict:
    return {
        "id": 30,
        "type": "Group",
        "name": name,
        "slug": slug,
        "bio": {"markdown": "Shared workbench."},
        "created_at": "2025-01-02T03:04:05Z",
        "updated_at": "2025-02-03T04:05:06Z",
        "user": {"id": 40, "type": "User", "name": "Owner", "slug": "owner"},
        "counts": {"channels": 5, "users": 6},
    }


def test_render_text_block_sources_matching_title_as_header_url() -> None:
    rendered = arena._render_block(
        {
            **_text_block(100),
            "created_at": "2026-05-20T10:00:00Z",
            "user": {"id": 4, "name": "Creator Person", "slug": "creator-person"},
            "source": {
                "url": "https://example.com/source",
                "title": "Block 100",
            },
        },
        include_comments=False,
        include_connections=False,
    )

    assert rendered == "\n".join(
        [
            "title   Block 100",
            "source  https://example.com/source",
            "created 2026-05-20T10:00Z by Creator Person",
            "---",
            "",
            "body 100",
        ]
    )


def test_render_text_block_sources_different_title_as_header_title_url() -> None:
    rendered = arena._render_block(
        {
            **_text_block(101),
            "source": {
                "url": "https://example.com/source",
                "title": "Original Page",
            },
        },
        include_comments=False,
        include_connections=False,
    )

    assert rendered == "\n".join(
        [
            "title   Block 101",
            "source  Original Page <https://example.com/source>",
            "---",
            "",
            "body 101",
        ]
    )


def test_render_image_block_keeps_image_url_and_sources_page_in_header() -> None:
    rendered = arena._render_block(
        {
            "id": 45704370,
            "type": "Image",
            "title": "Saved image",
            "source": {
                "url": "https://example.com/article",
                "title": "Original Article",
            },
            "image": {"src": "https://images.are.na/preview.jpg"},
        },
        include_comments=False,
        include_connections=False,
        include_media_descriptions=False,
    )

    assert rendered == "\n".join(
        [
            "title   Saved image",
            "source  Original Article <https://example.com/article>",
            "---",
            "",
            "[Image: Saved image]",
            "URL: https://images.are.na/preview.jpg",
        ]
    )


def test_render_link_block_does_not_duplicate_source_header() -> None:
    rendered = arena._render_block(
        {
            "id": 200,
            "type": "Link",
            "title": "Original Page",
            "source": {
                "url": "https://example.com/source",
                "title": "Original Page",
            },
        },
        include_comments=False,
        include_connections=False,
    )

    assert "source  " not in rendered
    assert rendered == "\n".join(
        [
            "title   Original Page",
            "---",
            "",
            "https://example.com/source",
        ]
    )


def test_resolve_channel_includes_root_channel_metadata(monkeypatch) -> None:
    channel = _channel(
        "root",
        1,
        "Root",
        0,
        description={"markdown": "A channel about roots."},
        created_at="2025-01-02T03:04:05Z",
        updated_at="2025-02-03T04:05:06Z",
    )

    monkeypatch.setattr(arena, "_fetch_channel", lambda slug: channel)
    monkeypatch.setattr(
        arena,
        "_fetch_channel_page",
        lambda slug, page, per=100: {"contents": [], "meta": {"total_pages": 1}},
    )

    docs = arena_plugin.resolve(
        "https://www.are.na/owner/root",
        {"use_cache": False},
    )

    assert len(docs) == 1
    assert docs[0]["content"] == "\n".join(
        [
            "[Channel: Root]",
            "Owner: Owner",
            "Blocks: 0",
            "Started: 2025-01-02T03:04Z",
            "Modified: 2025-02-03T04:05Z",
            "https://www.are.na/channel/root",
            "Description: A channel about roots.",
        ]
    )
    assert docs[0]["metadata"]["context_subpath"] == "root/_channel.md"
    assert docs[0]["metadata"]["source_created"] == "2025-01-02T03:04:05Z"
    assert docs[0]["metadata"]["source_modified"] == "2025-02-03T04:05:06Z"


def test_resolve_channel_multiline_description_has_no_spacer_line(monkeypatch) -> None:
    channel = _channel(
        "root",
        1,
        "Root",
        0,
        description={"markdown": "First paragraph.\n\nSecond paragraph."},
    )

    monkeypatch.setattr(arena, "_fetch_channel", lambda slug: channel)
    monkeypatch.setattr(
        arena,
        "_fetch_channel_page",
        lambda slug, page, per=100: {"contents": [], "meta": {"total_pages": 1}},
    )

    docs = arena_plugin.resolve(
        "https://www.are.na/owner/root",
        {"use_cache": False},
    )

    assert "Description:\nFirst paragraph." in docs[0]["content"]
    assert "Description:\n\nFirst paragraph." not in docs[0]["content"]


def test_resolve_channel_block_paths_do_not_double_prefix_slug(monkeypatch) -> None:
    channel = _channel("root", 1, "Root", 1)

    monkeypatch.setattr(arena, "_fetch_channel", lambda slug: channel)
    monkeypatch.setattr(
        arena,
        "_fetch_channel_page",
        lambda slug, page, per=100: {
            "contents": [_text_block(100)],
            "meta": {"total_pages": 1},
        },
    )

    docs = arena_plugin.resolve(
        "https://www.are.na/owner/root",
        {"use_cache": False},
    )

    assert docs[1]["metadata"]["source_path"] == "root/100"
    assert docs[1]["metadata"]["context_subpath"] == "root/100.md"


def test_resolve_channel_preserves_block_connection_context(monkeypatch) -> None:
    channel = _channel("root", 1, "Root", 1)
    block = {
        **_text_block(100),
        "connection": {
            "id": 9001,
            "position": 3,
            "connected_at": "2026-05-24T21:50:27Z",
            "connected_by": {
                "id": 2,
                "name": "Adder Person",
                "slug": "adder-person",
            },
        },
    }

    monkeypatch.setattr(arena, "_fetch_channel", lambda slug: channel)
    monkeypatch.setattr(
        arena,
        "_fetch_channel_page",
        lambda slug, page, per=100: {
            "contents": [block],
            "meta": {"total_pages": 1},
        },
    )

    _metadata, flat = resolve_channel(
        "root",
        use_cache=False,
        settings=ArenaSettings(sort_order="asc", include_connections=False),
    )

    resolved = flat[0][1]
    assert resolved["connected_at"] == "2026-05-24T21:50:27Z"
    assert resolved["connected_by"]["slug"] == "adder-person"
    assert resolved["_contextualize_channel_context"]["slug"] == "root"
    assert resolved["_contextualize_channel_context"]["owner"]["slug"] == "owner"


def test_list_targets_expands_user_channels_and_applies_exclusions(monkeypatch) -> None:
    def _fetch_owner_channel_page(kind, slug, page, *, per=100, sort="created_at_asc"):
        assert kind == "user"
        assert slug == "alice"
        assert sort == "created_at_asc"
        return {
            "data": [
                _channel("keep", 10, "Keep", 0),
                _channel("skip", 11, "Skip", 0),
            ],
            "meta": {"total_pages": 1},
        }

    monkeypatch.setattr(arena, "_fetch_owner_channel_page", _fetch_owner_channel_page)

    items = arena_plugin.list_targets(
        "arena:user:alice",
        {
            "use_cache": False,
            "overrides": {"arena": {"exclude-channels": ["skip"]}},
        },
    )

    assert items == [
        {
            "target": "https://www.are.na/channel/keep",
            "label": "Keep",
            "kind": "channel",
            "metadata": {"owner_kind": "user", "owner_slug": "alice"},
        }
    ]


def test_list_targets_channel_exposes_child_blocks(monkeypatch) -> None:
    monkeypatch.setattr(
        arena,
        "resolve_channel",
        lambda slug, **_kwargs: (
            _channel(slug, 1, "Root", 2),
            [
                ("root", _text_block(100)),
                ("root", _channel("nested", 200, "Nested", 0)),
            ],
        ),
    )

    items = arena_plugin.list_targets(
        "https://www.are.na/channel/root",
        {"use_cache": False},
    )

    assert items == [
        {
            "target": "https://www.are.na/block/100",
            "label": "Block 100",
            "kind": "block",
            "metadata": {
                "relation": "contains",
                "block_id": 100,
                "block_type": "Text",
                "channel_path": "root",
                "source_channel": "https://www.are.na/channel/root",
                "nested_depth": 0,
            },
        },
        {
            "target": "https://www.are.na/channel/nested",
            "label": "Nested",
            "kind": "channel",
            "metadata": {
                "relation": "contains",
                "block_id": 200,
                "block_type": "Channel",
                "channel_path": "root",
                "source_channel": "https://www.are.na/channel/root",
                "nested_depth": 0,
                "channel_slug": "nested",
                "owner": {"id": 1, "slug": "owner", "name": "Owner"},
            },
        },
    ]


def test_arena_digest_returns_channel_overview(monkeypatch) -> None:
    monkeypatch.setattr(
        arena,
        "resolve_channel",
        lambda slug, **_kwargs: (
            _channel(slug, 1, "Root", 4),
            [
                ("root", _text_block(100)),
                ("root", _channel("nested", 200, "Nested", 0)),
            ],
        ),
    )

    digest = arena_plugin.arena_digest(
        "https://www.are.na/channel/root",
        {"use_cache": False, "sample_size": 2},
    )

    assert digest["channel"]["slug"] == "root"
    assert digest["pagination"] == {
        "sampleSize": 2,
        "requestedSampleSize": 2,
        "totalCount": 4,
        "hasMore": True,
    }
    assert digest["typeBreakdown"] == [
        {"type": "Channel", "count": 1},
        {"type": "Text", "count": 1},
    ]
    assert digest["sampledItems"][0]["target"] == "https://www.are.na/block/100"


def test_arena_targets_includes_containing_channel_for_block(monkeypatch) -> None:
    monkeypatch.setattr(
        arena,
        "_fetch_block",
        lambda block_id: {
            **_text_block(block_id),
            "source": {"url": "https://example.com/source", "title": "Source"},
        },
    )
    monkeypatch.setattr(
        arena,
        "_fetch_block_connections",
        lambda block_id, **_kwargs: (
            [
                {
                    "id": 5,
                    "slug": "container",
                    "title": "Container",
                    "owner": {"id": 2, "slug": "bob", "name": "Bob"},
                    "connection": {
                        "id": 44,
                        "position": 7,
                        "connected_at": "2026-06-01T00:00:00Z",
                        "connected_by": {"id": 3, "slug": "adder", "name": "Adder"},
                    },
                }
            ],
            False,
        ),
    )

    graph = arena_plugin.arena_targets(
        "https://www.are.na/block/335",
        {"use_cache": False},
    )

    assert graph["targets"] == [
        {
            "target": "https://example.com/source",
            "label": "Source",
            "kind": "link",
            "metadata": {
                "source": "source",
                "block_id": 335,
                "relation": "resolves_to",
                "source_target": "https://www.are.na/block/335",
            },
        },
        {
            "target": "https://www.are.na/channel/container",
            "label": "Container",
            "kind": "channel",
            "metadata": {
                "relation": "contained_by",
                "source_target": "https://www.are.na/block/335",
                "source_block_id": 335,
                "source_block_type": "Text",
                "channel_id": 5,
                "channel_slug": "container",
                "owner": {"id": 2, "slug": "bob", "name": "Bob"},
                "connection_id": 44,
                "position": 7,
                "connected_at": "2026-06-01T00:00:00Z",
                "connected_by": {"id": 3, "slug": "adder", "name": "Adder"},
            },
        },
    ]


def test_list_targets_block_exposes_structured_children(monkeypatch) -> None:
    monkeypatch.setattr(
        arena,
        "_fetch_block",
        lambda block_id: {
            "id": block_id,
            "type": "Attachment",
            "title": "Paper",
            "attachment": {
                "filename": "paper.pdf",
                "content_type": "application/pdf",
                "file_extension": "pdf",
                "url": "https://attachments.are.na/335/paper.pdf?1",
                "size": 2048,
            },
            "source": {"url": "https://example.com/source", "title": "Source"},
            "embed": {"url": "https://youtu.be/example", "title": "Video"},
            "image": {"src": "https://images.are.na/preview.jpg"},
        },
    )

    items = arena_plugin.list_targets(
        "https://www.are.na/block/335",
        {"use_cache": False},
    )

    assert items == [
        {
            "target": "https://www.are.na/block/335?attachment=paper.pdf",
            "label": "paper.pdf",
            "kind": "attachment:pdf",
            "metadata": {
                "block_id": 335,
                "filename": "paper.pdf",
                "content_type": "application/pdf",
                "url": "https://attachments.are.na/335/paper.pdf?1",
                "bytes": 2048,
                "source_block": "https://www.are.na/block/335",
            },
        },
        {
            "target": "https://example.com/source",
            "label": "Source",
            "kind": "link",
            "metadata": {"source": "source", "block_id": 335},
        },
        {
            "target": "https://youtu.be/example",
            "label": "Video",
            "kind": "embed",
            "metadata": {"source": "embed", "block_id": 335},
        },
        {
            "target": "https://images.are.na/preview.jpg",
            "label": "preview.jpg",
            "kind": "image",
            "metadata": {
                "source": "image",
                "image_index": 0,
                "block_id": 335,
            },
        },
    ]


def test_list_targets_reuses_channel_block_for_structured_children(monkeypatch) -> None:
    arena_plugin._LISTED_BLOCK_CACHE.clear()
    block = {
        "id": 335,
        "type": "Attachment",
        "title": "Paper",
        "attachment": {
            "filename": "paper.pdf",
            "content_type": "application/pdf",
            "file_extension": "pdf",
            "url": "https://attachments.are.na/335/paper.pdf?1",
            "size": 2048,
        },
    }
    monkeypatch.setattr(
        arena,
        "resolve_channel",
        lambda slug, **_kwargs: (_channel(slug, 1, "Root", 1), [("root", block)]),
    )
    monkeypatch.setattr(
        arena,
        "_fetch_block",
        lambda block_id: pytest.fail(f"unexpected block fetch: {block_id}"),
    )

    channel_items = arena_plugin.list_targets(
        "https://www.are.na/channel/root",
        {"use_cache": False},
    )
    block_items = arena_plugin.list_targets(
        "https://www.are.na/block/335",
        {"use_cache": False},
    )

    assert channel_items[0]["target"] == "https://www.are.na/block/335"
    assert block_items[0]["target"] == "https://www.are.na/block/335?attachment=paper.pdf"


def test_channel_listed_attachment_materialize_reuses_cached_block(
    monkeypatch, tmp_path: Path
) -> None:
    arena_plugin._LISTED_BLOCK_CACHE.clear()
    block = {
        "id": 335,
        "type": "Attachment",
        "title": "Notes",
        "updated_at": "2026-05-15T18:47:00Z",
        "attachment": {
            "filename": "notes.txt",
            "content_type": "text/plain",
            "file_extension": "txt",
            "url": "https://attachments.are.na/335/notes.txt?1",
        },
    }
    monkeypatch.setattr(
        arena,
        "resolve_channel",
        lambda slug, **_kwargs: (_channel(slug, 1, "Root", 1), [("root", block)]),
    )
    monkeypatch.setattr(
        arena,
        "_fetch_block",
        lambda block_id: pytest.fail(f"unexpected block fetch: {block_id}"),
    )

    payload_path = tmp_path / "notes.txt"
    payload_path.write_text("cached attachment text", encoding="utf-8")

    def _download(url: str, **_kwargs) -> Path:
        assert url == "https://attachments.are.na/335/notes.txt?1"
        return payload_path

    monkeypatch.setattr(
        "cx_plugins.providers.shared.media.download_cached_media_to_temp",
        _download,
    )

    channel_items = arena_plugin.list_targets(
        "https://www.are.na/channel/root",
        {"use_cache": False},
    )
    block_items = arena_plugin.list_targets(
        channel_items[0]["target"],
        {"use_cache": False},
    )
    files = arena_plugin.materialize(block_items[0]["target"], {"use_cache": False})

    assert files[0]["content"] == b"cached attachment text"


def test_arena_attachment_materialize_downloads_bytes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        arena,
        "_fetch_block",
        lambda block_id: {
            "id": block_id,
            "type": "Attachment",
            "updated_at": "2026-05-15T18:47:00Z",
            "attachment": {
                "filename": "paper.pdf",
                "content_type": "application/pdf",
                "file_extension": "pdf",
                "url": "https://attachments.are.na/335/paper.pdf?1",
            },
        },
    )

    payload_path = tmp_path / "paper.pdf"
    payload_path.write_bytes(b"pdf-bytes")

    def _download(url: str, **kwargs) -> Path:
        assert url == "https://attachments.are.na/335/paper.pdf?1"
        expected_identity = (
            "arena:block:335:2026-05-15T18:47:00Z:attachment:"
            "https://attachments.are.na/335/paper.pdf?1"
        )
        assert (
            kwargs["cache_identity"]
            == expected_identity
        )
        return payload_path

    monkeypatch.setattr(
        "cx_plugins.providers.shared.media.download_cached_media_to_temp",
        _download,
    )

    files = arena_plugin.materialize(
        "https://www.are.na/block/335?attachment=paper.pdf",
        {"use_cache": False, "refresh_cache": True},
    )

    assert files == [
        {
            "source": "https://www.are.na/block/335?attachment=paper.pdf",
            "label": "paper.pdf",
            "filename": "paper.pdf",
            "content": b"pdf-bytes",
            "content_type": "application/pdf",
            "metadata": {
                "provider": "arena",
                "kind": "attachment",
                "sourceBlockUrl": "https://www.are.na/block/335",
                "attachmentUrl": "https://attachments.are.na/335/paper.pdf?1",
                "attachmentName": "paper.pdf",
                "blockId": 335,
                "bytes": 9,
            },
        }
    ]


def test_render_pdf_attachment_fallback_preserves_block_metadata() -> None:
    rendered = arena._render_block(
        {
            "id": 1001,
            "type": "Attachment",
            "created_at": "2026-05-24T21:50:27Z",
            "title": "Research Packet",
            "description": {"markdown": "Synthetic PDF packet."},
            "attachment": {
                "filename": "research-packet.pdf",
                "content_type": "application/pdf",
                "file_size": 403548,
                "file_extension": "pdf",
                "url": "https://attachments.are.na/1001/research-packet.pdf?1",
            },
        },
        include_comments=False,
        include_connections=False,
        include_pdf_content=False,
        include_media_descriptions=False,
    )

    assert rendered == "\n".join(
        [
            "title   Research Packet",
            "created 2026-05-24T21:50Z",
            "---",
            "",
            "Synthetic PDF packet.",
            "",
            "***",
            "",
            "[Attachment: research-packet.pdf]",
            "Type: application/pdf",
            "Size: 403548 bytes",
            "URL: https://attachments.are.na/1001/research-packet.pdf?1",
        ]
    )


def test_render_pdf_attachment_fallback_dedupes_title_description() -> None:
    rendered = arena._render_block(
        {
            "id": 1002,
            "type": "Attachment",
            "created_at": "2016-08-22T21:20:18Z",
            "title": "Shared Reading Notes",
            "description": {
                "markdown": "Shared Reading Notes\n"
            },
            "attachment": {
                "filename": "reading-notes.pdf",
                "content_type": "application/pdf",
                "file_size": 1011519,
                "file_extension": "pdf",
                "url": "https://attachments.are.na/1002/reading-notes.pdf?1",
            },
        },
        include_comments=False,
        include_connections=False,
        include_pdf_content=False,
        include_media_descriptions=False,
    )

    assert rendered.count("Shared Reading Notes") == 1
    assert "Size: 1011519 bytes" in rendered


def test_render_pdf_attachment_fallback_uses_preview_image_when_enabled(
    monkeypatch,
) -> None:
    calls = []

    def _render_preview(url: str, suffix: str, **kwargs) -> str:
        calls.append((url, suffix, kwargs))
        return "Image size: 1582 x 994\n\nA scanned title page."

    monkeypatch.setattr(arena, "_render_block_binary", _render_preview)

    rendered = arena._render_block(
        {
            "id": 1001,
            "type": "Attachment",
            "created_at": "2026-05-24T21:50:27Z",
            "title": "Research Packet",
            "attachment": {
                "filename": "paper.pdf",
                "content_type": "application/pdf",
                "file_size": 403548,
                "file_extension": "pdf",
                "url": "https://attachments.are.na/1001/paper.pdf?1",
            },
            "image": {
                "src": "https://images.are.na/preview.png",
                "width": 1582,
                "height": 994,
            },
        },
        include_comments=False,
        include_connections=False,
        include_pdf_content=False,
        include_media_descriptions=True,
    )

    assert calls[0][0] == "https://images.are.na/preview.png"
    assert calls[0][1] == ".png"
    assert (
        "## Block image description (auto-generated, dimensions: 1582x994)"
        in rendered
    )
    assert "A scanned title page." in rendered


def test_render_pdf_attachment_preview_ignores_old_render_cache(
    monkeypatch,
) -> None:
    seen_variants = []

    def _cached(_block_id, _updated_at, *, render_variant):
        seen_variants.append(render_variant)
        if "attachment-fallback=2" not in render_variant:
            return "[Attachment: stale.pdf]"
        return None

    monkeypatch.setattr(
        "cx_plugins.providers.arena.cache.get_cached_block_render",
        _cached,
    )
    monkeypatch.setattr(
        arena,
        "_render_block_binary",
        lambda *_args, **_kwargs: "Image size: 100 x 100\n\nFresh preview.",
    )

    rendered = arena._render_block(
        {
            "id": 1001,
            "type": "Attachment",
            "updated_at": "2026-05-24T22:44:17Z",
            "title": "Research Packet",
            "attachment": {
                "filename": "paper.pdf",
                "content_type": "application/pdf",
                "file_size": 403548,
                "file_extension": "pdf",
                "url": "https://attachments.are.na/1001/paper.pdf?1",
            },
            "image": {"src": "https://images.are.na/preview.png"},
        },
        include_comments=False,
        include_connections=False,
        include_pdf_content=False,
        include_media_descriptions=True,
    )

    assert seen_variants
    assert all("attachment-fallback=2" in variant for variant in seen_variants)
    assert "[Attachment: stale.pdf]" not in rendered
    assert "Fresh preview." in rendered


def test_render_channel_block_uses_header_added_line_and_other_channels(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "cx_plugins.providers.arena.cache.get_cached_block_connections",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "cx_plugins.providers.arena.cache.store_block_connections",
        lambda *_args, **_kwargs: None,
    )

    def _fetch_connections_page(block_id, page, per=100):
        assert block_id == 100
        assert page == 1
        return {
            "data": [
                _channel("root", 1, "Root", 2),
                {
                    **_channel("other", 2, "Other", 3),
                    "visibility": "public",
                    "owner": {
                        "id": 3,
                        "name": "Neighbor Owner",
                        "slug": "neighbor-owner",
                    },
                },
            ],
            "meta": {"total_pages": 1},
        }

    monkeypatch.setattr(arena, "_fetch_block_connections_page", _fetch_connections_page)
    block = {
        **_text_block(100),
        "created_at": "2026-05-20T10:00:00Z",
        "user": {"id": 4, "name": "Creator Person", "slug": "creator-person"},
        "_contextualize_channel_context": _channel("root", 1, "Root", 2),
        "connection": {
            "id": 9001,
            "position": 20,
            "connected_at": "2026-05-24T21:50:27Z",
            "connected_by": {
                "id": 2,
                "name": "Adder Person",
                "slug": "adder-person",
            },
        },
    }

    rendered = arena._render_block(
        block,
        include_comments=False,
        include_connections=True,
        connections_max_items=30,
    )

    assert rendered.startswith(
        "\n".join(
            [
                "title   Block 100",
                "created 2026-05-20T10:00Z by Creator Person",
                "added   2026-05-24T21:50Z by Adder Person",
                "---",
            ]
        )
    )
    assert "## Added to Channel" not in rendered
    assert "## Other channels" in rendered
    assert "- Other (Neighbor Owner, public; 3 blocks; other)" in rendered
    assert "- Root (Owner; 2 blocks; root)" not in rendered


def test_render_direct_block_shows_all_connected_channels(monkeypatch) -> None:
    monkeypatch.setattr(
        "cx_plugins.providers.arena.cache.get_cached_block_connections",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "cx_plugins.providers.arena.cache.store_block_connections",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        arena,
        "_fetch_block_connections_page",
        lambda block_id, page, per=100: {
            "data": [
                _channel("root", 1, "Root", 2),
                _channel("other", 2, "Other", 3),
            ],
            "meta": {"total_pages": 1},
        },
    )

    block = {
        **_text_block(100),
        "created_at": "2026-05-20T10:00:00Z",
        "user": {"id": 4, "name": "Creator Person", "slug": "creator-person"},
    }

    rendered = arena._render_block(
        block,
        include_comments=False,
        include_connections=True,
        connections_max_items=30,
    )

    assert "## Added to Channel" not in rendered
    assert rendered.startswith(
        "title   Block 100\ncreated 2026-05-20T10:00Z by Creator Person\n---"
    )
    assert "added   " not in rendered
    assert "## Channels" in rendered
    assert "- Root (Owner; 2 blocks; root)" in rendered
    assert "- Other (Owner; 3 blocks; other)" in rendered


def test_render_connected_channels_reports_limit(monkeypatch) -> None:
    monkeypatch.setattr(
        "cx_plugins.providers.arena.cache.get_cached_block_connections",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "cx_plugins.providers.arena.cache.store_block_connections",
        lambda *_args, **_kwargs: None,
    )
    channels = [
        _channel(f"channel-{index}", index, f"Channel {index}", 1)
        for index in range(1, 32)
    ]
    monkeypatch.setattr(
        arena,
        "_fetch_block_connections_page",
        lambda block_id, page, per=100: {
            "data": channels,
            "meta": {"total_pages": 1},
        },
    )

    rendered = arena._render_block(
        _text_block(100),
        include_comments=False,
        include_connections=True,
        connections_max_items=30,
    )

    limit_message = (
        "Showing first 30 channels; "
        "more omitted by limit."
    )
    assert limit_message in rendered
    assert "- Channel 30 (Owner; 1 block; channel-30)" in rendered
    assert "- Channel 31 " not in rendered


def test_resolve_arena_attachment_target_uses_attachment_url_reference(monkeypatch) -> None:
    monkeypatch.setattr(
        arena,
        "_fetch_block",
        lambda block_id: {
            "id": block_id,
            "type": "Attachment",
            "connected_at": "2026-05-15T18:47:00Z",
            "updated_at": "2026-05-15T18:48:00Z",
            "attachment": {
                "filename": "paper.pdf",
                "content_type": "application/pdf",
                "url": "https://attachments.are.na/335/paper.pdf?1",
            },
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
            assert url == "https://attachments.are.na/335/paper.pdf?1"
            assert format == "raw"
            assert label == "name"
            assert filename_override == "paper.pdf"
            assert use_cache is False
            assert refresh_cache is True
            assert plugin_overrides == {"transcribe": {"provider": "mistral"}}

        def read(self) -> str:
            return "paper text"

    monkeypatch.setattr("contextualize.references.url.URLReference", _DummyReference)

    docs = arena_plugin.resolve(
        "https://www.are.na/block/335?attachment=paper.pdf",
        {
            "use_cache": False,
            "refresh_cache": True,
            "overrides": {"transcribe": {"provider": "mistral"}},
        },
    )

    assert len(docs) == 1
    assert docs[0]["content"] == "paper text"
    assert docs[0]["metadata"]["kind"] == "attachment"
    assert docs[0]["metadata"]["blockId"] == 335
    assert docs[0]["metadata"]["attachmentName"] == "paper.pdf"


def test_resolve_user_emits_profile_doc_with_groups(monkeypatch) -> None:
    monkeypatch.setattr(
        arena,
        "fetch_owner_profile",
        lambda kind, slug, **_kwargs: _user_profile(slug, "Alice"),
    )
    monkeypatch.setattr(
        arena,
        "fetch_user_groups",
        lambda slug, **_kwargs: [
            {
                **_group_profile("studio", "Studio"),
                "counts": {"channels": 5, "users": 6},
            }
        ],
    )
    monkeypatch.setattr(arena, "fetch_owner_channels", lambda kind, slug, **_kwargs: [])

    docs = arena_plugin.resolve("arena:user:alice", {"use_cache": False})

    assert len(docs) == 1
    assert docs[0]["metadata"]["context_subpath"] == "users/alice/_user.md"
    assert docs[0]["content"] == "\n".join(
        [
            "[User: Alice]",
            "Joined: 2024-01-02T03:04Z",
            "Modified: 2024-02-03T04:05Z",
            "Channels: 2",
            "Followers: 3",
            "Following: 4",
            "https://www.are.na/alice",
            "Info: Building a garden.",
            "Groups:",
            "- Studio (@studio), Owner: Owner (@owner), Channels: 5, Members: 6",
        ]
    )


def test_resolve_group_emits_profile_doc_with_owner_and_counts(monkeypatch) -> None:
    monkeypatch.setattr(
        arena,
        "fetch_owner_profile",
        lambda kind, slug, **_kwargs: _group_profile(slug, "Studio"),
    )
    monkeypatch.setattr(arena, "fetch_owner_channels", lambda kind, slug, **_kwargs: [])

    docs = arena_plugin.resolve("arena:group:studio", {"use_cache": False})

    assert len(docs) == 1
    assert docs[0]["metadata"]["context_subpath"] == "groups/studio/_group.md"
    assert docs[0]["content"] == "\n".join(
        [
            "[Group: Studio]",
            "Owner: Owner (@owner)",
            "Channels: 5",
            "Members: 6",
            "Created: 2025-01-02T03:04Z",
            "Modified: 2025-02-03T04:05Z",
            "https://www.are.na/studio",
            "Info: Shared workbench.",
        ]
    )


def test_resolve_owner_profile_remains_when_channels_are_excluded(monkeypatch) -> None:
    monkeypatch.setattr(
        arena,
        "_fetch_owner_profile",
        lambda kind, slug: _user_profile(slug, "Alice"),
    )
    monkeypatch.setattr(
        arena,
        "_fetch_user_groups_page",
        lambda slug, page, *, per=100, sort="name_asc": {
            "data": [],
            "meta": {"total_pages": 1},
        },
    )
    monkeypatch.setattr(
        arena,
        "_fetch_owner_channel_page",
        lambda kind, slug, page, *, per=100, sort="created_at_asc": {
            "data": [_channel("skip", 10, "Skip", 0)],
            "meta": {"total_pages": 1},
        },
    )

    docs = arena_plugin.resolve(
        "arena:user:alice",
        {
            "use_cache": False,
            "overrides": {"arena": {"exclude-channels": ["skip"]}},
        },
    )

    assert [doc["metadata"]["context_subpath"] for doc in docs] == [
        "users/alice/_user.md"
    ]


def test_list_targets_bare_profile_url_falls_back_to_group(monkeypatch) -> None:
    calls: list[str] = []

    def _fetch_owner_channel_page(kind, slug, page, *, per=100, sort="created_at_asc"):
        calls.append(kind)
        assert slug == "media-working-group"
        if kind == "user":
            raise ValueError(
                "Are.na resource not found: /users/media-working-group/contents"
            )
        return {
            "data": [_channel("root", 10, "Root", 0)],
            "meta": {"total_pages": 1},
        }

    monkeypatch.setattr(arena, "_fetch_owner_channel_page", _fetch_owner_channel_page)

    items = arena_plugin.list_targets(
        "https://www.are.na/media-working-group",
        {"use_cache": False},
    )

    assert calls == ["user", "group"]
    assert items == [
        {
            "target": "https://www.are.na/channel/root",
            "label": "Root",
            "kind": "channel",
            "metadata": {
                "owner_kind": "group",
                "owner_slug": "media-working-group",
            },
        }
    ]


def test_explicit_user_target_does_not_fallback_to_group(monkeypatch) -> None:
    calls: list[str] = []

    def _fetch_owner_channel_page(kind, slug, page, *, per=100, sort="created_at_asc"):
        calls.append(kind)
        raise ValueError("Are.na resource not found: /users/alice/contents")

    monkeypatch.setattr(arena, "_fetch_owner_channel_page", _fetch_owner_channel_page)

    with pytest.raises(ValueError, match="/users/alice/contents"):
        arena_plugin.list_targets("arena:user:alice", {"use_cache": False})

    assert calls == ["user"]


def test_resolve_bare_profile_url_uses_group_prefix_after_fallback(monkeypatch) -> None:
    calls: list[str] = []

    def _fetch_owner_profile(kind, slug, **_kwargs):
        calls.append(kind)
        assert slug == "media-working-group"
        if kind == "user":
            raise ValueError("Are.na resource not found: /users/media-working-group")
        return _group_profile(slug, "Media Working Group")

    def _fetch_owner_channels(kind, slug, **_kwargs):
        assert slug == "media-working-group"
        assert kind == "group"
        return [_channel("root", 1, "Root", 1)]

    monkeypatch.setattr(arena, "fetch_owner_profile", _fetch_owner_profile)
    monkeypatch.setattr(arena, "fetch_owner_channels", _fetch_owner_channels)
    monkeypatch.setattr(
        arena,
        "resolve_channel",
        lambda slug, **_kwargs: (
            _channel(slug, 1, "Root", 1),
            [(slug, _text_block(100))],
        ),
    )

    docs = arena_plugin.resolve(
        "https://www.are.na/media-working-group",
        {"use_cache": False},
    )

    assert calls == ["user", "group"]
    paths = [doc["metadata"]["context_subpath"] for doc in docs]
    assert paths[0] == "groups/media-working-group/_group.md"
    assert paths[1] == "groups/media-working-group/root/_channel.md"
    assert all(path.startswith("groups/media-working-group/") for path in paths)


def test_resolve_user_prefixes_each_channel_under_owner_path(monkeypatch) -> None:
    monkeypatch.setattr(
        arena,
        "fetch_owner_profile",
        lambda kind, slug, **_kwargs: _user_profile(slug, "Alice"),
    )
    monkeypatch.setattr(arena, "fetch_user_groups", lambda slug, **_kwargs: [])
    monkeypatch.setattr(
        arena,
        "fetch_owner_channels",
        lambda kind, slug, **_kwargs: [_channel("root", 1, "Root", 1)],
    )
    monkeypatch.setattr(
        arena,
        "resolve_channel",
        lambda slug, **_kwargs: (
            _channel(slug, 1, "Root", 1),
            [(slug, _text_block(100))],
        ),
    )

    docs = arena_plugin.resolve("arena:user:alice", {"use_cache": False})

    paths = [doc["metadata"]["context_subpath"] for doc in docs]
    assert paths[0] == "users/alice/_user.md"
    assert paths[1] == "users/alice/root/_channel.md"
    assert all(path.startswith("users/alice/") for path in paths)


def test_excluded_nested_channel_prunes_branch_without_excluding_repeated_blocks(
    monkeypatch,
) -> None:
    root_block = _text_block(7)
    child_block = _text_block(7)

    monkeypatch.setattr(
        arena,
        "_fetch_channel",
        lambda slug: {
            "root": _channel("root", 1, "Root", 2),
            "skip": _channel("skip", 2, "Skip", 1),
        }[slug],
    )

    def _fetch_channel_page(slug, page, per=100):
        if slug == "skip":
            raise AssertionError("excluded channel should not be fetched")
        return {
            "root": {
                "contents": [_channel("skip", 2, "Skip", 1), root_block],
                "meta": {"total_pages": 1},
            },
            "skip": {"contents": [child_block], "meta": {"total_pages": 1}},
        }[slug]

    monkeypatch.setattr(arena, "_fetch_channel_page", _fetch_channel_page)

    _metadata, flat = resolve_channel(
        "root",
        use_cache=False,
        settings=ArenaSettings(
            max_depth=1,
            sort_order="asc",
            recurse_users=None,
            exclude_channels=parse_arena_channel_exclusions(["skip"]),
        ),
    )

    assert [block["id"] for _, block in flat] == [7]


def test_recurse_blocks_ratio_caps_nested_channels_and_keeps_stub(monkeypatch) -> None:
    child_blocks = [_text_block(i) for i in range(1000, 1100)]
    child_summary = _channel("child", 2, "Child", 100)
    child_metadata = _channel(
        "child",
        2,
        "Child",
        100,
        description={"markdown": "Nested channel metadata."},
        created_at="2025-03-04T05:06:07Z",
        updated_at="2025-04-05T06:07:08Z",
    )

    monkeypatch.setattr(
        arena,
        "_fetch_channel",
        lambda slug: {
            "root": _channel("root", 1, "Root", 1000),
            "child": child_metadata,
        }[slug],
    )
    monkeypatch.setattr(
        arena,
        "_fetch_channel_page",
        lambda slug, page, per=100: {
            "root": {
                "contents": [child_summary],
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
    assert flat[0][1]["description"] == {"markdown": "Nested channel metadata."}
    assert flat[0][1]["created_at"] == "2025-03-04T05:06:07Z"
    assert flat[0][1]["updated_at"] == "2025-04-05T06:07:08Z"
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
