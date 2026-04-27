from datetime import datetime, timezone

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
