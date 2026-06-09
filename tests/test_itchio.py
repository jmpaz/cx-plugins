from __future__ import annotations

from types import SimpleNamespace

import click
import requests

from cx_plugins.providers.itchio import cache, itchio, plugin
from cx_plugins.providers.itchio.itchio import ItchioMedia, ItchioSettings


GAME_URL = "https://demo.itch.io/widget"
DEVLOG_URL = "https://demo.itch.io/widget/devlog/7/update"
DEVLOG_URL_2 = "https://demo.itch.io/widget/devlog/8/fix"
COLLECTION_URL = "https://itch.io/c/123/synthetic-collection"
PROFILE_URL = "https://maker.itch.io/"
QUEUE_URL = "https://itch.io/queue/c/123/synthetic-collection?game_id=2"


GAME_HTML = """
<html>
  <head>
    <title>Widget - itch.io</title>
    <meta name="itch:path" content="games/42">
    <meta property="og:title" content="Widget">
    <meta name="theme-color" content="#223344">
    <style id="game_theme">
      :root {
        --itchio_bg_color: #112233;
        --itchio_fg_color: #eeeeee;
        --itchio_link_color: #44aaee;
      }
    </style>
    <link rel="alternate" type="application/rss+xml" href="/widget/devlog.rss">
  </head>
  <body>
    <h1 class="game_title">Widget</h1>
    <div class="formatted_description">
      <p>Build <strong>tiny</strong> tools ✨ with <a href="/widget/devlog/7/update">notes</a>.</p>
      <h5><a href="https://example.com/included">See what's included</a></h5>
      <img src="https://img.itch.zone/a/original.png" alt="Inline preview">
      <img src="https://img.itch.zone/a/generic.png" alt="image">
      <a href="https://example.com/patron">
        <img src="https://img.itch.zone/a/become_a_patron_button.png" alt="Become a Patron!">
      </a>
      <h6>You might have seen this</h6>
      <table class="feature_grid">
        <tr>
          <td><h3>Compatibility</h3><p>Works everywhere.</p></td>
          <td><h3>Questions</h3><p>Contact support.</p></td>
        </tr>
      </table>
    </div>
    <div class="game_info_panel_widget">
      <table>
        <tr><td>Status</td><td>Released</td></tr>
        <tr><td>Platforms</td><td>Windows, Linux</td></tr>
      </table>
    </div>
    <div class="uploads">
      <div class="upload"><a href="https://demo.itch.io/widget/download/main">widget.zip</a> 10 MB</div>
    </div>
    <div class="screenshot_list">
      <a href="https://img.itch.zone/a/original.png">
        <img src="https://img.itch.zone/a/thumb.png" alt="Screenshot one" width="640" height="480">
      </a>
      <a href="https://img.itch.zone/a/original/%2Fslash.png">
        <img src="https://img.itch.zone/a/second-thumb.png" alt="Screenshot two" width="640" height="480">
      </a>
    </div>
    <div class="game_devlog_widget">
      <a href="/widget/devlog/7/update">Version 1</a>
      <a href="/widget/devlog/8/fix">Fix notes</a>
    </div>
    <div class="community_post">
      <span class="post_author"><a>Ada</a></span>
      <time datetime="2026-01-01">Jan 1</time>
      <div class="post_body"><p>First comment.</p></div>
    </div>
    <div class="community_post">
      <span class="post_author"><a>Bea</a></span>
      <time datetime="2026-01-02">Jan 2</time>
      <div class="post_body"><p>Second comment.</p></div>
    </div>
    <div class="community_post">
      <span class="post_author"><a>Cy</a></span>
      <time datetime="2026-01-03">Jan 3</time>
      <div class="post_body"><p>Third comment.</p></div>
    </div>
  </body>
</html>
"""


DEVLOG_HTML = """
<html>
  <head>
    <meta property="og:title" content="Version 1">
  </head>
  <body class="game_devlog_post_page">
    <h1 class="post_title">Version 1</h1>
    <div class="post_body"><p>Changed the renderer.</p></div>
    <ul class="post_files">
      <li><a href="https://demo.itch.io/widget/download/patch">patch.zip</a></li>
    </ul>
  </body>
</html>
"""


DEVLOG_HTML_2 = """
<html>
  <head><meta property="og:title" content="Fix notes"></head>
  <body class="game_devlog_post_page">
    <h1 class="post_title">Fix notes</h1>
    <div class="post_body"><p>Fixed packaging.</p></div>
  </body>
</html>
"""


COLLECTION_HTML = """
<html>
  <head>
    <meta name="itch:path" content="collections/123">
    <meta property="og:title" content="Synthetic Collection">
    <link rel="next" href="/c/123/synthetic-collection?page=2">
  </head>
  <body>
    <h1 class="collection_title">Synthetic Collection</h1>
    <div class="grid_header">
      <div class="sub_header">
        a collection by <a href="https://maker.itch.io">Maker</a>
        · last updated <span class="date_format">2026-02-03 04:05:06</span>
      </div>
    </div>
    <div class="game_cell has_cover" data-game_id="1">
      <div class="game_thumb">
        <a class="thumb_link game_link" href="https://one.itch.io/tool">
          <img data-lazy_src="https://img.itch.zone/a/tool-cover.png" width="315" height="250">
        </a>
        <a href="/g/one/tool/add-to-collection">Add to collection</a>
      </div>
      <div class="game_cell_data">
        <div class="game_title">
          <a class="title game_link" href="https://one.itch.io/tool">Tool One</a>
          <div class="price_tag meta_tag" title="Pay $2 or more for this tool">
            <div class="price_value">$2</div>
          </div>
        </div>
        <div class="game_text" title="Short playable tool summary.">Short playable tool summary.</div>
        <div class="game_author"><a href="https://one.itch.io">One</a></div>
        <div class="game_genre">Puzzle</div>
        <div class="game_platform">
          <span class="web_flag">Play in browser</span>
          <span title="Download for Windows" class="icon icon-windows8"></span>
          <span title="Download for Linux" class="icon icon-tux"></span>
        </div>
        <div class="game_updated">Updated 10 days ago</div>
      </div>
    </div>
    <div class="game_cell">
      <a class="title" href="https://two.itch.io/game">Game Two</a>
      <span class="game_author">Two</span>
      <div class="game_text">Second summary.</div>
    </div>
  </body>
</html>
"""


PROFILE_HTML = """
<html>
  <head>
    <meta name="itch:path" content="users/88">
    <meta property="og:title" content="Maker">
    <style id="user_theme">:root { --itchio_bg_color: #101010; }</style>
  </head>
  <body>
    <div class="user_profile formatted"><p>Public maker profile.</p></div>
    <div class="game_cell">
      <a class="title" href="/profile-game">Profile Game</a>
    </div>
    <div class="collection_row">
      <a href="https://itch.io/c/456/profile-set">Profile Set</a>
    </div>
  </body>
</html>
"""


QUEUE_HTML = """
<html>
  <head>
    <meta property="og:title" content="Queue">
    <meta property="og:url" content="https://current.itch.io/game">
  </head>
  <body>
    <script>
      R.Queue.Viewer({
        "games_before": [{"title": "Before", "url": "https://before.itch.io/game"}],
        "game": {"title": "Current", "url": "https://current.itch.io/game"},
        "next_item": {"title": "Next", "url": "https://next.itch.io/game"},
        "games_after": [{"title": "After", "url": "https://after.itch.io/game"}]
      });
    </script>
  </body>
</html>
"""


def _fixture_fetch(fixtures: dict[str, str]):
    def fetch(url: str, **_kwargs) -> str:
        return fixtures[url]

    return fetch


def test_fetch_itchio_html_prefers_utf8_when_response_claims_latin1(monkeypatch) -> None:
    response = requests.Response()
    response.status_code = 200
    response.encoding = "ISO-8859-1"
    response._content = "Build tiny tools ✨".encode("utf-8")

    monkeypatch.setattr(itchio.requests, "get", lambda *_args, **_kwargs: response)

    assert itchio.fetch_itchio_html(GAME_URL, use_cache=False) == "Build tiny tools ✨"


def test_html_cache_identity_is_versioned(monkeypatch) -> None:
    reads: dict[str, object] = {}
    writes: dict[str, object] = {}

    def fake_read(root, identity, **kwargs):
        reads.update(root=root, identity=identity, kwargs=kwargs)
        return SimpleNamespace(value="cached-html", metadata={})

    def fake_write(root, identity, content, **kwargs):
        writes.update(root=root, identity=identity, content=content, kwargs=kwargs)

    monkeypatch.setattr(cache, "read_text_entry", fake_read)
    monkeypatch.setattr(cache, "write_text_entry", fake_write)

    assert cache.get_cached_html(GAME_URL) == "cached-html"
    cache.store_html(GAME_URL, "fresh-html")

    expected_identity = f"html-v{cache.HTML_CACHE_VERSION}:{GAME_URL}"
    assert reads["identity"] == expected_identity
    assert writes["identity"] == expected_identity
    assert writes["content"] == "fresh-html"
    assert writes["kwargs"]["extra_metadata"] == {
        "source_url": GAME_URL,
        "html_cache_version": cache.HTML_CACHE_VERSION,
    }


def test_settings_default_to_media_descriptions() -> None:
    assert itchio.build_itchio_settings(None).media_descriptions is True
    assert (
        itchio.build_itchio_settings({"media": {"describe": False}}).media_descriptions
        is False
    )


def test_resolve_game_page_renders_clean_media_tags_and_limited_comments(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        itchio,
        "fetch_itchio_html",
        _fixture_fetch({GAME_URL: GAME_HTML}),
    )
    monkeypatch.setattr(
        itchio,
        "_describe_media",
        lambda media: f"Generated description for {media.filename}.",
    )

    docs = itchio.resolve_itchio_target(
        GAME_URL,
        settings=ItchioSettings(comments_limit=2),
    )

    assert len(docs) == 1
    assert docs[0]["label"] == GAME_URL
    content = docs[0]["content"]
    assert "Build **tiny** tools ✨" in content
    assert "[See what's included](https://example.com/included)" in content
    assert "**You might have seen this**" in content
    assert "##### " not in content
    assert "###### " not in content
    assert (
        '<image filename="original.png" caption="Inline preview">\n'
        "Generated description for original.png.\n"
        "</image>"
    ) in content
    assert (
        '<image filename="generic.png">\n'
        "Generated description for generic.png.\n"
        "</image>"
    ) in content
    assert (
        '<image filename="become_a_patron_button.png" caption="Become a Patron!">\n'
        "Generated description for become_a_patron_button.png.\n"
        "</image>"
    ) in content
    assert "[<image" not in content
    assert (
        '<image filename="slash.png">\n'
        "Generated description for slash.png.\n"
        "</image>"
    ) in content
    assert 'caption="image"' not in content
    assert 'caption="Screenshot two"' not in content
    assert "![image]" not in content
    assert "itchio:media:" not in content
    assert "### Compatibility" in content
    assert "Works everywhere." in content
    assert "### Questions" in content
    assert "Contact support." in content
    assert "## Theme" in content
    assert "- bg-color: `#112233`" in content
    assert "## More Information" in content
    assert "- Status: Released" in content
    assert "## Files" in content
    assert "widget.zip" in content
    assert "## Media" in content
    assert "## Development Log" in content
    assert "[Version 1](devlog/7/update)" in content
    assert f"[Version 1]({DEVLOG_URL})" not in content
    assert "\n***\n\n## Comments" in content
    assert "First comment." in content
    assert "Second comment." in content
    assert "Third comment." not in content
    assert docs[0]["metadata"]["theme"]["bg-color"] == "#112233"


def test_list_targets_keeps_materializable_media_targets(monkeypatch) -> None:
    monkeypatch.setattr(
        itchio,
        "fetch_itchio_html",
        _fixture_fetch({GAME_URL: GAME_HTML}),
    )

    listing = plugin.list_targets(GAME_URL, {})

    media_targets = [
        item for item in listing["targets"] if item["kind"].startswith("media:")
    ]
    assert len(media_targets) == 1
    assert media_targets[0]["target"].startswith("itchio:media:game:games_42:0:")
    assert media_targets[0]["label"] == "slash.png"
    assert media_targets[0]["metadata"]["url"] == (
        "https://img.itch.zone/a/original/%2Fslash.png"
    )


def test_include_devlogs_returns_child_documents(monkeypatch) -> None:
    monkeypatch.setattr(
        itchio,
        "fetch_itchio_html",
        _fixture_fetch(
            {
                GAME_URL: GAME_HTML,
                DEVLOG_URL: DEVLOG_HTML,
                DEVLOG_URL_2: DEVLOG_HTML_2,
            }
        ),
    )

    docs = itchio.resolve_itchio_target(
        GAME_URL,
        settings=ItchioSettings(
            include_devlogs=True,
            devlogs_limit=1,
            media_descriptions=False,
        ),
    )

    assert [doc["label"] for doc in docs] == [GAME_URL, DEVLOG_URL]
    assert "Changed the renderer." in docs[1]["content"]
    assert "patch.zip" in docs[1]["content"]


def test_list_targets_collection_pages_game_grid_and_next_page(monkeypatch) -> None:
    monkeypatch.setattr(
        itchio,
        "fetch_itchio_html",
        _fixture_fetch({COLLECTION_URL: COLLECTION_HTML}),
    )
    describe_calls: list[ItchioMedia] = []
    monkeypatch.setattr(itchio, "_describe_media", describe_calls.append)

    listing = plugin.list_targets(COLLECTION_URL, {})

    targets = listing["targets"]
    game_targets = [item for item in targets if item["kind"] == "game"]
    assert [item["target"] for item in game_targets] == [
        "https://one.itch.io/tool",
        "https://two.itch.io/game",
    ]
    assert game_targets[0]["metadata"]["description"] == "Short playable tool summary."
    assert game_targets[0]["metadata"]["author"] == "One"
    assert game_targets[0]["metadata"]["genre"] == "Puzzle"
    assert game_targets[0]["metadata"]["platforms"] == [
        "Play in browser",
        "Windows",
        "Linux",
    ]
    assert "image" not in game_targets[0]["metadata"]
    assert game_targets[0]["metadata"]["price"] == "$2"
    assert game_targets[0]["metadata"]["lastUpdated"] == "Updated 10 days ago"
    assert not any("add-to-collection" in item["target"] for item in targets)
    assert listing["pagination"]["hasMore"] is True
    assert listing["pagination"]["next"] == (
        "https://itch.io/c/123/synthetic-collection?page=2"
    )
    assert listing["metadata"]["kind"] == "collection"

    docs = plugin.resolve(COLLECTION_URL, {})
    content = docs[0]["content"]
    assert "- Author: Maker" in content
    assert "- Last updated: 2026-02-03 04:05:06" in content
    assert "### [Tool One](https://one.itch.io/tool)" in content
    assert "tool-cover.png" not in content
    assert "<image" not in content
    assert "Short playable tool summary." in content
    assert "- Author: [One](https://one.itch.io)" in content
    assert "- Genre: Puzzle" in content
    assert "- Platforms: Play in browser, Windows, Linux" in content
    assert "- Price: $2" in content
    assert "- Last updated: Updated 10 days ago" in content
    assert "add-to-collection" not in content
    assert describe_calls == []


def test_profile_lists_games_and_collections(monkeypatch) -> None:
    monkeypatch.setattr(
        itchio,
        "fetch_itchio_html",
        _fixture_fetch({PROFILE_URL: PROFILE_HTML}),
    )

    listing = plugin.list_targets(PROFILE_URL, {})

    by_kind = {(item["kind"], item["label"]) for item in listing["targets"]}
    assert ("game", "Profile Game") in by_kind
    assert ("collection", "Profile Set") in by_kind
    assert listing["metadata"]["theme"]["bg-color"] == "#101010"


def test_queue_page_lists_neighbor_games(monkeypatch) -> None:
    monkeypatch.setattr(
        itchio,
        "fetch_itchio_html",
        _fixture_fetch({QUEUE_URL: QUEUE_HTML}),
    )

    listing = plugin.list_targets(QUEUE_URL, {})

    labels = [item["label"] for item in listing["targets"]]
    relations = [item["metadata"]["relation"] for item in listing["targets"]]
    assert labels == ["Before", "Current", "Next", "After"]
    assert relations == ["before", "current", "next", "after"]

    docs = plugin.resolve(QUEUE_URL, {})

    assert docs[0]["label"] == QUEUE_URL
    assert docs[0]["metadata"]["kind"] == "queue"
    assert f"url: {QUEUE_URL}" in docs[0]["content"]


def test_devlog_post_renders_body_and_files(monkeypatch) -> None:
    monkeypatch.setattr(
        itchio,
        "fetch_itchio_html",
        _fixture_fetch({DEVLOG_URL: DEVLOG_HTML}),
    )

    docs = plugin.resolve(DEVLOG_URL, {})

    assert len(docs) == 1
    assert docs[0]["label"] == DEVLOG_URL
    assert docs[0]["metadata"]["kind"] == "devlog-post"
    assert "Changed the renderer." in docs[0]["content"]
    assert "patch.zip" in docs[0]["content"]


def test_plugin_cli_overrides_and_classify() -> None:
    command = click.Command("cat")
    plugin.register_cli_options("cat", command)

    assert "itch_comments_limit" in {param.name for param in command.params}
    overrides = plugin.collect_cli_overrides(
        "cat",
        {
            "itch_theme": False,
            "itch_comments_limit": 4,
            "itch_comments_offset": 2,
            "itch_include_devlogs": True,
            "itch_devlogs_limit": 1,
            "itch_media": True,
            "itch_media_descriptions": True,
        },
    )
    assert overrides == {
        "theme": {"enabled": False},
        "comments": {"limit": 4, "offset": 2},
        "devlogs": {"include": True, "limit": 1},
        "media": {"enabled": True, "describe": True},
    }

    media_target = itchio.build_itchio_media_target(
        source_kind="game",
        source_id="games/42",
        index=0,
        media=ItchioMedia(kind="image", url="https://img.itch.zone/a/full.png"),
        source_url=GAME_URL,
    )
    assert plugin.classify_target(GAME_URL, {})["kind"] == "game"
    assert plugin.classify_target(COLLECTION_URL, {})["kind"] == "collection"
    assert plugin.classify_target(media_target, {})["kind"] == "media"


def test_materialize_media_target_uses_cached_bytes(monkeypatch) -> None:
    media_target = itchio.build_itchio_media_target(
        source_kind="game",
        source_id="games/42",
        index=0,
        media=ItchioMedia(
            kind="image",
            url="https://img.itch.zone/a/full.png",
            filename="full.png",
            caption="Full screenshot",
            content_type="image/png",
        ),
        source_url=GAME_URL,
    )
    monkeypatch.setattr(cache, "get_cached_media_bytes", lambda _identity: b"image-bytes")

    files = plugin.materialize(media_target, {"cache_only": True})

    assert files == [
        {
            "source": media_target,
            "label": "Full screenshot",
            "filename": "full.png",
            "content": b"image-bytes",
            "content_type": "image/png",
            "metadata": {
                "provider": "itchio",
                "kind": "image",
                "mediaUrl": "https://img.itch.zone/a/full.png",
                "sourceUrl": GAME_URL,
                "sourceKind": "game",
                "sourceId": "games_42",
                "bytes": 11,
            },
        }
    ]
