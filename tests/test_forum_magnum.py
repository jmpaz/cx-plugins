from __future__ import annotations

from typing import Any

from cx_plugins.providers.forum_magnum import forum_magnum
from cx_plugins.providers.forum_magnum import plugin as forum_magnum_plugin


def test_parse_forum_magnum_target_supports_requested_urls() -> None:
    ea = forum_magnum.parse_forum_magnum_target(
        "https://forum.effectivealtruism.org/posts/ckj6Moau9qpYArHWc/want-to-be-an-expert-build-deep-models"
    )
    assert ea is not None
    assert ea.platform == "ea_forum"
    assert ea.endpoint == "https://forum.effectivealtruism.org/graphql"
    assert ea.post_id == "ckj6Moau9qpYArHWc"

    lesswrong = forum_magnum.parse_forum_magnum_target(
        "https://www.lesswrong.com/posts/a2PHpr9huYkxpNNyx/the-most-important-meta-skill"
    )
    assert lesswrong is not None
    assert lesswrong.platform == "lesswrong"
    assert lesswrong.endpoint == "https://www.lesswrong.com/graphql"

    greaterwrong = forum_magnum.parse_forum_magnum_target(
        "https://greaterwrong.com/posts/a2PHpr9huYkxpNNyx/the-most-important-meta-skill"
    )
    assert greaterwrong is not None
    assert greaterwrong.platform == "lesswrong"
    assert greaterwrong.input_host == "greaterwrong.com"
    assert greaterwrong.endpoint == "https://www.lesswrong.com/graphql"

    af = forum_magnum.parse_forum_magnum_target(
        "https://www.alignmentforum.org/s/f2YA4eGskeztcJsqT/p/bxt7uCiHam4QXrQAA"
    )
    assert af is not None
    assert af.platform == "alignment_forum"
    assert af.post_id == "bxt7uCiHam4QXrQAA"


def test_parse_forum_magnum_target_supports_bare_and_www_hosts() -> None:
    host_expectations = {
        "lesswrong.com": "lesswrong",
        "www.lesswrong.com": "lesswrong",
        "greaterwrong.com": "lesswrong",
        "www.greaterwrong.com": "lesswrong",
        "alignmentforum.org": "alignment_forum",
        "www.alignmentforum.org": "alignment_forum",
        "forum.effectivealtruism.org": "ea_forum",
    }

    for host, platform in host_expectations.items():
        parsed = forum_magnum.parse_forum_magnum_target(
            f"https://{host}/posts/a2PHpr9huYkxpNNyx/the-most-important-meta-skill"
        )
        assert parsed is not None
        assert parsed.platform == platform
        assert parsed.input_host == host


def test_plugin_classifies_forum_magnum_urls() -> None:
    classified = forum_magnum_plugin.classify_target(
        "https://www.alignmentforum.org/s/f2YA4eGskeztcJsqT/p/bxt7uCiHam4QXrQAA",
        {},
    )

    assert classified == {
        "provider": "forum_magnum",
        "kind": "post",
        "is_external": True,
        "group_key": "alignment_forum",
        "metadata": {
            "platform": "alignment_forum",
            "post_id": "bxt7uCiHam4QXrQAA",
            "input_host": "www.alignmentforum.org",
        },
    }


def test_html_to_markdown_preserves_emphasis_boundary_spaces() -> None:
    markdown = forum_magnum._html_to_markdown(  # noqa: SLF001
        "<p><i>This piece is </i><a href='/post'>crossposted</a><i> on my blog.</i></p>",
        "https://forum.effectivealtruism.org/posts/post123/post",
    )

    assert (
        markdown
        == "*This piece is* [crossposted](https://forum.effectivealtruism.org/post) *on my blog.*"
    )


def test_resolve_forum_magnum_url_renders_post_and_comments(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_graphql(endpoint: str, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        calls.append({"endpoint": endpoint, "query": query, "variables": variables})
        if "GetPost" in query:
            return {
                "data": {
                    "post": {
                        "result": {
                            "_id": "post123",
                            "title": "Deep Models",
                            "slug": "deep-models",
                            "pageUrl": "https://forum.effectivealtruism.org/posts/post123/deep-models",
                            "postedAt": "2021-12-04T22:20:39.558Z",
                            "baseScore": 72,
                            "voteCount": 47,
                            "commentCount": 2,
                            "contents": {
                                "html": "<p>Hello <strong>world</strong>. <a href='/x'>relative</a></p>"
                            },
                            "user": {
                                "username": "author-handle",
                                "displayName": "Author Name",
                            },
                            "tags": [{"name": "Expertise"}],
                        }
                    }
                }
            }
        terms = variables["terms"]
        assert terms == {
            "postId": "post123",
            "limit": 2,
            "offset": 0,
            "view": "postCommentsOld",
        }
        return {
            "data": {
                "comments": {
                    "results": [
                        {
                            "_id": "comment1",
                            "postId": "post123",
                            "parentCommentId": None,
                            "pageUrl": "https://forum.effectivealtruism.org/posts/post123/deep-models#comment1",
                            "contents": {"html": "<p>First comment</p>"},
                            "baseScore": 5,
                            "postedAt": "2021-12-05T00:00:00.000Z",
                            "user": {"username": "c1", "displayName": "Commenter One"},
                        },
                        {
                            "_id": "comment2",
                            "postId": "post123",
                            "parentCommentId": "comment1",
                            "pageUrl": "https://forum.effectivealtruism.org/posts/post123/deep-models#comment2",
                            "contents": {"html": "<blockquote><p>Reply</p></blockquote>"},
                            "baseScore": 3,
                            "postedAt": "2021-12-05T01:00:00.000Z",
                            "user": {"username": "c2", "displayName": "Commenter Two"},
                        },
                    ]
                }
            }
        }

    monkeypatch.setattr(forum_magnum, "_graphql", fake_graphql)
    settings = forum_magnum.ForumMagnumSettings(max_comments=2)

    documents = forum_magnum.resolve_forum_magnum_url(
        "https://forum.effectivealtruism.org/posts/post123/deep-models",
        settings=settings,
        use_cache=False,
        cache_ttl=None,
        refresh_cache=False,
    )

    assert len(documents) == 3
    assert calls[0]["endpoint"] == "https://forum.effectivealtruism.org/graphql"
    assert documents[0].kind == "post"
    assert documents[0].canonical_id == "ea_forum:post:post123"
    assert documents[0].prose_authors == ("Author Name",)
    assert "# Deep Models" in documents[0].rendered
    assert "**world**" in documents[0].rendered
    assert "[relative](https://forum.effectivealtruism.org/x)" in documents[0].rendered
    assert "resolved_comment_count: 2" in documents[0].rendered
    assert documents[1].kind == "comment"
    assert documents[1].parent_comment_id is None
    assert documents[2].parent_comment_id == "comment1"
    assert "> Reply" in documents[2].rendered


def test_plugin_resolve_returns_contextualize_documents(monkeypatch) -> None:
    document = forum_magnum.ForumMagnumResolvedDocument(
        label="forum_magnum/lesswrong/post123",
        rendered="# Post",
        prose="Post",
        prose_authors=("Author",),
        trace_path="lesswrong/posts/post123",
        source_ref="lesswrong.com",
        source_path="lesswrong/posts/post123",
        context_subpath="forum-magnum/lesswrong/post123.md",
        source_created="2020-01-01T00:00:00.000Z",
        source_modified=None,
        kind="post",
        platform="lesswrong",
        canonical_id="lesswrong:post:post123",
        canonical_url="https://www.lesswrong.com/posts/post123/post",
        post_id="post123",
        comment_id=None,
        parent_comment_id=None,
        author="Author",
        score=10,
        dedupe_rank=0,
    )
    monkeypatch.setattr(
        forum_magnum,
        "resolve_forum_magnum_url",
        lambda *args, **kwargs: [document],
    )

    resolved = forum_magnum_plugin.resolve(
        "https://greaterwrong.com/posts/post123/post",
        {"overrides": {"forum_magnum": {"comments": False}}},
    )

    assert resolved == [
        {
            "source": "https://greaterwrong.com/posts/post123/post",
            "label": "forum_magnum/lesswrong/post123",
            "content": "# Post",
            "prose": "Post",
            "prose_authors": ["Author"],
            "metadata": {
                "trace_path": "lesswrong/posts/post123",
                "provider": "forum_magnum",
                "source_ref": "lesswrong.com",
                "source_path": "lesswrong/posts/post123",
                "context_subpath": "forum-magnum/lesswrong/post123.md",
                "source_created": "2020-01-01T00:00:00.000Z",
                "source_modified": None,
                "kind": "post",
                "platform": "lesswrong",
                "canonical_id": "lesswrong:post:post123",
                "canonical_url": "https://www.lesswrong.com/posts/post123/post",
                "post_id": "post123",
                "comment_id": None,
                "parent_comment_id": None,
                "author": "Author",
                "score": 10,
                "settings_key": (
                    ("include_comments", False),
                    ("max_comments", 500),
                    ("comment_batch_size", 100),
                    ("comment_view", "postCommentsOld"),
                ),
                "hydrate_dedupe": {
                    "mode": "canonical_symlink",
                    "key": (
                        "forum-magnum:lesswrong:post:post123:"
                        "(('include_comments', False), ('max_comments', 500), "
                        "('comment_batch_size', 100), "
                        "('comment_view', 'postCommentsOld')):lesswrong/posts/post123"
                    ),
                    "rank": 0,
                },
            },
        }
    ]
