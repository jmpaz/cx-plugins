from __future__ import annotations

import pytest

from cx_plugins.providers.xwitter import plugin as xwitter_plugin
from cx_plugins.providers.xwitter import xwitter
from cx_plugins.providers.ytdlp import plugin as ytdlp_plugin


_OEMBED_PAYLOAD = {
    "url": "https://x.com/jack/status/20",
    "author_name": "jack",
    "author_url": "https://x.com/jack",
    "html": (
        '<blockquote class="twitter-tweet">'
        '<p lang="en" dir="ltr">just setting up my twttr</p>'
        '&mdash; jack (@jack) '
        '<a href="https://x.com/jack/status/20?ref_src=twsrc%5Etfw">'
        "March 21, 2006</a>"
        "</blockquote>"
    ),
    "provider_name": "X",
    "provider_url": "https://x.com",
}

_OEMBED_LINK_PAYLOAD = {
    "url": "https://x.com/jack/status/20",
    "author_name": "jack",
    "author_url": "https://x.com/jack",
    "html": (
        '<blockquote class="twitter-tweet">'
        '<p lang="en" dir="ltr">'
        'read <a href="https://t.co/abc123">https://t.co/abc123</a> now'
        "</p>"
        '&mdash; jack (@jack) '
        '<a href="https://x.com/jack/status/20?ref_src=twsrc%5Etfw">'
        "March 21, 2006</a>"
        "</blockquote>"
    ),
    "provider_name": "X",
    "provider_url": "https://x.com",
}

_QUOTE_PARENT_PAYLOAD = {
    "url": "https://x.com/jack/status/20",
    "author_name": "jack",
    "author_url": "https://x.com/jack",
    "html": (
        '<blockquote class="twitter-tweet">'
        '<p lang="en" dir="ltr">'
        'look at this <a href="https://t.co/quote">https://t.co/quote</a>'
        "</p>"
        '&mdash; jack (@jack) '
        '<a href="https://x.com/jack/status/20?ref_src=twsrc%5Etfw">'
        "March 21, 2006</a>"
        "</blockquote>"
    ),
    "provider_name": "X",
    "provider_url": "https://x.com",
}

_QUOTE_CHILD_PAYLOAD = {
    "url": "https://x.com/quoted/status/222",
    "author_name": "quoted",
    "author_url": "https://x.com/quoted",
    "html": (
        '<blockquote class="twitter-tweet">'
        '<p lang="en" dir="ltr">quoted tweet body</p>'
        '&mdash; quoted (@quoted) '
        '<a href="https://x.com/quoted/status/222?ref_src=twsrc%5Etfw">'
        "March 20, 2006</a>"
        "</blockquote>"
    ),
    "provider_name": "X",
    "provider_url": "https://x.com",
}

_CYCLE_PARENT_PAYLOAD = {
    "url": "https://x.com/jack/status/30",
    "author_name": "jack",
    "author_url": "https://x.com/jack",
    "html": (
        '<blockquote class="twitter-tweet">'
        '<p lang="en" dir="ltr">'
        'parent <a href="https://t.co/to-child">https://t.co/to-child</a>'
        "</p>"
        '&mdash; jack (@jack) '
        '<a href="https://x.com/jack/status/30?ref_src=twsrc%5Etfw">'
        "March 21, 2006</a>"
        "</blockquote>"
    ),
    "provider_name": "X",
    "provider_url": "https://x.com",
}

_CYCLE_CHILD_PAYLOAD = {
    "url": "https://x.com/quoted/status/40",
    "author_name": "quoted",
    "author_url": "https://x.com/quoted",
    "html": (
        '<blockquote class="twitter-tweet">'
        '<p lang="en" dir="ltr">'
        'child <a href="https://t.co/to-parent">https://t.co/to-parent</a>'
        "</p>"
        '&mdash; quoted (@quoted) '
        '<a href="https://x.com/quoted/status/40?ref_src=twsrc%5Etfw">'
        "March 20, 2006</a>"
        "</blockquote>"
    ),
    "provider_name": "X",
    "provider_url": "https://x.com",
}

_ALIAS_HTML = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <link rel="canonical" href="https://x.com/jack/status/20"/>
    <meta property="og:url" content="https://x.com/jack/status/20"/>
    <meta property="twitter:creator" content="@jack"/>
    <meta property="og:title" content="jack (@jack)"/>
    <meta property="og:description" content="just setting up my twttr"/>
    <meta property="og:site_name" content="FixupX"/>
    <meta property="og:image" content="https://pbs.twimg.com/profile.jpg"/>
    <link
      rel="alternate"
      href="https://fixupx.com/owoembed?status=20&author=jack"
      type="application/json+oembed"
      title="jack"
    />
  </head>
  <body></body>
</html>
"""


def _frontmatter(rendered: str) -> dict[str, object]:
    import yaml

    end = rendered.find("\n---\n", 4)
    assert end != -1
    parsed = yaml.safe_load(rendered[4:end])
    assert isinstance(parsed, dict)
    return parsed


def _body(rendered: str) -> str:
    return rendered.split("\n---\n", 1)[1].split("\n***", 1)[0].strip()


def _patch_oembed_payloads(
    monkeypatch,
    payloads: dict[str, dict[str, object]],
) -> None:
    def _fake_http_get_json(_url: str, *, params, timeout: int):
        assert timeout == 30
        return dict(payloads[params["url"]])

    monkeypatch.setattr(xwitter, "_http_get_json", _fake_http_get_json)


def _patch_redirects(monkeypatch, redirects: dict[str, str]) -> None:
    def _fake_resolve_redirect(url: str, *, timeout: int) -> str:
        assert timeout == 30
        return redirects[url]

    monkeypatch.setattr(xwitter, "_http_resolve_redirect", _fake_resolve_redirect)


@pytest.mark.parametrize(
    ("url", "author"),
    [
        ("https://x.com/jack/status/20", "jack"),
        ("https://twitter.com/jack/status/20?s=20", "jack"),
        ("https://fixupx.com/jack/status/20", "jack"),
        ("https://twittpr.com/jack/statuses/20", "jack"),
        ("https://x.com/i/web/status/20", None),
    ],
)
def test_parse_xwitter_target_canonicalizes_supported_hosts(
    url: str,
    author: str | None,
) -> None:
    parsed = xwitter.parse_xwitter_target(url)
    assert parsed is not None
    assert parsed.kind == "tweet"
    assert parsed.tweet_id == "20"
    assert parsed.author == author
    if author is None:
        assert parsed.canonical_url == "https://x.com/i/web/status/20"
        assert parsed.source_path == "status/20"
    else:
        assert parsed.canonical_url == "https://x.com/jack/status/20"
        assert parsed.source_path == "jack/status/20"
    assert xwitter_plugin.can_resolve(url, {}) is True
    assert xwitter_plugin.classify_target(url, {}) == {
        "provider": "xwitter",
        "kind": "tweet",
        "is_external": True,
        "group_key": "tweet",
    }


def test_parse_xwitter_target_rejects_non_tweet_urls() -> None:
    assert xwitter.parse_xwitter_target("https://x.com/jack") is None
    assert xwitter.parse_xwitter_target("https://example.com/jack/status/20") is None
    assert xwitter_plugin.can_resolve("https://x.com/jack", {}) is False
    assert xwitter_plugin.classify_target("https://x.com/jack", {}) is None


def test_xwitter_oembed_url_normalizes_aliases() -> None:
    assert xwitter.xwitter_oembed_url("https://fixupx.com/jack/status/20") == (
        "https://publish.x.com/oembed?"
        "url=https%3A%2F%2Fx.com%2Fjack%2Fstatus%2F20&omit_script=true"
    )


def test_resolve_xwitter_url_renders_oembed_document(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def _fake_http_get_json(url: str, *, params, timeout: int):
        calls.append({"url": url, "params": dict(params), "timeout": timeout})
        return dict(_OEMBED_PAYLOAD)

    monkeypatch.setattr(xwitter, "_http_get_json", _fake_http_get_json)

    docs = xwitter.resolve_xwitter_url(
        "https://twitter.com/jack/status/20?s=20",
        use_cache=False,
    )

    assert calls == [
        {
            "url": "https://publish.x.com/oembed",
            "params": {
                "url": "https://x.com/jack/status/20",
                "omit_script": "true",
            },
            "timeout": 30,
        }
    ]
    assert len(docs) == 1
    doc = docs[0]
    assert doc.label == "xwitter/jack/status/20"
    assert doc.context_subpath == "xwitter/jack/status/20.md"
    assert doc.source_path == "jack/status/20"
    assert doc.canonical_url == "https://x.com/jack/status/20"
    assert doc.author == "jack"
    assert "author: jack" in doc.rendered
    assert "posted_at: March 21, 2006" in doc.rendered
    assert "fetched_via: x-oembed" in doc.rendered
    assert "just setting up my twttr" in doc.rendered


def test_resolve_xwitter_url_expands_tco_links_in_body_and_link_text(
    monkeypatch,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        xwitter,
        "_http_get_json",
        lambda *_args, **_kwargs: dict(_OEMBED_LINK_PAYLOAD),
    )

    def _fake_resolve_redirect(url: str, *, timeout: int) -> str:
        calls.append(url)
        assert timeout == 30
        return "https://trueurl.com/article"

    monkeypatch.setattr(xwitter, "_http_resolve_redirect", _fake_resolve_redirect)

    docs = xwitter.resolve_xwitter_url(
        "https://x.com/jack/status/20",
        use_cache=False,
    )

    assert calls == ["https://t.co/abc123"]
    assert "\n\nread https://trueurl.com/article now\n\n***" in docs[0].rendered
    assert "- [https://trueurl.com/article](https://t.co/abc123)" in docs[0].rendered


def test_resolve_xwitter_url_emits_nested_quote_documents(monkeypatch) -> None:
    _patch_oembed_payloads(
        monkeypatch,
        {
            "https://x.com/jack/status/20": _QUOTE_PARENT_PAYLOAD,
            "https://x.com/quoted/status/222": _QUOTE_CHILD_PAYLOAD,
        },
    )
    _patch_redirects(
        monkeypatch,
        {"https://t.co/quote": "https://twitter.com/quoted/status/222"},
    )

    docs = xwitter.resolve_xwitter_url(
        "https://x.com/jack/status/20",
        use_cache=False,
    )

    assert [doc.tweet_id for doc in docs] == ["20", "222"]
    assert docs[0].label == "xwitter/jack/status/20"
    assert docs[1].label == "xwitter/quoted/status/222"
    frontmatter = _frontmatter(docs[0].rendered)
    assert frontmatter["quoted_tweet_url"] == "https://x.com/quoted/status/222"
    assert frontmatter["quoted_tweet_id"] == "222"
    assert "look at this" in _body(docs[0].rendered)
    assert "https://t.co/quote" not in _body(docs[0].rendered)
    assert "https://x.com/quoted/status/222" not in _body(docs[0].rendered)
    assert "## Links" not in docs[0].rendered
    assert "quoted tweet body" in docs[1].rendered


def test_quote_depth_zero_keeps_quote_as_normal_link(monkeypatch) -> None:
    _patch_oembed_payloads(
        monkeypatch,
        {"https://x.com/jack/status/20": _QUOTE_PARENT_PAYLOAD},
    )
    _patch_redirects(
        monkeypatch,
        {"https://t.co/quote": "https://twitter.com/quoted/status/222"},
    )

    docs = xwitter.resolve_xwitter_url(
        "https://x.com/jack/status/20",
        settings=xwitter.build_xwitter_settings({"quote_depth": 0}),
        use_cache=False,
    )

    assert [doc.tweet_id for doc in docs] == ["20"]
    assert "quoted_tweet_id" not in _frontmatter(docs[0].rendered)
    assert (
        "look at this https://x.com/quoted/status/222"
        in _body(docs[0].rendered)
    )
    assert (
        "- [https://x.com/quoted/status/222](https://t.co/quote)"
        in docs[0].rendered
    )


def test_failed_quote_fetch_falls_back_to_normal_link(monkeypatch) -> None:
    def _fake_http_get_json(_url: str, *, params, timeout: int):
        assert timeout == 30
        if params["url"] == "https://x.com/jack/status/20":
            return dict(_QUOTE_PARENT_PAYLOAD)
        raise RuntimeError("quoted tweet unavailable")

    monkeypatch.setattr(xwitter, "_http_get_json", _fake_http_get_json)
    _patch_redirects(
        monkeypatch,
        {"https://t.co/quote": "https://twitter.com/quoted/status/222"},
    )

    docs = xwitter.resolve_xwitter_url(
        "https://x.com/jack/status/20",
        use_cache=False,
    )

    assert [doc.tweet_id for doc in docs] == ["20"]
    assert "quoted_tweet_id" not in _frontmatter(docs[0].rendered)
    assert (
        "look at this https://x.com/quoted/status/222"
        in _body(docs[0].rendered)
    )
    assert (
        "- [https://x.com/quoted/status/222](https://t.co/quote)"
        in docs[0].rendered
    )


def test_cyclic_quote_references_do_not_duplicate_documents(monkeypatch) -> None:
    _patch_oembed_payloads(
        monkeypatch,
        {
            "https://x.com/jack/status/30": _CYCLE_PARENT_PAYLOAD,
            "https://x.com/quoted/status/40": _CYCLE_CHILD_PAYLOAD,
        },
    )
    _patch_redirects(
        monkeypatch,
        {
            "https://t.co/to-child": "https://twitter.com/quoted/status/40",
            "https://t.co/to-parent": "https://twitter.com/jack/status/30",
        },
    )

    docs = xwitter.resolve_xwitter_url(
        "https://x.com/jack/status/30",
        settings=xwitter.build_xwitter_settings({"quote_depth": 2}),
        use_cache=False,
    )

    assert [doc.tweet_id for doc in docs] == ["30", "40"]
    assert _frontmatter(docs[0].rendered)["quoted_tweet_id"] == "40"
    assert _frontmatter(docs[1].rendered)["quoted_tweet_id"] == "30"


def test_xwitter_plugin_emits_canonical_dedupe_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        xwitter,
        "_http_get_json",
        lambda *_args, **_kwargs: _OEMBED_PAYLOAD,
    )

    result = xwitter_plugin.resolve(
        "https://fixupx.com/jack/status/20",
        {"use_cache": False},
    )

    assert result[0]["label"] == "xwitter/jack/status/20"
    metadata = result[0]["metadata"]
    assert metadata["provider"] == "xwitter"
    assert metadata["source_ref"] == "x.com"
    assert metadata["source_path"] == "jack/status/20"
    assert metadata["context_subpath"] == "xwitter/jack/status/20.md"
    assert metadata["canonical_url"] == "https://x.com/jack/status/20"
    assert metadata["tweet_id"] == "20"
    assert metadata["hydrate_dedupe"]["mode"] == "canonical_symlink"
    assert metadata["hydrate_dedupe"]["rank"] == 0
    assert "xwitter-tweet:20" in metadata["hydrate_dedupe"]["key"]


def test_xwitter_alias_metadata_fallback(monkeypatch) -> None:
    def _raise_oembed(*_args, **_kwargs):
        raise RuntimeError("oEmbed unavailable")

    monkeypatch.setattr(xwitter, "_http_get_json", _raise_oembed)
    monkeypatch.setattr(xwitter, "_http_get_text", lambda *_args, **_kwargs: _ALIAS_HTML)

    docs = xwitter.resolve_xwitter_url(
        "https://fixupx.com/jack/status/20",
        use_cache=False,
    )

    assert docs[0].canonical_url == "https://x.com/jack/status/20"
    assert docs[0].author == "jack"
    assert "fetched_via: fixupx.com-open-graph" in docs[0].rendered
    assert "provider_name: FixupX" in docs[0].rendered
    assert "image_url: https://pbs.twimg.com/profile.jpg" in docs[0].rendered
    assert "just setting up my twttr" in docs[0].rendered


def test_xwitter_plugin_priority_is_above_ytdlp() -> None:
    assert ytdlp_plugin.PLUGIN_PRIORITY < xwitter_plugin.PLUGIN_PRIORITY
