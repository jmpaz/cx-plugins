from __future__ import annotations

from urllib.parse import urlparse

import pytest

from cx_plugins.providers.twitter import plugin as twitter_plugin
from cx_plugins.providers.twitter import twitter
from cx_plugins.providers.ytdlp import plugin as ytdlp_plugin


_ROOT_ID = "2000000000000000001"
_QUOTE_ID = "2000000000000000002"
_NESTED_ID = "2000000000000000003"
_CYCLE_ID = "2000000000000000004"


def _fx_tweet(
    *,
    handle: str,
    tweet_id: str,
    text: str,
    raw_text: str | None = None,
    facets: list[dict[str, object]] | None = None,
    media: list[dict[str, object]] | None = None,
    quote: dict[str, object] | None = None,
) -> dict[str, object]:
    tweet: dict[str, object] = {
        "url": f"https://x.com/{handle}/status/{tweet_id}",
        "id": tweet_id,
        "text": text,
        "raw_text": {
            "text": raw_text if raw_text is not None else text,
            "display_text_range": [0, len(raw_text if raw_text is not None else text)],
            "facets": facets or [],
        },
        "author": {
            "screen_name": handle,
            "url": f"https://x.com/{handle}",
            "id": f"author-{handle}",
            "name": handle.replace("_", " ").title(),
        },
        "created_at": "Mon Jan 01 00:00:00 +0000 2024",
        "created_timestamp": 1704067200,
        "lang": "en",
        "source": "Synthetic",
        "provider": "twitter",
    }
    if media:
        tweet["media"] = {"all": media}
    if quote:
        tweet["quote"] = quote
    return tweet


def _fx_payload(tweet: dict[str, object]) -> dict[str, object]:
    return {"code": 200, "message": "OK", "tweet": tweet}


def _media_item(
    *,
    media_id: str = "synthetic-media-1",
    url: str = "https://pbs.twimg.invalid/media/synthetic.jpg?name=orig",
    width: int = 800,
    height: int = 600,
    alt: str | None = None,
) -> dict[str, object]:
    item: dict[str, object] = {
        "type": "photo",
        "id": media_id,
        "url": url,
        "width": width,
        "height": height,
    }
    if alt:
        item["alt"] = alt
    return item


def _frontmatter(rendered: str) -> dict[str, object]:
    import yaml

    end = rendered.find("\n---\n", 4)
    assert end != -1
    parsed = yaml.safe_load(rendered[4:end])
    assert isinstance(parsed, dict)
    return parsed


def _body(rendered: str) -> str:
    return rendered.split("\n---\n", 1)[1].split("\n***", 1)[0].strip()


def _patch_fx_payloads(
    monkeypatch,
    payloads: dict[str, dict[str, object]],
) -> None:
    def _fake_http_get_json(url: str, *, params=None, timeout: int):
        assert timeout == 30
        parsed = urlparse(url)
        if parsed.netloc == "api.fxtwitter.com":
            return _fx_payload(payloads[url])
        raise AssertionError(f"unexpected JSON fetch: {url} {params}")

    monkeypatch.setattr(twitter, "_http_get_json", _fake_http_get_json)


def _fx_url(handle: str, tweet_id: str) -> str:
    return f"https://api.fxtwitter.com/{handle}/status/{tweet_id}"


@pytest.mark.parametrize(
    ("url", "author"),
    [
        (f"https://x.com/synthetic_root/status/{_ROOT_ID}", "synthetic_root"),
        (f"https://twitter.com/synthetic_root/status/{_ROOT_ID}?s=20", "synthetic_root"),
        (f"https://fixupx.com/synthetic_root/status/{_ROOT_ID}", "synthetic_root"),
        (f"https://twittpr.com/synthetic_root/statuses/{_ROOT_ID}", "synthetic_root"),
        (f"https://fxtwitter.com/synthetic_root/status/{_ROOT_ID}", "synthetic_root"),
        (f"https://x.com/i/web/status/{_ROOT_ID}", None),
    ],
)
def test_parse_twitter_target_canonicalizes_supported_hosts(
    url: str,
    author: str | None,
) -> None:
    parsed = twitter.parse_twitter_target(url)
    assert parsed is not None
    assert parsed.kind == "tweet"
    assert parsed.tweet_id == _ROOT_ID
    assert parsed.author == author
    if author is None:
        assert parsed.canonical_url == f"https://x.com/i/web/status/{_ROOT_ID}"
        assert parsed.source_path == f"status/{_ROOT_ID}"
    else:
        assert parsed.canonical_url == f"https://x.com/synthetic_root/status/{_ROOT_ID}"
        assert parsed.source_path == f"synthetic_root/status/{_ROOT_ID}"
    assert twitter_plugin.can_resolve(url, {}) is True
    assert twitter_plugin.classify_target(url, {}) == {
        "provider": "twitter",
        "kind": "tweet",
        "is_external": True,
        "group_key": "tweet",
    }


def test_parse_twitter_target_rejects_non_tweet_urls() -> None:
    assert twitter.parse_twitter_target("https://x.com/synthetic_root") is None
    assert (
        twitter.parse_twitter_target(
            f"https://example.invalid/synthetic_root/status/{_ROOT_ID}"
        )
        is None
    )
    assert twitter_plugin.can_resolve("https://x.com/synthetic_root", {}) is False
    assert twitter_plugin.classify_target("https://x.com/synthetic_root", {}) is None


def test_twitter_oembed_url_normalizes_aliases() -> None:
    assert twitter.twitter_oembed_url(
        f"https://fixupx.com/synthetic_root/status/{_ROOT_ID}"
    ) == (
        "https://publish.x.com/oembed?"
        "url=https%3A%2F%2Fx.com%2Fsynthetic_root%2Fstatus%2F"
        f"{_ROOT_ID}&omit_script=true"
    )


def test_resolve_twitter_url_renders_fx_media_document(monkeypatch) -> None:
    tweet = _fx_tweet(
        handle="synthetic_root",
        tweet_id=_ROOT_ID,
        text="Synthetic root tweet",
        raw_text="Synthetic root tweet https://t.co/media",
        facets=[
            {
                "type": "media",
                "original": "https://t.co/media",
                "replacement": f"https://x.com/synthetic_root/status/{_ROOT_ID}/photo/1",
            }
        ],
        media=[_media_item()],
    )
    _patch_fx_payloads(monkeypatch, {_fx_url("synthetic_root", _ROOT_ID): tweet})
    monkeypatch.setattr(
        twitter,
        "_describe_media",
        lambda *_args, **_kwargs: "Generated image description.",
    )

    docs = twitter.resolve_twitter_url(
        f"https://twitter.com/synthetic_root/status/{_ROOT_ID}?s=20",
        use_cache=False,
    )

    assert len(docs) == 1
    doc = docs[0]
    assert doc.label == f"twitter/synthetic_root/status/{_ROOT_ID}"
    assert doc.context_subpath == f"twitter/synthetic_root/status/{_ROOT_ID}.md"
    assert doc.source_path == f"synthetic_root/status/{_ROOT_ID}"
    assert doc.canonical_url == f"https://x.com/synthetic_root/status/{_ROOT_ID}"
    assert _body(doc.rendered) == "Synthetic root tweet"
    frontmatter = _frontmatter(doc.rendered)
    assert frontmatter["provider_name"] == "FxTwitter"
    assert frontmatter["fetched_via"] == "fxtwitter-api"
    assert frontmatter["media_count"] == 1
    assert "likes" not in frontmatter
    assert "views" not in frontmatter
    assert "## Links" not in doc.rendered
    assert "<image " in doc.rendered
    assert 'url="https://pbs.twimg.invalid/media/synthetic.jpg?name=orig"' in doc.rendered
    assert "description: Generated image description." in doc.rendered


def test_fx_url_facets_are_resolved_without_links_section(monkeypatch) -> None:
    tweet = _fx_tweet(
        handle="synthetic_root",
        tweet_id=_ROOT_ID,
        text="Read https://example.invalid/article now",
        raw_text="Read https://t.co/article now",
        facets=[
            {
                "type": "url",
                "original": "https://t.co/article",
                "replacement": "https://example.invalid/article",
                "display": "example.invalid/article",
            }
        ],
    )
    _patch_fx_payloads(monkeypatch, {_fx_url("synthetic_root", _ROOT_ID): tweet})

    docs = twitter.resolve_twitter_url(
        f"https://x.com/synthetic_root/status/{_ROOT_ID}",
        use_cache=False,
    )

    assert _body(docs[0].rendered) == "Read https://example.invalid/article now"
    assert "## Links" not in docs[0].rendered


def test_default_quote_depth_fetches_direct_and_nested_quotes(monkeypatch) -> None:
    nested = _fx_tweet(
        handle="synthetic_nested",
        tweet_id=_NESTED_ID,
        text="Nested quote text",
    )
    quoted = _fx_tweet(
        handle="synthetic_quote",
        tweet_id=_QUOTE_ID,
        text="Quoted text",
        quote=nested,
    )
    root = _fx_tweet(
        handle="synthetic_root",
        tweet_id=_ROOT_ID,
        text="Root text",
        quote=quoted,
    )
    _patch_fx_payloads(monkeypatch, {_fx_url("synthetic_root", _ROOT_ID): root})

    docs = twitter.resolve_twitter_url(
        f"https://x.com/synthetic_root/status/{_ROOT_ID}",
        use_cache=False,
    )

    assert [doc.tweet_id for doc in docs] == [_ROOT_ID, _QUOTE_ID, _NESTED_ID]
    assert _frontmatter(docs[0].rendered)["quoted_tweet_id"] == _QUOTE_ID
    assert _frontmatter(docs[1].rendered)["quoted_tweet_id"] == _NESTED_ID
    assert "quoted_tweet_id" not in _frontmatter(docs[2].rendered)


def test_quote_depth_one_stops_before_nested_quote(monkeypatch) -> None:
    nested = _fx_tweet(
        handle="synthetic_nested",
        tweet_id=_NESTED_ID,
        text="Nested quote text",
    )
    quoted = _fx_tweet(
        handle="synthetic_quote",
        tweet_id=_QUOTE_ID,
        text="Quoted text",
        quote=nested,
    )
    root = _fx_tweet(
        handle="synthetic_root",
        tweet_id=_ROOT_ID,
        text="Root text",
        quote=quoted,
    )
    _patch_fx_payloads(monkeypatch, {_fx_url("synthetic_root", _ROOT_ID): root})

    docs = twitter.resolve_twitter_url(
        f"https://x.com/synthetic_root/status/{_ROOT_ID}",
        settings=twitter.build_twitter_settings({"quote_depth": 1}),
        use_cache=False,
    )

    assert [doc.tweet_id for doc in docs] == [_ROOT_ID, _QUOTE_ID]
    assert _frontmatter(docs[1].rendered)["quoted_tweet_id"] == _NESTED_ID


def test_cyclic_quote_references_do_not_duplicate_documents(monkeypatch) -> None:
    root_stub = _fx_tweet(
        handle="synthetic_root",
        tweet_id=_ROOT_ID,
        text="Root text",
    )
    cycle = _fx_tweet(
        handle="synthetic_cycle",
        tweet_id=_CYCLE_ID,
        text="Cycle text",
        quote=root_stub,
    )
    root = _fx_tweet(
        handle="synthetic_root",
        tweet_id=_ROOT_ID,
        text="Root text",
        quote=cycle,
    )
    _patch_fx_payloads(monkeypatch, {_fx_url("synthetic_root", _ROOT_ID): root})

    docs = twitter.resolve_twitter_url(
        f"https://x.com/synthetic_root/status/{_ROOT_ID}",
        use_cache=False,
    )

    assert [doc.tweet_id for doc in docs] == [_ROOT_ID, _CYCLE_ID]
    assert _frontmatter(docs[1].rendered)["quoted_tweet_id"] == _ROOT_ID


def test_fx_failure_degrades_to_oembed_document(monkeypatch) -> None:
    oembed_payload = {
        "url": f"https://x.com/synthetic_root/status/{_ROOT_ID}",
        "author_name": "Synthetic Root",
        "author_url": "https://x.com/synthetic_root",
        "html": (
            '<blockquote class="twitter-tweet">'
            '<p lang="en" dir="ltr">'
            'read <a href="https://t.co/article">https://t.co/article</a>'
            "</p>"
            "&mdash; Synthetic Root (@synthetic_root) "
            f'<a href="https://x.com/synthetic_root/status/{_ROOT_ID}?ref_src=twsrc%5Etfw">'
            "January 1, 2024</a>"
            "</blockquote>"
        ),
        "provider_name": "X",
        "provider_url": "https://x.com",
    }

    def _fake_http_get_json(url: str, *, params=None, timeout: int):
        assert timeout == 30
        if url.startswith("https://api.fxtwitter.com/"):
            raise RuntimeError("synthetic rich fetch failure")
        assert url == "https://publish.x.com/oembed"
        return dict(oembed_payload)

    monkeypatch.setattr(twitter, "_http_get_json", _fake_http_get_json)
    monkeypatch.setattr(
        twitter,
        "_http_resolve_redirect",
        lambda _url, *, timeout: "https://example.invalid/article",
    )

    docs = twitter.resolve_twitter_url(
        f"https://x.com/synthetic_root/status/{_ROOT_ID}",
        use_cache=False,
    )

    frontmatter = _frontmatter(docs[0].rendered)
    assert frontmatter["fetched_via"] == "x-oembed"
    assert "FxTwitter API failed" in frontmatter["resolution_error"]
    assert _body(docs[0].rendered) == "read https://example.invalid/article"
    assert "## Links" not in docs[0].rendered


def test_media_description_failure_keeps_media_block(monkeypatch) -> None:
    tweet = _fx_tweet(
        handle="synthetic_root",
        tweet_id=_ROOT_ID,
        text="Synthetic root tweet",
        media=[_media_item()],
    )
    _patch_fx_payloads(monkeypatch, {_fx_url("synthetic_root", _ROOT_ID): tweet})

    def _raise_description(*_args, **_kwargs):
        raise RuntimeError("description unavailable")

    monkeypatch.setattr(twitter, "_describe_media", _raise_description)

    docs = twitter.resolve_twitter_url(
        f"https://x.com/synthetic_root/status/{_ROOT_ID}",
        use_cache=False,
    )

    assert "<image " in docs[0].rendered
    assert 'url="https://pbs.twimg.invalid/media/synthetic.jpg?name=orig"' in docs[
        0
    ].rendered
    assert "Description unavailable." in docs[0].rendered


def test_twitter_plugin_emits_canonical_dedupe_metadata(monkeypatch) -> None:
    tweet = _fx_tweet(
        handle="synthetic_root",
        tweet_id=_ROOT_ID,
        text="Synthetic root tweet",
    )
    _patch_fx_payloads(monkeypatch, {_fx_url("synthetic_root", _ROOT_ID): tweet})

    result = twitter_plugin.resolve(
        f"https://fixupx.com/synthetic_root/status/{_ROOT_ID}",
        {"use_cache": False},
    )

    assert result[0]["label"] == f"twitter/synthetic_root/status/{_ROOT_ID}"
    metadata = result[0]["metadata"]
    assert metadata["provider"] == "twitter"
    assert metadata["source_ref"] == "x.com"
    assert metadata["source_path"] == f"synthetic_root/status/{_ROOT_ID}"
    assert (
        metadata["context_subpath"]
        == f"twitter/synthetic_root/status/{_ROOT_ID}.md"
    )
    assert metadata["canonical_url"] == f"https://x.com/synthetic_root/status/{_ROOT_ID}"
    assert metadata["tweet_id"] == _ROOT_ID
    assert metadata["hydrate_dedupe"]["mode"] == "canonical_symlink"
    assert metadata["hydrate_dedupe"]["rank"] == 0
    assert "twitter-tweet:" in metadata["hydrate_dedupe"]["key"]
    assert "xwitter" not in metadata["hydrate_dedupe"]["key"]


def test_twitter_plugin_priority_is_above_ytdlp() -> None:
    assert ytdlp_plugin.PLUGIN_PRIORITY < twitter_plugin.PLUGIN_PRIORITY
