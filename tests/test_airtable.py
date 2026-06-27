from __future__ import annotations

import cx_plugins.providers.airtable.plugin as airtable_plugin
from cx_plugins.providers.airtable import airtable


APP = "appTESTbase000001"
SHARE = "shrTESTshare00001"
PAGE = "pagTESTpage000001"
TABLE = "tblTESTtable00001"


class FakeResponse:
    def __init__(self, *, text: str = "", status_code: int = 200, payload=None) -> None:
        self.text = text
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        if self._payload is None:
            raise ValueError("no json payload")
        return self._payload


class FakeSession:
    def __init__(self, routes: list[tuple[str, FakeResponse]]) -> None:
        self.headers: dict[str, str] = {}
        self._routes = routes
        self.calls: list[str] = []

    def get(self, url: str, headers=None, timeout=None) -> FakeResponse:
        self.calls.append(url)
        for matcher, response in self._routes:
            if matcher in url:
                return response
        raise AssertionError(f"unexpected URL: {url}")


def _install_session(monkeypatch, routes: list[tuple[str, FakeResponse]]) -> FakeSession:
    session = FakeSession(routes)
    monkeypatch.setattr("requests.Session", lambda: session)
    return session


def _page_html(*, prefetch: str | None = None, og_title: str | None = None) -> str:
    parts = ['window.initData = {"pageLoadId":"pglTEST","codeVersion":"abc123def456"};']
    if og_title is not None:
        parts.append(f'<meta property="og:title" content="{og_title}" />')
    if prefetch is not None:
        parts.append(f'<script>fetchPrefetch({{ urlWithParams: "{prefetch}" }})</script>')
    return "<!DOCTYPE html><html><head>" + "".join(parts) + "</head></html>"


# --------------------------------------------------------------------------- #
# Target parsing                                                              #
# --------------------------------------------------------------------------- #


def test_parse_shared_view_with_table() -> None:
    parsed = airtable.parse_airtable_target(
        f"https://airtable.com/{APP}/{SHARE}/{TABLE}"
    )
    assert parsed is not None
    assert parsed.app_id == APP
    assert parsed.share_id == SHARE
    assert parsed.table_id == TABLE
    assert parsed.page_id is None
    assert parsed.canonical_id == f"{APP}:{SHARE}:{TABLE}"
    assert parsed.page_url == f"https://airtable.com/{APP}/{SHARE}/{TABLE}"


def test_parse_embed_and_query_are_normalized() -> None:
    parsed = airtable.parse_airtable_target(
        f"https://airtable.com/embed/{APP}/{SHARE}?detail=abc&x=allRecords"
    )
    assert parsed is not None
    assert parsed.is_embed is True
    assert parsed.share_id == SHARE
    assert parsed.page_url == f"https://airtable.com/embed/{APP}/{SHARE}"


def test_parse_interface_form_route() -> None:
    parsed = airtable.parse_airtable_target(
        f"https://airtable.com/{APP}/{PAGE}/form"
    )
    assert parsed is not None
    assert parsed.page_id == PAGE
    assert parsed.is_form_route is True
    assert parsed.canonical_id == f"{APP}:{PAGE}:form"


def test_parse_rejects_non_airtable_and_bare_app() -> None:
    assert airtable.parse_airtable_target("https://example.com/appX/shrY") is None
    assert airtable.parse_airtable_target(f"https://airtable.com/{APP}") is None
    assert airtable.is_airtable_target(f"https://airtable.com/{APP}/{SHARE}") is True


def test_plugin_classifies_airtable_links() -> None:
    target = f"https://airtable.com/{APP}/{SHARE}"
    assert airtable_plugin.can_resolve(target, {}) is True
    assert airtable_plugin.classify_target(target, {}) == {
        "provider": "airtable",
        "kind": "database",
        "is_external": True,
        "group_key": "database",
    }


def test_runtime_overrides_parse_manifest_aliases() -> None:
    parsed = airtable_plugin._airtable_runtime_overrides(  # noqa: SLF001
        {
            "include-media": False,
            "max-rows": 50,
            "timeout": 12,
        }
    )
    assert parsed == {
        "include_media": False,
        "max_rows": 50,
        "timeout_seconds": 12,
    }


# --------------------------------------------------------------------------- #
# Cell rendering                                                              #
# --------------------------------------------------------------------------- #


def test_render_value_resolves_select_choices_and_lists() -> None:
    choices = {"selX": {"id": "selX", "name": "Policy"}}
    assert airtable._render_value("selX", choices) == "Policy"
    assert airtable._render_value(["selX", "selX"], choices) == "Policy, Policy"
    assert airtable._render_value(True, None) == "Yes"
    assert airtable._render_value(3.0, None) == "3"
    assert airtable._render_value({"foreignRowDisplayName": "Linked"}, None) == "Linked"


# --------------------------------------------------------------------------- #
# Resolution                                                                  #
# --------------------------------------------------------------------------- #


def test_resolve_shared_grid_view(monkeypatch) -> None:
    prefetch = (
        "/v0.3/view/viwTEST/readSharedViewData"
        "?stringifiedObjectParams=%7B%7D&requestId=req1&accessPolicy=signed"
    )
    grid_payload = {
        "data": {
            "table": {
                "name": "Fallback Name",
                "primaryColumnId": "fld1",
                "columns": [
                    {"id": "fld1", "name": "Title", "type": "text"},
                    {
                        "id": "fld2",
                        "name": "Area",
                        "type": "select",
                        "typeOptions": {"choices": {"selX": {"name": "Policy"}}},
                    },
                    {"id": "fld3", "name": "Photos", "type": "multipleAttachment"},
                ],
                "rows": [
                    {
                        "cellValuesByColumnId": {
                            "fld1": "Row One",
                            "fld2": "selX",
                            "fld3": [
                                {
                                    "url": "https://example.com/y.png",
                                    "filename": "y.png",
                                    "type": "image/png",
                                }
                            ],
                        }
                    }
                ],
            }
        }
    }
    _install_session(
        monkeypatch,
        [
            (f"/{APP}/{SHARE}", FakeResponse(text=_page_html(prefetch=prefetch, og_title="My View - Airtable"))),
            ("readSharedViewData", FakeResponse(payload=grid_payload)),
        ],
    )

    docs = airtable_plugin.resolve(
        f"https://airtable.com/{APP}/{SHARE}", {"use_cache": False}
    )
    assert len(docs) == 1
    doc = docs[0]
    assert doc["label"] == "airtable/My_View"
    assert "# My View" in doc["content"]
    assert "## Row One" in doc["content"]
    assert "- **Area**: Policy" in doc["content"]
    assert '<image filename="y.png" url="https://example.com/y.png" caption="Photos" />' in doc["content"]
    assert "Title: Row One" in doc["prose"]
    metadata = doc["metadata"]
    assert metadata["provider"] == "airtable"
    assert metadata["kind"] == "table"
    assert metadata["media_count"] == 1
    assert metadata["hydrate_dedupe"]["mode"] == "canonical_symlink"


def test_resolve_shared_form(monkeypatch) -> None:
    prefetch = "/v0.3/view/viwFORM/readSharedFormData?requestId=req2&accessPolicy=signed"
    form_payload = {
        "data": {
            "formTable": {
                "name": "Table",
                "columns": [
                    {"id": "c1", "name": "Email", "type": "text"},
                    {
                        "id": "c2",
                        "name": "Pick",
                        "type": "select",
                        "typeOptions": {
                            "choiceOrder": ["a", "b"],
                            "choices": {"a": {"name": "A"}, "b": {"name": "B"}},
                        },
                    },
                ],
                "views": [
                    {
                        "id": "viwFORM",
                        "type": "form",
                        "name": "My Form",
                        "columnOrder": [
                            {"columnId": "c1", "visibility": True},
                            {"columnId": "c2", "visibility": True},
                        ],
                        "metadata": {
                            "form": {
                                "description": "Fill this out",
                                "fieldsByColumnId": {
                                    "c1": {
                                        "title": "Your email",
                                        "required": True,
                                        "description": "work email please",
                                    }
                                },
                            }
                        },
                    }
                ],
            }
        }
    }
    _install_session(
        monkeypatch,
        [
            (f"/{APP}/{SHARE}", FakeResponse(text=_page_html(prefetch=prefetch))),
            ("readSharedFormData", FakeResponse(payload=form_payload)),
        ],
    )

    docs = airtable_plugin.resolve(
        f"https://airtable.com/{APP}/{SHARE}", {"use_cache": False}
    )
    doc = docs[0]
    content = doc["content"]
    assert doc["metadata"]["kind"] == "form"
    assert "# My Form" in content
    assert "Fill this out" in content
    assert "### Your email *(required)*" in content
    assert "_work email please_" in content
    assert "### Pick" in content
    assert "*Single select:*\n- [ ] A\n- [ ] B" in content


def test_resolve_interface_form_renders_schema_with_note(monkeypatch) -> None:
    interface_payload = {
        "data": {
            "tableSchemas": [
                {
                    "id": "tblZ",
                    "name": "Applicants",
                    "columns": [
                        {"id": "c1", "name": "Name", "type": "text"},
                        {"id": "c2", "name": "Email", "type": "text"},
                    ],
                }
            ],
            "tableDatas": [],
            "pageBundles": [],
        }
    }
    _install_session(
        monkeypatch,
        [
            (f"/{APP}/{PAGE}/form", FakeResponse(text=_page_html())),
            ("readForPages", FakeResponse(payload=interface_payload)),
        ],
    )

    docs = airtable_plugin.resolve(
        f"https://airtable.com/{APP}/{PAGE}/form", {"use_cache": False}
    )
    doc = docs[0]
    assert doc["metadata"]["kind"] == "interface"
    assert "Interface record data is not included" in doc["content"]
    assert "- **Name** (text)" in doc["content"]


def test_resolve_interface_form_renders_full_form(monkeypatch) -> None:
    form_payload = {
        "data": {
            "tableSchemas": [
                {
                    "id": "tblX",
                    "columns": [
                        {"id": "fldName", "name": "Full Name", "type": "text"},
                        {
                            "id": "fldChoice",
                            "name": "Pick",
                            "type": "select",
                            "typeOptions": {
                                "choiceOrder": ["o1", "o2"],
                                "choices": {"o1": {"name": "Alpha"}, "o2": {"name": "Beta"}},
                            },
                        },
                    ],
                }
            ],
            "tableDatas": [],
            "pages": [
                {
                    "id": PAGE,
                    "publishedLayout": {
                        "elementById": {
                            "fc": {
                                "id": "fc",
                                "type": "formContainer",
                                "requiredColumnIds": ["fldName"],
                                "description": [
                                    {"insert": "Welcome to "},
                                    {"insert": "the form", "attributes": {"link": "https://ex.com"}},
                                    {"insert": "!\n"},
                                ],
                            },
                            "sec": {
                                "id": "sec",
                                "type": "section",
                                "title": "Basics",
                                "shouldDisplayTitle": True,
                            },
                            "row1": {"id": "row1", "type": "sectionGridRow"},
                            "row2": {"id": "row2", "type": "sectionGridRow"},
                            "ceName": {
                                "id": "ceName",
                                "type": "cellEditor",
                                "source": {"columnId": "fldName"},
                                "label": {"isEnabled": True, "value": "What is your name?"},
                                "description": [{"insert": "Legal name please\n"}],
                            },
                            "ceChoice": {
                                "id": "ceChoice",
                                "type": "cellEditor",
                                "source": {"columnId": "fldChoice"},
                                "label": {"isEnabled": False},
                            },
                        },
                        "slotElementsById": {
                            "s1": {"parentId": "fc", "elementId": "sec", "index": "a0"},
                            "s2": {"parentId": "sec", "elementId": "row1", "index": "a0"},
                            "s3": {"parentId": "sec", "elementId": "row2", "index": "a1"},
                            "s4": {"parentId": "row1", "elementId": "ceName", "index": "a0"},
                            "s5": {"parentId": "row2", "elementId": "ceChoice", "index": "a0"},
                        },
                    },
                }
            ],
        }
    }
    _install_session(
        monkeypatch,
        [
            (f"/{APP}/{PAGE}/form", FakeResponse(text=_page_html(og_title="My Application - Airtable"))),
            ("readForPages", FakeResponse(payload=form_payload)),
        ],
    )

    docs = airtable_plugin.resolve(
        f"https://airtable.com/{APP}/{PAGE}/form", {"use_cache": False}
    )
    content = docs[0]["content"]
    assert docs[0]["metadata"]["kind"] == "form"
    assert "# My Application" in content
    assert "Welcome to [the form](https://ex.com)!" in content
    assert "## Basics" in content
    assert "### What is your name? *(required)*" in content
    assert "_Legal name please_" in content
    assert "### Pick" in content
    assert "*Single select:*\n- [ ] Alpha\n- [ ] Beta" in content
    assert "Interface record data is not included" not in content
    # ordering: name field before choice field
    assert content.index("What is your name?") < content.index("### Pick")


def test_resolve_failure_returns_error_document(monkeypatch) -> None:
    _install_session(
        monkeypatch,
        [(f"/{APP}/{SHARE}", FakeResponse(text="<html>no bootstrap</html>"))],
    )

    docs = airtable_plugin.resolve(
        f"https://airtable.com/{APP}/{SHARE}", {"use_cache": False}
    )
    doc = docs[0]
    assert doc["label"].endswith("-error")
    assert "generic webpage fallback was suppressed" in doc["content"]
    assert "resolution_error" in doc["metadata"]
