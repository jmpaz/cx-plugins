from __future__ import annotations

from typing import Any

PLUGIN_API_VERSION = "1"
PLUGIN_NAME = "arxiv"
PLUGIN_PRIORITY = 100


def normalize_manifest_config(
    raw_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if raw_config is None:
        return None
    if not isinstance(raw_config, dict):
        raise ValueError("arxiv config must be a mapping")
    return dict(raw_config)


def _arxiv_runtime_overrides(raw: dict[str, Any]) -> dict[str, Any] | None:
    result: dict[str, Any] = {}

    for key in ("format", "include_tex_sidecars", "max_tex_sidecars"):
        if key in raw:
            result[key] = raw[key]

    for config_key, result_key in (
        ("paper-format", "format"),
        ("include-tex-sidecars", "include_tex_sidecars"),
        ("max-tex-sidecars", "max_tex_sidecars"),
    ):
        if config_key in raw:
            result[result_key] = raw.get(config_key)

    paper_cfg = raw.get("paper")
    if paper_cfg is not None:
        if not isinstance(paper_cfg, dict):
            raise ValueError("arxiv paper config must be a mapping")
        if "format" in paper_cfg:
            result["format"] = paper_cfg.get("format")

    source_cfg = raw.get("source")
    if source_cfg is not None:
        if not isinstance(source_cfg, dict):
            raise ValueError("arxiv source config must be a mapping")
        for config_key, result_key in (
            ("tex-sidecars", "include_tex_sidecars"),
            ("max-tex-sidecars", "max_tex_sidecars"),
        ):
            if config_key in source_cfg:
                result[result_key] = source_cfg.get(config_key)

    return result or None


def _arxiv_overrides(context: dict[str, Any]) -> dict[str, Any] | None:
    overrides = context.get("overrides")
    if not isinstance(overrides, dict):
        return None
    value = overrides.get("arxiv")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("arxiv overrides must be a mapping")
    return _arxiv_runtime_overrides(value)


def can_resolve(target: str, context: dict[str, Any]) -> bool:
    from .arxiv import is_arxiv_paper_target

    return is_arxiv_paper_target(target)


def classify_target(target: str, context: dict[str, Any]) -> dict[str, Any] | None:
    from .arxiv import parse_arxiv_paper_target

    parsed = parse_arxiv_paper_target(target)
    if parsed is None:
        return None
    return {
        "provider": PLUGIN_NAME,
        "kind": "paper",
        "is_external": True,
        "group_key": "paper",
    }


def resolve(target: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    from .arxiv import (
        arxiv_settings_cache_key,
        build_arxiv_settings,
        resolve_arxiv_paper,
    )

    settings = build_arxiv_settings(_arxiv_overrides(context))
    settings_key = arxiv_settings_cache_key(settings)
    documents = resolve_arxiv_paper(
        target,
        settings=settings,
        use_cache=bool(context.get("use_cache", True)),
        cache_ttl=context.get("cache_ttl"),
        refresh_cache=bool(context.get("refresh_cache", False)),
    )

    out: list[dict[str, Any]] = []
    for document in documents:
        out.append(
            {
                "source": target,
                "label": document.label,
                "content": document.rendered,
                "metadata": {
                    "trace_path": document.label,
                    "provider": PLUGIN_NAME,
                    "source_ref": "arxiv.org",
                    "source_path": document.source_path,
                    "context_subpath": document.context_subpath,
                    "source_created": document.source_created,
                    "source_modified": document.source_modified,
                    "kind": document.kind,
                    "canonical_id": document.canonical_id,
                    "settings_key": settings_key,
                    "hydrate_dedupe": {
                        "mode": "canonical_symlink",
                        "key": (
                            f"arxiv-paper:{document.canonical_id}:"
                            f"{settings_key}:{document.source_path}"
                        ),
                        "rank": document.dedupe_rank,
                    },
                },
            }
        )
    return out
