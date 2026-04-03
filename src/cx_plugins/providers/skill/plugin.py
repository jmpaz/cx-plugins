from __future__ import annotations

from typing import Any

PLUGIN_API_VERSION = "1"
PLUGIN_NAME = "skill"
PLUGIN_PRIORITY = 50


def normalize_manifest_config(
    raw_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if raw_config is None:
        return None
    if not isinstance(raw_config, dict):
        raise ValueError("skill config must be a mapping")
    return dict(raw_config)


def can_resolve(target: str, context: dict[str, Any]) -> bool:
    from .skill import is_skill_target

    return is_skill_target(target)


def classify_target(target: str, context: dict[str, Any]) -> dict[str, Any] | None:
    from .skill import parse_skill_target

    parsed = parse_skill_target(target)
    if parsed is None:
        return None
    return {
        "provider": PLUGIN_NAME,
        "kind": "skill",
        "is_external": False,
        "group_key": "skill",
    }


def resolve(target: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    from .skill import resolve_skill

    return resolve_skill(target, context)
