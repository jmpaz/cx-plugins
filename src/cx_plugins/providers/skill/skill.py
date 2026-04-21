from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SKILL_SCHEME_RE = re.compile(r"^skill:(.+)$")
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?\n)---\s*\n", re.DOTALL)
_SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".venv"}

SKILL_FILENAME = "SKILL.md"
VALID_MODIFIERS = {"map", "bare"}

DEFAULT_SEARCH_PATHS = (
    Path.home() / ".claude" / "skills",
    Path.home() / ".codex" / "skills",
)


@dataclass(frozen=True)
class SkillTarget:
    mode: str  # "named" or "path"
    value: str | Path  # skill name or resolved directory
    modifiers: frozenset[str]  # e.g. {"map", "bare"}


def _parse_modifiers(raw: str) -> tuple[str, frozenset[str]]:
    parts = raw.split(":")
    if len(parts) == 1:
        return parts[0], frozenset()
    name_or_path = parts[0]
    mod_str = ":".join(parts[1:])
    mods = frozenset(m.strip() for m in mod_str.split(",") if m.strip())
    unknown = mods - VALID_MODIFIERS
    if unknown:
        return raw, frozenset()
    return name_or_path, mods


def is_skill_target(target: str) -> bool:
    if _SKILL_SCHEME_RE.match(target):
        return True
    path = Path(target).expanduser()
    if path.is_dir() and (path / SKILL_FILENAME).is_file():
        return True
    return False


def parse_skill_target(target: str) -> SkillTarget | None:
    m = _SKILL_SCHEME_RE.match(target)
    if m:
        rest = m.group(1)
        value, modifiers = _parse_modifiers(rest)
        expanded = Path(value).expanduser()
        if expanded.is_absolute() or value.startswith("."):
            resolved = expanded.resolve()
            if resolved.is_dir() and (resolved / SKILL_FILENAME).is_file():
                return SkillTarget("path", resolved, modifiers)
            if resolved.is_file() and resolved.name == SKILL_FILENAME:
                return SkillTarget("path", resolved.parent, modifiers)
        return SkillTarget("named", value, modifiers)

    path = Path(target).expanduser().resolve()
    if path.is_dir() and (path / SKILL_FILENAME).is_file():
        return SkillTarget("path", path, frozenset())
    return None


def find_named_skill(name: str, extra_paths: list[Path] | None = None) -> Path | None:
    search = list(DEFAULT_SEARCH_PATHS)
    if extra_paths:
        search = list(extra_paths) + search

    for base in search:
        candidate = base / name / SKILL_FILENAME
        if candidate.is_file():
            return candidate.parent
    return None


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text

    import yaml

    raw = m.group(1)
    body = text[m.end() :]
    try:
        fm = yaml.safe_load(raw)
    except Exception:
        return {}, text

    if not isinstance(fm, dict):
        return {}, text
    return fm, body


def read_skill(skill_dir: Path) -> tuple[dict[str, Any], str]:
    skill_path = skill_dir / SKILL_FILENAME
    text = skill_path.read_text(encoding="utf-8")
    return parse_frontmatter(text)


_HARNESS_KEYS = {"context", "agent", "model", "effort", "allowed-tools"}


def _virtual_frontmatter(fm: dict[str, Any]) -> str:
    reader_fields = {k: v for k, v in fm.items() if k not in _HARNESS_KEYS}
    if not reader_fields:
        return ""
    lines = ["---"]
    for key, value in reader_fields.items():
        s = str(value)
        if "\n" in s or len(s) > 60:
            indented = "\n".join(f"  {line}" for line in s.splitlines())
            lines.append(f"{key}: |\n{indented}")
        else:
            lines.append(f"{key}: {s}")
    lines.append("---\n")
    return "\n".join(lines)


def _collect_subfiles(skill_dir: Path) -> list[Path]:
    result: list[Path] = []
    for item in sorted(skill_dir.rglob("*")):
        if item.is_dir():
            continue
        if item.name == SKILL_FILENAME and item.parent == skill_dir:
            continue
        if any(
            part.startswith(".") or part in _SKIP_DIRS
            for part in item.relative_to(skill_dir).parts
        ):
            continue
        result.append(item)
    return result


def _build_tree(skill_dir: Path, subfiles: list[Path]) -> str:
    name = skill_dir.name
    lines = [f"{name}/"]
    all_paths = [skill_dir / SKILL_FILENAME] + subfiles
    for i, path in enumerate(all_paths):
        rel = path.relative_to(skill_dir)
        prefix = (
            "\u2514\u2500\u2500 " if i == len(all_paths) - 1 else "\u251c\u2500\u2500 "
        )
        size = path.stat().st_size
        if size < 1024:
            size_str = f"{size}B"
        elif size < 1024 * 1024:
            size_str = f"{size / 1024:.1f}K"
        else:
            size_str = f"{size / (1024 * 1024):.1f}M"
        lines.append(f"  {prefix}{rel}  ({size_str})")
    return "\n".join(lines)


def _read_file_safe(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def resolve_skill(target: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    parsed = parse_skill_target(target)
    if parsed is None:
        return []

    if parsed.mode == "named":
        extra = _extra_search_paths(context)
        skill_dir = find_named_skill(str(parsed.value), extra_paths=extra)
        if skill_dir is None:
            return []
    else:
        skill_dir = parsed.value

    frontmatter, body = read_skill(skill_dir)
    name = frontmatter.get("name", skill_dir.name)
    subfiles = _collect_subfiles(skill_dir)
    mods = parsed.modifiers

    include_skill = "map" not in mods or "bare" in mods or not mods - {"bare"}
    include_subfiles = "bare" not in mods and "map" not in mods
    include_map = "map" in mods

    # bare alone → skill only
    # map alone → map only
    # bare,map → skill + map
    # (none) → skill + subfiles
    if mods == frozenset({"map"}):
        include_skill = False
        include_subfiles = False
        include_map = True
    elif mods == frozenset({"bare"}):
        include_skill = True
        include_subfiles = False
        include_map = False
    elif mods == frozenset({"bare", "map"}):
        include_skill = True
        include_subfiles = False
        include_map = True
    else:
        include_skill = True
        include_subfiles = True
        include_map = False

    docs: list[dict[str, Any]] = []

    if include_map:
        tree = _build_tree(skill_dir, subfiles)
        docs.append(
            {
                "source": target,
                "label": f"skill: {name} (map)",
                "content": tree,
                "metadata": {
                    "trace_path": f"skill/{name}/map",
                    "provider": "skill",
                    "source_path": str(skill_dir),
                    "kind": "skill-map",
                    "frontmatter": frontmatter,
                },
            }
        )

    if include_skill:
        vfm = _virtual_frontmatter(frontmatter)
        content = vfm + body.lstrip("\n") if vfm else body.lstrip("\n")
        docs.append(
            {
                "source": target,
                "label": f"skill: {name}",
                "content": content,
                "metadata": {
                    "trace_path": f"skill/{name}/{SKILL_FILENAME}",
                    "provider": "skill",
                    "source_path": str(skill_dir / SKILL_FILENAME),
                    "kind": "skill",
                    "frontmatter": frontmatter,
                },
            }
        )

    if include_subfiles:
        for subfile in subfiles:
            content = _read_file_safe(subfile)
            if content is None:
                continue
            rel = subfile.relative_to(skill_dir)
            docs.append(
                {
                    "source": target,
                    "label": f"skill: {name}/{rel}",
                    "content": content,
                    "metadata": {
                        "trace_path": f"skill/{name}/{rel}",
                        "provider": "skill",
                        "source_path": str(subfile),
                        "kind": "skill-file",
                    },
                }
            )

    return docs


def _extra_search_paths(context: dict[str, Any]) -> list[Path] | None:
    overrides = context.get("overrides")
    if not isinstance(overrides, dict):
        return None
    skill_cfg = overrides.get("skill")
    if not isinstance(skill_cfg, dict):
        return None
    raw = skill_cfg.get("search_paths") or skill_cfg.get("search-paths")
    if not raw:
        return None
    if isinstance(raw, str):
        raw = [raw]
    return [Path(p).expanduser() for p in raw if isinstance(p, str)]
