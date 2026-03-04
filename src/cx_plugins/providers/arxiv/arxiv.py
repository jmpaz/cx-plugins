from __future__ import annotations

import io
import re
import tarfile
from dataclasses import dataclass
from datetime import timedelta
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import quote, unquote, urlparse
import xml.etree.ElementTree as ET

_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"

_IDENTIFIER_NEW_RE = re.compile(r"^(?P<base>\d{4}\.\d{4,5})(?P<version>v\d+)?$")
_IDENTIFIER_OLD_RE = re.compile(
    r"^(?P<base>[A-Za-z\-]+(?:\.[A-Za-z]{2})?/\d{7})(?P<version>v\d+)?$"
)
_INCLUDE_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")
_VALID_FORMATS = frozenset({"md", "tex"})
_BEGIN_DOCUMENT_RE = re.compile(r"\\begin\{document\}")
_END_DOCUMENT_RE = re.compile(r"\\end\{document\}")
_SETUP_LINE_RE = re.compile(
    r"^\s*\\(?:documentclass|usepackage|RequirePackage|PassOptionsToPackage|"
    r"newcommand|renewcommand|providecommand|newtheorem|theoremstyle|setlength|"
    r"addtolength|hypersetup|bibliographystyle|bibliography|pagestyle|"
    r"thispagestyle|title|author|date|icmltitlerunning|icmlsetsymbol|icmlkeywords|"
    r"icmlcorrespondingauthor|icmlcode|icmldata|icmladdress|maketitle)\b"
)
_SETUP_INPUT_RE = re.compile(
    r"^\s*\\input\{[^}]*?(?:preamble|math_commands)[^}]*\}\s*$", re.IGNORECASE
)
_GRAPHICS_MARKER_RE = re.compile(
    r"(?im)^[ \t]*<\s*g\s*r\s*a\s*p\s*h\s*i\s*c\s*s\s*>[ \t]*$"
)
_LATEX_COMMAND_TOKEN_RE = re.compile(r"\\[A-Za-z@]+")
_FIGURE_BEGIN_RE = re.compile(r"\\begin\{figure\*?\}")
_FIGURE_END_RE = re.compile(r"\\end\{figure\*?\}")
_DATE_LINE_RE = re.compile(r"^[A-Za-z]+ \d{1,2}, \d{4}$")
_UNDERLINE_LINE_RE = re.compile(r"^=+$")
_INDENTED_LIST_RE = re.compile(r"^\s{4}(?:[-*+] |\d+\.\s)")
_SECTION_START_RE = re.compile(
    r"(?im)^\s*(?:§\s+[^\n]+|#{1,6}\s+[^\n]+|\\section\*?\{)"
)


@dataclass(frozen=True)
class ParsedArxivTarget:
    raw_target: str
    base_id: str
    version: str | None

    @property
    def canonical_id(self) -> str:
        if self.version:
            return f"{self.base_id}{self.version}"
        return self.base_id


@dataclass(frozen=True)
class ArxivSettings:
    format: str = "md"
    include_tex_sidecars: bool = False
    max_tex_sidecars: int = 8


@dataclass(frozen=True)
class ArxivEntry:
    entry_id: str
    title: str
    summary: str
    authors: tuple[str, ...]
    published: str | None
    updated: str | None
    primary_category: str | None
    categories: tuple[str, ...]
    pdf_url: str


@dataclass(frozen=True)
class ArxivResolvedDocument:
    label: str
    rendered: str
    source_path: str
    context_subpath: str
    canonical_id: str
    kind: str
    dedupe_rank: int
    source_created: str | None
    source_modified: str | None


@dataclass(frozen=True)
class _SourceFile:
    path: str
    text: str


@dataclass(frozen=True)
class _SourceBundle:
    main_path: str
    main_text: str
    sidecars: tuple[_SourceFile, ...]


def _parse_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if not cleaned:
            return default
        return cleaned not in {"0", "false", "no", "off"}
    return default


def _parse_positive_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value if value > 0 else default
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return default
        try:
            parsed = int(cleaned)
        except ValueError:
            return default
        return parsed if parsed > 0 else default
    return default


def _parse_format(value: Any, *, default: str) -> str:
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in _VALID_FORMATS:
            return cleaned
    return default


def build_arxiv_settings(overrides: dict[str, Any] | None) -> ArxivSettings:
    raw = overrides or {}
    return ArxivSettings(
        format=_parse_format(raw.get("format"), default="md"),
        include_tex_sidecars=_parse_bool(
            raw.get("include_tex_sidecars"), default=False
        ),
        max_tex_sidecars=_parse_positive_int(raw.get("max_tex_sidecars"), default=8),
    )


def arxiv_settings_cache_key(settings: ArxivSettings) -> tuple[Any, ...]:
    return (settings.format, settings.include_tex_sidecars, settings.max_tex_sidecars)


def _normalize_identifier(identifier: str) -> ParsedArxivTarget | None:
    cleaned = identifier.strip()
    if not cleaned:
        return None

    match_new = _IDENTIFIER_NEW_RE.fullmatch(cleaned)
    if match_new:
        version = match_new.group("version")
        return ParsedArxivTarget(
            raw_target=identifier,
            base_id=match_new.group("base"),
            version=version.lower() if version else None,
        )

    match_old = _IDENTIFIER_OLD_RE.fullmatch(cleaned)
    if match_old:
        version = match_old.group("version")
        return ParsedArxivTarget(
            raw_target=identifier,
            base_id=match_old.group("base"),
            version=version.lower() if version else None,
        )

    return None


def _identifier_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if not host.endswith("arxiv.org"):
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return None

    prefix = parts[0].lower()
    candidate = "/".join(parts[1:])
    if prefix == "pdf" and candidate.lower().endswith(".pdf"):
        candidate = candidate[:-4]
    if prefix != "abs" and prefix != "pdf":
        return None
    candidate = unquote(candidate).strip()
    return candidate or None


def parse_arxiv_paper_target(target: str) -> ParsedArxivTarget | None:
    cleaned = target.strip()
    if not cleaned:
        return None

    if cleaned.lower().startswith("arxiv://"):
        remainder = cleaned.split("://", 1)[1]
        normalized = remainder.lstrip("/")
        return _normalize_identifier(normalized)

    if cleaned.lower().startswith("arxiv:"):
        remainder = cleaned.split(":", 1)[1]
        normalized = remainder.lstrip("/")
        return _normalize_identifier(normalized)

    identifier_from_url = _identifier_from_url(cleaned)
    if identifier_from_url is not None:
        return _normalize_identifier(identifier_from_url)

    return _normalize_identifier(cleaned)


def is_arxiv_paper_target(target: str) -> bool:
    return parse_arxiv_paper_target(target) is not None


def _http_get(
    url: str,
    *,
    timeout: int,
) -> Any:
    import requests

    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "contextualize/arxiv"},
    )
    response.raise_for_status()
    return response


def _entry_text(node: ET.Element, tag: str) -> str | None:
    value = node.findtext(f"{{{_ATOM_NS}}}{tag}")
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _entry_pdf_url(node: ET.Element, fallback: str) -> str:
    for link in node.findall(f"{{{_ATOM_NS}}}link"):
        href = link.attrib.get("href")
        if not isinstance(href, str) or not href:
            continue
        title = (link.attrib.get("title") or "").strip().lower()
        rel = (link.attrib.get("rel") or "").strip().lower()
        content_type = (link.attrib.get("type") or "").strip().lower()
        if title == "pdf" or content_type == "application/pdf":
            return href
        if rel == "related" and href.endswith(".pdf"):
            return href
    return fallback


def _parse_api_entry(xml_text: str, *, fallback_id: str) -> ArxivEntry:
    root = ET.fromstring(xml_text)
    entry = root.find(f"{{{_ATOM_NS}}}entry")
    if entry is None:
        raise ValueError(f"arXiv paper not found: {fallback_id}")

    entry_id = _entry_text(entry, "id") or ""
    title = _entry_text(entry, "title") or fallback_id
    summary = _entry_text(entry, "summary") or ""
    published = _entry_text(entry, "published")
    updated = _entry_text(entry, "updated")

    authors: list[str] = []
    for author in entry.findall(f"{{{_ATOM_NS}}}author"):
        name = author.findtext(f"{{{_ATOM_NS}}}name")
        if not isinstance(name, str):
            continue
        cleaned = name.strip()
        if cleaned:
            authors.append(cleaned)

    primary_category = None
    primary = entry.find(f"{{{_ARXIV_NS}}}primary_category")
    if primary is not None:
        term = primary.attrib.get("term")
        if isinstance(term, str) and term.strip():
            primary_category = term.strip()

    categories: list[str] = []
    for category in entry.findall(f"{{{_ATOM_NS}}}category"):
        term = category.attrib.get("term")
        if not isinstance(term, str):
            continue
        cleaned = term.strip()
        if cleaned:
            categories.append(cleaned)

    fallback_pdf = f"https://arxiv.org/pdf/{quote(fallback_id, safe='/')}.pdf"
    return ArxivEntry(
        entry_id=entry_id,
        title=title,
        summary=summary,
        authors=tuple(authors),
        published=published,
        updated=updated,
        primary_category=primary_category,
        categories=tuple(categories),
        pdf_url=_entry_pdf_url(entry, fallback=fallback_pdf),
    )


def _safe_path_segment(value: str, *, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def _canonical_dir_name(canonical_id: str) -> str:
    return _safe_path_segment(canonical_id, fallback="paper")


def _decode_source_bytes(data: bytes) -> str | None:
    for encoding in ("utf-8", "latin-1"):
        try:
            decoded = data.decode(encoding)
        except Exception:
            continue
        text = decoded.strip()
        if text:
            return decoded
    return None


def _normalize_tar_member_path(name: str) -> str | None:
    value = name.strip().replace("\\", "/")
    if not value:
        return None
    try:
        normalized = str(PurePosixPath(value))
    except Exception:
        return None
    if normalized.startswith("/"):
        return None
    if ".." in PurePosixPath(normalized).parts:
        return None
    return normalized


def _extract_source_tex_files(data: bytes) -> dict[str, str]:
    extracted: dict[str, str] = {}

    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            for member in archive.getmembers():
                if not member.isfile() or member.size <= 0 or member.size > 2_500_000:
                    continue
                member_path = _normalize_tar_member_path(member.name)
                if member_path is None or not member_path.lower().endswith(".tex"):
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                raw = handle.read()
                text = _decode_source_bytes(raw)
                if text is None:
                    continue
                extracted[member_path] = text
    except tarfile.ReadError:
        maybe_tex = _decode_source_bytes(data)
        if maybe_tex is not None and "\\" in maybe_tex:
            extracted["main.tex"] = maybe_tex

    return extracted


def _main_tex_priority(path: str, text: str) -> tuple[int, int, int, str]:
    name = PurePosixPath(path).name.lower()
    depth = len(PurePosixPath(path).parts)
    has_documentclass = 0 if "\\documentclass" in text else 1
    preferred = 1
    if name in {"main.tex", "paper.tex", "ms.tex", "article.tex"}:
        preferred = 0
    return (preferred, has_documentclass, depth, name)


def _find_main_tex_path(files: dict[str, str]) -> str | None:
    if not files:
        return None
    return min(
        files.keys(),
        key=lambda path: _main_tex_priority(path, files[path]),
    )


def _resolve_include_path(current_path: str, include_token: str) -> str | None:
    token = include_token.strip()
    if not token:
        return None
    if token.startswith("/"):
        return None
    candidate = PurePosixPath(token)
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".tex")
    parent = PurePosixPath(current_path).parent
    resolved = str((parent / candidate).as_posix())
    normalized = _normalize_tar_member_path(resolved)
    return normalized


def _expand_tex_includes(main_path: str, files: dict[str, str]) -> str:
    cache: dict[str, str] = {}

    def _expand(path: str, active: set[str]) -> str:
        cached = cache.get(path)
        if cached is not None:
            return cached

        source = files.get(path)
        if source is None:
            return ""

        out_parts: list[str] = []
        for line in source.splitlines(keepends=True):
            cleaned = _line_without_unescaped_comments(line)
            matches = list(_INCLUDE_RE.finditer(cleaned))
            if not matches:
                out_parts.append(line)
                continue

            cursor = 0
            for match in matches:
                out_parts.append(line[cursor : match.start()])
                include_raw = line[match.start() : match.end()]
                include_target = match.group(1)
                resolved = _resolve_include_path(path, include_target)
                if resolved is None or resolved not in files or resolved in active:
                    out_parts.append(include_raw)
                else:
                    out_parts.append(_expand(resolved, active | {resolved}))
                cursor = match.end()
            out_parts.append(line[cursor:])

        expanded = "".join(out_parts)
        cache[path] = expanded
        return expanded

    return _expand(main_path, {main_path})


def _select_tex_sidecars(
    files: dict[str, str],
    *,
    main_path: str,
    max_items: int,
) -> tuple[_SourceFile, ...]:
    selected_paths: list[str] = [main_path]
    seen = {main_path}

    queue = [main_path]
    while queue and len(selected_paths) < max_items:
        current = queue.pop(0)
        current_text = files.get(current)
        if current_text is None:
            continue
        for include_match in _INCLUDE_RE.findall(current_text):
            resolved = _resolve_include_path(current, include_match)
            if resolved is None or resolved in seen:
                continue
            if resolved not in files:
                continue
            seen.add(resolved)
            selected_paths.append(resolved)
            queue.append(resolved)
            if len(selected_paths) >= max_items:
                break

    if len(selected_paths) < max_items:
        for path in sorted(files.keys()):
            if path in seen:
                continue
            selected_paths.append(path)
            seen.add(path)
            if len(selected_paths) >= max_items:
                break

    return tuple(_SourceFile(path=path, text=files[path]) for path in selected_paths)


def _fetch_source_bundle(
    parsed: ParsedArxivTarget,
    *,
    max_tex_sidecars: int,
) -> _SourceBundle | None:
    source_url = f"https://arxiv.org/e-print/{quote(parsed.canonical_id, safe='/')}"
    try:
        response = _http_get(source_url, timeout=30)
    except Exception:
        return None

    tex_files = _extract_source_tex_files(response.content)
    if not tex_files:
        return None

    main_path = _find_main_tex_path(tex_files)
    if main_path is None:
        return None
    main_text = tex_files.get(main_path)
    if not isinstance(main_text, str) or not main_text.strip():
        return None

    expanded_main_text = _expand_tex_includes(main_path, tex_files).strip()
    if expanded_main_text:
        main_text = expanded_main_text

    sidecar_limit = max(1, max_tex_sidecars)
    sidecars = _select_tex_sidecars(
        tex_files,
        main_path=main_path,
        max_items=sidecar_limit,
    )
    return _SourceBundle(main_path=main_path, main_text=main_text, sidecars=sidecars)


def _fetch_pdf_text(
    pdf_url: str,
    *,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
) -> str | None:
    from contextualize.references.url import URLReference

    reference = URLReference(
        pdf_url,
        format="raw",
        label="relative",
        inject=False,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )
    text = reference.read()
    cleaned = text.strip()
    return cleaned or None


def _extract_document_body(tex: str) -> str:
    begin = _BEGIN_DOCUMENT_RE.search(tex)
    if begin is None:
        return tex
    end = _END_DOCUMENT_RE.search(tex, begin.end())
    if end is None:
        return tex[begin.end() :]
    return tex[begin.end() : end.start()]


def _strip_setup_lines(tex: str) -> str:
    kept: list[str] = []
    for line in tex.splitlines():
        if _SETUP_LINE_RE.match(line):
            continue
        if _SETUP_INPUT_RE.match(line):
            continue
        kept.append(line)
    return "\n".join(kept)


def _extract_figure_blocks(tex: str) -> tuple[str, ...]:
    ast_blocks = _extract_figure_blocks_via_ast(tex)
    if ast_blocks:
        return ast_blocks
    return _extract_figure_blocks_via_scan(tex)


def _extract_figure_blocks_via_ast(tex: str) -> tuple[str, ...]:
    try:
        from pylatexenc.latexwalker import LatexEnvironmentNode, LatexWalker
    except Exception:
        return ()

    try:
        nodes, _, _ = LatexWalker(tex).get_latex_nodes(pos=0)
    except Exception:
        return ()

    blocks: list[str] = []

    def _walk(node_list: list[Any]) -> None:
        for node in node_list:
            if isinstance(node, LatexEnvironmentNode):
                env_name = getattr(node, "environmentname", "")
                if env_name in {"figure", "figure*"}:
                    start = getattr(node, "pos", None)
                    length = getattr(node, "len", None)
                    if (
                        isinstance(start, int)
                        and isinstance(length, int)
                        and length > 0
                    ):
                        block = tex[start : start + length].strip()
                        if block:
                            blocks.append(block)

            child_nodes = getattr(node, "nodelist", None)
            if isinstance(child_nodes, list) and child_nodes:
                _walk(child_nodes)

            nodeargd = getattr(node, "nodeargd", None)
            arg_nodes = getattr(nodeargd, "argnlist", None) if nodeargd else None
            if isinstance(arg_nodes, list):
                for arg in arg_nodes:
                    if arg is None:
                        continue
                    nested = getattr(arg, "nodelist", None)
                    if isinstance(nested, list) and nested:
                        _walk(nested)

    _walk(nodes)
    return tuple(blocks)


def _line_without_unescaped_comments(line: str) -> str:
    escaped = False
    for idx, char in enumerate(line):
        if char == "\\":
            escaped = not escaped
            continue
        if char == "%" and not escaped:
            return line[:idx]
        escaped = False
    return line


def _extract_figure_blocks_via_scan(tex: str) -> tuple[str, ...]:
    blocks: list[str] = []
    line_starts: list[int] = []
    cursor = 0
    lines = tex.splitlines(keepends=True)
    for line in lines:
        line_starts.append(cursor)
        cursor += len(line)

    depth = 0
    start_offset: int | None = None
    for line_index, line in enumerate(lines):
        cleaned = _line_without_unescaped_comments(line)

        begin_count = len(_FIGURE_BEGIN_RE.findall(cleaned))
        end_count = len(_FIGURE_END_RE.findall(cleaned))

        if depth == 0 and begin_count > 0:
            start_offset = line_starts[line_index]

        if begin_count > 0:
            depth += begin_count

        if end_count > 0 and depth > 0:
            depth -= end_count
            if depth <= 0 and start_offset is not None:
                end_offset = line_starts[line_index] + len(line)
                block = tex[start_offset:end_offset].strip()
                if block:
                    blocks.append(block)
                start_offset = None
                depth = 0

    return tuple(blocks)


def _prepare_latex_for_conversion(tex: str) -> str:
    body = _extract_document_body(tex)
    filtered = _strip_setup_lines(body)
    return filtered.strip()


def _latex_to_text(source: str) -> str:
    from pylatexenc.latex2text import LatexNodes2Text

    return LatexNodes2Text().latex_to_text(source)


def _postprocess_converted_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    lines = _strip_maketitle_artifacts(lines)
    lines = _normalize_prose_indentation(lines)
    out: list[str] = []
    blank_count = 0
    for line in lines:
        if line.strip():
            blank_count = 0
            out.append(line)
            continue
        blank_count += 1
        if blank_count <= 2:
            out.append("")
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def _strip_maketitle_artifacts(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped in {"[NO \\title GIVEN]", "[NO \\author GIVEN]"}:
            continue
        cleaned.append(line)

    first_nonempty = 0
    while first_nonempty < len(cleaned) and not cleaned[first_nonempty].strip():
        first_nonempty += 1

    if first_nonempty >= len(cleaned):
        return cleaned

    trailing = cleaned[first_nonempty:]
    if not trailing:
        return cleaned

    idx = 0
    while idx < len(trailing):
        stripped = trailing[idx].strip()
        if stripped and (
            _DATE_LINE_RE.fullmatch(stripped) or _UNDERLINE_LINE_RE.fullmatch(stripped)
        ):
            idx += 1
            continue
        break

    if idx > 0:
        return cleaned[:first_nonempty] + trailing[idx:]
    return cleaned


def _normalize_prose_indentation(lines: list[str]) -> list[str]:
    normalized: list[str] = []
    in_fence = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            normalized.append(line.lstrip())
            continue

        if in_fence or not line.startswith("    "):
            normalized.append(line)
            continue

        if _INDENTED_LIST_RE.match(line):
            normalized.append(line)
            continue

        candidate = line[4:]
        if candidate.startswith((">", "|", "`")):
            normalized.append(line)
            continue
        normalized.append(candidate)

    return normalized


def _inject_figure_blocks_at_markers(
    text: str, figure_blocks: tuple[str, ...]
) -> tuple[str, tuple[str, ...]]:
    if not figure_blocks:
        return text, ()

    lines = text.split("\n")
    out: list[str] = []
    next_index = 0
    i = 0

    while i < len(lines):
        line = lines[i]
        if _GRAPHICS_MARKER_RE.fullmatch(line) is None:
            out.append(line)
            i += 1
            continue

        if next_index >= len(figure_blocks):
            out.append(line)
            i += 1
            continue

        block = figure_blocks[next_index]
        next_index += 1
        out.extend(["```tex", block, "```"])
        i += 1

        caption = _extract_caption_text(block)
        if not caption:
            continue

        scan = i
        while scan < len(lines) and not lines[scan].strip():
            scan += 1
        if scan >= len(lines):
            continue

        if _texts_equivalent_caption(lines[scan], caption):
            i = scan + 1

    replaced = "\n".join(out)
    return replaced, figure_blocks[next_index:]


def _extract_caption_text(figure_block: str) -> str:
    marker = "\\caption{"
    start = figure_block.find(marker)
    if start < 0:
        return ""

    i = start + len(marker)
    depth = 1
    out_chars: list[str] = []
    while i < len(figure_block):
        char = figure_block[i]
        if char == "{":
            depth += 1
            out_chars.append(char)
            i += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                break
            out_chars.append(char)
            i += 1
            continue
        out_chars.append(char)
        i += 1

    if depth != 0:
        return ""

    raw = "".join(out_chars)
    cleaned = re.sub(r"\\ref\{[^}]+\}", "<ref>", raw)
    cleaned = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", "", cleaned)
    cleaned = cleaned.replace("{", "").replace("}", "")
    return " ".join(cleaned.split()).strip()


def _normalize_captionish_text(text: str) -> str:
    collapsed = " ".join(text.split()).lower()
    collapsed = collapsed.replace("“", '"').replace("”", '"').replace("’", "'")
    collapsed = re.sub(r"\\ref\{[^}]+\}", "<ref>", collapsed)
    collapsed = re.sub(r"[`\"'.,;:!?()\[\]{}]", "", collapsed)
    return collapsed.strip()


def _texts_equivalent_caption(candidate: str, caption: str) -> bool:
    cand_norm = _normalize_captionish_text(candidate)
    cap_norm = _normalize_captionish_text(caption)
    if not cand_norm or not cap_norm:
        return False
    if cand_norm == cap_norm:
        return True
    return cand_norm.startswith(cap_norm) or cap_norm.startswith(cand_norm)


def _append_figure_fallback(text: str, figure_blocks: tuple[str, ...]) -> str:
    if not figure_blocks:
        return text

    parts = [text.rstrip(), "", "## Figures (LaTeX)", ""]
    for block in figure_blocks:
        parts.extend(["```tex", block, "```", ""])
    return "\n".join(parts).strip()


def _conversion_is_low_quality(converted: str, source: str) -> bool:
    cleaned = converted.strip()
    if not cleaned:
        return True

    source_clean = source.strip()
    source_len = len(source_clean)
    min_chars = 8 if source_len < 1000 else max(60, source_len // 40)
    if len(cleaned) < min_chars:
        return True

    word_count = len(cleaned.split())
    command_count = len(_LATEX_COMMAND_TOKEN_RE.findall(cleaned))
    if command_count >= 20 and command_count > max(10, word_count // 2):
        return True
    return False


def _convert_latex_source_to_markdown(source_text: str) -> str | None:
    prepared = _prepare_latex_for_conversion(source_text)
    if not prepared:
        return None

    figure_blocks = _extract_figure_blocks(prepared)

    try:
        converted_raw = _latex_to_text(prepared)
    except Exception:
        return None

    converted = _postprocess_converted_text(converted_raw)
    if _conversion_is_low_quality(converted, prepared):
        return None
    with_inline, unmatched = _inject_figure_blocks_at_markers(converted, figure_blocks)
    return _append_figure_fallback(with_inline, unmatched)


def _render_source_main_text(source_text: str, settings: ArxivSettings) -> str:
    cleaned = source_text.strip()
    if settings.format == "tex":
        return cleaned

    converted = _convert_latex_source_to_markdown(cleaned)
    if converted is None:
        return cleaned
    return converted


def _format_main_document(
    *,
    entry: ArxivEntry,
    canonical_id: str,
    paper_text: str,
) -> str:
    abstract_text = (entry.summary or "").strip()
    body_text = _body_from_converted_paper_text(paper_text, abstract_text)

    lines: list[str] = [f"title: {entry.title}"]
    lines.append(f"url: https://arxiv.org/abs/{canonical_id}")
    if entry.published:
        lines.append(f"published: {entry.published}")
    if entry.updated:
        lines.append(f"updated: {entry.updated}")
    if entry.categories:
        lines.append(f"categories: {', '.join(entry.categories)}")
    if entry.authors:
        lines.append(f"authors: {', '.join(entry.authors)}")
    lines.extend(["", "§ ABSTRACT", ""])
    if abstract_text:
        lines.append(abstract_text)
    elif body_text:
        lines.append(body_text)
        lines.append("")
        return "\n".join(lines)
    lines.extend(["", body_text.strip()])
    lines.append("")
    return "\n".join(lines)


def _body_from_converted_paper_text(paper_text: str, abstract_text: str) -> str:
    cleaned = paper_text.strip()
    if not cleaned:
        return ""

    section_match = _SECTION_START_RE.search(cleaned)
    if section_match is not None:
        return cleaned[section_match.start() :].strip()

    if _looks_like_summary_prefix(cleaned, abstract_text):
        parts = re.split(r"\n\s*\n", cleaned, maxsplit=1)
        if len(parts) == 2:
            return parts[1].strip()

    return cleaned


def _looks_like_summary_prefix(text: str, summary: str) -> bool:
    if not text or not summary:
        return False
    return _normalize_space(text).startswith(_normalize_space(summary))


def _normalize_space(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _safe_source_relpath(path: str) -> str:
    parts = [
        _safe_path_segment(segment, fallback="part")
        for segment in path.replace("\\", "/").split("/")
        if segment.strip()
    ]
    if not parts:
        return "source.tex"
    return "/".join(parts)


def _entry_identifier_from_api(entry: ArxivEntry) -> ParsedArxivTarget | None:
    parsed = urlparse(entry.entry_id)
    path = parsed.path or ""
    if "/abs/" not in path:
        return None
    identifier = path.split("/abs/", 1)[1]
    return _normalize_identifier(identifier)


def resolve_arxiv_paper(
    target: str,
    *,
    settings: ArxivSettings,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
) -> list[ArxivResolvedDocument]:
    parsed = parse_arxiv_paper_target(target)
    if parsed is None:
        raise ValueError(f"Unsupported arXiv target: {target}")

    query_id = parsed.canonical_id
    api_url = f"https://export.arxiv.org/api/query?id_list={quote(query_id, safe='/')}"
    response = _http_get(api_url, timeout=20)
    entry = _parse_api_entry(response.text, fallback_id=query_id)

    api_identifier = _entry_identifier_from_api(entry)
    if parsed.version is None and api_identifier is not None:
        canonical_id = api_identifier.base_id
    else:
        canonical_id = parsed.canonical_id

    source_bundle = _fetch_source_bundle(
        parsed,
        max_tex_sidecars=settings.max_tex_sidecars,
    )

    paper_text: str | None = None
    if source_bundle is not None and source_bundle.main_text.strip():
        paper_text = _render_source_main_text(source_bundle.main_text, settings)

    if not paper_text:
        paper_text = _fetch_pdf_text(
            entry.pdf_url,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            refresh_cache=refresh_cache,
        )

    if not paper_text:
        raise ValueError(f"Unable to resolve arXiv paper content: {target}")

    dir_name = _canonical_dir_name(canonical_id)
    documents: list[ArxivResolvedDocument] = [
        ArxivResolvedDocument(
            label=f"arXiv/{canonical_id}",
            rendered=_format_main_document(
                entry=entry,
                canonical_id=canonical_id,
                paper_text=paper_text,
            ),
            source_path=canonical_id,
            context_subpath=f"arxiv/{dir_name}/paper.md",
            canonical_id=canonical_id,
            kind="paper",
            dedupe_rank=0,
            source_created=entry.published,
            source_modified=entry.updated,
        )
    ]

    if settings.include_tex_sidecars and source_bundle is not None:
        for rank, sidecar in enumerate(source_bundle.sidecars, start=1):
            source_relpath = _safe_source_relpath(sidecar.path)
            documents.append(
                ArxivResolvedDocument(
                    label=f"{canonical_id}/source/{source_relpath}",
                    rendered=sidecar.text.strip(),
                    source_path=f"{canonical_id}/source/{sidecar.path}",
                    context_subpath=f"arxiv/{dir_name}/source/{source_relpath}",
                    canonical_id=canonical_id,
                    kind="source_tex",
                    dedupe_rank=rank,
                    source_created=entry.published,
                    source_modified=entry.updated,
                )
            )

    return documents
