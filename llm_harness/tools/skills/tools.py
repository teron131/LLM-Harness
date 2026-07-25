"""Workspace skills tools."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
from typing import Any

from langchain.tools import tool
from langchain_core.embeddings import Embeddings
import yaml

from ...clients import OpenAIEmbeddings

SKILL_FILENAME = "SKILL.md"
DEFAULT_MAX_FILES = 20
DEFAULT_MAX_TOKENS = 50_000
CHARS_PER_TOKEN_RATIO = 4
DEFAULT_MAX_CHARS_PER_FILE = DEFAULT_MAX_TOKENS * CHARS_PER_TOKEN_RATIO
DEFAULT_SEARCH_SKILLS_MODEL = "openai/text-embedding-3-small"
DEFAULT_SKILLS_TOP_K = 5
DEFAULT_SKILLS_SCORE_THRESHOLD = 0.2
IGNORED_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
}
SKILLS_NAME_PATTERN = re.compile(r"^[a-z0-9-]{1,64}$")


@dataclass(frozen=True)
class SkillsFile:
    """Parsed skills file content plus stable path metadata."""

    path: Path
    relative_path: str
    skills_root: Path
    frontmatter: dict[str, Any]
    body: str


@dataclass(frozen=True)
class SkillsSearchEntry:
    """Indexed skills metadata plus one cached embedding vector."""

    payload: dict[str, Any]
    embedding: list[float]


@dataclass(frozen=True)
class SkillsSearchIndex:
    """One cached embedding index for workspace skills search."""

    signature: tuple[tuple[str, int, int], ...]
    entries: list[SkillsSearchEntry]


_SEARCH_SKILLS_CACHE: dict[tuple[str, str], SkillsSearchIndex] = {}


def _resolve_search_path(
    *,
    root_dir: Path,
    path: str,
) -> Path:
    """Resolve one search path from the current workspace or an absolute path."""
    cleaned_path = path.strip() or "."
    candidate_path = Path(cleaned_path).expanduser()
    if not candidate_path.is_absolute():
        candidate_path = root_dir / candidate_path
    candidate_path = candidate_path.resolve()
    return candidate_path


def _resolve_skills_root(
    *,
    search_path: Path,
    root_dir: Path,
) -> Path:
    """Choose the relative-path root for one search path."""
    try:
        search_path.relative_to(root_dir)
    except ValueError:
        return search_path if search_path.is_dir() else search_path.parent
    return root_dir


def _discover_skills_for_path(path: str) -> tuple[list[SkillsFile], Path]:
    """Return discovered skills entries plus the root used for relative skills paths."""
    root_dir = Path.cwd().resolve()
    search_path = _resolve_search_path(
        root_dir=root_dir,
        path=path,
    )
    skills_root = _resolve_skills_root(
        search_path=search_path,
        root_dir=root_dir,
    )
    return _discover_skills_files(search_path, root_dir=skills_root), skills_root


def _search_skills_signature(
    skills_files: list[SkillsFile],
) -> tuple[tuple[str, int, int], ...]:
    """Return one stable cache signature for the current skills files."""
    return tuple(
        (
            skills_file.relative_path,
            skills_file.path.stat().st_mtime_ns,
            skills_file.path.stat().st_size,
        )
        for skills_file in skills_files
    )


def _iter_skills_paths(search_path: Path) -> list[Path]:
    """Return stable SKILL.md paths under one rooted search path."""
    if search_path.is_file():
        return [search_path] if search_path.name == SKILL_FILENAME else []

    discovered_paths: list[Path] = []
    for current_root, dir_names, file_names in os.walk(search_path):
        dir_names[:] = sorted(
            name for name in dir_names if name not in IGNORED_DIR_NAMES
        )
        for file_name in sorted(file_names):
            if file_name == SKILL_FILENAME:
                discovered_paths.append(Path(current_root) / file_name)
    return discovered_paths


def _parse_skills_file(
    path: Path,
    *,
    root_dir: Path,
) -> SkillsFile:
    """Read one skills file and return parsed frontmatter plus markdown body."""
    text = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text)
    return SkillsFile(
        path=path,
        relative_path=str(path.relative_to(root_dir)),
        skills_root=path.parent,
        frontmatter=frontmatter,
        body=body,
    )


def _discover_skills_files(
    search_path: Path,
    *,
    root_dir: Path,
) -> list[SkillsFile]:
    """Return parsed skills files under one rooted search path."""
    return [
        _parse_skills_file(
            path,
            root_dir=root_dir,
        )
        for path in _iter_skills_paths(search_path)
    ]


def _split_frontmatter(
    text: str,
) -> tuple[
    dict[str, Any],
    str,
]:
    """Return parsed YAML frontmatter plus the remaining markdown body."""
    if not text.startswith("---\n"):
        return {}, text

    end_marker = text.find("\n---\n", 4)
    if end_marker == -1:
        return {}, text

    frontmatter_text = text[4:end_marker]
    body = text[end_marker + 5 :]
    try:
        parsed_frontmatter = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError:
        return {}, body
    if not isinstance(parsed_frontmatter, dict):
        return {}, body
    return parsed_frontmatter, body


def _iter_resource_paths(skills_root: Path, resource_dir_name: str) -> list[Path]:
    """Return stable file paths under one optional skills resource directory."""
    resource_root = skills_root / resource_dir_name
    if not resource_root.is_dir():
        return []

    discovered_paths: list[Path] = []
    for current_root, dir_names, file_names in os.walk(resource_root):
        dir_names[:] = sorted(
            name for name in dir_names if name not in IGNORED_DIR_NAMES
        )
        for file_name in sorted(file_names):
            discovered_paths.append(Path(current_root) / file_name)
    return discovered_paths


def _skills_metadata(
    skills_file: SkillsFile,
) -> tuple[
    dict[str, Any] | None,
    str | None,
]:
    """Build one standards-aligned metadata payload for a skills file."""
    skills_name = skills_file.frontmatter.get("name")
    description = skills_file.frontmatter.get("description")

    if not isinstance(skills_name, str) or not skills_name.strip():
        return (
            None,
            f"{skills_file.relative_path}: missing required frontmatter field 'name'",
        )
    if (
        not SKILLS_NAME_PATTERN.fullmatch(skills_name)
        or skills_name.startswith("-")
        or skills_name.endswith("-")
        or "--" in skills_name
    ):
        return None, f"{skills_file.relative_path}: invalid skills name '{skills_name}'"
    if skills_file.skills_root.name != skills_name:
        return (
            None,
            f"{skills_file.relative_path}: skills name '{skills_name}' must match parent directory '{skills_file.skills_root.name}'",
        )
    if not isinstance(description, str) or not description.strip():
        return (
            None,
            f"{skills_file.relative_path}: missing required frontmatter field 'description'",
        )

    return {
        "path": skills_file.relative_path,
        "name": skills_name,
        "description": description,
    }, None


def _read_text_file(
    path: Path,
    *,
    root_dir: Path,
    max_chars: int,
) -> dict[str, Any]:
    """Return bounded UTF-8 text content for one file."""
    text = path.read_text(encoding="utf-8")
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(root_dir)),
        "content": text[:max_chars],
        "truncated": len(text) > max_chars,
    }


def _load_resource_group(
    paths: list[Path],
    *,
    root_dir: Path,
    max_chars_per_file: int,
) -> list[dict[str, Any]]:
    """Load bounded text payloads for one resource group."""
    return [
        _read_text_file(
            path,
            root_dir=root_dir,
            max_chars=max_chars_per_file,
        )
        for path in paths
    ]


def _load_skills_entry(
    skills_file: SkillsFile,
    *,
    root_dir: Path,
    max_chars_per_file: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """Load one skills entry with instructions plus scripts and references."""
    metadata, error_message = _skills_metadata(skills_file)
    if metadata is None:
        return None, error_message

    scripts = _iter_resource_paths(skills_file.skills_root, "scripts")
    references = _iter_resource_paths(skills_file.skills_root, "references")
    return {
        **metadata,
        "skills_path": str(skills_file.path),
        "instructions": skills_file.body[:max_chars_per_file],
        "instructions_truncated": len(skills_file.body) > max_chars_per_file,
        "scripts": _load_resource_group(
            scripts,
            root_dir=root_dir,
            max_chars_per_file=max_chars_per_file,
        ),
        "references": _load_resource_group(
            references,
            root_dir=root_dir,
            max_chars_per_file=max_chars_per_file,
        ),
    }, None


def _matching_skills_files(
    skills_files: list[SkillsFile],
    names: list[str],
) -> tuple[list[SkillsFile], list[str]]:
    """Return selected skills entries plus diagnostics for unknown names."""
    if not names:
        return skills_files, []

    skills_files_by_name: dict[str, SkillsFile] = {}
    diagnostics: list[str] = []
    for skills_file in skills_files:
        metadata, error_message = _skills_metadata(skills_file)
        if metadata is not None:
            skills_files_by_name[metadata["name"]] = skills_file
        elif error_message is not None:
            diagnostics.append(error_message)

    selected_skills_files: list[SkillsFile] = []
    for name in names:
        matched_skills_file = skills_files_by_name.get(name)
        if matched_skills_file is None:
            diagnostics.append(f"Skills not found: {name}")
            continue
        selected_skills_files.append(matched_skills_file)
    return selected_skills_files, diagnostics


def _result_payload(
    skills: list[dict[str, Any]],
    diagnostics: list[str],
) -> dict[str, Any]:
    """Build the shared result shape for skills tools."""
    result: dict[str, Any] = {
        "status": "ok",
        "skills": skills,
    }
    if diagnostics:
        result["diagnostics"] = diagnostics
    return result


def _searchable_skills_entry(
    skills_file: SkillsFile,
) -> tuple[tuple[dict[str, Any], str] | None, str | None]:
    """Return one searchable metadata payload plus embedding text."""
    metadata, error_message = _skills_metadata(skills_file)
    if metadata is None:
        return None, error_message

    aliases = skills_file.frontmatter.get("aliases")
    alias_text = (
        ", ".join(
            alias for alias in aliases if isinstance(alias, str) and alias.strip()
        )
        if isinstance(aliases, list)
        else ""
    )
    tags = skills_file.frontmatter.get("tags")
    tag_text = (
        ", ".join(tag for tag in tags if isinstance(tag, str) and tag.strip())
        if isinstance(tags, list)
        else ""
    )
    parts = [
        f"Description: {metadata['description']}",
        f"Skills: {metadata['name']}",
    ]
    if alias_text:
        parts.append(f"Aliases: {alias_text}")
    if tag_text:
        parts.append(f"Tags: {tag_text}")
    return (metadata, "\n".join(parts)), None


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity for two embedding vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0

    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=False)
    )
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def _build_search_skills_index(
    *,
    skills_files: list[SkillsFile],
    skills_root: Path,
    embeddings: Embeddings,
) -> tuple[SkillsSearchIndex, list[str]]:
    """Build or reuse one cached embedding index for the current skills files."""
    diagnostics: list[str] = []
    signature = _search_skills_signature(skills_files)
    cache_key = (str(skills_root), getattr(embeddings, "model", ""))
    cached_index = _SEARCH_SKILLS_CACHE.get(cache_key)
    if cached_index is not None and cached_index.signature == signature:
        return cached_index, diagnostics

    searchable_entries: list[tuple[dict[str, Any], str]] = []
    for skills_file in skills_files:
        searchable_entry, error_message = _searchable_skills_entry(skills_file)
        if searchable_entry is not None:
            searchable_entries.append(searchable_entry)
        elif error_message is not None:
            diagnostics.append(error_message)

    if not searchable_entries:
        empty_index = SkillsSearchIndex(signature=signature, entries=[])
        _SEARCH_SKILLS_CACHE[cache_key] = empty_index
        return empty_index, diagnostics

    search_texts = [search_text for _, search_text in searchable_entries]
    vectors = embeddings.embed_documents(search_texts)
    entries = [
        SkillsSearchEntry(
            payload=payload,
            embedding=[float(value) for value in vector],
        )
        for (payload, _), vector in zip(searchable_entries, vectors, strict=False)
    ]
    index = SkillsSearchIndex(signature=signature, entries=entries)
    _SEARCH_SKILLS_CACHE[cache_key] = index
    return index, diagnostics


@tool(parse_docstring=True)
def list_skills(
    path: str = ".",
    max_files: int = DEFAULT_MAX_FILES,
) -> dict[str, Any]:
    """List skills from the workspace.

    Args:
        path: Relative directory or file path from the current working directory, or an absolute path. Defaults to the current working directory.
        max_files: Maximum number of skills entries to return.
    """
    skills_files, _ = _discover_skills_for_path(path)
    bounded_files = skills_files[: max(0, max_files)]
    skills: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    for skills_file in bounded_files:
        payload, error_message = _skills_metadata(skills_file)
        if payload is not None:
            skills.append(payload)
        elif error_message is not None:
            diagnostics.append(error_message)
    return _result_payload(skills, diagnostics)


@tool(parse_docstring=True)
def search_skills(
    query: str,
    path: str = ".",
    top_k: int = DEFAULT_SKILLS_TOP_K,
    score_threshold: float = DEFAULT_SKILLS_SCORE_THRESHOLD,
    model: str = DEFAULT_SEARCH_SKILLS_MODEL,
) -> dict[str, Any]:
    """Search workspace skills semantically from descriptions and metadata only.

    Args:
        query: Natural-language task or question used to find relevant skills.
        path: Relative directory or file path from the current working directory, or an absolute path. Defaults to the current working directory.
        top_k: Maximum number of matching skills entries to return.
        score_threshold: Minimum cosine similarity score required to include a match.
        model: Embedding model name understood by the configured OpenAI-compatible endpoint.
    """
    if not query.strip():
        return _result_payload([], ["Missing required query"])

    skills_files, skills_root = _discover_skills_for_path(path)
    try:
        embeddings = OpenAIEmbeddings(
            model=model,
            check_embedding_ctx_length=False,
        )
        index, diagnostics = _build_search_skills_index(
            skills_files=skills_files,
            skills_root=skills_root,
            embeddings=embeddings,
        )
        query_embedding = [float(value) for value in embeddings.embed_query(query)]
    except Exception as exc:
        return {
            "status": "error",
            "skills": [],
            "diagnostics": [str(exc)],
        }

    bounded_top_k = max(0, top_k)
    scored_skills = []
    for entry in index.entries:
        score = _cosine_similarity(
            entry.embedding, [float(value) for value in query_embedding]
        )
        if score < score_threshold:
            continue
        scored_skills.append(
            {
                **entry.payload,
                "score": round(score, 6),
            }
        )

    scored_skills.sort(key=lambda skills: float(skills["score"]), reverse=True)
    result = _result_payload(scored_skills[:bounded_top_k], diagnostics)
    result["indexed_skills"] = len(index.entries)
    result["model"] = model
    result["score_threshold"] = score_threshold
    return result


@tool(parse_docstring=True)
def load_skills(
    path: str = ".",
    skills: str = "",
) -> dict[str, Any]:
    """Load one selected skills entry from the workspace, including instructions, scripts, and references.

    Args:
        path: Relative directory or file path from the current working directory, or an absolute path. Defaults to the current working directory.
        skills: Skills entry name to load.
    """
    skills_files, skills_root = _discover_skills_for_path(path)
    if not skills.strip():
        return _result_payload([], ["Missing required skills"])

    selected_skills_files, diagnostics = _matching_skills_files(
        skills_files,
        names=[skills],
    )
    loaded_skills: list[dict[str, Any]] = []
    for skills_file in selected_skills_files[:1]:
        payload, error_message = _load_skills_entry(
            skills_file,
            root_dir=skills_root,
            max_chars_per_file=DEFAULT_MAX_CHARS_PER_FILE,
        )
        if payload is not None:
            loaded_skills.append(payload)
        elif error_message is not None:
            diagnostics.append(error_message)
    return _result_payload(loaded_skills, diagnostics)
