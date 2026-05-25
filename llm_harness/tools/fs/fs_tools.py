"""Minimal filesystem tools for tool-calling.

Sandbox-oriented with root_dir + traversal protection.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
import subprocess

from langchain.tools import tool

from .apply_patch import apply_patch_chunks_to_text, parse_single_file_patch_with_stats

PATH_TRAVERSAL_ERROR = "Path traversal not allowed"
PATH_OUTSIDE_ROOT_ERROR = "Path outside root"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SandboxFS:
    """Sandboxed filesystem wrapper with path traversal protection."""

    root_dir: Path

    def resolve(self, user_path: str) -> Path:
        """Resolve a relative path against the configured root."""
        cleaned_path = user_path.strip()
        if not cleaned_path:
            raise ValueError("Empty path")
        if cleaned_path.startswith("~"):
            raise ValueError(PATH_TRAVERSAL_ERROR)

        virtual_path = cleaned_path if cleaned_path.startswith("/") else f"/{cleaned_path}"
        if ".." in virtual_path:
            raise ValueError(PATH_TRAVERSAL_ERROR)

        resolved_path = (self.root_dir / virtual_path.lstrip("/")).resolve()
        try:
            resolved_path.relative_to(self.root_dir)
        except ValueError as e:
            raise ValueError(PATH_OUTSIDE_ROOT_ERROR) from e
        return resolved_path

    def require_file(self, path: str) -> Path:
        """Return a path only when it points to an existing file."""
        file_path = self.resolve(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return file_path

    def read_text(self, path: str) -> str:
        """Read text from a rooted file path."""
        return self.require_file(path).read_text(encoding="utf-8")

    def write_text(self, path: str, text: str) -> None:
        """Write text to a rooted file path."""
        file_path = self.resolve(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(text, encoding="utf-8")

    def apply_patch(
        self,
        patch: str,
        *,
        target_path: str | None = None,
    ) -> str:
        """Apply a unified patch inside the rooted filesystem."""
        file_patch, patch_stats = parse_single_file_patch_with_stats(
            patch_text=patch,
            target_path=target_path,
        )
        path = f"/{file_patch.path.lstrip('/')}"
        original_text = self.read_text(path)
        updated_text = apply_patch_chunks_to_text(
            original_text=original_text,
            file_path=path,
            chunks=file_patch.chunks,
        )
        self.write_text(path, updated_text)
        logger.info(f"[FS_PATCH] {path} c={patch_stats.chunk_count} r={patch_stats.lines_removed} i={patch_stats.lines_inserted} t={patch_stats.lines_touched}")
        return f"Patched {path}"


def make_fs_tools(*, root_dir: str | Path):
    """Create sandboxed filesystem tools for file operations."""
    fs = SandboxFS(Path(root_dir).resolve())

    @tool(parse_docstring=True)
    def fs_read_text(path: str) -> str:
        """Read a UTF-8 text file from the sandboxed workspace.

        Args:
            path: File path relative to the sandbox root (or virtual absolute like "/foo.txt").
        """

        return fs.read_text(path)

    @tool(parse_docstring=True)
    def fs_write_text(path: str, text: str) -> str:
        """Write a UTF-8 text file into the sandboxed workspace.

        Creates parent directories as needed.

        Args:
            path: File path relative to the sandbox root (or virtual absolute like "/out.txt").
            text: Full file contents.
        """

        fs.write_text(path, text)
        return f"Wrote {path}"

    @tool(parse_docstring=True)
    def fs_patch(patch: str) -> str:
        """Apply a single-file patch to an existing UTF-8 text file.

        Args:
            patch: Unified patch text for a single existing file.
        """

        return fs.apply_patch(patch)

    @tool(parse_docstring=True)
    def fs_edit_with_ed(path: str, script: str) -> str:
        """Edit a file by running an `ed` script against it.

        This is well-suited for LLM-generated, line-oriented patches (including multi-line inserts).

        Args:
            path: File path relative to the sandbox root (or virtual absolute like "/foo.txt").
            script: An ed program (commands + replacement lines). Prefer ending with `wq`.
        """

        fp = fs.require_file(path)

        ed_path = shutil.which("ed")
        if not ed_path:
            raise RuntimeError("`ed` not found on PATH")

        proc = subprocess.run(  # noqa: S603
            [ed_path, "-s", str(fp)],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"ed failed (code={proc.returncode}): {err}")

        return f"Edited {path}"

    return [
        fs_read_text,
        fs_write_text,
        fs_patch,
        fs_edit_with_ed,
    ]
