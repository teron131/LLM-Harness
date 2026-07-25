"""Upgrade dependency minimum versions in pyproject.toml using uv.

This script follows the uv workflow of removing and re-adding dependencies so that uv
recomputes lower bounds (e.g., `pkg>=<latest>`) and refreshes the lockfile.

Run with the project interpreter, e.g.:

    uv run python update_deps.py
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
from time import sleep


def _requirement_to_name(req: str) -> str:
    """Best-effort extraction of the package name (including extras) from a PEP 508 requirement."""

    base = req.split(";", 1)[0].strip()
    if not base:
        return ""

    if " @ " in base:
        return base.split(" @ ", 1)[0].strip()

    # Trim at the first version operator (==, >=, ~=, etc.). Keep extras.
    m = re.search(r"(===|==|!=|<=|>=|~=|<|>)", base)
    return (base[: m.start()] if m else base).strip()


def _run(cmd: list[str]) -> None:
    """Helper for run."""
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)  # noqa: S603


def _run_with_retry(
    cmd: list[str], *, attempts: int = 3, delay_seconds: float = 2.0
) -> None:
    """Run a command with a small retry loop for transient failures."""
    for attempt in range(1, attempts + 1):
        try:
            _run(cmd)
            return
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise

            print(
                f"uv sync failed on attempt {attempt}/{attempts}; retrying in {delay_seconds:.1f}s...",
                file=sys.stderr,
            )
            sleep(delay_seconds)
            delay_seconds *= 2


def _get_list(dct: dict, *path: str) -> list[str]:
    """Return list."""
    cur: object = dct
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return []
        cur = cur[key]
    return cur if isinstance(cur, list) else []


def main() -> int:
    """Run the Dependency update the project command-line entrypoint."""
    try:
        import tomllib
    except ModuleNotFoundError:
        print(
            "This script requires Python 3.11+ (tomllib). Run via: uv run python update_deps.py",
            file=sys.stderr,
        )
        return 2

    repo_root = Path(__file__).resolve().parent
    pyproject_path = repo_root / "pyproject.toml"
    if not pyproject_path.exists():
        print(f"pyproject.toml not found at {pyproject_path}", file=sys.stderr)
        return 2

    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    project_deps = _get_list(data, "project", "dependencies")
    dep_groups: dict[str, list[str]] = (
        data.get("dependency-groups", {})
        if isinstance(data.get("dependency-groups"), dict)
        else {}
    )
    optional_deps: dict[str, list[str]] = (
        data.get("project", {}).get("optional-dependencies", {})
        if isinstance(data.get("project", {}).get("optional-dependencies"), dict)
        else {}
    )

    # 1) Project dependencies
    project_names = [n for n in (_requirement_to_name(r) for r in project_deps) if n]
    if project_names:
        _run(["uv", "remove", "--no-sync", *project_names])
        _run(["uv", "add", "--no-sync", "--bounds", "lower", *project_names])

    # 2) Dependency groups (if present)
    for group, reqs in dep_groups.items():
        if not isinstance(reqs, list):
            continue
        names = [n for n in (_requirement_to_name(r) for r in reqs) if n]
        if not names:
            continue
        _run(["uv", "remove", "--group", group, "--no-sync", *names])
        _run(["uv", "add", "--group", group, "--no-sync", "--bounds", "lower", *names])

    # 3) Optional dependency extras (if present)
    for extra, reqs in optional_deps.items():
        if not isinstance(reqs, list):
            continue
        names = [n for n in (_requirement_to_name(r) for r in reqs) if n]
        if not names:
            continue
        _run(["uv", "remove", "--optional", extra, "--no-sync", *names])
        _run(
            ["uv", "add", "--optional", extra, "--no-sync", "--bounds", "lower", *names]
        )

    # Ensure lockfile and environment are up to date.
    _run(["uv", "lock"])
    if os.environ.get("UV_NO_SYNC") not in {"1", "true", "TRUE"}:
        _run_with_retry(["uv", "sync"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
