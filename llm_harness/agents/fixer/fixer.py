"""Public entrypoint for the generic file fixer workflow."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from ...tools.fs.fs_tools import SandboxFS
from .graph import create_fixer_graph
from .prompts import DEFAULT_FIXER_SYSTEM_PROMPT
from .state import DEFAULT_FIXER_MAX_ITERATIONS, FixerInput

logger = logging.getLogger(__name__)


def _resolve_model_name(model: str | None) -> str:
    """Resolve the fixer model name from the explicit arg or environment."""
    resolved_model = model or os.getenv("FAST_LLM")
    if not resolved_model:
        raise ValueError(
            "No model configured. Pass `fixer_model=...` or set `FAST_LLM`."
        )
    return resolved_model


def _resolve_target(
    *,
    path: str | Path,
    root_dir: str | Path | None,
) -> tuple[Path, Path, str]:
    """Resolve the target path and its sandbox root."""
    target_path = Path(path).expanduser().resolve()
    if not target_path.is_file():
        raise FileNotFoundError(f"Target file does not exist: {target_path}")

    if root_dir is None:
        root_path = target_path.parent
        relative_target = target_path.name
        return target_path, root_path, relative_target

    root_path = Path(root_dir).expanduser().resolve()
    relative_target = target_path.relative_to(root_path).as_posix()
    return target_path, root_path, relative_target


def fix_file(
    *,
    path: str | Path,
    root_dir: str | Path | None = None,
    fixer_model: str | None = None,
    fixer_context: str | None = None,
    fixer_system_prompt: str | None = None,
    max_iterations: int = DEFAULT_FIXER_MAX_ITERATIONS,
    restore_best_on_failure: bool = True,
) -> dict[str, object]:
    """Run the fixer graph on a target UTF-8 text file."""
    target_path, resolved_root_dir, target_file = _resolve_target(
        path=path, root_dir=root_dir
    )

    fixer_input = FixerInput(
        root_dir=str(resolved_root_dir),
        target_file=target_file,
        fixer_model=_resolve_model_name(fixer_model),
        fixer_context=fixer_context or "",
        fixer_system_prompt=fixer_system_prompt or DEFAULT_FIXER_SYSTEM_PROMPT,
        max_iterations=max_iterations,
        restore_best_on_failure=restore_best_on_failure,
    )

    logger.info(
        "[FIXER] Start file=%s model=%s root=%s",
        target_path,
        fixer_input.fixer_model,
        fixer_input.root_dir,
    )
    result = create_fixer_graph().invoke(fixer_input.model_dump())
    logger.info(
        "[FIXER] Done ok=%s turns=%s fixer_tokens=(%s,%s) cost=%s",
        result.get("fixer_completed", False),
        result.get("iteration", 0),
        result.get("fixer_tokens_in", 0),
        result.get("fixer_tokens_out", 0),
        result.get("fixer_cost", 0.0),
    )
    if result.get("fixer_last_text"):
        logger.warning("[FIXER] Last message: %s", result.get("fixer_last_text"))
    return result


def fix_text(
    *,
    text: str,
    fixer_model: str | None = None,
    fixer_context: str | None = None,
    fixer_system_prompt: str | None = None,
    max_iterations: int = DEFAULT_FIXER_MAX_ITERATIONS,
    restore_best_on_failure: bool = True,
    sandbox_file_name: str = "input.txt",
) -> str:
    """Run the fixer graph on in-memory text via a temporary sandbox file."""
    with TemporaryDirectory(prefix="llm-harness-fixer-") as temp_dir:
        fs = SandboxFS(root_dir=Path(temp_dir))
        sandbox_path = f"/{sandbox_file_name.lstrip('/')}"
        fs.write_text(sandbox_path, text)
        fix_file(
            path=fs.resolve(sandbox_path),
            root_dir=temp_dir,
            fixer_model=fixer_model,
            fixer_context=fixer_context,
            fixer_system_prompt=fixer_system_prompt,
            max_iterations=max_iterations,
            restore_best_on_failure=restore_best_on_failure,
        )
        return fs.read_text(sandbox_path)
