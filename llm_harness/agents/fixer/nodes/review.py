"""Fixer node that reviews progress and decides loop continuation."""

from __future__ import annotations

from langchain.messages import HumanMessage, SystemMessage

from ....clients.openai import ChatOpenAI
from ....clients.parser import get_metadata
from ..prompts import (
    CLEAN_TASK_LOG,
    DEFAULT_FIXER_SYSTEM_PROMPT,
    build_fixer_progress_prompt,
    build_review_system_prompt,
)
from ..state import FixerState
from .common import (
    MAX_REPEAT_REMAINING_REVIEWS,
    _add_usage,
    _build_runtime,
    _continue_or_finalize,
    _FixerProgress,
    _FixerRuntime,
    logger,
)
from .task_log import (
    _normalized_remaining_block,
    _stop_reason_for_task_log,
    _task_log_score,
)

STOP_KIND_STATUS = {
    "no_change": ("_no_change", "stalled_no_change"),
    "empty_edit": ("_empty_edit", "stalled_empty_edit"),
}

NON_TERMINAL_REVIEW_LOGS = {
    "no_change": "[FIXER] No-op pass %s; remaining work still logged",
    "empty_edit": "[FIXER] Empty edit pass %s; remaining work still logged",
}


def _strip_code_fences(text: str) -> str:
    """Remove wrapping markdown fences from a model response."""
    stripped = text.strip()
    if not stripped.startswith("```") or not stripped.endswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _run_review_snapshot(
    *,
    state: FixerState,
    progress: _FixerProgress,
    current_text: str,
) -> None:
    """Ask the model for a compact progress checklist of the current file."""
    response = ChatOpenAI(
        model=state.fixer_model,
        temperature=0,
        reasoning_effort="low",
    ).invoke(
        [
            SystemMessage(
                content=build_review_system_prompt(
                    state.fixer_system_prompt or DEFAULT_FIXER_SYSTEM_PROMPT,
                )
            ),
            HumanMessage(
                content=build_fixer_progress_prompt(
                    target_file=state.target_file,
                    current_text=current_text,
                )
            ),
        ]
    )
    tokens_in, tokens_out, cost = get_metadata(response)
    progress.fixer_notes = (
        _strip_code_fences(str(response.content or "")) or CLEAN_TASK_LOG
    )
    _add_usage(
        progress,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost=cost,
    )
    candidate_score = _task_log_score(progress.fixer_notes)
    if progress.best_text is None:
        progress.best_text = current_text
        progress.best_notes = progress.fixer_notes
        progress.best_score = candidate_score
        return
    if candidate_score is None:
        return
    if progress.best_score is None or candidate_score <= progress.best_score:
        progress.best_text = current_text
        progress.best_notes = progress.fixer_notes
        progress.best_score = candidate_score


def _review_and_maybe_stop(
    *,
    state: FixerState,
    progress: _FixerProgress,
    current_text: str,
    turn: int,
    stop_kind: str,
    done_last_text: str,
) -> dict[str, object] | None:
    """Review a non-patched terminal signal and stop if the file is clean enough."""
    _run_review_snapshot(state=state, progress=progress, current_text=current_text)
    current_remaining_block = _normalized_remaining_block(progress.fixer_notes)
    if (
        current_remaining_block
        and current_remaining_block == progress.last_remaining_block
    ):
        progress.repeated_remaining_reviews += 1
    else:
        progress.repeated_remaining_reviews = 0
        progress.last_remaining_block = current_remaining_block

    done_suffix, stalled_reason = STOP_KIND_STATUS[stop_kind]
    if stop_reason := _stop_reason_for_task_log(progress.fixer_notes):
        logger.info(
            "[FIXER] Stop reason=%s%s at pass=%s", stop_reason, done_suffix, turn
        )
        return progress.build_result(
            iteration=turn,
            last_text=done_last_text,
            completed=True,
        )
    if progress.repeated_remaining_reviews >= MAX_REPEAT_REMAINING_REVIEWS:
        logger.info("[FIXER] Stop reason=%s at pass=%s", stalled_reason, turn)
        return progress.build_result(
            iteration=turn,
            last_text="stalled",
            completed=False,
        )
    return None


def _review_patched_text(
    *,
    runtime: _FixerRuntime,
    state: FixerState,
    progress: _FixerProgress,
    iteration: int,
    current_text: str,
) -> dict[str, object]:
    """Review the file after a successful patch application."""
    _run_review_snapshot(state=state, progress=progress, current_text=current_text)
    progress.last_remaining_block = _normalized_remaining_block(progress.fixer_notes)
    progress.repeated_remaining_reviews = 0
    task_log = progress.fixer_notes.replace("\n", " | ").strip()
    logger.info("[FIXER] Task log after pass %s: %s", iteration, task_log)
    if stop_reason := _stop_reason_for_task_log(progress.fixer_notes):
        logger.info(
            "[FIXER] Stop reason=%s_after_patch at pass=%s", stop_reason, iteration
        )
        return (
            progress.state_update()
            | progress.build_result(
                iteration=iteration,
                last_text="done",
                completed=True,
            )
            | {"review_kind": ""}
        )
    logger.info("[FIXER] Applied pass %s", iteration)
    return _continue_or_finalize(
        runtime=runtime,
        progress=progress,
        iteration=iteration,
        restore_best_on_failure=state.restore_best_on_failure,
        max_iterations=state.max_iterations,
    )


def _review_initial_text(
    *,
    state: FixerState,
    progress: _FixerProgress,
    current_text: str,
) -> dict[str, object]:
    """Review the initial file state before the first edit pass."""
    _run_review_snapshot(state=state, progress=progress, current_text=current_text)
    progress.last_remaining_block = _normalized_remaining_block(progress.fixer_notes)
    progress.repeated_remaining_reviews = 0
    task_log = progress.fixer_notes.replace("\n", " | ").strip()
    logger.info("[FIXER] Initial task log: %s", task_log)
    if stop_reason := _stop_reason_for_task_log(progress.fixer_notes):
        logger.info("[FIXER] Stop reason=%s_before_fix", stop_reason)
        return (
            progress.state_update()
            | progress.build_result(
                iteration=state.iteration,
                last_text="done",
                completed=True,
            )
            | {"review_kind": ""}
        )
    return progress.state_update() | {"review_kind": ""}


def review_node(state: FixerState) -> dict[str, object]:
    """Review the current file state and decide whether to continue."""
    runtime = _build_runtime(state)
    progress = _FixerProgress.from_state(state)
    current_text = runtime.fs.read_text(runtime.target_path)

    if state.review_kind == "":
        return _review_initial_text(
            state=state,
            progress=progress,
            current_text=current_text,
        )

    if state.review_kind == "patched":
        return _review_patched_text(
            runtime=runtime,
            state=state,
            progress=progress,
            iteration=state.iteration,
            current_text=current_text,
        )

    stop_result = _review_and_maybe_stop(
        state=state,
        progress=progress,
        current_text=current_text,
        turn=state.iteration,
        stop_kind=state.review_kind,
        done_last_text=state.fixer_last_text,
    )
    if stop_result:
        return progress.state_update() | stop_result | {"review_kind": ""}

    logger.info(NON_TERMINAL_REVIEW_LOGS[state.review_kind], state.iteration)
    return _continue_or_finalize(
        runtime=runtime,
        progress=progress,
        iteration=state.iteration,
        restore_best_on_failure=state.restore_best_on_failure,
        max_iterations=state.max_iterations,
    )
