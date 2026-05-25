"""Fixer node that generates and applies patch edit passes."""

from __future__ import annotations

import json

from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ....clients.openai import ChatOpenAI
from ....clients.parser import StructuredOutput, get_metadata
from ..prompts import (
    DEFAULT_FIXER_SYSTEM_PROMPT,
    build_fixer_agent_prompt,
    build_fixer_pass_prompt,
)
from ..state import FixerState
from .common import (
    EMPTY_EDIT_SENTINEL,
    _add_usage,
    _append_write_note,
    _build_runtime,
    _continue_or_finalize,
    _FixerProgress,
    _FixerRuntime,
    _FixPassResult,
    _WriteApplyResult,
    logger,
)


class PatchEditResponse(BaseModel):
    """Single-file patch returned by one fixer pass."""

    patch: str | None = Field(
        default=None,
        description="Apply-patch text for the target file, or null when no changes are needed.",
    )


def _summarize_write_error(error: ValueError) -> str:
    """Extract a short single-line summary from an edit error."""
    first_line = str(error).splitlines()[0].strip()
    return first_line or error.__class__.__name__


def _build_edit_llm(state: FixerState):
    """Build the structured fixer model used for edit and repair passes."""
    return ChatOpenAI(
        model=state.fixer_model,
        temperature=0,
        reasoning_effort="low",
    ).with_structured_output(PatchEditResponse, include_raw=True)


def _parse_edit_response(response: object) -> tuple[PatchEditResponse, int, int, float]:
    """Parse a structured fixer response and extract usage metadata."""
    structured_response = StructuredOutput.model_validate(response)
    return (
        PatchEditResponse.model_validate(structured_response.parsed),
        *get_metadata(structured_response.raw),
    )


def _write_patch(
    *,
    runtime: _FixerRuntime,
    current_text: str,
    patch: str | None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost: float = 0.0,
) -> _WriteApplyResult:
    """Apply one patch while preserving shared no-op handling."""
    if patch is None or not patch.strip():
        logger.info("[FIXER] Treating empty edit as no-op for %s", runtime.target_path)
        return _WriteApplyResult(
            after_text=current_text,
            write_error=EMPTY_EDIT_SENTINEL,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
        )

    runtime.fs.apply_patch(patch, target_path=runtime.target_path)
    updated_text = runtime.fs.read_text(runtime.target_path)
    try:
        json.loads(current_text)
    except json.JSONDecodeError:
        pass
    else:
        try:
            json.loads(updated_text)
        except json.JSONDecodeError as exc:
            runtime.fs.write_text(runtime.target_path, current_text)
            raise ValueError(f"write broke JSON validity: {exc}") from exc

    if updated_text == current_text:
        logger.info("[FIXER] Treating empty patch as no-op for %s", runtime.target_path)
        return _WriteApplyResult(
            after_text=current_text,
            write_error=EMPTY_EDIT_SENTINEL,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
        )

    return _WriteApplyResult(
        after_text=updated_text,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost=cost,
    )


def _run_fix_pass(
    *,
    state: FixerState,
    progress: _FixerProgress,
    current_text: str,
    turn: int,
) -> _FixPassResult:
    """Run one fixer model pass against the current file contents."""
    llm = _build_edit_llm(state)
    prompt = build_fixer_pass_prompt(
        target_file=state.target_file,
        current_text=current_text,
        pass_number=turn,
        max_turns=state.max_iterations,
        task_log=progress.fixer_notes,
    )
    logger.info("[FIXER] Pass %s/%s chars=%s", turn, state.max_iterations, len(current_text))

    system_prompt = state.fixer_system_prompt or DEFAULT_FIXER_SYSTEM_PROMPT
    base_prompt = build_fixer_agent_prompt(
        target_file=state.target_file,
        fixer_context=state.fixer_context,
        max_turns=state.max_iterations,
    )
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{base_prompt}\n\n{prompt}"),
        ]
    )
    edit_response, tokens_in, tokens_out, cost = _parse_edit_response(response)
    return _FixPassResult(
        patch=edit_response.patch,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost=cost,
    )


def _handle_write_result(
    *,
    runtime: _FixerRuntime,
    state: FixerState,
    progress: _FixerProgress,
    current_text: str,
    turn: int,
    write_result: _WriteApplyResult,
) -> dict[str, object]:
    """Update fixer state after edit application, rollback, or no-op."""
    _add_usage(
        progress,
        tokens_in=write_result.tokens_in,
        tokens_out=write_result.tokens_out,
        cost=write_result.cost,
    )

    if write_result.write_error == EMPTY_EDIT_SENTINEL and write_result.after_text == current_text:
        return progress.state_update() | {
            "iteration": turn,
            "review_kind": "empty_edit",
            "fixer_last_text": "done",
        }

    if write_result.after_text is None:
        if write_result.write_error is not None:
            progress.fixer_notes = _append_write_note(progress.fixer_notes, write_result.write_error)
        return _continue_or_finalize(
            runtime=runtime,
            progress=progress,
            iteration=turn,
            restore_best_on_failure=state.restore_best_on_failure,
            max_iterations=state.max_iterations,
        )

    if write_result.after_text == current_text:
        logger.info("[FIXER] Edit pass %s made no changes; remaining work still logged", turn)
        return _continue_or_finalize(
            runtime=runtime,
            progress=progress,
            iteration=turn,
            restore_best_on_failure=state.restore_best_on_failure,
            max_iterations=state.max_iterations,
        )

    return progress.state_update() | {
        "iteration": turn,
        "review_kind": "patched",
        "fixer_last_text": state.fixer_last_text,
    }


def fix_node(state: FixerState) -> dict[str, object]:
    """Run one fixer pass and queue the next review state."""
    runtime = _build_runtime(state)
    progress = _FixerProgress.from_state(state)
    turn = state.iteration + 1

    if state.iteration == 0:
        logger.info(
            "[FIXER] Direct loop start file=%s model=%s max_turns=%s",
            state.target_file,
            state.fixer_model,
            state.max_iterations,
        )

    current_text = runtime.fs.read_text(runtime.target_path)
    pass_result = _run_fix_pass(
        state=state,
        progress=progress,
        current_text=current_text,
        turn=turn,
    )
    _add_usage(
        progress,
        tokens_in=pass_result.tokens_in,
        tokens_out=pass_result.tokens_out,
        cost=pass_result.cost,
    )
    if pass_result.patch is None:
        return progress.state_update() | {
            "iteration": turn,
            "review_kind": "no_change",
            "fixer_last_text": "no_change",
        }

    try:
        write_result = _write_patch(
            runtime=runtime,
            current_text=current_text,
            patch=pass_result.patch,
        )
    except ValueError as error:
        logger.warning("[FIXER] Patch edit rejected: %s", _summarize_write_error(error))
        write_result = _WriteApplyResult(after_text=None, write_error=str(error))
    return _handle_write_result(
        runtime=runtime,
        state=state,
        progress=progress,
        current_text=current_text,
        turn=turn,
        write_result=write_result,
    )
