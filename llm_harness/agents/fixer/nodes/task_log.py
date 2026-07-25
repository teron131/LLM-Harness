"""Structured task-log helpers for fixer review decisions."""

from __future__ import annotations

from ..prompts import (
    ALLOWED_FIXER_ACTIONS,
    CLEAN_TASK_LOG,
    HIGH_PRIORITY_FIXER_ACTIONS,
    LOW_PRIORITY_FIXER_ACTIONS,
)

ALLOWED_FIXER_ACTION_PREFIXES = tuple(f"{action}:" for action in ALLOWED_FIXER_ACTIONS)


def _task_log_sections(task_log: str) -> tuple[list[str], list[str]] | None:
    """Split a task log into DONE and REMAINING bullet lists."""
    done_lines: list[str] = []
    remaining_lines: list[str] = []
    section: str | None = None
    found_section = False
    for raw_line in task_log.splitlines():
        line = raw_line.strip()
        if line == "DONE:":
            section = "done"
            found_section = True
            continue
        if line == "REMAINING:":
            section = "remaining"
            found_section = True
            continue
        if not line.startswith("- "):
            continue
        if section == "done":
            done_lines.append(line)
        elif section == "remaining":
            remaining_lines.append(line)
    return (done_lines, remaining_lines) if found_section else None


def _normalized_remaining_block(task_log: str) -> str:
    """Normalize the remaining-work block for repeat-detection comparisons."""
    sections = _task_log_sections(task_log)
    if sections is None:
        return ""
    _, remaining_lines = sections
    return "\n".join(remaining_lines)


def _remaining_action_names(task_log: str) -> list[str]:
    """Extract normalized action names from remaining task-log bullets."""
    action_names: list[str] = []
    sections = _task_log_sections(task_log)
    if sections is None:
        return action_names
    _, remaining_lines = sections
    for line in remaining_lines:
        action = line[2:].strip()
        if action.startswith(ALLOWED_FIXER_ACTION_PREFIXES):
            action_name, _, _ = action.partition(":")
            action_names.append(action_name)
    return action_names


def _stop_reason_for_task_log(task_log: str) -> str | None:
    """Return a stop reason when the task log shows only acceptable remaining work."""
    if task_log.strip() == CLEAN_TASK_LOG:
        return "clean"
    remaining_action_names = _remaining_action_names(task_log)
    if not remaining_action_names:
        return "clean_enough"
    if all(action in LOW_PRIORITY_FIXER_ACTIONS for action in remaining_action_names):
        return "soft_remaining_only"
    return None


def _task_log_score(task_log: str) -> tuple[int, int] | None:
    """Score remaining work to decide whether a snapshot is the best so far."""
    sections = _task_log_sections(task_log)
    if sections is None:
        return None
    done_lines, _ = sections
    remaining_action_names = _remaining_action_names(task_log)
    strong_remaining = sum(
        1 for action in remaining_action_names if action in HIGH_PRIORITY_FIXER_ACTIONS
    )
    soft_remaining = sum(
        1 for action in remaining_action_names if action in LOW_PRIORITY_FIXER_ACTIONS
    )
    return strong_remaining, soft_remaining - len(done_lines)
