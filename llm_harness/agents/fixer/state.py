"""State definitions for the generic file fixer workflow."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

type FixerReviewKind = Literal["", "no_change", "empty_edit", "patched"]
DEFAULT_FIXER_MAX_ITERATIONS = 3


class FixerInput(BaseModel):
    """Inputs for one fixer run."""

    root_dir: str = Field(description="Directory containing the target file")
    target_file: str = Field(description="File path relative to root_dir")
    fixer_model: str = Field(description="Model used by the fixer workflow")
    fixer_context: str = Field(
        default="", description="Additional domain context for the fixer"
    )
    fixer_system_prompt: str = Field(
        description="System prompt that defines the target behavior"
    )
    max_iterations: int = Field(
        default=DEFAULT_FIXER_MAX_ITERATIONS,
        description="Maximum number of direct-fix passes",
    )
    restore_best_on_failure: bool = Field(
        default=True,
        description="Restore the best reviewed snapshot when the fixer runs out of turns",
    )


class FixerOutput(BaseModel):
    """Public output for one fixer run."""

    fixer_tokens_in: int = 0
    fixer_tokens_out: int = 0
    fixer_cost: float = 0.0
    fixer_notes: str = ""
    iteration: int = 0
    fixer_completed: bool = False
    fixer_last_text: str = ""


class FixerState(FixerInput, FixerOutput):
    """Persisted graph state for the file fixer workflow."""

    best_text: str | None = None
    best_notes: str = ""
    best_score: tuple[int, int] | None = None
    repeated_remaining_reviews: int = 0
    last_remaining_block: str = ""
    review_kind: FixerReviewKind = ""
