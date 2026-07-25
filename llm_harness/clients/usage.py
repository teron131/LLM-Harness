"""Token usage and cost tracking for workflow execution.

Uses `contextvars.ContextVar` so usage is isolated per async task and per thread.
Requires `reset()` to be called at the start of each workflow run.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any


@dataclass
class UsageMetadata:
    """Token usage and cost metadata."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0

    def __add__(self, other: UsageMetadata) -> UsageMetadata:
        """Combine two usage summaries field by field."""
        return UsageMetadata(
            total_input_tokens=self.total_input_tokens + other.total_input_tokens,
            total_output_tokens=self.total_output_tokens + other.total_output_tokens,
            total_cost=self.total_cost + other.total_cost,
        )

    def to_dict(self) -> dict[str, int | float]:
        """Serialize the usage summary to a dict."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UsageMetadata:
        """Build a usage summary from a dict payload."""
        return cls(
            total_input_tokens=int(data.get("total_input_tokens", 0)),
            total_output_tokens=int(data.get("total_output_tokens", 0)),
            total_cost=float(data.get("total_cost", 0.0)),
        )

    def format(self) -> str:
        """Format the usage summary for display."""
        summary = (
            f"Input: {self.total_input_tokens:,}, Output: {self.total_output_tokens:,}"
        )
        if self.total_cost > 0:
            summary = f"{summary}, Cost: ${self.total_cost:.4f}"
        return summary


EMPTY_USAGE = UsageMetadata()


_usage: ContextVar[UsageMetadata | None] = ContextVar("llm_harness_usage", default=None)


def _new_usage() -> UsageMetadata:
    """Helper for new usage."""
    return UsageMetadata()


def _get_storage() -> UsageMetadata:
    """Get per-execution storage, initializing if needed."""
    storage = _usage.get()
    if storage is None:
        storage = _new_usage()
        _usage.set(storage)
    return storage


def reset_usage() -> None:
    """Reset the usage tracker for the current execution context."""
    _usage.set(_new_usage())


def track_usage(input_tokens: int, output_tokens: int, cost: float) -> None:
    """Add token usage and cost to the current execution context."""
    storage = _get_storage()
    storage.total_input_tokens += int(input_tokens or 0)
    storage.total_output_tokens += int(output_tokens or 0)
    storage.total_cost += float(cost or 0.0)


def get_usage() -> dict[str, int | float]:
    """Get current usage totals for the current execution context."""
    return _get_storage().to_dict()


def get_accumulated_usage() -> UsageMetadata:
    """Get current usage as UsageMetadata object."""
    return _get_storage()


def create_reset_usage_node():
    """Factory for LangGraph node that resets and returns zeroed usage fields."""

    def reset_usage_node(state) -> dict[str, int | float]:
        """Reset the tracked usage counters in graph state."""
        _ = state
        reset_usage()
        return {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0,
        }

    return reset_usage_node


def create_capture_usage_node():
    """Factory for LangGraph node that captures usage from state."""

    def capture_usage_node(state: UsageMetadata) -> dict[str, int | float]:
        """Capture one model call usage record into graph state."""
        return {
            "total_input_tokens": state.total_input_tokens,
            "total_output_tokens": state.total_output_tokens,
            "total_cost": state.total_cost,
        }

    return capture_usage_node
