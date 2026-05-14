"""Agent package exports and tool registry."""

from langchain.tools import BaseTool, tool

from ..tools.web import webloader, webloader_tool
from .orchestrator import (
    Orchestrator,
    StageInvocation,
    StageResult,
    coerce_stage_result,
    make_orchestrator_tools,
    create_orchestrator,
    stage_tool_from_callable,
    stage_tool_from_graph,
    stage_tool_from_react_agent,
)
from .youtube import youtubeloader


@tool(parse_docstring=True)
def youtubeloader_tool(url: str) -> str:
    """Load YouTube transcript and metadata from a video URL.

    Args:
        url: YouTube video URL to load.
    """
    return youtubeloader(url)


def get_tools() -> list[BaseTool]:
    """Return the default tool set used by harness agents."""
    return [webloader_tool, youtubeloader_tool]


__all__ = [
    "Orchestrator",
    "StageInvocation",
    "StageResult",
    "coerce_stage_result",
    "create_orchestrator",
    "get_tools",
    "make_orchestrator_tools",
    "stage_tool_from_callable",
    "stage_tool_from_graph",
    "stage_tool_from_react_agent",
    "webloader",
    "webloader_tool",
    "youtubeloader",
    "youtubeloader_tool",
]
