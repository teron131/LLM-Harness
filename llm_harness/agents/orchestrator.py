"""Reusable stage-tool orchestration templates for chatbot-first agents."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from ..tools import (
    list_skills,
    load_skills,
    make_fs_tools,
    make_sql_tools,
    make_tabular_tools,
    scrape_youtube_tool,
    search_skills,
    webloader_tool,
)


class StageInvocation(BaseModel):
    """Minimal input exposed by each coarse stage tool."""

    message: str = Field(description="Task or request for this stage.")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional compact context from the parent orchestrator.",
    )


class StageResult(BaseModel):
    """Compact result returned from a stage tool to the parent orchestrator."""

    status: str = Field(
        default="ok",
        description="Stage status such as ok, blocked, error, or complete.",
    )
    content: str = Field(default="", description="Short stage-facing result text.")
    artifact: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured payload worth preserving outside the chat text.",
    )
    trace: list[str] = Field(
        default_factory=list, description="Compact stage trace messages."
    )


StageCallable = Callable[
    [StageInvocation], StageResult | BaseModel | Mapping[str, Any] | str
]
GraphInputBuilder = Callable[[StageInvocation], Mapping[str, Any] | BaseModel]
GraphOutputBuilder = Callable[[Any], StageResult | BaseModel | Mapping[str, Any] | str]


def _tool_map(tools: Sequence[BaseTool]) -> dict[str, BaseTool]:
    """Return tools keyed by their LangChain tool name."""
    return {tool.name: tool for tool in tools}


def _selected_tools(
    tools: Sequence[BaseTool],
    *,
    tool_names: Sequence[str] | None,
) -> list[BaseTool]:
    """Return all tools or one named subset in caller-provided order."""
    if tool_names is None:
        return list(tools)
    tools_by_name = _tool_map(tools)
    missing_names = [name for name in tool_names if name not in tools_by_name]
    if missing_names:
        raise ValueError(f"Unknown orchestrator tool(s): {', '.join(missing_names)}")
    return [tools_by_name[name] for name in tool_names]


def make_orchestrator_tools(
    *,
    root_dir: str | Path | None = None,
    include_fs_writes: bool = False,
    include_sql: bool = True,
    include_tabular: bool = True,
    include_skills: bool = True,
    include_web: bool = True,
    include_youtube: bool = True,
    tool_names: Sequence[str] | None = None,
) -> list[BaseTool]:
    """Compose the general-purpose tools exported by the repo for an orchestrator."""
    resolved_root_dir = (
        Path.cwd() if root_dir is None else Path(root_dir).expanduser().resolve()
    )
    tools: list[BaseTool] = []
    if include_fs_writes:
        tools.extend(make_fs_tools(root_dir=resolved_root_dir))
    else:
        fs_tools = _tool_map(make_fs_tools(root_dir=resolved_root_dir))
        tools.append(fs_tools["fs_read_text"])
    if include_sql:
        tools.extend(make_sql_tools(root_dir=resolved_root_dir))
    if include_tabular:
        tools.extend(make_tabular_tools(root_dir=resolved_root_dir))
    if include_skills:
        tools.extend([list_skills, search_skills, load_skills])
    if include_web:
        tools.append(webloader_tool)
    if include_youtube:
        tools.append(scrape_youtube_tool)
    return _selected_tools(tools, tool_names=tool_names)


def _orchestrator_tools(
    *,
    tools: Sequence[BaseTool] | None = None,
    stages: Sequence[BaseTool] | None = None,
    root_dir: str | Path | None = None,
) -> list[BaseTool]:
    """Return the complete tool set for the parent orchestrator."""
    if tools is None and stages is None:
        tools = make_orchestrator_tools(root_dir=root_dir)
    selected_tools = [*(tools or ()), *(stages or ())]
    if not selected_tools:
        raise ValueError("Orchestrator requires at least one tool or stage.")
    return selected_tools


def _message_content(message: BaseMessage) -> str:
    """Return readable text from a LangChain message."""
    if isinstance(message.content, str):
        return message.content.strip()
    return str(message.content or "").strip()


def coerce_stage_result(
    value: StageResult | BaseModel | Mapping[str, Any] | str,
) -> StageResult:
    """Normalize any supported stage return value into the shared result contract."""
    if isinstance(value, StageResult):
        return value
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, str):
        return StageResult(content=value)

    payload = dict(value)
    known_keys = {"status", "content", "artifact", "trace"}
    artifact = payload.get("artifact")
    if not isinstance(artifact, dict):
        artifact = {key: item for key, item in payload.items() if key not in known_keys}
    trace = payload.get("trace") or []
    if not isinstance(trace, list):
        trace = [str(trace)]
    return StageResult(
        status=str(payload.get("status") or "ok"),
        content=str(payload.get("content") or payload.get("summary") or ""),
        artifact=artifact,
        trace=[str(item) for item in trace],
    )


def stage_tool_from_callable(
    *,
    name: str,
    description: str,
    run: StageCallable,
) -> BaseTool:
    """Wrap a plain Python stage as a coarse LangChain tool."""

    def stage_tool(
        message: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Run the wrapped stage with a compact message and optional context."""
        result = coerce_stage_result(
            run(StageInvocation(message=message, context=context or {}))
        )
        return result.model_dump(mode="json")

    return StructuredTool.from_function(
        func=stage_tool,
        name=name,
        description=description,
        args_schema=StageInvocation,
    )


def stage_tool_from_graph(
    *,
    name: str,
    description: str,
    graph: CompiledStateGraph,
    build_input: GraphInputBuilder | None = None,
    build_output: GraphOutputBuilder | None = None,
) -> BaseTool:
    """Wrap a compiled LangGraph as a coarse LangChain tool."""

    def run_graph(invocation: StageInvocation) -> StageResult:
        graph_input = (
            build_input(invocation)
            if build_input
            else invocation.model_dump(mode="python")
        )
        graph_result = graph.invoke(graph_input)
        result_payload = build_output(graph_result) if build_output else graph_result
        return coerce_stage_result(result_payload)

    return stage_tool_from_callable(
        name=name,
        description=description,
        run=run_graph,
    )


def _react_agent_result(response: Mapping[str, Any]) -> StageResult:
    """Convert a LangChain create_agent response into a compact stage result."""
    messages = list(response.get("messages") or [])
    content = ""
    if messages:
        content = _message_content(messages[-1])

    structured_response = response.get("structured_response")
    artifact: dict[str, Any] = {}
    if isinstance(structured_response, BaseModel):
        artifact["structured_response"] = structured_response.model_dump(mode="json")
    elif structured_response is not None:
        artifact["structured_response"] = structured_response

    tool_trace = []
    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", None) or []:
                tool_trace.append(str(tool_call.get("name") or "tool"))

    return StageResult(
        status="ok",
        content=content,
        artifact=artifact,
        trace=[f"called {tool_name}" for tool_name in tool_trace],
    )


def stage_tool_from_react_agent(
    *,
    name: str,
    description: str,
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    system_prompt: str,
) -> BaseTool:
    """Wrap a LangChain ReAct-style create_agent worker as a coarse stage tool."""
    agent = create_agent(
        model=llm,
        tools=list(tools),
        system_prompt=system_prompt,
        name=name,
    )

    def run_agent(invocation: StageInvocation) -> StageResult:
        response = agent.invoke(
            {"messages": [HumanMessage(content=invocation.message)]}
        )
        return _react_agent_result(response)

    return stage_tool_from_callable(
        name=name,
        description=description,
        run=run_agent,
    )


def create_orchestrator(
    *,
    llm: BaseChatModel,
    system_prompt: str,
    tools: Sequence[BaseTool] | None = None,
    stages: Sequence[BaseTool] | None = None,
    root_dir: str | Path | None = None,
    name: str = "orchestrator",
) -> CompiledStateGraph:
    """Create the parent ReAct orchestrator that selects and calls tools."""
    return create_agent(
        model=llm,
        tools=_orchestrator_tools(
            tools=tools,
            stages=stages,
            root_dir=root_dir,
        ),
        system_prompt=system_prompt,
        name=name,
    )


class Orchestrator:
    """Small runnable wrapper around the parent orchestrator graph."""

    def __init__(
        self,
        *,
        llm: BaseChatModel,
        system_prompt: str,
        tools: Sequence[BaseTool] | None = None,
        stages: Sequence[BaseTool] | None = None,
        root_dir: str | Path | None = None,
        name: str = "orchestrator",
    ):
        self.graph = create_orchestrator(
            llm=llm,
            system_prompt=system_prompt,
            tools=tools,
            stages=stages,
            root_dir=root_dir,
            name=name,
        )

    def invoke(self, message: str) -> dict[str, Any]:
        """Run one chatbot turn through the orchestrator."""
        return self.graph.invoke({"messages": [HumanMessage(content=message)]})

    def answer(self, message: str) -> str:
        """Return the final assistant text for one chatbot turn."""
        result = self.invoke(message)
        messages = list(result.get("messages") or [])
        if not messages:
            return ""
        return _message_content(messages[-1])
