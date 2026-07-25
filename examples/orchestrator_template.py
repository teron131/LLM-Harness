"""Runnable template for a ReAct orchestrator with stage tools."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from llm_harness.agents.orchestrator import (
    Orchestrator,
    StageInvocation,
    StageResult,
    make_orchestrator_tools,
    stage_tool_from_callable,
    stage_tool_from_graph,
    stage_tool_from_react_agent,
)


class ToolBindableFakeChatModel(FakeMessagesListChatModel):
    """Fake chat model that can run through create_agent tool-binding paths."""

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return self so the fake model can drive deterministic tool calls."""
        _ = tools, tool_choice, kwargs
        return self


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_tool(name: str) -> BaseTool:
    """Return one non-destructive repo tool for this runnable template."""
    tools_by_name = {
        tool.name: tool
        for tool in make_orchestrator_tools(
            root_dir=REPO_ROOT,
            include_sql=False,
            include_tabular=False,
            include_skills=False,
            include_web=False,
            include_youtube=False,
        )
    }
    return tools_by_name[name]


def build_plain_stage():
    """Build a plain Python stage tool."""

    def run(invocation: StageInvocation) -> StageResult:
        topic = invocation.context.get("topic", "stage template")
        return StageResult(
            content=f"Plain stage received `{invocation.message}` for {topic}.",
            artifact={"kind": "plain", "message_length": len(invocation.message)},
            trace=["plain stage completed"],
        )

    return stage_tool_from_callable(
        name="plain_stage",
        description="Use for deterministic Python work that does not need an LLM.",
        run=run,
    )


def build_react_stage():
    """Build a stage backed by another ReAct worker agent and repo tool."""
    worker_llm = ToolBindableFakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "fs_read_text",
                        "args": {"path": "pyproject.toml"},
                        "id": "worker_call_1",
                    }
                ],
            ),
            AIMessage(content="The worker used the repo filesystem tool and finished."),
        ]
    )
    return stage_tool_from_react_agent(
        name="react_worker_stage",
        description="Use for a bounded task that benefits from a private ReAct worker.",
        llm=worker_llm,
        tools=[repo_tool("fs_read_text")],
        system_prompt="You are a small worker. Use tools when they directly answer the stage request.",
    )


class DemoGraphInput(BaseModel):
    """Compulsory input fields for the demo LangGraph stage."""

    message: str


class DemoGraphOutput(BaseModel):
    """Public output fields for the demo LangGraph stage."""

    status: str = "pending"
    content: str = ""
    artifact: dict[str, Any] = Field(default_factory=dict)
    trace: list[str] = Field(default_factory=list)


class DemoGraphState(DemoGraphInput, DemoGraphOutput):
    """Internal graph state with extra fields callers do not need to provide."""

    normalized_message: str = ""
    character_count: int = 0


def normalize_node(state: DemoGraphState) -> dict[str, Any]:
    """Normalize the message before the answer node."""
    normalized_message = state.message.strip().lower()
    return {
        "normalized_message": normalized_message,
        "character_count": len(normalized_message),
        "trace": [*state.trace, "normalized message"],
    }


def answer_node(state: DemoGraphState) -> dict[str, Any]:
    """Return a compact graph-stage result."""
    return {
        "status": "complete",
        "content": f"Graph stage normalized {state.character_count} characters.",
        "artifact": {
            "kind": "langgraph",
            "normalized_message": state.normalized_message,
            "character_count": state.character_count,
        },
        "trace": [*state.trace, "graph stage completed"],
    }


def build_graph_stage():
    """Build a stage backed by a deterministic LangGraph."""
    builder = StateGraph(
        DemoGraphState,
        input_schema=DemoGraphInput,
        output_schema=DemoGraphOutput,
    )
    builder.add_node("normalize", normalize_node)
    builder.add_node("answer", answer_node)
    builder.add_edge(START, "normalize")
    builder.add_edge("normalize", "answer")
    builder.add_edge("answer", END)
    graph = builder.compile(name="demo_graph_stage")
    return stage_tool_from_graph(
        name="graph_stage",
        description="Use for deterministic multi-step stateful workflow stages.",
        graph=graph,
    )


def build_orchestrator() -> Orchestrator:
    """Build the parent ReAct orchestrator with ordinary tools and stage tools."""
    orchestrator_llm = ToolBindableFakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "fs_read_text",
                        "args": {"path": "README.md"},
                        "id": "parent_call_1",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "plain_stage",
                        "args": {
                            "message": "prepare deterministic context",
                            "context": {"topic": "stage orchestration"},
                        },
                        "id": "parent_call_2",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "react_worker_stage",
                        "args": {
                            "message": "ask the private worker to inspect package metadata"
                        },
                        "id": "parent_call_3",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "graph_stage",
                        "args": {"message": "Normalize This Message In A Graph"},
                        "id": "parent_call_4",
                    }
                ],
            ),
            AIMessage(
                content="Template run complete: repo tools plus plain, ReAct-worker, and LangGraph stages all returned results."
            ),
        ]
    )
    return Orchestrator(
        llm=orchestrator_llm,
        tools=[
            *make_orchestrator_tools(
                root_dir=REPO_ROOT,
                include_sql=False,
                include_tabular=False,
                include_skills=False,
                include_web=False,
                include_youtube=False,
                tool_names=["fs_read_text"],
            ),
            build_plain_stage(),
            build_react_stage(),
            build_graph_stage(),
        ],
        system_prompt=(
            "You are the major chatbot orchestrator. Use ordinary tools directly when they are enough, "
            "choose coarse stage tools when a stage should own its internals, and finish with a concise result."
        ),
    )


def main() -> None:
    """Run the orchestrator template."""
    orchestrator = build_orchestrator()
    result = orchestrator.invoke("Run the staged template.")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
