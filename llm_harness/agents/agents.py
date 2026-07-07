"""Pre-configured agents for common LLM tasks."""

import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from exa_py import Exa
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.messages import HumanMessage
from langchain.tools import BaseTool
from pydantic import BaseModel

from ..clients.multimodal import MediaMessage
from ..clients.openai import ChatOpenAI
from ..tools.web import webloader_tool
from .fixer import fix_file as run_fix_file, fix_text as run_fix_text
from .fixer.prompts import DEFAULT_FIXER_TASK_PROMPT
from .fixer.state import DEFAULT_FIXER_MAX_ITERATIONS
from .youtube.schemas import Summary
from .youtube.summarizer import summarize_video
from .youtube.summarizer_gemini import summarize_video as summarize_video_gemini
from .youtube.summarizer_react import summarize_video_react

load_dotenv()

ReasoningEffort = Literal["minimal", "low", "medium", "high"]


class ExaAgent:
    """Exa API as web search subagent."""

    def __init__(self, system_prompt: str, output_schema: type[BaseModel]):
        """Initialize the agent with its runtime configuration."""
        self.exa = Exa(api_key=os.getenv("EXA_API_KEY"))
        self.system_prompt = system_prompt
        self.output_schema = output_schema

    def invoke(self, query: str) -> BaseModel:
        """Run the configured agent and return its response."""
        result = self.exa.answer(
            query=query,
            system_prompt=self.system_prompt,
            output_schema=self.output_schema.model_json_schema(),
        )
        if not hasattr(result, "answer"):
            raise ValueError("Exa answer response did not include an answer")
        return self.output_schema.model_validate(result.answer)


class BaseHarnessAgent:
    """Base class for harness agents."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0,
        reasoning_effort: ReasoningEffort = "medium",
        system_prompt: str | None = None,
        response_format: type[BaseModel] | None = None,
        tools: list[BaseTool] | None = None,
        **model_kwargs: Any,
    ):
        """Initialize the agent with its runtime configuration."""
        model = model or os.getenv("FAST_LLM")
        if not model:
            raise ValueError("No model configured. Pass `model=...` or set `FAST_LLM`.")
        self.system_prompt = system_prompt
        self.response_format = response_format
        self.model = ChatOpenAI(
            model=model,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            **model_kwargs,
        )
        self.agent = create_agent(
            model=self.model,
            tools=tools or [],
            system_prompt=self.system_prompt,
            response_format=ToolStrategy(self.response_format) if self.response_format else None,
        )

    def _process_response(self, response: dict[str, Any]) -> BaseModel | str:
        """Extract either structured output or the final text message."""
        if self.response_format:
            structured_response = response.get("structured_response")
            if structured_response is None:
                raise ValueError("Agent did not return structured response")
            return self.response_format.model_validate(structured_response)

        messages = response.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("Agent response did not include messages")
        return str(messages[-1].content)


class WebLoaderAgent(BaseHarnessAgent):
    """Agent that loads web content from URLs and processes it using tools."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0,
        reasoning_effort: ReasoningEffort = "medium",
        system_prompt: str | None = None,
        response_format: type[BaseModel] | None = None,
        **model_kwargs: Any,
    ):
        """Initialize web loader agent with tool capabilities."""

        super().__init__(
            model=model,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            system_prompt=system_prompt,
            response_format=response_format,
            tools=[webloader_tool],
            **model_kwargs,
        )

    def invoke(self, user_input: str) -> BaseModel | str:
        """Load and process web content based on user input."""
        response = self.agent.invoke({"messages": [HumanMessage(content=user_input)]})
        return self._process_response(response)


class ImageAnalysisAgent(BaseHarnessAgent):
    """Agent that accepts image inputs and returns structured or plain responses."""

    def invoke(
        self,
        image_paths: str | Path | bytes | list[str | Path | bytes],
        description: str = "",
    ) -> BaseModel | str:
        """Analyze one or more images with an optional description/prompt."""
        response = self.agent.invoke({"messages": [MediaMessage(paths=image_paths, description=description)]})
        return self._process_response(response)


class FixerAgent:
    """High-level fixer for either raw text or a file on disk."""

    def __init__(
        self,
        model: str | None = None,
        system_prompt: str | None = None,
        default_prompt: str = DEFAULT_FIXER_TASK_PROMPT,
        max_iterations: int = DEFAULT_FIXER_MAX_ITERATIONS,
        restore_best_on_failure: bool = True,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.default_prompt = default_prompt
        self.max_iterations = max_iterations
        self.restore_best_on_failure = restore_best_on_failure

    def _resolve_context(self, context: str) -> str:
        """Return the caller context or the default fixer task prompt."""
        return context or self.default_prompt

    def invoke(
        self,
        target: str | Path,
        *,
        context: str = "",
    ) -> str:
        """Fix either an existing file path or raw text content."""
        if isinstance(target, Path):
            return self.fix_file(target, context=context)

        target_path = Path(target).expanduser()
        if target_path.exists() and target_path.is_file():
            return self.fix_file(target_path, context=context)
        return self.fix_text(target, context=context)

    def fix_file(
        self,
        path: str | Path,
        *,
        context: str = "",
    ) -> str:
        """Fix a file in place and return its final text."""
        resolved_path = Path(path).expanduser().resolve()
        run_fix_file(
            path=path,
            fixer_model=self.model,
            fixer_context=self._resolve_context(context),
            fixer_system_prompt=self.system_prompt,
            max_iterations=self.max_iterations,
            restore_best_on_failure=self.restore_best_on_failure,
        )
        return resolved_path.read_text(encoding="utf-8")

    def fix_text(
        self,
        text: str,
        *,
        context: str = "",
    ) -> str:
        """Fix raw UTF-8 text content inside a temporary sandbox."""
        return run_fix_text(
            text=text,
            fixer_model=self.model,
            fixer_context=self._resolve_context(context),
            fixer_system_prompt=self.system_prompt,
            max_iterations=self.max_iterations,
            restore_best_on_failure=self.restore_best_on_failure,
        )


class YouTubeSummarizerReAct:
    """ReAct-based YouTube summarizer using the LangGraph workflow."""

    def __init__(self, target_language: str | None = None):
        """Initialize the agent with its runtime configuration."""
        self.target_language = target_language

    def invoke(self, transcript_or_url: str) -> Summary:
        """Summarize a transcript or YouTube URL."""
        return summarize_video_react(
            transcript_or_url=transcript_or_url,
            target_language=self.target_language,
        )


class YouTubeSummarizer:
    """Lightweight YouTube summarizer."""

    def __init__(self, target_language: str | None = None):
        """Initialize the agent with its runtime configuration."""
        self.target_language = target_language

    def invoke(self, transcript_or_url: str) -> Summary:
        """Summarize a transcript or YouTube URL."""
        return summarize_video(
            transcript_or_url=transcript_or_url,
            target_language=self.target_language,
        )


class YouTubeSummarizerGemini:
    """Gemini multimodal YouTube summarizer."""

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        thinking_level: ReasoningEffort = "medium",
        target_language: str = "auto",
        api_key: str | None = None,
    ):
        """Initialize the agent with its runtime configuration."""
        self.model = model
        self.thinking_level = thinking_level
        self.target_language = target_language
        self.api_key = api_key

    def invoke(self, video_url: str) -> Summary | None:
        """Summarize a YouTube URL using Gemini multimodal input."""
        return summarize_video_gemini(
            video_url=video_url,
            model=self.model,
            thinking_level=self.thinking_level,
            target_language=self.target_language,
            api_key=self.api_key,
        )
