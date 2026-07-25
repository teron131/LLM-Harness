"""YouTube video transcript summarization using LangChain with LangGraph self-checking workflow."""

from collections.abc import Generator
from typing import Any

from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from ...clients.openai import ChatOpenAI
from ...tools.fs.fast_copy import filter_content, tag_content, untag_content
from ...tools.youtube.scraper import get_transcript
from ...utils.youtube_utils import is_youtube_url
from .prompts import (
    get_garbage_filter_prompt,
    get_langchain_summary_prompt,
    get_quality_check_prompt,
)
from .schemas import (
    GarbageIdentification,
    Quality,
    Summary,
)

# ============================================================================
# Configuration
# ============================================================================

SUMMARY_MODEL = "x-ai/grok-4.1-fast"
QUALITY_MODEL = "x-ai/grok-4.1-fast"
FAST_MODEL = "google/gemini-2.5-flash-lite-preview-09-2025"
MIN_QUALITY_SCORE = 80
MAX_ITERATIONS = 2
TARGET_LANGUAGE = "en"  # ISO language code (en, es, fr, de, etc.)


# ============================================================================
# Data Models
# ============================================================================


class SummarizerState(BaseModel):
    """State schema for the summarization graph."""

    transcript: str | None = None
    summary: Summary | None = None
    quality: Quality | None = None
    target_language: str | None = None
    iteration_count: int = 0
    is_complete: bool = False


class SummarizerOutput(BaseModel):
    """Output schema for the summarization graph."""

    summary: Summary
    quality: Quality | None = None
    iteration_count: int
    transcript: str | None = None


# ============================================================================
# Graph Nodes
# ============================================================================


def garbage_filter_node(state: SummarizerState) -> dict[str, Any]:
    """Identify and remove garbage from the transcript."""
    if state.transcript is None:
        raise ValueError("Transcript is required before garbage filtering")

    # Tag the transcript for identification
    tagged_transcript = tag_content(state.transcript)

    llm = ChatOpenAI(
        model=FAST_MODEL,
        temperature=0,
        reasoning_effort="low",
    ).with_structured_output(GarbageIdentification)

    system_prompt = get_garbage_filter_prompt()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=tagged_transcript),
    ]

    raw_garbage = llm.invoke(messages)
    garbage = GarbageIdentification.model_validate(raw_garbage)

    if garbage.garbage_ranges:
        filtered_transcript = filter_content(tagged_transcript, garbage.garbage_ranges)
        # Untag for the next stage (summary)
        cleaned_transcript = untag_content(filtered_transcript)
        print(f"🧹 Removed {len(garbage.garbage_ranges)} garbage sections.")
        return {"transcript": cleaned_transcript}

    return {}


def summary_node(state: SummarizerState) -> dict[str, Any]:
    """Generate summary from transcript."""
    llm = ChatOpenAI(
        model=SUMMARY_MODEL,
        temperature=0,
        reasoning_effort="medium",
    ).with_structured_output(Summary)

    system_prompt = get_langchain_summary_prompt(
        target_language=state.target_language,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Transcript:\n{state.transcript}"),
    ]

    summary = llm.invoke(messages)

    return {
        "summary": summary,
        "iteration_count": state.iteration_count + 1,
    }


def quality_node(state: SummarizerState) -> dict[str, Any]:
    """Assess quality of summary."""
    llm = ChatOpenAI(
        model=QUALITY_MODEL,
        temperature=0,
        reasoning_effort="low",
    ).with_structured_output(Quality)

    system_prompt = get_quality_check_prompt(target_language=state.target_language)

    summary_json = (
        state.summary.model_dump_json() if state.summary else "No summary provided"
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Transcript:\n{state.transcript}\n\nSummary:\n{summary_json}"
        ),
    ]

    raw_quality = llm.invoke(messages)
    quality = Quality.model_validate(raw_quality)

    return {
        "quality": quality,
        "is_complete": quality.is_acceptable,
    }


# ============================================================================
# Graph Construction
# ============================================================================


def should_continue(state: SummarizerState) -> str:
    """Determine next step in workflow."""
    quality_percent = state.quality.percentage_score if state.quality else None
    quality_display = f"{quality_percent}%" if quality_percent is not None else "N/A"

    if state.is_complete:
        print(f"✅ Complete: quality {quality_display}")
        return END

    if (
        state.quality
        and not state.quality.is_acceptable
        and state.iteration_count < MAX_ITERATIONS
    ):
        print(
            f"🔄 Refining: quality {quality_display} < {MIN_QUALITY_SCORE}% (iteration {state.iteration_count + 1})"
        )
        return "summary"

    print(f"⚠️ Stopping: quality {quality_display}, {state.iteration_count} iterations")
    return END


def create_graph() -> CompiledStateGraph:
    """Create the summarization workflow graph with conditional routing."""
    builder = StateGraph(
        SummarizerState,
        output_schema=SummarizerOutput,
    )

    builder.add_node("garbage_filter", garbage_filter_node)
    builder.add_node("summary", summary_node)
    builder.add_node("quality", quality_node)

    builder.add_edge(START, "garbage_filter")
    builder.add_edge("garbage_filter", "summary")
    builder.add_edge("summary", "quality")

    builder.add_conditional_edges(
        "quality",
        should_continue,
        {
            "summary": "summary",
            END: END,
        },
    )

    return builder.compile()


# ============================================================================
# Helper Functions
# ============================================================================


def _extract_transcript(transcript_or_url: str) -> str:
    """Extract transcript from URL or return text directly."""
    if is_youtube_url(transcript_or_url):
        return get_transcript(transcript_or_url)

    if not transcript_or_url or not transcript_or_url.strip():
        raise ValueError("Transcript cannot be empty")

    return transcript_or_url


# ============================================================================
# Public API
# ============================================================================


def summarize_video_react(
    transcript_or_url: str,
    target_language: str | None = None,
) -> Summary:
    """Summarize YouTube video or text transcript with quality self-checking."""
    graph = create_graph()
    transcript = _extract_transcript(transcript_or_url)

    initial_state = SummarizerState(
        transcript=transcript,
        target_language=target_language or TARGET_LANGUAGE,
    )
    result: dict = graph.invoke(initial_state.model_dump())
    output = SummarizerOutput.model_validate(result)

    quality_percent = output.quality.percentage_score if output.quality else None
    quality_display = f"{quality_percent}%" if quality_percent is not None else "N/A"
    print(f"🎯 Final: quality {quality_display}, {output.iteration_count} iterations")
    return output.summary


def stream_summarize_video_react(
    transcript_or_url: str,
    target_language: str | None = None,
) -> Generator[SummarizerState]:
    """Stream the summarization process with progress updates."""
    graph = create_graph()
    transcript = _extract_transcript(transcript_or_url)

    initial_state = SummarizerState(
        transcript=transcript,
        target_language=target_language or TARGET_LANGUAGE,
    )
    for chunk in graph.stream(initial_state.model_dump(), stream_mode="values"):
        yield SummarizerState.model_validate(chunk)
