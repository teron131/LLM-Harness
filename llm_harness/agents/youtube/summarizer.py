"""YouTube video transcript summarization using LangChain ReAct agent with structured output."""

from collections.abc import Callable

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.agents.structured_output import ToolStrategy
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools.tool_node import ToolCallRequest

from ...clients.openai import ChatOpenAI
from ...tools.fs.fast_copy import (
    filter_content,
    tag_content,
    untag_content,
)
from ...tools.youtube import scrape_youtube_tool
from ...utils.youtube_utils import is_youtube_url
from .prompts import get_garbage_filter_prompt, get_langchain_summary_prompt
from .schemas import GarbageIdentification, Summary

load_dotenv()

DEFAULT_MODEL = "google/gemini-3-flash-preview"
FAST_MODEL = "google/gemini-2.5-flash-lite-preview-09-2025"


@wrap_tool_call
def garbage_filter_middleware(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage],
) -> ToolMessage:
    """Middleware to filter garbage from tool results (like transcripts)."""
    result = handler(request)

    # Only filter if it's the scrape_youtube_tool and the call succeeded
    if request.tool_call["name"] == "scrape_youtube_tool" and result.status != "error":
        transcript = result.content
        if isinstance(transcript, str) and transcript.strip():
            # Apply the tagging/filtering mechanism
            tagged_transcript = tag_content(transcript)

            llm = ChatOpenAI(
                model=FAST_MODEL,
                temperature=0,
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
                cleaned_transcript = untag_content(filtered_transcript)
                print(f"🧹 Middleware removed {len(garbage.garbage_ranges)} garbage sections from tool result.")
                # Update the result content
                result.content = cleaned_transcript

    return result


def create_summarizer_agent(
    target_language: str | None = None,
):
    """Create a ReAct agent for summarizing video transcripts with structured output."""
    llm = ChatOpenAI(
        model=DEFAULT_MODEL,
        temperature=0,
        reasoning_effort="medium",
    )

    system_prompt = get_langchain_summary_prompt(
        target_language=target_language,
    )

    agent = create_agent(
        model=llm,
        tools=[scrape_youtube_tool],
        system_prompt=system_prompt,
        middleware=[garbage_filter_middleware],  # Add the garbage filter middleware
        response_format=ToolStrategy(Summary),  # Use ToolStrategy for better error handling
    )

    return agent


def summarize_video(
    transcript_or_url: str,
    target_language: str | None = None,
) -> Summary:
    """Summarize video transcript or YouTube URL.

    Args:
        transcript_or_url: Transcript text or YouTube URL
        target_language: Optional target language code (e.g., "en", "es", "fr")

    Returns:
        Structured summary schema
    """
    agent = create_summarizer_agent(
        target_language=target_language,
    )

    # If it's a YouTube URL, let the agent use the tool to fetch it
    # Otherwise, provide the transcript directly
    prompt = f"Summarize this YouTube video: {transcript_or_url}" if is_youtube_url(transcript_or_url) else f"Transcript:\n{transcript_or_url}"

    response = agent.invoke({"messages": [HumanMessage(content=prompt)]})

    # Extract structured response
    summary = response.get("structured_response")
    if summary is None:
        raise ValueError("Agent did not return structured response")

    return Summary.model_validate(summary)
