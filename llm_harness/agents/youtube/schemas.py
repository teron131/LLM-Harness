"""Structured output schemas for YouTube summarization."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ...tools.fs.fast_copy import TagRange
from ...utils.text_utils import s2hk


class Chapter(BaseModel):
    """Represents a single chapter in the video summary."""

    title: str = Field(
        description="A concise chapter heading.",
    )
    description: str = Field(
        description="A substantive chapter description grounded in the content. Include key facts (numbers/names/steps) when present. Avoid meta-language like 'the video', 'the author', 'the speaker says'—state the content directly.",
    )
    start_time: str | None = Field(
        None,
        description="Optional chapter start timestamp in the format MM:SS.",
    )
    end_time: str | None = Field(
        None,
        description="Optional chapter end timestamp matching the same format as start_time.",
    )

    @field_validator("title", "description")
    def convert_string_to_hk(cls, value: str) -> str:
        """Convert string fields to Traditional Chinese if needed (helper hook)."""
        # Note: Actual conversion logic usually depends on context, but maintaining
        # the pattern from original files where simple s2hk was used.
        return s2hk(value)


class Summary(BaseModel):
    """Complete analysis of a YouTube video."""

    overview: str = Field(
        description="An end-to-end summary of the whole content (main thesis + arc), written in direct statements without meta-language.",
    )
    chapters: list[Chapter] = Field(
        min_length=1,
        description="Chronological, non-overlapping chapters covering the core content.",
    )

    @field_validator("overview")
    def convert_string_to_hk(cls, value: str) -> str:
        """Convert string fields to Traditional Chinese."""
        return s2hk(value)

    def to_text(self) -> str:
        """Format the summary as a readable string."""
        lines = [
            "=" * 80,
            "SUMMARY:",
            "=" * 80,
            f"\nOverview:\n{self.overview}",
            f"\nChapters ({len(self.chapters)}):",
        ]

        for i, chapter in enumerate(self.chapters, 1):
            lines.append(f"\n  Chapter {i}: {chapter.title}")
            lines.append(f"    Summary: {chapter.description}")
            if chapter.start_time or chapter.end_time:
                time_range = f"{chapter.start_time or '?'} - {chapter.end_time or '?'}"
                lines.append(f"    Time: {time_range}")

        return "\n".join(lines)


class GarbageIdentification(BaseModel):
    """List of identified garbage sections in a transcript."""

    garbage_ranges: list[TagRange] = Field(
        description="List of line ranges identified as promotional or irrelevant content"
    )


class Rate(BaseModel):
    """Quality rating for a single aspect."""

    rate: Literal["Fail", "Refine", "Pass"] = Field(
        description="Score for the quality aspect"
    )
    reason: str = Field(description="Reason for the score")


class Quality(BaseModel):
    """Quality assessment of the summary."""

    completeness: Rate = Field(
        description="Rate for completeness: The entire transcript has been considered"
    )
    structure: Rate = Field(
        description="Rate for structure: The result is in desired structures"
    )
    no_garbage: Rate = Field(
        description="Rate for no_garbage: The promotional and meaningless content such as cliche intros, outros, filler, sponsorships, and other irrelevant segments are effectively removed"
    )
    meta_language_avoidance: Rate = Field(
        description="Rate for meta-language avoidance: No phrases like 'This chapter introduces', 'This section covers', etc."
    )
    useful_keywords: Rate = Field(
        description="Rate for keywords: The keywords are useful for highlighting the summary"
    )
    correct_language: Rate = Field(
        description="Rate for language: Match the original language of the transcript or user requested"
    )

    @property
    def all_aspects(self) -> list[Rate]:
        """Return all quality aspects as a list."""
        return [
            self.completeness,
            self.structure,
            self.no_garbage,
            self.meta_language_avoidance,
            self.useful_keywords,
            self.correct_language,
        ]

    @property
    def percentage_score(self) -> int:
        """Calculate percentage score based on Pass/Refine/Fail ratings."""
        aspects = self.all_aspects
        pass_count = sum(1 for a in aspects if a.rate == "Pass")
        refine_count = sum(1 for a in aspects if a.rate == "Refine")
        # Pass = 100%, Refine = 50%, Fail = 0%
        return int((pass_count * 100 + refine_count * 50) / len(aspects))

    @property
    def is_acceptable(self) -> bool:
        """Check if quality score meets minimum threshold."""
        return self.percentage_score >= 80  # Default threshold, can be overridden
