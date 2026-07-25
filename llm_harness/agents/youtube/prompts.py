"""Prompt builders for YouTube summarization."""


def _build_context_block(
    title: str | None = None,
    description: str | None = None,
) -> str:
    """Helper to build the contextual information block."""
    metadata_parts = []
    if title:
        metadata_parts.append(f"Video Title: {title}")
    if description:
        metadata_parts.append(f"Video Description: {description}")

    if metadata_parts:
        return "\n# CONTEXTUAL INFORMATION:\n" + "\n".join(metadata_parts) + "\n"
    return ""


def get_gemini_summary_prompt(
    target_language: str = "auto",
    title: str | None = None,
    description: str | None = None,
) -> str:
    """Build the system prompt for video analysis (Gemini)."""
    # Language descriptions mapping
    lang_descriptions = {
        "auto": "Use the same language as the video, or English if the language is unclear",
        "en": "English (US)",
        "zh-TW": "Traditional Chinese (繁體中文)",
    }

    # Determine language instruction
    lang_desc = lang_descriptions.get(target_language, target_language)
    instruction = (
        lang_desc
        if target_language == "auto"
        else f"Write ALL output in {lang_desc}. Do not use English or any other language."
    )

    language_instruction = f"- OUTPUT LANGUAGE (REQUIRED): {instruction}"

    metadata = _build_context_block(title, description)

    prompt_lines = [
        "Create a grounded, chronological summary.",
        metadata,
        language_instruction,
        "",
        "SOURCE: Given the full video. Use BOTH spoken content and visuals (on-screen text/slides/charts/code/UI). Do not invent details that are not clearly supported by what you can see/hear.",
        "",
        "Return JSON only (no extra text) with:",
        "- overview: string",
        "- chapters: array of { title: string, description: string, start_time?: string, end_time?: string }",
        "(start_time/end_time are optional MM:SS; omit if unsure)",
        "",
        "Rules:",
        "- Chapters must be chronological and non-overlapping",
        "- Avoid meta-language (no 'this video...' framing)",
        "- Exclude sponsors/promos/calls to action entirely",
    ]

    return "\n".join(prompt_lines)


def get_langchain_summary_prompt(
    target_language: str | None = None,
    title: str | None = None,
    description: str | None = None,
) -> str:
    """Build the system prompt for transcript summarization (LangChain)."""
    metadata = _build_context_block(title, description)

    prompt_parts = [
        "Create a grounded, chronological summary of the transcript.",
        metadata,
        "Rules:",
        "- Ground every claim in the transcript; do not add unsupported details",
        "- Exclude sponsors/ads/promos/calls to action entirely",
        "- Avoid meta-language (no 'this video...', 'the speaker...', etc.)",
        "- Prefer concrete facts, names, numbers, and steps when present",
        "- Ensure output matches the provided response schema",
        "- Return JSON only with overview + chapters",
    ]

    if target_language:
        prompt_parts.append(f"\nOUTPUT LANGUAGE (REQUIRED): {target_language}")

    return "\n".join(prompt_parts)


def get_garbage_filter_prompt() -> str:
    """Build the system prompt for garbage filtering."""
    return (
        "Identify transcript lines that are NOT part of the core content and should be removed.\n"
        "Focus on: sponsors/ads/promos, discount codes, affiliate links, subscribe/like/call to action blocks, filler intros/outros, housekeeping, and other irrelevant segments.\n"
        "The transcript contains line tags like [L1], [L2], etc.\n"
        "Return ONLY the line ranges to remove (garbage_ranges).\n"
        "If unsure about a segment, prefer excluding it."
    )


def get_quality_check_prompt(target_language: str | None = None) -> str:
    """Build the system prompt for quality assessment."""
    system_prompt = (
        "Evaluate the summary against the transcript.\n"
        "For each aspect in the response schema, return a rating (Fail/Refine/Pass) and a specific, actionable reason.\n"
        "Rules:\n"
        "- Be strict about transcript grounding\n"
        "- Treat any sponsor/promo/call to action content as a failure for no_garbage\n"
        "- Treat meta-language as a failure for meta_language_avoidance"
    )
    if target_language:
        system_prompt += f"\nVerify the output language matches: {target_language}"

    return system_prompt
