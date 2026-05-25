# Codemap: `llm_harness/tools/fs`

## Scope

- Filesystem and transcript-line manipulation helpers designed for tool-calling workflows.

## What Module Is For

- This module provides sandboxed filesystem operations and transcript line-tag filtering helpers.

## High-signal locations

- `llm_harness/tools/fs/fs_tools.py -> SandboxFS/make_fs_tools`
- `llm_harness/tools/fs/apply_patch.py -> parse_single_file_patch_with_stats/apply_patch_chunks_to_text`
- `llm_harness/tools/fs/fast_copy.py -> TagRange/tag_content/filter_content/untag_content`

## Repository snapshot

- High-signal files listed below form the stable architecture anchors for this module.
- Keep imports and exports aligned with these anchors when extending behavior.

## Symbol Inventory

- Primary symbols are enumerated in the high-signal locations and syntax relationship sections.
- Preserve existing exported names unless changing a public contract intentionally.

## Key takeaways per location

- `SandboxFS.resolve` is the trust boundary: strips/normalizes user paths, rejects traversal, and enforces root containment.
- `make_fs_tools` creates LangChain tools for read/write/patch/`ed` operations inside the sandbox.
- `apply_patch.py` parses the single-file patch format and applies chunks with tolerant whitespace and punctuation matching.
- `fast_copy.py` provides deterministic line-tag transforms used to remove identified garbage transcript ranges.

## Project-specific conventions and rationale

- Keep sandbox path checks strict and centralized in `SandboxFS`.
- Preserve UTF-8 text semantics for read/write helpers.
- Keep `apply_patch.py` single-file scoped; callers should not assume multi-file patch support.
- `ed`-based editing exists for LLM-friendly line patching; non-zero `ed` exit codes must surface as runtime errors.
- Tag filtering operates on inclusive ranges and keeps untouched order stable.

## Syntax Relationships

- `fs_tools.py -> make_fs_tools -> fs_read_text/fs_write_text/fs_patch/fs_edit_with_ed`
- `fs_tools.py -> SandboxFS.apply_patch -> apply_patch.parse_single_file_patch_with_stats/apply_patch.apply_patch_chunks_to_text`
- `fs_tools.py -> fs_edit_with_ed -> subprocess.run([ed, ...])`
- `fast_copy.py -> filter_content -> TagRange(start_tag/end_tag)`
- `agents/youtube/summarizer.py -> tag_content/filter_content/untag_content`

## General approach (not rigid checklist)

- Modify path behavior only through `SandboxFS.resolve` to avoid split-brain security rules.
- Keep `fast_copy` transforms pure and side-effect free.
- Treat these functions as low-level primitives reused by higher-level agents.

## Validation commands

- `/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh`
- `uv run ruff check .`
