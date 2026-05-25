# Codemap: `llm_harness/tools`

## Scope

- Tool-callable integration boundaries: filesystem editing, structured data inspection, standalone SQL querying, web loading, and YouTube transcript retrieval.

## What Module Is For

- This module defines side-effect boundaries for filesystem, structured data inspection, standalone SQL querying, web loading, and YouTube transcript retrieval.

## High-signal locations

- `llm_harness/tools/__init__.py -> package re-exports`
- `llm_harness/tools/tabular/tools.py -> inspect_tabular/profile_tabular/extract_tabular/make_tabular_tools`
- `llm_harness/tools/sql/tools.py -> sql_list/sql_describe/sql_query/make_sql_tools`
- `llm_harness/tools/sql/query.py -> standalone SQLite query helpers`
- `llm_harness/tools/fs/fs_tools.py -> make_fs_tools`
- `llm_harness/tools/fs/apply_patch.py -> patch parser/application`
- `llm_harness/tools/fs/fast_copy.py -> TagRange/tag_content/filter_content/untag_content`
- `llm_harness/tools/web/webloader.py -> webloader/webloader_tool`
- `llm_harness/tools/youtube/scraper.py -> scrape_youtube/get_transcript`

## Repository snapshot

- High-signal files listed below form the stable architecture anchors for this module.
- Keep imports and exports aligned with these anchors when extending behavior.

## Symbol Inventory

- Primary symbols are enumerated in the high-signal locations and syntax relationship sections.
- Preserve existing exported names unless changing a public contract intentionally.

## Key takeaways per location

- `tools/__init__.py` is the consolidated import surface consumed by agents and downstream callers.
- `tabular/tools.py` provides table-aware inspection over CSV/XLSX files as shared block-based tabular normalization.
- `sql/tools.py` exposes standalone SQL navigation and query tools that are not coupled to file ingestion.
- `sql/query.py` contains the reusable SQLite listing/describe/query behavior behind the SQL tools.
- `fs/fs_tools.py` enforces sandbox root + traversal guards for file operations and exposes patch/`ed` editing tools.
- `fs/apply_patch.py` keeps the tool-facing patch format parsing isolated from higher-level agents.
- `fs/fast_copy.py` enables line-tag based filtering workflow used by YouTube summarizers.
- `web/webloader.py` converts URLs to markdown concurrently and sanitizes noisy artifacts.
- `youtube/scraper.py` abstracts transcript provider fallback and normalizes result shape.

## Project-specific conventions and rationale

- Keep side-effecting operations in tools, not in core agent orchestration.
- Keep data interpretation tools domain-specific (`data`) instead of expanding low-level filesystem primitives.
- Keep SQL querying separate from ingestion/extraction modules when it can stand alone cleanly.
- Preserve safety boundaries in filesystem tools (`SandboxFS.resolve` invariants).
- Keep transcript acquisition provider-agnostic via fallback logic (ScrapeCreators -> Supadata).
- Preserve tool-friendly function signatures and return types for LangChain integration.

## Syntax Relationships

- `agents/youtube/summarizer.py -> tools.fs.fast_copy + tools.youtube.scraper`
- `tools/__init__.py -> re-exports tabular/sql/fs/web/youtube symbols`
- `agents/__init__.py -> tools.web.webloader_tool`
- `tools/web/webloader.py -> @tool webloader_tool -> webloader`

## General approach (not rigid checklist)

- Extend tool behavior behind existing exported functions before adding new entrypoints.
- Keep tool modules domain-focused (`fs`, `web`, `youtube`) and avoid cross-domain coupling.
- When adding a tool, export it in `tools/__init__.py` only after interface is stable.

## Validation commands

- `/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh`
- `uv run ruff check .`
