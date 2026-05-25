# Codemap: `llm_harness/agents/fixer`

## Scope

- Generic single-file fixer workflow that iteratively edits UTF-8 text using structured apply-patch edits and review-driven loop control.

## What Module Is For

- This module runs a bounded LangGraph cleanup loop for one existing text file using the active system prompt as the success target.

## High-signal locations

- `llm_harness/agents/fixer/fixer.py -> fix_file`
- `llm_harness/agents/fixer/graph.py -> create_fixer_graph`
- `llm_harness/agents/fixer/state.py -> FixerInput/FixerOutput/FixerState`
- `llm_harness/agents/fixer/prompts.py -> DEFAULT_FIXER_SYSTEM_PROMPT/build_*_prompt`
- `llm_harness/agents/fixer/nodes/fix.py -> fix_node`
- `llm_harness/agents/fixer/nodes/review.py -> review_node`
- `llm_harness/agents/fixer/nodes/common.py -> _continue_or_finalize/_restore_best_snapshot`
- `llm_harness/agents/fixer/nodes/task_log.py -> _stop_reason_for_task_log/_task_log_score`

## Repository snapshot

- Primary API: `fix_file`
- Stable control loop: `review -> fix -> review`
- Edit primitive: structured apply-patch edit applied through `SandboxFS`

## Key takeaways per location

- `fixer.py` normalizes the target path and launches the graph with a sandbox root.
- `graph.py` keeps the loop bounded and explicit; stop conditions live in graph state, not only in prompts.
- `nodes/fix.py` asks for a structured apply-patch edit and applies it after format checks.
- `nodes/review.py` converts the current file into a compact DONE/REMAINING checklist and decides whether to continue or stop.
- `nodes/common.py` keeps usage totals and restores the best reviewed in-memory snapshot if the fixer runs out of turns.

## Project-specific conventions and rationale

- Keep edits local and line-anchored; avoid full-file rewrites unless the current file state clearly demands them.
- Preserve format validity for structured files.
- Prefer extending the prompt/task-log heuristics over adding ad-hoc stop flags.
- This module is for local file cleanup only, not broader extraction or validation workflows.
