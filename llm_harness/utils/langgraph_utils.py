"""Helpers for rendering artifact files from LangGraph graphs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

from langchain_core.runnables.graph import Edge, Graph
from langchain_core.runnables.graph_mermaid import draw_mermaid, draw_mermaid_png
from langgraph.graph.state import CompiledStateGraph

DEFAULT_GRAPH_DIR = Path("artifacts/graphs")
MermaidDirection = Literal["TD", "LR"]
_SENTINEL_NODES = {"__start__", "__end__"}
EdgeLabels = dict[tuple[str, str], str]


def _normalize_branch_label(label: str) -> str:
    """Convert LangGraph branch labels into readable Mermaid edge labels."""
    if label == "__end__":
        return "end"
    return label.replace("_", " ")


def _edge_labels_from_graph(graph: CompiledStateGraph) -> EdgeLabels:
    """Build edge labels from LangGraph branch specs when available."""
    branches = graph.builder.branches
    if not branches:
        return {}

    edge_labels: EdgeLabels = {}
    for source, branch_map in branches.items():
        for branch_spec in branch_map.values():
            ends = cast(Mapping[str, str] | None, branch_spec.ends) or {}
            for label, target in ends.items():
                edge_labels[(source, target)] = _normalize_branch_label(label)
    return edge_labels


def build_langgraph_mermaid(
    graph: CompiledStateGraph,
    *,
    direction: MermaidDirection = "LR",
    show_edge_labels: bool = False,
) -> str:
    """Build a Mermaid flowchart from a compiled LangGraph graph."""
    graph_spec: Graph = graph.get_graph()
    edge_labels = _edge_labels_from_graph(graph) if show_edge_labels else {}
    visible_nodes = {
        node_id: node
        for node_id, node in graph_spec.nodes.items()
        if node_id not in _SENTINEL_NODES
    }
    labeled_edges = [
        Edge(
            source=edge.source,
            target=edge.target,
            data=edge_labels.get((edge.source, edge.target)),
            conditional=False,
        )
        for edge in graph_spec.edges
        if edge.source not in _SENTINEL_NODES and edge.target not in _SENTINEL_NODES
    ]
    mermaid_syntax = draw_mermaid(
        visible_nodes,
        labeled_edges,
        wrap_label_n_words=3,
    )
    return mermaid_syntax.replace("graph TD;", f"graph {direction};", 1)


def write_langgraph_artifacts(
    graph: CompiledStateGraph,
    *,
    filename_stem: str = "langgraph",
    output_dir: str | Path | None = None,
    show_edge_labels: bool = False,
) -> dict[str, str]:
    """Write Mermaid and PNG artifacts for any compiled LangGraph graph."""
    resolved_output_dir = (
        (Path(output_dir) if output_dir is not None else DEFAULT_GRAPH_DIR)
        .expanduser()
        .resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    mermaid_path = resolved_output_dir / f"{filename_stem}.mmd"
    png_path = resolved_output_dir / f"{filename_stem}.png"
    mermaid_syntax = build_langgraph_mermaid(
        graph,
        show_edge_labels=show_edge_labels,
    )
    mermaid_path.write_text(mermaid_syntax, encoding="utf-8")
    png_path.write_bytes(draw_mermaid_png(mermaid_syntax))
    return {
        "mermaid_path": str(mermaid_path),
        "png_path": str(png_path),
    }
