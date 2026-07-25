"""SQL tool exports."""

from .tools import make_sql_tools

__all__ = [
    "SQLAgent",
    "answer_sql_question",
    "create_sql_graph",
    "make_llm_planner",
    "make_sql_tools",
]


def __getattr__(name: str):
    """Lazily load SQL tools from the package namespace."""
    if name in {
        "SQLAgent",
        "answer_sql_question",
        "create_sql_graph",
        "make_llm_planner",
    }:
        from . import sql_agent

        return getattr(sql_agent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
