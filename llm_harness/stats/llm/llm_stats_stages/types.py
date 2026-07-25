"""Shared types for the native Python LLM stats stages."""

from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class LlmStatsStageConfig(TypedDict, total=False):
    """Configuration for Typed models for LLM stats selection."""

    matcher: dict[str, Any]
    openrouter: dict[str, Any]
    final: dict[str, Any]
    scoring: dict[str, Any]


class ModelStatsSelectedModel(TypedDict, total=False):
    """Pydantic model for Typed models for LLM stats selection."""

    id: str | None
    name: str | None
    provider: str | None
    logo: str
    attachment: bool | None
    reasoning: bool | None
    release_date: str | None
    modalities: Any
    open_weights: bool | None
    cost: Any
    context_window: Any
    speed: dict[str, Any]
    intelligence: Any
    intelligence_index_cost: Any
    evaluations: Any
    scores: Any
    relative_scores: Any


class ModelStatsSelectedPayload(TypedDict):
    """Payload for Typed models for LLM stats selection."""

    fetched_at_epoch_seconds: int | None
    models: list[ModelStatsSelectedModel]


class ModelStatsSelectedOptions(TypedDict, total=False):
    """Options for Typed models for LLM stats selection."""

    id: str | None


class LlmSourceData(TypedDict):
    """Llm Source Data for Typed models for LLM stats selection."""

    artificial_analysis_payload: dict[str, Any]
    scraped_payload: dict[str, Any]
    models_dev_payload: dict[str, Any]
    scraped_rows: list[dict[str, Any]]
    preferred_models_dev_models: list[dict[str, Any]]
    models_dev_by_id: dict[str, dict[str, Any]]
    api_by_slug: dict[str, dict[str, Any]]
    scraped_by_slug: dict[str, dict[str, Any]]


class EnrichedRows(TypedDict):
    """Enriched Rows for Typed models for LLM stats selection."""

    rows: list[dict[str, Any]]
    openrouter_speed_by_id: dict[str, dict[str, Any]]
    openrouter_pricing_by_id: dict[str, dict[str, Any]]
    speed_output_token_anchors: list[int]


class MatcherConfigModel(BaseModel):
    """Pydantic model for Typed models for LLM stats selection."""

    model_config = ConfigDict(extra="allow")

    variant_tokens: list[str] = Field(default_factory=list)


class OpenRouterConfigModel(BaseModel):
    """Pydantic model for Typed models for LLM stats selection."""

    model_config = ConfigDict(extra="allow")

    speed_concurrency: int = 8


class FinalStageConfigModel(BaseModel):
    """Pydantic model for Typed models for LLM stats selection."""

    model_config = ConfigDict(extra="allow")

    null_field_prune_threshold: float = 0.5
    null_field_prune_recent_lookback_days: int = 90


class ScoringConfigModel(BaseModel):
    """Pydantic model for Typed models for LLM stats selection."""

    model_config = ConfigDict(extra="allow")

    intelligence_benchmark_keys: list[str] = Field(default_factory=list)
    agentic_benchmark_keys: list[str] = Field(default_factory=list)
    default_speed_output_token_anchors: list[int] = Field(
        default_factory=lambda: [200, 500, 1_000, 2_000, 8_000]
    )
    speed_output_token_range_min: int = 200
    speed_output_token_range_max: int = 8_000
    speed_anchor_quantiles: list[float] = Field(
        default_factory=lambda: [0.25, 0.5, 0.75]
    )
    weighted_price_input_ratio: float = 0.75
    weighted_price_output_ratio: float = 0.25


class LlmStatsStageConfigModel(BaseModel):
    """Pydantic model for Typed models for LLM stats selection."""

    model_config = ConfigDict(extra="allow")

    matcher: MatcherConfigModel
    openrouter: OpenRouterConfigModel
    final: FinalStageConfigModel
    scoring: ScoringConfigModel


class ModelStatsSelectedOptionsModel(BaseModel):
    """Pydantic model for Typed models for LLM stats selection."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = None


class ModelStatsSelectedModelModel(BaseModel):
    """Pydantic model for Typed models for LLM stats selection."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    name: str | None = None
    provider: str | None = None
    logo: str = ""
    attachment: bool | None = None
    reasoning: bool | None = None
    release_date: str | None = None
    modalities: Any = None
    open_weights: bool | None = None
    cost: Any = None
    context_window: Any = None
    speed: dict[str, Any] = Field(default_factory=dict)
    intelligence: Any = None
    intelligence_index_cost: Any = None
    evaluations: Any = None
    scores: Any = None
    relative_scores: Any = None


class ModelStatsSelectedPayloadModel(BaseModel):
    """Pydantic model for Typed models for LLM stats selection."""

    model_config = ConfigDict(extra="forbid")

    fetched_at_epoch_seconds: int | None = None
    models: list[ModelStatsSelectedModelModel] = Field(default_factory=list)


class LlmSourceDataModel(BaseModel):
    """Pydantic model for Typed models for LLM stats selection."""

    model_config = ConfigDict(extra="forbid")

    artificial_analysis_payload: dict[str, Any]
    scraped_payload: dict[str, Any]
    models_dev_payload: dict[str, Any]
    scraped_rows: list[dict[str, Any]] = Field(default_factory=list)
    preferred_models_dev_models: list[dict[str, Any]] = Field(default_factory=list)
    models_dev_by_id: dict[str, dict[str, Any]] = Field(default_factory=dict)
    api_by_slug: dict[str, dict[str, Any]] = Field(default_factory=dict)
    scraped_by_slug: dict[str, dict[str, Any]] = Field(default_factory=dict)


class EnrichedRowsModel(BaseModel):
    """Pydantic model for Typed models for LLM stats selection."""

    model_config = ConfigDict(extra="forbid")

    rows: list[dict[str, Any]] = Field(default_factory=list)
    openrouter_speed_by_id: dict[str, dict[str, Any]] = Field(default_factory=dict)
    openrouter_pricing_by_id: dict[str, dict[str, Any]] = Field(default_factory=dict)
    speed_output_token_anchors: list[int] = Field(default_factory=list)
