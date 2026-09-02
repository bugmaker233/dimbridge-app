from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class DatasetSummary(BaseModel):
    dataset_id: str
    filename: str
    row_count: int
    column_count: int
    columns: list[str]
    numeric_columns: list[str]
    renderable_row_count: int
    preview: list[dict[str, Any]]
    records: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    python_version: str
    torch_available: bool
    torch_cuda_available: bool
    torch_version: str | None = None


class PredicateRequest(BaseModel):
    dataset_id: str | None = None
    records: list[dict[str, Any]] | None = None
    selected_masks: list[list[bool]] = Field(default_factory=list)
    attribute_names: list[str] | None = None

    @model_validator(mode="after")
    def validate_data_source(self) -> "PredicateRequest":
        if self.dataset_id is None and self.records is None:
            raise ValueError("Either dataset_id or records must be provided.")
        if not self.selected_masks:
            raise ValueError("selected_masks must contain at least one selection mask.")
        return self


class PredicateResponse(BaseModel):
    columns: list[str]
    predicates: list[Any]
    qualities: list[dict[str, float]] | None = None


class ProjectionRequest(BaseModel):
    dataset_id: str
    factor_columns: list[str] = Field(default_factory=list)
    objective_columns: list[str] = Field(default_factory=list)
    target_column: str | None = None
    method: Literal["pca", "tsne", "umap"] = "umap"
    supervision: Literal["unsupervised", "supervised"] = "unsupervised"
    standardize: bool = True
    n_neighbors: int = Field(default=20, ge=2)
    min_dist: float = Field(default=0.1, ge=0.0, le=1.0)
    metric: str = "euclidean"
    target_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    perplexity: float = Field(default=30.0, gt=0.0)
    learning_rate: float = Field(default=200.0, gt=0.0)
    max_iterations: int = Field(default=1000, ge=250)

    @model_validator(mode="after")
    def validate_projection_request(self) -> "ProjectionRequest":
        if not self.factor_columns:
            raise ValueError("factor_columns must contain at least one numeric column.")
        if self.method != "umap" and self.supervision == "supervised":
            raise ValueError("supervision is only supported for UMAP.")
        if self.method == "umap" and self.supervision == "supervised" and not self.target_column:
            raise ValueError("target_column is required when UMAP supervision is 'supervised'.")
        return self


class ProjectionLegendItem(BaseModel):
    value: str
    rgb: list[int]


class ProjectionResponse(BaseModel):
    dataset_id: str
    factor_columns: list[str]
    objective_columns: list[str] = Field(default_factory=list)
    target_column: str | None = None
    method: Literal["pca", "tsne", "umap"]
    supervision: Literal["unsupervised", "supervised"]
    projection_strategy: Literal["pca", "tsne", "umap", "fallback"]
    standardize: bool
    row_count: int
    dropped_row_count: int
    x: list[float]
    y: list[float]
    records: list[dict[str, Any]]
    color_mode: Literal["constant", "continuous", "categorical"]
    color_values: list[Any] = Field(default_factory=list)
    color_label: str
    color_legend: list[ProjectionLegendItem] = Field(default_factory=list)
