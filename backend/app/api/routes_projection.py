from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas import ProjectionRequest, ProjectionResponse
from app.services.dataset_service import dataset_store
from app.services.projection_service import build_projection_bundle


router = APIRouter(prefix="/api/projection", tags=["projection"])


def _run_projection(payload: ProjectionRequest) -> ProjectionResponse:
    record = dataset_store.get(payload.dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    bundle = build_projection_bundle(
        dataset_id=payload.dataset_id,
        dataframe=record.dataframe.copy(),
        factor_columns=payload.factor_columns,
        objective_columns=payload.objective_columns,
        target_column=payload.target_column,
        method=payload.method,
        supervision=payload.supervision,
        standardize=payload.standardize,
        n_neighbors=payload.n_neighbors,
        min_dist=payload.min_dist,
        metric=payload.metric,
        target_weight=payload.target_weight,
        perplexity=payload.perplexity,
        learning_rate=payload.learning_rate,
        max_iterations=payload.max_iterations,
    )

    return ProjectionResponse(
        dataset_id=bundle.dataset_id,
        factor_columns=bundle.factor_columns,
        objective_columns=bundle.objective_columns,
        target_column=bundle.target_column,
        method=bundle.method,
        supervision=bundle.supervision,
        projection_strategy=bundle.projection_strategy,
        standardize=bundle.standardize,
        row_count=bundle.row_count,
        dropped_row_count=bundle.dropped_row_count,
        x=bundle.x,
        y=bundle.y,
        records=bundle.records,
        color_mode=bundle.color_mode,
        color_values=bundle.color_values,
        color_label=bundle.color_label,
        color_legend=bundle.color_legend,
    )


@router.post("/run", response_model=ProjectionResponse)
def run_projection(payload: ProjectionRequest) -> ProjectionResponse:
    return _run_projection(payload)


@router.post("/umap", response_model=ProjectionResponse)
def run_umap_projection(payload: ProjectionRequest) -> ProjectionResponse:
    payload.method = "umap"
    return _run_projection(payload)
