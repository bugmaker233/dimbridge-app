from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from app.core.predicate_engine import compute_predicate_sequence
from app.schemas import PredicateRequest, PredicateResponse
from app.services.dataset_service import (
    dataframe_from_records,
    dataset_store,
    numeric_columns,
    renderable_numeric_frame,
)


router = APIRouter(prefix="/api/predicate", tags=["predicate"])


def _resolve_dataframe(payload: PredicateRequest):
    if payload.dataset_id is not None:
        record = dataset_store.get(payload.dataset_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        return record.dataframe.copy()
    return dataframe_from_records(payload.records or [])


def _resolve_columns(dataframe, requested: list[str] | None) -> list[str]:
    columns = requested or numeric_columns(dataframe)
    if not columns:
        raise HTTPException(status_code=400, detail="No numeric columns available for predicate computation.")
    missing = [col for col in columns if col not in dataframe.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Unknown columns requested: {missing}")
    non_numeric = [col for col in columns if col not in numeric_columns(dataframe)]
    if non_numeric:
        raise HTTPException(status_code=400, detail=f"Requested columns are not numeric: {non_numeric}")
    return columns


def _prepare_numeric_frame(dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    numeric_frame = renderable_numeric_frame(dataframe, columns)
    if numeric_frame.empty:
        raise HTTPException(
            status_code=400,
            detail="No renderable numeric rows remain after filtering invalid values.",
        )
    return numeric_frame


def _validate_masks(selected_masks: list[list[bool]], row_count: int) -> np.ndarray:
    masks = np.asarray(selected_masks, dtype=bool)
    if masks.ndim != 2:
        raise HTTPException(status_code=400, detail="selected_masks must be a 2D boolean array.")
    if masks.shape[1] != row_count:
        raise HTTPException(
            status_code=400,
            detail=f"selected_masks width {masks.shape[1]} does not match dataset row count {row_count}.",
        )
    if np.any(masks.sum(axis=1) == 0):
        raise HTTPException(status_code=400, detail="Every selection mask must select at least one row.")
    if np.any(masks.sum(axis=1) == row_count):
        raise HTTPException(status_code=400, detail="A selection mask cannot select all rows.")
    return masks


@router.post("/data-extent", response_model=PredicateResponse)
def predicate_data_extent(payload: PredicateRequest) -> PredicateResponse:
    dataframe = _resolve_dataframe(payload)
    columns = _resolve_columns(dataframe, payload.attribute_names)
    numeric_frame = _prepare_numeric_frame(dataframe, columns)
    masks = _validate_masks(payload.selected_masks, len(numeric_frame))

    predicates: list[list[dict[str, Any]]] = []
    for mask in masks:
        subset = numeric_frame[mask]
        predicate = [
            dict(
                dim=index,
                attribute=column,
                interval=[float(subset[column].min()), float(subset[column].max())],
            )
            for index, column in enumerate(columns)
        ]
        predicates.append(predicate)

    return PredicateResponse(columns=columns, predicates=predicates, qualities=None)


@router.post("/regression", response_model=PredicateResponse)
def predicate_regression(payload: PredicateRequest) -> PredicateResponse:
    dataframe = _resolve_dataframe(payload)
    columns = _resolve_columns(dataframe, payload.attribute_names)
    numeric_frame = _prepare_numeric_frame(dataframe, columns)
    masks = _validate_masks(payload.selected_masks, len(numeric_frame))

    predicates, qualities, _ = compute_predicate_sequence(
        numeric_frame.to_numpy(dtype=float),
        masks,
        attribute_names=columns,
    )

    normalized_qualities = [
        {
            "brush": float(item["brush"]),
            "accuracy": float(item["accuracy"]),
            "precision": float(item["precision"]),
            "recall": float(item["recall"]),
            "f1": float(item["f1"]),
        }
        for item in qualities
    ]

    return PredicateResponse(
        columns=columns,
        predicates=predicates,
        qualities=normalized_qualities,
    )
