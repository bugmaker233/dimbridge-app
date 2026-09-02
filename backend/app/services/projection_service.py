from __future__ import annotations

from dataclasses import dataclass
import inspect
import os
import tempfile
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from app.services.dataset_service import numeric_columns

ProjectionMethod = Literal["pca", "tsne", "umap"]
SupervisionMode = Literal["unsupervised", "supervised"]
ProjectionStrategy = Literal["pca", "tsne", "umap", "fallback"]


@dataclass
class ProjectionBundle:
    dataset_id: str
    factor_columns: list[str]
    objective_columns: list[str]
    target_column: str | None
    method: ProjectionMethod
    supervision: SupervisionMode
    projection_strategy: ProjectionStrategy
    standardize: bool
    records: list[dict[str, Any]]
    x: list[float]
    y: list[float]
    color_mode: Literal["constant", "continuous", "categorical"]
    color_values: list[Any]
    color_label: str
    color_legend: list[dict[str, Any]]
    row_count: int
    dropped_row_count: int


def _validate_factor_columns(dataframe: pd.DataFrame, factor_columns: list[str]) -> list[str]:
    missing = [column for column in factor_columns if column not in dataframe.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Unknown factor columns requested: {missing}")

    numeric_set = set(numeric_columns(dataframe))
    non_numeric = [column for column in factor_columns if column not in numeric_set]
    if non_numeric:
        raise HTTPException(status_code=400, detail=f"Factor columns must be numeric: {non_numeric}")

    return factor_columns


def _validate_objective_columns(dataframe: pd.DataFrame, objective_columns: list[str]) -> list[str]:
    missing = [column for column in objective_columns if column not in dataframe.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Unknown objective columns requested: {missing}")

    numeric_set = set(numeric_columns(dataframe))
    non_numeric = [column for column in objective_columns if column not in numeric_set]
    if non_numeric:
        raise HTTPException(status_code=400, detail=f"Objective columns must be numeric: {non_numeric}")

    return objective_columns


def _resolve_target_series(
    dataframe: pd.DataFrame,
    target_column: str | None,
    method: ProjectionMethod,
    supervision: SupervisionMode,
) -> tuple[pd.Series | None, Literal["constant", "continuous", "categorical"]]:
    if target_column is None:
        return None, "constant"

    if target_column not in dataframe.columns:
        raise HTTPException(status_code=400, detail=f"Unknown target column requested: {target_column}")

    raw_series = dataframe[target_column]
    numeric_series = pd.to_numeric(raw_series, errors="coerce")
    numeric_valid = np.isfinite(numeric_series.to_numpy(dtype=float))

    if method == "umap" and supervision == "supervised":
        if not numeric_valid.any():
            raise HTTPException(
                status_code=400,
                detail="Supervised UMAP currently requires a numeric target column.",
            )
        if not numeric_valid.all():
            # Missing or invalid target rows will be removed later.
            pass
        return numeric_series, "continuous"

    if numeric_valid.any() and numeric_valid.sum() == raw_series.notna().sum():
        return numeric_series, "continuous"

    categorical_series = raw_series.astype("string")
    return categorical_series, "categorical"


def _valid_target_mask(
    target_series: pd.Series | None,
    color_mode: Literal["constant", "continuous", "categorical"],
) -> np.ndarray:
    if target_series is None:
        return np.ones(0, dtype=bool)

    if color_mode == "continuous":
        numeric_values = pd.to_numeric(target_series, errors="coerce").to_numpy(dtype=float)
        return np.isfinite(numeric_values)

    values = target_series.astype("string")
    return values.notna().to_numpy() & (values != "").to_numpy()


def _fallback_embedding(feature_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_rows, n_features = feature_matrix.shape
    if n_rows == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    if n_rows == 1:
        return np.array([0.0]), np.array([0.0])
    if n_features == 1:
        return feature_matrix[:, 0].astype(float), np.zeros(n_rows, dtype=float)
    return feature_matrix[:, 0].astype(float), feature_matrix[:, 1].astype(float)


def _prepare_features(feature_matrix: np.ndarray, standardize: bool) -> np.ndarray:
    if not standardize:
        return feature_matrix
    return StandardScaler().fit_transform(feature_matrix)


def _compute_pca_embedding(feature_matrix: np.ndarray) -> tuple[np.ndarray, ProjectionStrategy]:
    n_rows, n_features = feature_matrix.shape
    if n_rows < 3 or n_features < 2:
        x, y = _fallback_embedding(feature_matrix)
        return np.column_stack([x, y]), "fallback"

    projector = PCA(n_components=2, svd_solver="auto")
    return projector.fit_transform(feature_matrix), "pca"


def _compute_tsne_embedding(
    feature_matrix: np.ndarray,
    perplexity: float,
    learning_rate: float,
    max_iterations: int,
) -> tuple[np.ndarray, ProjectionStrategy]:
    n_rows, n_features = feature_matrix.shape
    if n_rows < 4:
        x, y = _fallback_embedding(feature_matrix)
        return np.column_stack([x, y]), "fallback"

    effective_perplexity = min(perplexity, max(1.0, (n_rows - 1) / 3))
    tsne_kwargs: dict[str, Any] = dict(
        n_components=2,
        perplexity=effective_perplexity,
        learning_rate=learning_rate,
        init="pca" if n_features >= 2 else "random",
        metric="euclidean",
        random_state=42,
    )
    iteration_param = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    tsne_kwargs[iteration_param] = max_iterations

    projector = TSNE(**tsne_kwargs)
    return projector.fit_transform(feature_matrix), "tsne"


def _compute_umap_embedding(
    feature_matrix: np.ndarray,
    target_values: np.ndarray | None,
    supervision: SupervisionMode,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    target_weight: float,
) -> tuple[np.ndarray, ProjectionStrategy]:
    n_rows = feature_matrix.shape[0]
    if n_rows < 3:
        x, y = _fallback_embedding(feature_matrix)
        return np.column_stack([x, y]), "fallback"

    effective_neighbors = min(max(2, n_neighbors), max(2, n_rows - 1))

    if "NUMBA_CACHE_DIR" not in os.environ:
        os.environ["NUMBA_CACHE_DIR"] = os.path.join(
            tempfile.gettempdir(),
            "dimbridge-numba-cache",
        )

    from umap import UMAP

    projector_kwargs = dict(
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    if supervision == "supervised":
        projector_kwargs["target_metric"] = "l2"
        projector_kwargs["target_weight"] = target_weight

    projector = UMAP(**projector_kwargs)

    if supervision == "supervised":
        embedding = projector.fit_transform(feature_matrix, y=target_values)
    else:
        embedding = projector.fit_transform(feature_matrix)

    return embedding, "umap"


def _compute_embedding(
    feature_matrix: np.ndarray,
    target_values: np.ndarray | None,
    method: ProjectionMethod,
    supervision: SupervisionMode,
    standardize: bool,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    target_weight: float,
    perplexity: float,
    learning_rate: float,
    max_iterations: int,
) -> tuple[np.ndarray, ProjectionStrategy]:
    features = _prepare_features(feature_matrix, standardize)

    if method == "pca":
        return _compute_pca_embedding(features)
    if method == "tsne":
        return _compute_tsne_embedding(features, perplexity, learning_rate, max_iterations)
    return _compute_umap_embedding(
        features,
        target_values,
        supervision,
        n_neighbors,
        min_dist,
        metric,
        target_weight,
    )


def _build_color_payload(
    target_series: pd.Series | None,
    color_mode: Literal["constant", "continuous", "categorical"],
    target_column: str | None,
) -> tuple[list[Any], str, list[dict[str, Any]]]:
    if target_series is None:
        return [], "Selection", []

    if color_mode == "continuous":
        values = pd.to_numeric(target_series, errors="coerce").to_numpy(dtype=float)
        return values.astype(float).tolist(), target_column or "Target", []

    categories = target_series.astype("string").fillna("")
    unique_values = [value for value in categories.drop_duplicates().tolist() if value != ""]
    palette = [
        [31, 119, 180],
        [255, 127, 14],
        [44, 160, 44],
        [214, 39, 40],
        [148, 103, 189],
        [140, 86, 75],
        [227, 119, 194],
        [127, 127, 127],
        [188, 189, 34],
        [23, 190, 207],
    ]
    legend = []
    color_lookup: dict[str, list[int]] = {}
    for index, value in enumerate(unique_values):
        rgb = palette[index % len(palette)]
        color_lookup[value] = rgb
        legend.append({"value": str(value), "rgb": rgb})

    color_values = [color_lookup.get(str(value), [160, 160, 160]) for value in categories.tolist()]
    return color_values, target_column or "Target", legend


def build_projection_bundle(
    *,
    dataset_id: str,
    dataframe: pd.DataFrame,
    factor_columns: list[str],
    objective_columns: list[str],
    target_column: str | None,
    method: ProjectionMethod,
    supervision: SupervisionMode,
    standardize: bool,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    target_weight: float,
    perplexity: float,
    learning_rate: float,
    max_iterations: int,
) -> ProjectionBundle:
    factor_columns = _validate_factor_columns(dataframe, factor_columns)
    objective_columns = _validate_objective_columns(dataframe, objective_columns)
    target_series, inferred_color_mode = _resolve_target_series(
        dataframe,
        target_column,
        method,
        supervision,
    )

    numeric_frame = dataframe[factor_columns].apply(pd.to_numeric, errors="coerce")
    valid_mask = np.isfinite(numeric_frame.to_numpy(dtype=float)).all(axis=1)

    if target_series is not None:
        target_valid_mask = _valid_target_mask(target_series, inferred_color_mode)
        if target_valid_mask.shape[0] != len(dataframe):
            raise HTTPException(status_code=500, detail="Target mask length does not match dataset rows.")
        valid_mask &= target_valid_mask

    filtered_features = numeric_frame.loc[valid_mask].reset_index(drop=True)
    if filtered_features.empty:
        raise HTTPException(
            status_code=400,
            detail="No rows remain after filtering invalid factor/target values for projection.",
        )

    filtered_target = None
    if target_series is not None:
        filtered_target = target_series.loc[valid_mask].reset_index(drop=True)

    target_values = None
    if filtered_target is not None and inferred_color_mode == "continuous":
        target_values = pd.to_numeric(filtered_target, errors="coerce").to_numpy(dtype=float)

    embedding, projection_strategy = _compute_embedding(
        filtered_features.to_numpy(dtype=float),
        target_values,
        method,
        supervision,
        standardize,
        n_neighbors,
        min_dist,
        metric,
        target_weight,
        perplexity,
        learning_rate,
        max_iterations,
    )

    color_values, color_label, color_legend = _build_color_payload(
        filtered_target,
        inferred_color_mode,
        target_column,
    )

    record_columns = list(dict.fromkeys(factor_columns + objective_columns))
    filtered_records = dataframe.loc[valid_mask, record_columns].reset_index(drop=True)

    return ProjectionBundle(
        dataset_id=dataset_id,
        factor_columns=factor_columns,
        objective_columns=objective_columns,
        target_column=target_column,
        method=method,
        supervision=supervision,
        projection_strategy=projection_strategy,
        standardize=standardize,
        records=filtered_records.where(pd.notnull(filtered_records), None).to_dict(orient="records"),
        x=embedding[:, 0].astype(float).tolist(),
        y=embedding[:, 1].astype(float).tolist(),
        color_mode="constant" if target_column is None else inferred_color_mode,
        color_values=color_values,
        color_label=color_label,
        color_legend=color_legend,
        row_count=int(valid_mask.sum()),
        dropped_row_count=int((~valid_mask).sum()),
    )
