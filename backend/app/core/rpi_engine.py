from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BaseClause:
    index: int
    dim: int
    attribute: str
    bin_index: int
    lower: float
    upper: float
    mask: np.ndarray


@dataclass(frozen=True)
class PredicateState:
    clause_indices: tuple[int, ...]
    mask: np.ndarray
    f1: float


def _classification_metrics(predicted: np.ndarray, label: np.ndarray) -> dict[str, float]:
    predicted = predicted.astype(bool, copy=False)
    label = label.astype(bool, copy=False)
    tp = int(np.logical_and(predicted, label).sum())
    fp = int(np.logical_and(predicted, ~label).sum())
    fn = int(np.logical_and(~predicted, label).sum())
    correct = int((predicted == label).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": correct / max(1, label.size),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _build_base_clauses(
    x: np.ndarray,
    attribute_names: list[str],
    n_bins: int,
) -> tuple[list[BaseClause], dict[str, int]]:
    clauses: list[BaseClause] = []
    bins_per_attribute: dict[str, int] = {}

    for dim, attribute in enumerate(attribute_names):
        values = x[:, dim]
        quantiles = np.quantile(values, np.linspace(0, 1, n_bins + 1))
        edges = [float(values.min())]
        for quantile in quantiles[1:-1]:
            lower_values = values[values < quantile]
            upper_values = values[values >= quantile]
            if lower_values.size == 0 or upper_values.size == 0:
                continue
            lower_observation = float(lower_values.max())
            upper_observation = float(upper_values.min())
            boundary = lower_observation + (upper_observation - lower_observation) / 2
            if boundary > edges[-1]:
                edges.append(boundary)
        maximum = float(values.max())
        if maximum > edges[-1]:
            edges.append(maximum)
        edges = np.asarray(edges, dtype=float)
        if edges.size < 2:
            bins_per_attribute[attribute] = 0
            continue

        bins_per_attribute[attribute] = int(edges.size - 1)
        for bin_index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            # Interior edges are midpoints between observations, so closed intervals do not
            # assign a row to two adjacent bins.
            mask = (values >= lower) & (values <= upper)
            clauses.append(
                BaseClause(
                    index=len(clauses),
                    dim=dim,
                    attribute=attribute,
                    bin_index=bin_index,
                    lower=float(lower),
                    upper=float(upper),
                    mask=mask,
                )
            )

    return clauses, bins_per_attribute


def _state_signature(clause_indices: tuple[int, ...]) -> str:
    return "&".join(str(index) for index in clause_indices)


def _rank_key(state: PredicateState) -> tuple[float, int, int, tuple[int, ...]]:
    return (-state.f1, len(state.clause_indices), -int(state.mask.sum()), state.clause_indices)


def _search_one_brush(
    base_clauses: list[BaseClause],
    selected: np.ndarray,
    *,
    max_depth: int,
    max_solutions: int | None,
    max_states: int | None,
    min_support: int,
    min_f1_improvement: float,
    global_min: np.ndarray,
    global_max: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frontier: dict[str, PredicateState] = {}
    evaluated_states = 0
    truncated = False

    for clause in base_clauses:
        support = int(clause.mask.sum())
        if support < min_support:
            continue
        state = PredicateState(
            clause_indices=(clause.index,),
            mask=clause.mask,
            f1=_classification_metrics(clause.mask, selected)["f1"],
        )
        frontier[_state_signature(state.clause_indices)] = state
        evaluated_states += 1
        if max_states is not None and evaluated_states >= max_states:
            truncated = len(frontier) < len(base_clauses)
            break

    endpoints: dict[str, PredicateState] = {}
    deepest_level = 1 if frontier else 0

    while frontier:
        current_depth = len(next(iter(frontier.values())).clause_indices)
        deepest_level = max(deepest_level, current_depth)
        if current_depth >= max_depth:
            endpoints.update(frontier)
            break

        next_frontier: dict[str, PredicateState] = {}
        budget_exhausted = False
        for state in sorted(frontier.values(), key=_rank_key):
            used_dims = {base_clauses[index].dim for index in state.clause_indices}
            improved = False
            fully_expanded = True

            for clause in base_clauses:
                if clause.dim in used_dims:
                    continue
                if max_states is not None and evaluated_states >= max_states:
                    truncated = True
                    budget_exhausted = True
                    fully_expanded = False
                    break

                child_indices = tuple(sorted((*state.clause_indices, clause.index)))
                signature = _state_signature(child_indices)
                existing_child = next_frontier.get(signature)
                if existing_child is not None:
                    if existing_child.f1 > state.f1 + min_f1_improvement:
                        improved = True
                    continue

                child_mask = state.mask & clause.mask
                support = int(child_mask.sum())
                evaluated_states += 1
                if support < min_support:
                    continue

                child_f1 = _classification_metrics(child_mask, selected)["f1"]
                if child_f1 > state.f1 + min_f1_improvement:
                    next_frontier[signature] = PredicateState(
                        clause_indices=child_indices,
                        mask=child_mask,
                        f1=child_f1,
                    )
                    improved = True

            if not improved or not fully_expanded:
                endpoints[_state_signature(state.clause_indices)] = state
            if budget_exhausted:
                break

        if budget_exhausted:
            endpoints.update(next_frontier)
            break
        frontier = next_frontier

    ranked_states = sorted(
        (state for state in endpoints.values() if state.f1 > 0),
        key=_rank_key,
    )
    if not ranked_states:
        all_rows = np.ones(selected.size, dtype=bool)
        ranked_states = [PredicateState((), all_rows, _classification_metrics(all_rows, selected)["f1"])]

    total_endpoints = len(ranked_states)
    if max_solutions is not None:
        ranked_states = ranked_states[:max_solutions]

    candidates: list[dict[str, Any]] = []
    for rank, state in enumerate(ranked_states, start=1):
        metrics = _classification_metrics(state.mask, selected)
        clauses: list[dict[str, Any]] = []
        for clause_index in state.clause_indices:
            clause = base_clauses[clause_index]
            other_indices = tuple(index for index in state.clause_indices if index != clause_index)
            if other_indices:
                membership_without = np.logical_and.reduce(
                    [base_clauses[index].mask for index in other_indices]
                )
            else:
                membership_without = np.ones(selected.size, dtype=bool)
            metrics_without = _classification_metrics(membership_without, selected)
            span = float(global_max[clause.dim] - global_min[clause.dim])
            range_reduction = (
                max(0.0, min(1.0, 1 - (clause.upper - clause.lower) / span))
                if span > 0
                else 0.0
            )
            clauses.append(
                {
                    "dim": clause.dim,
                    "attribute": clause.attribute,
                    "interval": [clause.lower, clause.upper],
                    "importance": max(0.0, metrics["f1"] - metrics_without["f1"]),
                    "f1_without": metrics_without["f1"],
                    "range_reduction": range_reduction,
                    "bin": clause.bin_index,
                }
            )

        clauses.sort(
            key=lambda item: (
                -item["importance"],
                -item["range_reduction"],
                item["dim"],
            )
        )
        for clause_rank, clause in enumerate(clauses, start=1):
            clause["rank"] = clause_rank

        true_positive_count = int(np.logical_and(state.mask, selected).sum())
        candidates.append(
            {
                "rank": rank,
                "predicate": clauses,
                "quality": metrics,
                "selected_count": int(selected.sum()),
                "predicted_count": int(state.mask.sum()),
                "true_positive_count": true_positive_count,
                "signature": _state_signature(state.clause_indices) or "TRUE",
            }
        )

    diagnostics = {
        "evaluated_state_count": evaluated_states,
        "endpoint_count": total_endpoints,
        "returned_solution_count": len(candidates),
        "solutions_limited": len(candidates) < total_endpoints,
        "deepest_level": deepest_level,
        "depth_limited": (
            deepest_level >= max_depth
            and max_depth < len({clause.dim for clause in base_clauses})
        ),
        "truncated": truncated,
    }
    return candidates, diagnostics


def compute_recursive_predicates(
    x: np.ndarray,
    selected: np.ndarray,
    attribute_names: list[str] | None = None,
    *,
    n_bins: int = 8,
    max_depth: int | None = None,
    max_solutions: int | None = 20,
    max_states: int | None = 20_000,
    min_support: int = 1,
    min_f1_improvement: float = 1e-9,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, float]], list[list[dict[str, Any]]], dict[str, Any]]:
    """Reconstruct PIXAL RPI with DimBridge's F1 scoring adaptation."""

    attribute_names = attribute_names or [f"feature_{index}" for index in range(x.shape[1])]
    base_clauses, bins_per_attribute = _build_base_clauses(x, attribute_names, n_bins)
    effective_max_depth = min(max_depth or x.shape[1], x.shape[1])
    global_min = x.min(axis=0)
    global_max = x.max(axis=0)

    candidate_solutions: list[list[dict[str, Any]]] = []
    brush_diagnostics: list[dict[str, Any]] = []
    predicates: list[list[dict[str, Any]]] = []
    qualities: list[dict[str, float]] = []

    for brush_index, brush_mask in enumerate(selected):
        candidates, diagnostics = _search_one_brush(
            base_clauses,
            brush_mask,
            max_depth=effective_max_depth,
            max_solutions=max_solutions,
            max_states=max_states,
            min_support=min_support,
            min_f1_improvement=min_f1_improvement,
            global_min=global_min,
            global_max=global_max,
        )
        primary = candidates[0]
        predicates.append(primary["predicate"])
        qualities.append({"brush": float(brush_index), **primary["quality"]})
        candidate_solutions.append(candidates)
        brush_diagnostics.append({"brush": brush_index, **diagnostics})

    diagnostics = {
        "paper_algorithm": "PIXAL recursive predicate induction with DimBridge F1 scoring",
        "numeric_binning": "quantile",
        "requested_bins": n_bins,
        "bins_per_attribute": bins_per_attribute,
        "base_predicate_count": len(base_clauses),
        "max_depth": effective_max_depth,
        "max_solutions": max_solutions,
        "max_states": max_states,
        "min_support": min_support,
        "min_f1_improvement": min_f1_improvement,
        "brushes": brush_diagnostics,
        "official_source_available": False,
    }
    return predicates, qualities, candidate_solutions, diagnostics
