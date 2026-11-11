from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class IntervalAxis:
    dim: int
    attribute: str
    values: np.ndarray
    codes: np.ndarray
    unique_values: np.ndarray
    boundaries: np.ndarray

    @property
    def interval_count(self) -> int:
        value_count = int(self.unique_values.size)
        return max(0, value_count * (value_count + 1) // 2 - 1)


@dataclass(frozen=True)
class IntervalClause:
    dim: int
    attribute: str
    start: int
    end: int
    lower: float
    upper: float


@dataclass(frozen=True)
class PredicateState:
    clauses: tuple[IntervalClause, ...]
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


def _build_interval_axes(
    x: np.ndarray,
    attribute_names: list[str],
) -> tuple[list[IntervalAxis], dict[str, int], dict[str, int]]:
    """Build observed-value midpoint boundaries from the official interval-tree notebook."""

    axes: list[IntervalAxis] = []
    unique_values_per_attribute: dict[str, int] = {}
    intervals_per_attribute: dict[str, int] = {}

    for dim, attribute in enumerate(attribute_names):
        values = np.asarray(x[:, dim], dtype=float)
        unique_values, codes = np.unique(values, return_inverse=True)
        unique_values_per_attribute[attribute] = int(unique_values.size)
        if unique_values.size < 2:
            intervals_per_attribute[attribute] = 0
            continue

        interior = unique_values[:-1] + (unique_values[1:] - unique_values[:-1]) / 2
        lower = np.nextafter(unique_values[0], -np.inf)
        upper = np.nextafter(unique_values[-1], np.inf)
        boundaries = np.concatenate(([lower], interior, [upper])).astype(float, copy=False)
        axis = IntervalAxis(
            dim=dim,
            attribute=attribute,
            values=values,
            codes=codes.astype(np.int32, copy=False),
            unique_values=unique_values,
            boundaries=boundaries,
        )
        axes.append(axis)
        intervals_per_attribute[attribute] = axis.interval_count

    return axes, unique_values_per_attribute, intervals_per_attribute


def _mask_signature(mask: np.ndarray) -> bytes:
    return np.packbits(mask, bitorder="little").tobytes()


def _state_signature(clauses: tuple[IntervalClause, ...]) -> str:
    return "&".join(f"{clause.dim}:{clause.start}:{clause.end}" for clause in clauses)


def _state_rank_key(state: PredicateState) -> tuple[float, int, int, str]:
    return (-state.f1, len(state.clauses), -int(state.mask.sum()), _state_signature(state.clauses))


def _interval_mask(axis: IntervalAxis, start: int, end: int) -> np.ndarray:
    return (axis.codes >= start) & (axis.codes <= end)


def _top_interval_children(
    state: PredicateState,
    axis: IntervalAxis,
    selected: np.ndarray,
    *,
    max_intervals: int | None,
    min_support: int,
    min_f1_improvement: float,
) -> tuple[list[PredicateState], int, int, bool]:
    """Evaluate every observed-value interval and retain the best distinct child masks."""

    value_count = int(axis.unique_values.size)
    parent_support_by_value = np.bincount(axis.codes[state.mask], minlength=value_count)
    true_positive_by_value = np.bincount(
        axis.codes[np.logical_and(state.mask, selected)],
        minlength=value_count,
    )
    support_prefix = np.concatenate(([0], np.cumsum(parent_support_by_value)))
    true_positive_prefix = np.concatenate(([0], np.cumsum(true_positive_by_value)))
    selected_count = int(selected.sum())
    root_state = len(state.clauses) == 0
    scored: list[tuple[float, int, int, int]] = []
    evaluated_interval_count = 0

    for start in range(value_count):
        ends = np.arange(start, value_count, dtype=np.int32)
        if start == 0 and ends.size and ends[-1] == value_count - 1:
            ends = ends[:-1]
        if ends.size == 0:
            continue

        evaluated_interval_count += int(ends.size)
        support = support_prefix[ends + 1] - support_prefix[start]
        true_positive = true_positive_prefix[ends + 1] - true_positive_prefix[start]
        denominator = support + selected_count
        f1 = np.divide(
            2 * true_positive,
            denominator,
            out=np.zeros_like(denominator, dtype=float),
            where=denominator > 0,
        )
        if root_state:
            eligible = (support >= min_support) & (f1 > 0)
        else:
            eligible = (support >= min_support) & (f1 > state.f1 + min_f1_improvement)

        for offset in np.flatnonzero(eligible):
            scored.append(
                (
                    float(f1[offset]),
                    int(support[offset]),
                    start,
                    int(ends[offset]),
                )
            )

    scored.sort(key=lambda item: (-item[0], -item[1], item[3] - item[2], item[2], item[3]))
    children: list[PredicateState] = []
    seen_masks: set[bytes] = set()
    branch_limit_applied = False
    for f1, _, start, end in scored:
        child_mask = np.logical_and(state.mask, _interval_mask(axis, start, end))
        mask_key = _mask_signature(child_mask)
        if mask_key in seen_masks:
            continue
        seen_masks.add(mask_key)
        if max_intervals is not None and len(children) >= max_intervals:
            branch_limit_applied = True
            break
        clause = IntervalClause(
            dim=axis.dim,
            attribute=axis.attribute,
            start=start,
            end=end,
            lower=float(axis.boundaries[start]),
            upper=float(axis.boundaries[end + 1]),
        )
        clauses = tuple(sorted((*state.clauses, clause), key=lambda item: item.dim))
        children.append(PredicateState(clauses=clauses, mask=child_mask, f1=f1))

    return children, evaluated_interval_count, len(scored), branch_limit_applied


def _mask_for_clauses(
    clauses: tuple[IntervalClause, ...],
    axes_by_dim: dict[int, IntervalAxis],
    row_count: int,
) -> np.ndarray:
    mask = np.ones(row_count, dtype=bool)
    for clause in clauses:
        mask &= _interval_mask(axes_by_dim[clause.dim], clause.start, clause.end)
    return mask


def _simplify_state(
    state: PredicateState,
    axes_by_dim: dict[int, IntervalAxis],
    selected: np.ndarray,
) -> PredicateState:
    """Remove clauses whose deletion preserves or improves the final rule F1."""

    clauses = list(state.clauses)
    current_mask = state.mask
    current_f1 = state.f1
    changed = True
    while changed and len(clauses) > 1:
        changed = False
        best_removal: tuple[float, int, np.ndarray] | None = None
        for index in range(len(clauses)):
            remaining = tuple(clauses[:index] + clauses[index + 1 :])
            candidate_mask = _mask_for_clauses(remaining, axes_by_dim, selected.size)
            candidate_f1 = _classification_metrics(candidate_mask, selected)["f1"]
            if candidate_f1 + 1e-12 < current_f1:
                continue
            if best_removal is None or candidate_f1 > best_removal[0] + 1e-12:
                best_removal = (candidate_f1, index, candidate_mask)
        if best_removal is not None:
            current_f1, index, current_mask = best_removal
            clauses.pop(index)
            changed = True

    return PredicateState(
        clauses=tuple(sorted(clauses, key=lambda item: item.dim)),
        mask=current_mask,
        f1=current_f1,
    )


def _search_one_brush(
    axes: list[IntervalAxis],
    selected: np.ndarray,
    *,
    max_depth: int,
    max_solutions: int | None,
    max_states: int | None,
    max_intervals_per_factor: int | None,
    beam_width: int | None,
    min_support: int,
    min_f1_improvement: float,
    global_min: np.ndarray,
    global_max: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    axes_by_dim = {axis.dim: axis for axis in axes}
    all_rows = np.ones(selected.size, dtype=bool)
    root = PredicateState(
        clauses=(),
        mask=all_rows,
        f1=_classification_metrics(all_rows, selected)["f1"],
    )
    frontier = [root]
    candidate_states: dict[str, PredicateState] = {}
    endpoint_signatures: set[str] = set()
    generated_states = 0
    expanded_states = 0
    evaluated_intervals = 0
    eligible_intervals = 0
    deepest_level = 0
    truncated = False
    interval_branch_limited = False
    interval_branch_limited_expansions = 0
    beam_limited = False
    beam_discarded_states = 0

    while frontier:
        current_depth = len(frontier[0].clauses)
        deepest_level = max(deepest_level, current_depth)
        if current_depth >= max_depth:
            for state in frontier:
                if state.clauses:
                    endpoint_signatures.add(_state_signature(state.clauses))
            break

        next_states: dict[str, PredicateState] = {}
        budget_exhausted = False
        for state in sorted(frontier, key=_state_rank_key):
            expanded_states += 1
            used_dims = {clause.dim for clause in state.clauses}
            state_children: list[PredicateState] = []
            for axis in axes:
                if axis.dim in used_dims:
                    continue
                children, interval_count, eligible_count, branch_limited = _top_interval_children(
                    state,
                    axis,
                    selected,
                    max_intervals=max_intervals_per_factor,
                    min_support=min_support,
                    min_f1_improvement=min_f1_improvement,
                )
                evaluated_intervals += interval_count
                eligible_intervals += eligible_count
                if branch_limited:
                    interval_branch_limited = True
                    interval_branch_limited_expansions += 1
                state_children.extend(children)

            if not state_children and state.clauses:
                endpoint_signatures.add(_state_signature(state.clauses))

            for child in sorted(state_children, key=_state_rank_key):
                signature = _state_signature(child.clauses)
                existing = next_states.get(signature)
                if existing is not None and _state_rank_key(existing) <= _state_rank_key(child):
                    continue
                if max_states is not None and generated_states >= max_states:
                    truncated = True
                    budget_exhausted = True
                    break
                next_states[signature] = child
                candidate_states[signature] = child
                generated_states += 1
            if budget_exhausted:
                break

        ranked_next = sorted(next_states.values(), key=_state_rank_key)
        if beam_width is not None and len(ranked_next) > beam_width:
            beam_limited = True
            beam_discarded_states += len(ranked_next) - beam_width
            ranked_next = ranked_next[:beam_width]
        frontier = ranked_next
        if budget_exhausted:
            break

    simplified_by_factor_set: dict[tuple[int, ...], PredicateState] = {}
    for state in candidate_states.values():
        simplified = _simplify_state(state, axes_by_dim, selected)
        if not simplified.clauses or simplified.f1 <= 0:
            continue
        factor_set = tuple(clause.dim for clause in simplified.clauses)
        existing = simplified_by_factor_set.get(factor_set)
        if existing is None or _state_rank_key(simplified) < _state_rank_key(existing):
            simplified_by_factor_set[factor_set] = simplified

    ranked_states = sorted(simplified_by_factor_set.values(), key=_state_rank_key)
    if not ranked_states:
        ranked_states = [root]
    total_candidates = len(ranked_states)
    if max_solutions is not None:
        ranked_states = ranked_states[:max_solutions]

    candidates: list[dict[str, Any]] = []
    for rank, state in enumerate(ranked_states, start=1):
        metrics = _classification_metrics(state.mask, selected)
        clauses: list[dict[str, Any]] = []
        for clause in state.clauses:
            remaining = tuple(item for item in state.clauses if item.dim != clause.dim)
            membership_without = _mask_for_clauses(remaining, axes_by_dim, selected.size)
            metrics_without = _classification_metrics(membership_without, selected)
            span = float(global_max[clause.dim] - global_min[clause.dim])
            display_lower = max(float(global_min[clause.dim]), clause.lower)
            display_upper = min(float(global_max[clause.dim]), clause.upper)
            range_reduction = (
                max(0.0, min(1.0, 1 - (display_upper - display_lower) / span))
                if span > 0
                else 0.0
            )
            clauses.append(
                {
                    "dim": clause.dim,
                    "attribute": clause.attribute,
                    "interval": [display_lower, display_upper],
                    "importance": max(0.0, metrics["f1"] - metrics_without["f1"]),
                    "f1_without": metrics_without["f1"],
                    "range_reduction": range_reduction,
                    "interval_id": f"{clause.dim}:{clause.start}:{clause.end}",
                    "value_index_range": [clause.start, clause.end],
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

        candidates.append(
            {
                "rank": rank,
                "predicate": clauses,
                "quality": metrics,
                "selected_count": int(selected.sum()),
                "predicted_count": int(state.mask.sum()),
                "true_positive_count": int(np.logical_and(state.mask, selected).sum()),
                "signature": _state_signature(state.clauses) or "TRUE",
            }
        )

    depth_limited = deepest_level >= max_depth and max_depth < len(axes)
    diagnostics = {
        "evaluated_interval_count": evaluated_intervals,
        "eligible_interval_count": eligible_intervals,
        "evaluated_state_count": generated_states,
        "expanded_state_count": expanded_states,
        "endpoint_count": len(endpoint_signatures),
        "candidate_factor_set_count": total_candidates,
        "returned_solution_count": len(candidates),
        "solutions_limited": len(candidates) < total_candidates,
        "deepest_level": deepest_level,
        "depth_limited": depth_limited,
        "truncated": truncated,
        "interval_branch_limited": interval_branch_limited,
        "interval_branch_limited_expansion_count": interval_branch_limited_expansions,
        "beam_limited": beam_limited,
        "beam_discarded_state_count": beam_discarded_states,
        "search_complete": not any(
            (truncated, interval_branch_limited, beam_limited, depth_limited)
        ),
    }
    return candidates, diagnostics


def compute_recursive_predicates(
    x: np.ndarray,
    selected: np.ndarray,
    attribute_names: list[str] | None = None,
    *,
    max_depth: int | None = None,
    max_solutions: int | None = 20,
    max_states: int | None = 20_000,
    max_intervals_per_factor: int | None = 5,
    beam_width: int | None = 100,
    min_support: int = 1,
    min_f1_improvement: float = 1e-9,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, float]], list[list[dict[str, Any]]], dict[str, Any]]:
    """Search conjunctions of exact observed-value intervals using DimBridge F1 scoring."""

    attribute_names = attribute_names or [f"feature_{index}" for index in range(x.shape[1])]
    axes, unique_values, intervals_per_attribute = _build_interval_axes(x, attribute_names)
    effective_max_depth = min(max_depth or x.shape[1], x.shape[1])
    global_min = x.min(axis=0)
    global_max = x.max(axis=0)

    candidate_solutions: list[list[dict[str, Any]]] = []
    brush_diagnostics: list[dict[str, Any]] = []
    predicates: list[list[dict[str, Any]]] = []
    qualities: list[dict[str, float]] = []

    for brush_index, brush_mask in enumerate(selected):
        candidates, diagnostics = _search_one_brush(
            axes,
            brush_mask,
            max_depth=effective_max_depth,
            max_solutions=max_solutions,
            max_states=max_states,
            max_intervals_per_factor=max_intervals_per_factor,
            beam_width=beam_width,
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
        "interval_generation": "all contiguous observed-value intervals",
        "interval_boundary": "midpoints between adjacent unique observations",
        "interval_source": "dim-bridge/notebooks/interval-tree.ipynb::all_intervals",
        "unique_values_per_attribute": unique_values,
        "intervals_per_attribute": intervals_per_attribute,
        "base_interval_count": int(sum(intervals_per_attribute.values())),
        "max_depth": effective_max_depth,
        "max_solutions": max_solutions,
        "max_states": max_states,
        "max_intervals_per_factor": max_intervals_per_factor,
        "beam_width": beam_width,
        "min_support": min_support,
        "min_f1_improvement": min_f1_improvement,
        "recursive_acceptance": "strict F1 improvement",
        "brushes": brush_diagnostics,
        "official_interval_generator_available": True,
        "official_rpi_source_available": False,
    }
    return predicates, qualities, candidate_solutions, diagnostics
