from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn, optim

from app.core.predicate_engine import classification_metrics


def _components(
    x: torch.Tensor,
    a: torch.Tensor,
    mu: torch.Tensor,
    exponent: int,
) -> torch.Tensor:
    return (a.abs() * (x - mu).abs()).pow(exponent)


def _predict(
    x: torch.Tensor,
    a: torch.Tensor,
    mu: torch.Tensor,
    exponent: int,
) -> torch.Tensor:
    return 1 / (1 + _components(x, a, mu, exponent).sum(dim=1))


def _crisp_membership(x: np.ndarray, clauses: list[dict[str, Any]]) -> np.ndarray:
    membership = np.ones(x.shape[0], dtype=bool)
    for clause in clauses:
        lower, upper = clause["interval"]
        membership &= (x[:, clause["dim"]] >= lower) & (x[:, clause["dim"]] <= upper)
    return membership


def _numpy_metrics(predicted: np.ndarray, label: np.ndarray) -> dict[str, float]:
    predicted_tensor = torch.from_numpy(predicted.astype(bool))
    label_tensor = torch.from_numpy(label.astype(bool))
    return classification_metrics(predicted_tensor, label_tensor)


def _extract_predicates(
    x0: np.ndarray,
    selected: np.ndarray,
    x: torch.Tensor,
    label: torch.Tensor,
    a: torch.Tensor,
    mu: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    attribute_names: list[str],
    exponent: int,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, float]]]:
    n_features = x0.shape[1]
    global_min = x0.min(axis=0)
    global_max = x0.max(axis=0)
    predicates: list[list[dict[str, Any]]] = []
    qualities: list[dict[str, float]] = []

    with torch.no_grad():
        for brush_index, selection in enumerate(selected):
            components = _components(x, a[brush_index], mu[brush_index], exponent)
            total_distance = components.sum(dim=1)
            membership = torch.nan_to_num(
                1 / (1 + total_distance),
                nan=0.5,
                posinf=1.0,
                neginf=0.0,
            )
            proxy_metrics = classification_metrics(membership > 0.5, label[brush_index])

            radius = 1 / a[brush_index].abs().clamp_min(1e-12)
            clauses: list[dict[str, Any]] = []
            for feature_index in range(n_features):
                radius_raw = float((radius[feature_index] * scale[feature_index]).item())
                center_raw = float(
                    (mu[brush_index, feature_index] * scale[feature_index] + mean[feature_index]).item()
                )
                lower = center_raw - radius_raw
                upper = center_raw + radius_raw
                if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                    continue

                span = float(global_max[feature_index] - global_min[feature_index])
                epsilon = max(span * 1e-9, 1e-12)
                covers_extent = (
                    lower <= global_min[feature_index] + epsilon
                    and upper >= global_max[feature_index] - epsilon
                )
                if covers_extent:
                    continue

                display_lower = max(lower, float(global_min[feature_index]))
                display_upper = min(upper, float(global_max[feature_index]))
                if display_lower >= display_upper:
                    continue

                distance_without = (total_distance - components[:, feature_index]).clamp_min(0)
                membership_without = torch.nan_to_num(
                    1 / (1 + distance_without),
                    nan=0.5,
                    posinf=1.0,
                    neginf=0.0,
                )
                metrics_without = classification_metrics(
                    membership_without > 0.5,
                    label[brush_index],
                )
                range_reduction = (
                    max(0.0, min(1.0, 1 - (display_upper - display_lower) / span))
                    if span > 0
                    else 0.0
                )
                clauses.append(
                    {
                        "dim": feature_index,
                        "attribute": attribute_names[feature_index],
                        "interval": [display_lower, display_upper],
                        "importance": max(0.0, proxy_metrics["f1"] - metrics_without["f1"]),
                        "f1_without": metrics_without["f1"],
                        "range_reduction": range_reduction,
                    }
                )

            clauses.sort(
                key=lambda clause: (
                    -clause["importance"],
                    -clause["range_reduction"],
                    clause["dim"],
                )
            )
            for rank, clause in enumerate(clauses, start=1):
                clause["rank"] = rank

            crisp_metrics = _numpy_metrics(_crisp_membership(x0, clauses), selection)
            qualities.append(
                {
                    "brush": float(brush_index),
                    **proxy_metrics,
                    "predicate_accuracy": crisp_metrics["accuracy"],
                    "predicate_precision": crisp_metrics["precision"],
                    "predicate_recall": crisp_metrics["recall"],
                    "predicate_f1": crisp_metrics["f1"],
                }
            )
            predicates.append(clauses)

    return predicates, qualities


def compute_paper_predicate_sequence(
    x0: np.ndarray,
    selected: np.ndarray,
    attribute_names: list[str] | None = None,
    *,
    n_iter: int = 1000,
    learning_rate: float = 0.01,
    exponent: int = 4,
    gamma_l1: float = 0.01,
    gamma_a: float = 0.05,
    gamma_mu: float = 0.01,
    class_balance: bool = True,
    random_seed: int = 0,
    force_cpu: bool = False,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, float]], dict[str, Any]]:
    """Optimize equations 5-10 from the DimBridge paper."""

    device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    attribute_names = attribute_names or [f"feature_{index}" for index in range(x0.shape[1])]
    torch.manual_seed(random_seed)

    try:
        n_points = x0.shape[0]
        x = torch.from_numpy(x0.astype(np.float32)).to(device)
        label = torch.from_numpy(selected.astype(np.float32)).to(device)
        label_mask = label.bool()

        mean = x.mean(dim=0)
        scale = x.std(dim=0, correction=0).clamp_min(1e-6)
        normalized_x = (x - mean) / scale

        selection_centroids = torch.stack(
            [normalized_x[label_mask[index]].mean(dim=0) for index in range(len(selected))],
            dim=0,
        )
        selection_std = torch.stack(
            [
                normalized_x[label_mask[index]].std(dim=0, correction=0)
                for index in range(len(selected))
            ],
            dim=0,
        ).clamp(min=0.05, max=20.0)

        mu = selection_centroids.detach().clone().to(device).requires_grad_(True)
        a = (1 / selection_std).clamp(max=20.0).detach().clone().to(device).requires_grad_(True)

        losses: list[nn.Module] = []
        for brush_index, selection in enumerate(selected):
            if class_balance:
                selected_count = int(selection.sum())
                unselected_count = n_points - selected_count
                weights = torch.empty(n_points, dtype=torch.float32, device=device)
                weights[label_mask[brush_index]] = n_points / (2 * selected_count)
                weights[~label_mask[brush_index]] = n_points / (2 * unselected_count)
                losses.append(nn.BCELoss(weight=weights))
            else:
                losses.append(nn.BCELoss())

        optimizer = optim.SGD(
            [mu, a],
            lr=learning_rate,
            momentum=0.8,
            nesterov=True,
        )

        initial_loss = None
        final_terms: dict[str, float] = {}
        for _ in range(n_iter):
            bce = torch.zeros((), device=device)
            for brush_index in range(len(selected)):
                probability = _predict(
                    normalized_x,
                    a[brush_index],
                    mu[brush_index],
                    exponent,
                )
                probability = torch.nan_to_num(
                    probability,
                    nan=0.5,
                    posinf=1.0,
                    neginf=0.0,
                ).clamp(1e-6, 1 - 1e-6)
                bce = bce + losses[brush_index](probability, label[brush_index])

            l1 = gamma_l1 * a.abs().sum()
            smooth_a = torch.zeros((), device=device)
            smooth_mu = torch.zeros((), device=device)
            if len(selected) > 1:
                smooth_a = gamma_a * (a[1:] - a[:-1]).pow(2).sum()
                smooth_mu = gamma_mu * (mu[1:] - mu[:-1]).pow(2).sum()

            total_loss = bce + l1 + smooth_a + smooth_mu
            if not torch.isfinite(total_loss):
                raise RuntimeError("Predicate Regression optimization produced a non-finite loss.")
            if initial_loss is None:
                initial_loss = float(total_loss.detach().cpu().item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        with torch.no_grad():
            final_bce = torch.zeros((), device=device)
            for brush_index in range(len(selected)):
                final_probability = _predict(
                    normalized_x,
                    a[brush_index],
                    mu[brush_index],
                    exponent,
                )
                final_probability = torch.nan_to_num(
                    final_probability,
                    nan=0.5,
                    posinf=1.0,
                    neginf=0.0,
                ).clamp(1e-6, 1 - 1e-6)
                final_bce = final_bce + losses[brush_index](
                    final_probability,
                    label[brush_index],
                )
            final_l1 = gamma_l1 * a.abs().sum()
            final_smooth_a = torch.zeros((), device=device)
            final_smooth_mu = torch.zeros((), device=device)
            if len(selected) > 1:
                final_smooth_a = gamma_a * (a[1:] - a[:-1]).pow(2).sum()
                final_smooth_mu = gamma_mu * (mu[1:] - mu[:-1]).pow(2).sum()
            final_total = final_bce + final_l1 + final_smooth_a + final_smooth_mu
            final_terms = {
                "total": float(final_total.cpu().item()),
                "bce": float(final_bce.cpu().item()),
                "l1": float(final_l1.cpu().item()),
                "smooth_a": float(final_smooth_a.cpu().item()),
                "smooth_mu": float(final_smooth_mu.cpu().item()),
            }

        a = a.detach()
        mu = mu.detach()
        predicates, qualities = _extract_predicates(
            x0,
            selected,
            normalized_x,
            label,
            a,
            mu,
            mean,
            scale,
            attribute_names,
            exponent,
        )
        diagnostics = {
            "paper_equations": [5, 6, 8, 9, 10],
            "device": device,
            "iterations": n_iter,
            "exponent": exponent,
            "learning_rate": learning_rate,
            "gamma_l1": gamma_l1,
            "gamma_a": gamma_a,
            "gamma_mu": gamma_mu,
            "class_balance": class_balance,
            "initial_loss": initial_loss,
            "final_loss": final_terms,
            "optimizer": "SGD(momentum=0.8,nesterov=True)",
        }
        return predicates, qualities, diagnostics
    except RuntimeError as exc:
        if device == "cuda" and not force_cpu:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return compute_paper_predicate_sequence(
                x0,
                selected,
                attribute_names,
                n_iter=n_iter,
                learning_rate=learning_rate,
                exponent=exponent,
                gamma_l1=gamma_l1,
                gamma_a=gamma_a,
                gamma_mu=gamma_mu,
                class_balance=class_balance,
                random_seed=random_seed,
                force_cpu=True,
            )
        raise
