from __future__ import annotations

from textwrap import dedent

import numpy as np
import torch
from torch import nn, optim


def predicate_components(x: torch.Tensor, a: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    r"""
    Per-factor contributions to the UMAP-inspired predicate distance.
    """

    b = 4
    return (a.abs() * (x - mu).abs()).pow(b)


def predict(x: torch.Tensor, a: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    r"""
    UMAP-inspired prediction function.
    """

    return 1 / (1 + predicate_components(x, a, mu).sum(1))


def classification_metrics(predicted: torch.Tensor, label: torch.Tensor) -> dict[str, float]:
    predicted = predicted.bool()
    label = label.bool()
    n_points = max(1, label.numel())
    correct = (predicted == label).float().sum().item()
    tp = (predicted & label).float().sum().item()
    fp = (predicted & ~label).float().sum().item()
    fn = (~predicted & label).float().sum().item()
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 / (1 / precision + 1 / recall) if precision > 0 and recall > 0 else 0
    return {
        "accuracy": correct / n_points,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_predicate_sequence(
    x0: np.ndarray,
    selected: np.ndarray,
    attribute_names: list[str] | None = None,
    n_iter: int = 1000,
    force_cpu: bool = False,
):
    """
    Parameters
    ----------
    x0:
        Numpy array with shape [n_points, n_features].
    selected:
        Boolean array with shape [n_brushes, n_points].
    """

    device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        attribute_names = attribute_names or [f"feature_{i}" for i in range(x0.shape[1])]

        n_points, n_features = x0.shape
        vmin = x0.min(0)
        vmax = x0.max(0)

        x = torch.from_numpy(x0.astype(np.float32)).to(device)
        label = torch.from_numpy(selected).float().to(device)

        mean = x.mean(0)
        scale = x.std(0) + 1e-6
        x = (x - mean) / scale

        selection_centroids = torch.stack([x[sel_t].mean(0) for sel_t in selected], 0)
        selection_std = torch.stack([x[sel_t].std(0) for sel_t in selected], 0).clamp_min(1e-6)

        mu = selection_centroids.to(device)
        a = (1 / selection_std).to(device)
        a.requires_grad_(True)
        mu.requires_grad_(True)

        bce_per_brush = []
        for st in selected:
            n_selected = st.sum()
            n_unselected = n_points - n_selected
            instance_weight = torch.ones(x.shape[0]).to(device)
            instance_weight[st] = n_points / n_selected
            instance_weight[~st] = 2 * n_points / n_unselected
            bce_per_brush.append(nn.BCELoss(weight=instance_weight))

        optimizer = optim.SGD(
            [
                {"params": mu, "weight_decay": 0},
                {"params": a, "weight_decay": 0.25},
            ],
            lr=1e-2,
            momentum=0.8,
            nesterov=True,
        )

        for _ in range(n_iter):
            loss_per_brush = []
            smoothness_loss = 0
            for t, _ in enumerate(selected):
                pred = predict(x, a[t], mu[t])
                pred = torch.nan_to_num(pred, nan=0.5, posinf=1.0, neginf=0.0).clamp(1e-6, 1 - 1e-6)
                loss = bce_per_brush[t](pred, label[t])
                loss_per_brush.append(loss)
                if len(selected) == 2:
                    smoothness_loss += 5 * (a[1:] - a[:-1]).pow(2).mean()
                elif len(selected) > 2:
                    smoothness_loss += 50 * (a[1:] - a[:-1]).pow(2).mean()

            total_loss = sum(loss_per_brush) + smoothness_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        a.detach_()
        mu.detach_()

        qualities = []
        factor_importances = []
        factor_f1_without = []
        with torch.no_grad():
            for t, _ in enumerate(selected):
                components = predicate_components(x, a[t], mu[t])
                total_distance = components.sum(1)
                membership = torch.nan_to_num(
                    1 / (1 + total_distance),
                    nan=0.5,
                    posinf=1.0,
                    neginf=0.0,
                )
                metrics = classification_metrics(membership > 0.5, label[t])
                qualities.append(dict(brush=t, **metrics))

                brush_importances = []
                brush_f1_without = []
                for k in range(n_features):
                    distance_without_factor = (total_distance - components[:, k]).clamp_min(0)
                    membership_without_factor = torch.nan_to_num(
                        1 / (1 + distance_without_factor),
                        nan=0.5,
                        posinf=1.0,
                        neginf=0.0,
                    )
                    metrics_without_factor = classification_metrics(
                        membership_without_factor > 0.5,
                        label[t],
                    )
                    f1_without = metrics_without_factor["f1"]
                    brush_f1_without.append(f1_without)
                    brush_importances.append(max(0.0, metrics["f1"] - f1_without))
                factor_importances.append(brush_importances)
                factor_f1_without.append(brush_f1_without)

        predicates = []
        for t, st in enumerate(selected):
            r = 1 / a[t].abs().clamp_min(1e-6)
            predicate_clauses = []
            for k in range(n_features):
                vmin_selected = x0[st, k].min()
                vmax_selected = x0[st, k].max()

                r_k = (r[k] * scale[k]).item()
                mu_k = (mu[t, k] * scale[k] + mean[k]).item()
                ci = [mu_k - r_k, mu_k + r_k]

                if not ci[0] < ci[1]:
                    continue

                should_include = not (ci[0] <= vmin[k] and ci[1] >= vmax[k])
                if ci[0] < vmin[k]:
                    ci[0] = vmin[k]
                if ci[1] > vmax[k]:
                    ci[1] = vmax[k]

                if should_include:
                    if ci[0] < vmin_selected:
                        ci[0] = vmin_selected
                    if ci[1] > vmax_selected:
                        ci[1] = vmax_selected
                    global_width = float(vmax[k] - vmin[k])
                    interval_width = float(ci[1] - ci[0])
                    range_reduction = (
                        max(0.0, min(1.0, 1 - interval_width / global_width))
                        if global_width > 0
                        else 0.0
                    )
                    predicate_clauses.append(
                        dict(
                            dim=k,
                            interval=[float(ci[0]), float(ci[1])],
                            attribute=attribute_names[k],
                            importance=float(factor_importances[t][k]),
                            f1_without=float(factor_f1_without[t][k]),
                            range_reduction=range_reduction,
                        )
                    )

            predicate_clauses.sort(
                key=lambda clause: (
                    -clause["importance"],
                    -clause["range_reduction"],
                    clause["dim"],
                )
            )
            for rank, clause in enumerate(predicate_clauses, start=1):
                clause["rank"] = rank

            predicates.append(predicate_clauses)

        return predicates, qualities, dict(mu=mu, a=a)
    except RuntimeError as exc:
        # CUDA kernel asserts poison the current CUDA context; retrying on CPU keeps API usable.
        if device == "cuda" and not force_cpu and "CUDA" in str(exc):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return compute_predicate_sequence(
                x0,
                selected,
                attribute_names=attribute_names,
                n_iter=n_iter,
                force_cpu=True,
            )
        raise


def format_quality_summary(qualities: list[dict[str, float]]) -> str:
    return "\n".join(
        dedent(
            f"""
            brush = {quality['brush']}
            accuracy = {quality['accuracy']}
            precision = {quality['precision']}
            recall = {quality['recall']}
            f1 = {quality['f1']}
            """
        ).strip()
        for quality in qualities
    )
