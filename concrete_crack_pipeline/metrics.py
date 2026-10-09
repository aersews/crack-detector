from __future__ import annotations

from itertools import product
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef, cohen_kappa_score, brier_score_loss,
)


def specificity(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if tn + fp else np.nan


def calculate_metrics(y_true, probabilities, threshold=0.5):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "specificity": specificity(y_true, predictions),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "mcc": matthews_corrcoef(y_true, predictions),
        "kappa": cohen_kappa_score(y_true, predictions),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "brier": brier_score_loss(y_true, probabilities),
        "confusion_matrix": cm,
        "predictions": predictions,
    }


def find_best_accuracy_threshold(y_true, probabilities, minimum=0.05, maximum=0.95, steps=181):
    best_threshold, best_accuracy = 0.5, -np.inf
    records = []
    for threshold in np.linspace(minimum, maximum, steps):
        metrics = calculate_metrics(y_true, probabilities, float(threshold))
        records.append({"threshold": threshold, **{k: metrics[k] for k in ("accuracy", "balanced_accuracy", "f1", "precision", "recall", "specificity", "mcc")}})
        if metrics["accuracy"] > best_accuracy:
            best_accuracy, best_threshold = metrics["accuracy"], threshold
    return float(best_threshold), pd.DataFrame(records)


def expected_calibration_error(y_true, probabilities, n_bins=10):
    y_true, probabilities = np.asarray(y_true), np.asarray(probabilities)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = ((probabilities >= bins[i]) & (probabilities <= bins[i + 1])) if i == n_bins - 1 else ((probabilities >= bins[i]) & (probabilities < bins[i + 1]))
        if mask.any():
            ece += (mask.sum() / len(y_true)) * abs(y_true[mask].mean() - probabilities[mask].mean())
    return float(ece)


def _group_indices(groups):
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    return unique, members


def cluster_bootstrap_metrics(y_true, probabilities, groups, threshold, metric_names, n_bootstrap, seed, logger=None):
    """Cluster bootstrap all requested metrics in one pass.

    The previous implementation recalculated the complete metric suite once for
    every metric requested, multiplying the CPU cost by ~10x. This implementation
    samples the same cluster bootstrap replicates and calculates the suite once
    per replicate, preserving the statistical procedure while greatly reducing
    redundant work.
    """
    y_true, probabilities, groups = map(np.asarray, (y_true, probabilities, groups))
    unique_groups, members = _group_indices(groups)
    rng = np.random.default_rng(seed)
    requested = list(metric_names)
    values = {name: [] for name in requested}
    n_groups = len(unique_groups)
    progress_step = max(1, n_bootstrap // 10)
    for iteration in range(n_bootstrap):
        sampled = rng.integers(0, n_groups, size=n_groups)
        indices = np.concatenate([members[i] for i in sampled])
        try:
            metrics = calculate_metrics(y_true[indices], probabilities[indices], threshold)
            for name in requested:
                value = metrics[name]
                if np.isfinite(value):
                    values[name].append(float(value))
        except (ValueError, ZeroDivisionError):
            continue
        if logger and (iteration + 1) % progress_step == 0:
            logger(f"Bootstrap progress: {iteration + 1}/{n_bootstrap}")
    return {
        name: (
            float(np.percentile(vals, 2.5)) if vals else np.nan,
            float(np.percentile(vals, 97.5)) if vals else np.nan,
        )
        for name, vals in values.items()
    }


def cluster_bootstrap_metric(y_true, probabilities, groups, threshold, metric_name, n_bootstrap, seed):
    return cluster_bootstrap_metrics(y_true, probabilities, groups, threshold, [metric_name], n_bootstrap, seed)[metric_name]


def paired_cluster_bootstrap_difference(y_true, pred_a, pred_b, groups, metric="accuracy", n_bootstrap=2000, seed=42):
    y_true, pred_a, pred_b, groups = map(np.asarray, (y_true, pred_a, pred_b, groups))
    if metric not in {"accuracy", "f1"}:
        raise ValueError("Supported paired metrics are 'accuracy' and 'f1'.")
    unique_groups, members = _group_indices(groups)
    rng = np.random.default_rng(seed)
    scorer = accuracy_score if metric == "accuracy" else lambda y, p: f1_score(y, p, zero_division=0)
    observed = float(scorer(y_true, pred_a) - scorer(y_true, pred_b))
    values = np.empty(n_bootstrap, dtype=np.float64)
    n_groups = len(unique_groups)
    for i in range(n_bootstrap):
        sampled = rng.integers(0, n_groups, size=n_groups)
        indices = np.concatenate([members[j] for j in sampled])
        values[i] = float(scorer(y_true[indices], pred_a[indices]) - scorer(y_true[indices], pred_b[indices]))
    return {
        "observed_difference": observed,
        "ci_lower": float(np.percentile(values, 2.5)),
        "ci_upper": float(np.percentile(values, 97.5)),
        "p_value": float(min(1.0, 2 * min(np.mean(values <= 0), np.mean(values >= 0)))),
    }


def parameter_counts(model):
    total = model.count_params()
    trainable = sum(np.prod(v.shape) for v in model.trainable_variables)
    return {"total": int(total), "trainable": int(trainable), "non_trainable": int(total - trainable), "total_millions": float(total / 1e6)}


def summarize_internal_results(results):
    rows = []
    for r in results:
        p = r["parameters"]
        rows.append({"Model": r["model"], "OuterFold": r["outer_fold"], "Seed": r["seed"], "ValidationAccuracy": r["validation_accuracy"], "TestAccuracy": r["test_accuracy"], "BalancedAccuracy": r["test_balanced_accuracy"], "Precision": r["test_precision"], "Recall": r["test_recall"], "Specificity": r["test_specificity"], "F1": r["test_f1"], "MCC": r["test_mcc"], "Kappa": r["test_kappa"], "ROC_AUC": r["test_roc_auc"], "PR_AUC": r["test_pr_auc"], "Brier": r["test_brier"], "ECE": r["test_ece"], "Threshold": r["threshold"], "Parameters": p["total"], "Parameters_M": p["total_millions"], "TrainingSeconds": r["training_seconds"]})
    return pd.DataFrame(rows)
