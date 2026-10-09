from __future__ import annotations

import json
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import calculate_metrics, cluster_bootstrap_metrics, expected_calibration_error, paired_cluster_bootstrap_difference, parameter_counts
from .plots import calibration, confusion_matrix, precision_recall, roc
from .training import build_dataset, train_model


def evaluate_outer_fold(model_name, builder, train_df, validation_df, test_df, params, seed, outer_fold, training_cfg, analysis_cfg, dirs, store, logger, class_weight=None):
    key = f"evaluation/{model_name}/outer-{outer_fold}/seed-{seed}"
    metrics_path = dirs["predictions"] / f"{model_name}_outer{outer_fold}_seed{seed}_metrics.json"
    prediction_path = dirs["predictions"] / f"{model_name}_outer{outer_fold}_seed{seed}_test_predictions.csv"
    if store.is_completed(key) and metrics_path.exists() and prediction_path.exists():
        logger.info("RESUME: loading completed %s", key)
        return load_saved_result(metrics_path, prediction_path)

    train_ds = build_dataset(train_df.path.values, train_df.label.values, training_cfg.img_size, training_cfg.batch_size, True, seed)
    val_ds = build_dataset(validation_df.path.values, validation_df.label.values, training_cfg.img_size, training_cfg.batch_size, False, seed)
    test_ds = build_dataset(test_df.path.values, test_df.label.values, training_cfg.img_size, training_cfg.batch_size, False, seed)
    from .environment import cleanup_tensorflow_memory
    model = builder(**params)
    if model is None:
        raise RuntimeError(f"Model {model_name} failed to build.")
    prefix = f"{model_name}_outer{outer_fold}_seed{seed}"
    start = time.time()
    try:
        model = train_model(model, model_name, train_ds, val_ds, prefix, dirs, training_cfg, logger, class_weight)
        validation_probabilities = model.predict(val_ds, verbose=0).ravel()
        threshold, threshold_df = _threshold(validation_df.label.values, validation_probabilities, analysis_cfg)
        threshold_df.to_csv(dirs["predictions"] / f"{prefix}_threshold.csv", index=False)
        validation_metrics = calculate_metrics(validation_df.label.values, validation_probabilities, threshold)
        test_probabilities = model.predict(test_ds, verbose=0).ravel()
        test_metrics = calculate_metrics(test_df.label.values, test_probabilities, threshold)
        ece = expected_calibration_error(test_df.label.values, test_probabilities)
        logger.info("BOOTSTRAP START | %s | outer=%d | seed=%d | samples=%d", model_name, outer_fold, seed, analysis_cfg.bootstrap_samples)
        bootstrap_values = cluster_bootstrap_metrics(
            test_df.label.values, test_probabilities, test_df.group_id.values, threshold,
            ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "mcc", "roc_auc", "average_precision", "brier"],
            analysis_cfg.bootstrap_samples, seed + outer_fold, logger=lambda msg: logger.info("%s | %s", model_name, msg)
        )
        bootstrap = {metric_name: {"lower": low, "upper": high} for metric_name, (low, high) in bootstrap_values.items()}
        logger.info("BOOTSTRAP COMPLETE | %s | outer=%d | seed=%d", model_name, outer_fold, seed)

        result = {
            "model": model_name, "outer_fold": int(outer_fold), "seed": int(seed), "threshold": float(threshold),
            "validation_accuracy": float(validation_metrics["accuracy"]), "validation_f1": float(validation_metrics["f1"]),
            "test_accuracy": float(test_metrics["accuracy"]), "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
            "test_precision": float(test_metrics["precision"]), "test_recall": float(test_metrics["recall"]),
            "test_specificity": float(test_metrics["specificity"]), "test_f1": float(test_metrics["f1"]),
            "test_mcc": float(test_metrics["mcc"]), "test_kappa": float(test_metrics["kappa"]),
            "test_roc_auc": float(test_metrics["roc_auc"]), "test_pr_auc": float(test_metrics["average_precision"]),
            "test_brier": float(test_metrics["brier"]), "test_ece": float(ece),
            "training_seconds": float(time.time() - start), "parameters": parameter_counts(model), "hyperparameters": params,
            "confusion_matrix": test_metrics["confusion_matrix"].tolist(), "bootstrap": bootstrap,
        }
        prediction_df = pd.DataFrame({"path": test_df.path.values, "label": test_df.label.values, "group_id": test_df.group_id.values, "probability": test_probabilities, "prediction": test_metrics["predictions"], "threshold": threshold})
        prediction_df.to_csv(prediction_path, index=False)
        metrics_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        store.mark_completed(key, artifact=str(metrics_path))
        return {**result, "y_true": test_df.label.values, "groups": test_df.group_id.values, "probabilities": test_probabilities, "predictions": test_metrics["predictions"], "paths": test_df.path.values}
    except Exception as exc:
        store.mark_failed(key, repr(exc)); raise
    finally:
        del model
        cleanup_tensorflow_memory()


def _threshold(y_true, probabilities, config):
    from .metrics import find_best_accuracy_threshold
    return find_best_accuracy_threshold(y_true, probabilities, config.threshold_min, config.threshold_max, config.threshold_steps)


def load_saved_result(metrics_path: Path, prediction_path: Path):
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    pred = pd.read_csv(prediction_path)
    return {**result, "y_true": pred.label.to_numpy(), "groups": pred.group_id.to_numpy(), "probabilities": pred.probability.to_numpy(), "predictions": pred.prediction.to_numpy(), "paths": pred.path.to_numpy()}


def evaluate_external_model(model, model_name, external_df, threshold, output_prefix, training_cfg, analysis_cfg, dirs, logger):
    external_ds = build_dataset(external_df.path.values, external_df.label.values, training_cfg.img_size, training_cfg.batch_size, False, 42)
    probabilities = model.predict(external_ds, verbose=0).ravel()
    metrics = calculate_metrics(external_df.label.values, probabilities, threshold)
    result = {"model": model_name, "threshold_from_CCIC_validation": float(threshold), "accuracy": float(metrics["accuracy"]), "balanced_accuracy": float(metrics["balanced_accuracy"]), "precision": float(metrics["precision"]), "recall": float(metrics["recall"]), "specificity": float(metrics["specificity"]), "f1": float(metrics["f1"]), "mcc": float(metrics["mcc"]), "kappa": float(metrics["kappa"]), "roc_auc": float(metrics["roc_auc"]), "pr_auc": float(metrics["average_precision"]), "brier": float(metrics["brier"]), "ece": float(expected_calibration_error(external_df.label.values, probabilities)), "confusion_matrix": metrics["confusion_matrix"].tolist()}
    bootstrap_values = cluster_bootstrap_metrics(
        external_df.label.values, probabilities, external_df.group_id.values, threshold,
        ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "mcc", "roc_auc", "average_precision", "brier"],
        analysis_cfg.bootstrap_samples, 999, logger=lambda msg: logger.info("%s | %s", output_prefix, msg)
    )
    bootstrap = {metric_name: {"lower": low, "upper": high} for metric_name, (low, high) in bootstrap_values.items()}
    result["bootstrap"] = bootstrap
    (dirs["external"] / f"{output_prefix}_metrics.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    pd.DataFrame({"path": external_df.path.values, "label": external_df.label.values, "group_id": external_df.group_id.values, "surface": external_df.surface.values, "probability": probabilities, "prediction": metrics["predictions"], "threshold": threshold}).to_csv(dirs["external"] / f"{output_prefix}_predictions.csv", index=False)
    confusion_matrix(metrics["confusion_matrix"], f"SDNET2018 - {model_name}", dirs["external"] / f"{output_prefix}_confusion_matrix.png")
    roc(external_df.label.values, probabilities, model_name, dirs["external"] / f"{output_prefix}_roc.png")
    precision_recall(external_df.label.values, probabilities, model_name, dirs["external"] / f"{output_prefix}_pr.png")
    calibration(external_df.label.values, probabilities, f"SDNET2018 calibration - {model_name}", dirs["external"] / f"{output_prefix}_calibration.png")
    return result


def pairwise_model_comparisons(results, output_path, bootstrap_samples):
    names = sorted({r["model"] for r in results}); grouped = defaultdict(dict)
    for result in results: grouped[(result["outer_fold"], result["seed"])][result["model"]] = result
    rows = []
    for model_a, model_b in product(names, names):
        if model_a >= model_b: continue
        for key, mapping in grouped.items():
            if model_a not in mapping or model_b not in mapping: continue
            comparison = paired_cluster_bootstrap_difference(mapping[model_a]["y_true"], mapping[model_a]["predictions"], mapping[model_b]["predictions"], mapping[model_a]["groups"], "accuracy", bootstrap_samples, 10000 + key[0] * 100 + key[1])
            rows.append({"Model_A": model_a, "Model_B": model_b, "OuterFold": key[0], "Seed": key[1], **comparison})
    frame = pd.DataFrame(rows); frame.to_csv(output_path, index=False); return frame
