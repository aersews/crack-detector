from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from .analysis import evaluate_external_model, evaluate_outer_fold, pairwise_model_comparisons
from .checkpoint import CheckpointStore
from .config import AppConfig
from .data import (
    add_exact_hashes, add_perceptual_hashes, audit_exact_duplicates, audit_images,
    create_dataset_groups, load_ccic_dataset, make_group_cv, prepare_external_dataset,
    remove_exact_duplicates, resolve_dataset_path, save_dataset_manifest, save_split,
    verify_group_disjointness,
)
from .environment import cleanup_tensorflow_memory, configure_environment
from .logging_utils import configure_logging
from .metrics import summarize_internal_results
from .models import build_models
from .plots import summary_plots
from .training import tune_model_inner, build_dataset, train_model


class Pipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        config.ensure_directories()
        self.logger = configure_logging(config.dirs["logs"])
        self.store = CheckpointStore(config.dirs["checkpoints"] / "pipeline_state.json")
        self.config.save(config.dirs["manifests"] / "config.json")

    def _save_json(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        temp.replace(path)

    def prepare_ccic(self) -> tuple[pd.DataFrame, dict, dict | None]:
        dirs = self.config.dirs
        index_path = dirs["manifests"] / "dataset_index.csv"
        manifest_path = dirs["manifests"] / "dataset_manifest.json"
        key = "dataset/ccic_prepared"
        if self.store.is_completed(key) and index_path.exists() and manifest_path.exists():
            self.logger.info("RESUME: loading prepared CCIC dataset")
            return pd.read_csv(index_path), json.loads(manifest_path.read_text()), self._class_weights(pd.read_csv(index_path))

        root = resolve_dataset_path(self.config.datasets.ccic_path, self.config.datasets.ccic_slug)
        self.logger.info("CCIC dataset: %s", root)
        df = load_ccic_dataset(root)
        class_weight = self._class_weights(df)
        df = audit_images(df, dirs["manifests"] / "invalid_images.csv", self.logger.info)
        df = add_exact_hashes(df, self.logger.info)
        duplicate_groups = audit_exact_duplicates(df, dirs["duplicates"] / "exact_duplicates.json")
        self.logger.info("Exact duplicate groups: %d", len(duplicate_groups))
        df = remove_exact_duplicates(df, self.config.audit.remove_exact_duplicates)
        df = add_perceptual_hashes(df, self.config.audit.phash_size, self.config.audit.phash_low_frequency_size)
        df = create_dataset_groups(df, self.config.datasets.source_group_csv, self.config.audit.phash_threshold, self.logger.info)
        manifest = save_dataset_manifest(df, dirs["manifests"], self.config.audit.phash_threshold, self.config.datasets.source_group_csv is not None)
        self.store.mark_completed(key, images=len(df), groups=manifest["n_groups"])
        return df, manifest, class_weight

    @staticmethod
    def _class_weights(df: pd.DataFrame):
        counts = df["label"].value_counts()
        if len(counts) < 2:
            raise ValueError("Dataset must contain both positive and negative labels.")
        ratio = counts.max() / counts.min()
        if ratio <= 1.1:
            return None
        weights = compute_class_weight("balanced", classes=np.unique(df["label"]), y=df["label"])
        return dict(enumerate(weights))

    def _outer_split(self, df, outer_fold):
        split_path = self.config.dirs["splits"] / f"outer_fold_{outer_fold}.csv"
        if split_path.exists():
            saved = pd.read_csv(split_path)
            # Preserve deterministic ordering from saved split rather than recomputing indices.
            return saved[saved.split == "train"], saved[saved.split == "validation"], saved[saved.split == "test"]
        outer_splits = make_group_cv(df, self.config.training.outer_folds, self.config.training.base_seed)
        outer_train_idx, outer_test_idx = outer_splits[outer_fold - 1]
        outer_train_df, outer_test_df = df.iloc[outer_train_idx].copy(), df.iloc[outer_test_idx].copy()
        inner = make_group_cv(outer_train_df, self.config.training.inner_folds, self.config.training.base_seed + outer_fold)
        train_idx, val_idx = inner[0]
        train_df, val_df = outer_train_df.iloc[train_idx].copy(), outer_train_df.iloc[val_idx].copy()
        combined = pd.concat([train_df, val_df, outer_test_df], ignore_index=True)
        verify_group_disjointness(combined, np.arange(len(train_df)), np.arange(len(train_df), len(train_df)+len(val_df)), np.arange(len(train_df)+len(val_df), len(combined)))
        save_split(combined, np.arange(len(train_df)), np.arange(len(train_df), len(train_df)+len(val_df)), np.arange(len(train_df)+len(val_df), len(combined)), split_path)
        return train_df, val_df, outer_test_df

    def run(self) -> dict:
        self.store.set_status("running")
        environment_manifest = configure_environment(self.config.training.base_seed, self.config.dirs["manifests"] / "environment.json", self.config.training, self.logger)
        self.logger.info("Starting concrete crack research pipeline")
        try:
            df, manifest, class_weight = self.prepare_ccic()
            builders = build_models(self.config.training.img_size)
            all_results = []
            tuning_records = []

            for outer_fold in range(1, self.config.training.outer_folds + 1):
                train_df, validation_df, test_df = self._outer_split(df, outer_fold)
                self.logger.info("OUTER FOLD %d/%d", outer_fold, self.config.training.outer_folds)
                for model_name, builder in builders.items():
                    # Hyperparameters are selected once per model/outer fold using a
                    # fixed tuning seed, then held constant across the independent
                    # model seeds. This preserves nested, group-aware selection while
                    # removing a 3x redundant tuning workload.
                    tuning_seed = self.config.training.tuning_seed
                    self.logger.info("MODEL START | %s | outer=%d/%d | tuning once with seed=%d", model_name, outer_fold, self.config.training.outer_folds, tuning_seed)
                    try:
                        best_params, tuning_df = tune_model_inner(model_name, builder, pd.concat([train_df, validation_df], ignore_index=True), tuning_seed, outer_fold, self.config.training, self.config.dirs, self.store, self.logger)
                        tuning_records.append({"model": model_name, "outer_fold": outer_fold, "seed": tuning_seed, "tuning_df": tuning_df})
                        self.logger.info("TUNING COMPLETE | %s | outer=%d | best=%s", model_name, outer_fold, best_params)
                    except Exception as exc:
                        key = f"tuning/{model_name}/outer-{outer_fold}/seed-{tuning_seed}"
                        self.store.mark_failed(key, repr(exc))
                        self.logger.error("Tuning failed: %s\n%s", key, traceback.format_exc())
                        continue

                    for seed in self.config.training.model_seeds:
                        try:
                            self.logger.info("EVALUATION START | %s | outer=%d/%d | seed=%d/%d | using shared hyperparameters", model_name, outer_fold, self.config.training.outer_folds, seed, len(self.config.training.model_seeds))
                            result = evaluate_outer_fold(model_name, builder, train_df, validation_df, test_df, best_params, seed, outer_fold, self.config.training, self.config.analysis, self.config.dirs, self.store, self.logger, class_weight)
                            all_results.append(result)
                            self.logger.info("EVALUATION COMPLETE | %s | outer=%d | seed=%d", model_name, outer_fold, seed)
                        except Exception as exc:
                            key = f"evaluation/{model_name}/outer-{outer_fold}/seed-{seed}"
                            self.store.mark_failed(key, repr(exc))
                            self.logger.error("Task failed: %s\n%s", key, traceback.format_exc())
                            continue

            if not all_results:
                raise RuntimeError("No successful model evaluations were produced.")
            internal_summary = summarize_internal_results(all_results)
            internal_summary.to_csv(self.config.output_dir / "internal_outer_fold_results.csv", index=False)
            summary_plots(internal_summary, self.config.dirs["plots"])

            ranking = internal_summary.groupby("Model")["ValidationAccuracy"].agg(["mean", "std"]).sort_values("mean", ascending=False)
            ranking.to_csv(self.config.dirs["statistics"] / "architecture_selection.csv")
            best_model = ranking.index[0]
            self.logger.info("Selected architecture: %s", best_model)
            pairwise_model_comparisons(all_results, self.config.dirs["statistics"] / "paired_cluster_bootstrap_accuracy.csv", self.config.analysis.bootstrap_samples)

            selected_tuning = [r["tuning_df"] for r in tuning_records if r["model"] == best_model]
            if not selected_tuning:
                raise RuntimeError("No tuning records for selected model.")
            combined = pd.concat(selected_tuning, ignore_index=True)
            grouped = combined.groupby(["dropout", "l2_reg"])["mean_validation_accuracy"].mean().sort_values(ascending=False)
            best = grouped.index[0]
            final_params = {"dropout": float(best[0]), "l2_reg": float(best[1])}
            self._save_json(self.config.dirs["manifests"] / "final_model_configuration.json", {"selected_model": best_model, "hyperparameters": final_params, "primary_metric": "accuracy", "selection_rule": "highest mean inner validation accuracy"})

            external_root = resolve_dataset_path(self.config.datasets.sdnet_path, self.config.datasets.sdnet_slug)
            external_manifest_path = self.config.dirs["external"] / "sdnet2018_manifest.csv"
            if external_manifest_path.exists() and self.store.is_completed("dataset/sdnet_prepared"):
                external_df = pd.read_csv(external_manifest_path)
            else:
                external_df = prepare_external_dataset(external_root, self.config.dirs, self.config.audit, self.logger.info)
                external_df.to_csv(external_manifest_path, index=False)
                self.store.mark_completed("dataset/sdnet_prepared", images=len(external_df))

            final_model_path = self.config.output_dir / f"FINAL_{best_model}.keras"
            final_key = f"final_model/{best_model}"
            if self.store.is_completed(final_key) and final_model_path.exists():
                from tensorflow import keras
                final_model = keras.models.load_model(final_model_path)
                self.logger.info("RESUME: loaded final model %s", final_model_path)
            else:
                cleanup_tensorflow_memory()
                original_batch = self.config.training.batch_size
                # Final training intentionally uses the original pipeline's reduced batch size for OOM protection.
                final_cfg = self.config.training.__class__(**{**self.config.training.__dict__, "batch_size": self.config.training.final_batch_size})
                splits = make_group_cv(df, self.config.training.outer_folds, self.config.training.base_seed)
                final_train = df.iloc[splits[0][0]]
                final_val = df.iloc[splits[0][1]]
                train_ds = build_dataset(final_train.path.values, final_train.label.values, final_cfg.img_size, final_cfg.batch_size, True, final_cfg.base_seed)
                val_ds = build_dataset(final_val.path.values, final_val.label.values, final_cfg.img_size, final_cfg.batch_size, False, final_cfg.base_seed)
                model = builders[best_model](**final_params)
                final_model = train_model(model, best_model, train_ds, val_ds, f"FINAL_{best_model}", self.config.dirs, final_cfg, self.logger, class_weight)
                final_model.save(final_model_path)
                self.store.mark_completed(final_key, artifact=str(final_model_path))

            selected_results = [r for r in all_results if r["model"] == best_model]
            external_threshold = float(selected_results[int(np.argmax([r["validation_accuracy"] for r in selected_results]))]["threshold"])

            external_key = f"external/{best_model}"
            external_metrics_path = self.config.dirs["external"] / f"SDNET2018_{best_model}_metrics.json"
            external_predictions_path = self.config.dirs["external"] / f"SDNET2018_{best_model}_predictions.csv"
            if self.store.is_completed(external_key) and external_metrics_path.exists() and external_predictions_path.exists():
                external_result = json.loads(external_metrics_path.read_text(encoding="utf-8"))
                self.logger.info("RESUME: external evaluation already complete | %s", best_model)
            else:
                self.logger.info("EXTERNAL EVALUATION START | %s | SDNET2018", best_model)
                external_result = evaluate_external_model(final_model, best_model, external_df, external_threshold, f"SDNET2018_{best_model}", self.config.training, self.config.analysis, self.config.dirs, self.logger)
                self.store.mark_completed(external_key, artifact=str(external_metrics_path))
                self.logger.info("EXTERNAL EVALUATION COMPLETE | %s", best_model)

            if self.config.analysis.run_gradcam:
                gradcam_key = f"gradcam/{best_model}"
                gradcam_marker = self.config.dirs["gradcam"] / f"SDNET2018_{best_model}.complete"
                if self.store.is_completed(gradcam_key) and gradcam_marker.exists():
                    self.logger.info("RESUME: Grad-CAM already complete | %s", best_model)
                else:
                    try:
                        from .gradcam import save_examples
                        self.logger.info("GRADCAM START | %s", best_model)
                        external_ds = build_dataset(external_df.path.values, external_df.label.values, self.config.training.img_size, self.config.training.batch_size, False, 42)
                        external_probabilities = final_model.predict(external_ds, verbose=0).ravel()
                        save_examples(final_model, best_model, external_df, external_probabilities, external_threshold, self.config.dirs["gradcam"], self.config.analysis.gradcam_examples_per_case, f"SDNET2018_{best_model}", self.logger.info)
                        gradcam_marker.write_text("complete\n", encoding="utf-8")
                        self.store.mark_completed(gradcam_key, artifact=str(gradcam_marker))
                        self.logger.info("GRADCAM COMPLETE | %s", best_model)
                    except Exception:
                        self.logger.error("Grad-CAM stage failed; preserving completed model/results.\n%s", traceback.format_exc())

            latency = {}
            if self.config.analysis.run_latency:
                latency_key = f"latency/{best_model}"
                latency_path = self.config.dirs["external"] / "final_model_latency.json"
                if self.store.is_completed(latency_key) and latency_path.exists():
                    latency = json.loads(latency_path.read_text(encoding="utf-8"))
                    self.logger.info("RESUME: CPU latency benchmark already complete")
                else:
                    self.logger.info("LATENCY START | CPU | samples=%d", self.config.analysis.latency_samples)
                    external_ds = build_dataset(external_df.path.values, external_df.label.values, self.config.training.img_size, self.config.training.batch_size, False, 42)
                    samples = []
                    for batch_x, _ in external_ds:
                        samples.extend(batch_x.numpy())
                        if len(samples) >= self.config.analysis.latency_samples: break
                    samples = np.asarray(samples[:self.config.analysis.latency_samples])
                    for _ in range(10): final_model(samples[:1], training=False).numpy()
                    times = []
                    import time
                    import tensorflow as tf
                    with tf.device("/CPU:0"):
                        for image in samples:
                            start_time = time.perf_counter(); final_model(image[None, ...], training=False).numpy(); times.append((time.perf_counter()-start_time)*1000)
                    if times:
                        latency = {"cpu_latency_mean_ms": float(np.mean(times)), "cpu_latency_median_ms": float(np.median(times)), "cpu_latency_p95_ms": float(np.percentile(times,95)), "cpu_latency_std_ms": float(np.std(times)), "cpu_fps": float(1000/np.mean(times))}
                        self._save_json(latency_path, latency)
                        self.store.mark_completed(latency_key, artifact=str(latency_path))
                    self.logger.info("LATENCY COMPLETE | %s", latency)

            internal_model_summary = internal_summary.groupby("Model").agg({"ValidationAccuracy":["mean","std"],"TestAccuracy":["mean","std"],"F1":["mean","std"],"MCC":["mean","std"],"ROC_AUC":["mean","std"],"PR_AUC":["mean","std"]})
            internal_model_summary.to_csv(self.config.output_dir / "internal_model_summary.csv")
            report = {"study":{"task":"Binary visible concrete-surface crack classification","primary_metric":"accuracy","architecture_selection":"mean inner validation accuracy","external_dataset":"SDNET2018"},"ccic":manifest,"experimental_design":{"outer_folds":self.config.training.outer_folds,"inner_folds":self.config.training.inner_folds,"model_seeds":list(self.config.training.model_seeds),"group_aware":True,"grouping_method":"true source groups" if self.config.datasets.source_group_csv else "perceptual-hash connected components; approximate source grouping"},"selected_model":best_model,"final_hyperparameters":final_params,"external_threshold":external_threshold,"external_result":external_result,"latency":latency,"environment":environment_manifest,"performance": {"batch_size": self.config.training.batch_size, "final_batch_size": self.config.training.final_batch_size, "mixed_precision": self.config.training.use_mixed_precision, "xla": self.config.training.enable_xla, "tuning_once_per_model_outer_fold": True, "tuning_seed": self.config.training.tuning_seed, "tuning_max_configs": self.config.training.tuning_max_configs, "bootstrap_samples": self.config.analysis.bootstrap_samples}}
            self._save_json(self.config.output_dir / "final_report.json", report)
            self.store.set_status("complete")
            self.logger.info("PIPELINE COMPLETE | selected=%s | SDNET accuracy=%.4f | F1=%.4f", best_model, external_result["accuracy"], external_result["f1"])
            return report
        except BaseException as exc:
            self.store.set_status("interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed")
            self.logger.error("Pipeline terminated: %s", exc)
            raise
