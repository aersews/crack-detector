from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .checkpoint import CheckpointStore
from .metrics import calculate_metrics, find_best_accuracy_threshold, parameter_counts
from .plots import training_history


def build_dataset(paths, labels, img_size, batch_size, training=False, seed=42, device=None):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    paths = np.asarray(paths, dtype=str)
    labels = np.asarray(labels, dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, img_size, method="bilinear")
        return tf.cast(image, tf.float32), tf.cast(label, tf.float32)

    # Keep image decoding/resizing in tf.data, but perform augmentation inside the
    # model. On NVIDIA GPUs this avoids a CPU-bound augmentation map becoming the
    # training bottleneck while preserving the same augmentation distribution.
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    dataset = dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return dataset


def _backup_callback(backup_dir: Path):
    from tensorflow.keras.callbacks import BackupAndRestore
    backup_dir.mkdir(parents=True, exist_ok=True)
    return BackupAndRestore(str(backup_dir), delete_checkpoint=True)


def train_model(model, model_name, train_ds, validation_ds, output_prefix, dirs, training_cfg, logger, class_weight=None):
    from tensorflow import keras
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

    checkpoint_dir = dirs["checkpoints"] / output_prefix
    backup_root = checkpoint_dir / "backup"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stage1_path = checkpoint_dir / "stage1_best.keras"
    stage2_path = checkpoint_dir / "stage2_best.keras"

    def run_stage(stage, epochs, lr, patience, best_path):
        from .models import compile_model
        stage_backup = backup_root / stage
        # A best model with no active BackupAndRestore directory means the stage
        # completed successfully before a possible process-level interruption.
        if best_path.exists() and not (stage_backup.exists() and any(stage_backup.iterdir())):
            logger.info("RESUME: stage %s already complete; loading %s", stage, best_path)
            return keras.models.load_model(best_path)
        compile_model(model, lr)
        callbacks = [
            _backup_callback(stage_backup),
            EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2 if stage == "stage1" else 3, min_lr=1e-7 if stage == "stage2" else 1e-6, verbose=1),
            ModelCheckpoint(str(best_path), monitor="val_loss", save_best_only=True, mode="min", verbose=1),
            CSVLogger(dirs["histories"] / f"{output_prefix}_{stage}.csv", append=True),
        ]
        logger.info("Training %s %s for up to %d epochs", output_prefix, stage, epochs)
        start = time.time()
        history = model.fit(train_ds, validation_data=validation_ds, epochs=epochs, callbacks=callbacks, verbose=1, class_weight=class_weight)
        logger.info("Completed %s in %.1f seconds", stage, time.time() - start)
        training_history(history, dirs["histories"] / f"{output_prefix}_{stage}.png", stage.replace("_", " ").title())
        if not best_path.exists():
            raise RuntimeError(f"Expected checkpoint was not created: {best_path}")
        return keras.models.load_model(best_path)

    model = run_stage("stage1", training_cfg.stage1_epochs if model_name != "TinyCNN" else 35, training_cfg.stage1_lr, training_cfg.early_stop_stage1, stage1_path)
    if model_name == "TinyCNN":
        return model

    backbone = model.get_layer("backbone")
    backbone.trainable = True
    for layer in backbone.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    model = run_stage("stage2", training_cfg.stage2_epochs, training_cfg.stage2_lr, training_cfg.early_stop_stage2, stage2_path)
    return model


def tune_model_inner(model_name, builder, outer_train_df, seed, outer_fold, config, dirs, store: CheckpointStore, logger):
    from sklearn.model_selection import ParameterGrid
    from .data import make_group_cv
    from .environment import cleanup_tensorflow_memory
    from .models import compile_model
    from tensorflow.keras.callbacks import EarlyStopping

    inner_splits = make_group_cv(outer_train_df, config.inner_folds, seed + 1000 + outer_fold)
    grid = list(ParameterGrid({"dropout": config.dropout_values, "l2_reg": config.l2_values}))
    if config.tuning_max_configs is not None:
        grid = grid[:config.tuning_max_configs]
    records = []

    total_jobs = len(grid) * len(inner_splits)
    completed_jobs = 0
    logger.info("TUNING START | %s | outer=%d | seed=%d | configs=%d | inner_folds=%d | jobs=%d", model_name, outer_fold, seed, len(grid), len(inner_splits), total_jobs)
    for config_index, params in enumerate(grid):
        fold_scores = []
        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_splits, start=1):
            key = f"tuning/{model_name}/outer-{outer_fold}/seed-{seed}/config-{config_index}/inner-{inner_fold}"
            score_path = dirs["predictions"] / f"tuning_{model_name}_outer{outer_fold}_seed{seed}_config{config_index}_inner{inner_fold}.json"
            if store.is_completed(key) and score_path.exists():
                fold_scores.append(float(json.loads(score_path.read_text())["accuracy"]))
                logger.info("RESUME: skipping completed %s", key)
                continue

            cleanup_tensorflow_memory()
            inner_train, inner_val = outer_train_df.iloc[inner_train_idx], outer_train_df.iloc[inner_val_idx]
            train_ds = build_dataset(inner_train.path.values, inner_train.label.values, config.img_size, config.batch_size, True, seed)
            val_ds = build_dataset(inner_val.path.values, inner_val.label.values, config.img_size, config.batch_size, False, seed)
            model = builder(**params)
            if model is None:
                raise RuntimeError(f"Model {model_name} failed to build.")
            try:
                compile_model(model, config.stage1_lr)
                model.fit(train_ds, validation_data=val_ds, epochs=config.tuning_epochs, callbacks=[_backup_callback(dirs["checkpoints"] / "tuning_backup" / model_name / f"outer{outer_fold}_seed{seed}_config{config_index}_inner{inner_fold}"), EarlyStopping(monitor="val_loss", patience=config.tuning_patience, restore_best_weights=True)], verbose=0)
                probabilities = model.predict(val_ds, verbose=0).ravel()
                threshold, _ = find_best_accuracy_threshold(inner_val.label.values, probabilities, config.threshold_min if hasattr(config, 'threshold_min') else 0.05, config.threshold_max if hasattr(config, 'threshold_max') else 0.95, config.threshold_steps if hasattr(config, 'threshold_steps') else 181)
                accuracy = float(calculate_metrics(inner_val.label.values, probabilities, threshold)["accuracy"])
                score_path.write_text(json.dumps({"accuracy": accuracy, "threshold": threshold}), encoding="utf-8")
                store.mark_completed(key, accuracy=accuracy, threshold=threshold, artifact=str(score_path))
                fold_scores.append(accuracy)
                completed_jobs += 1
                logger.info("TUNING %s | config %d/%d | inner %d/%d DONE | accuracy=%.5f | %d/%d jobs", model_name, config_index + 1, len(grid), inner_fold, len(inner_splits), accuracy, completed_jobs, total_jobs)
            except Exception as exc:
                store.mark_failed(key, repr(exc))
                raise
            finally:
                del model
                cleanup_tensorflow_memory()

        records.append({**params, "config_index": config_index, "mean_validation_accuracy": float(np.mean(fold_scores)), "std_validation_accuracy": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0, "fold_accuracies": fold_scores})

    tuning_df = pd.DataFrame(records)
    output = dirs["predictions"] / f"tuning_{model_name}_outer{outer_fold}_seed{seed}.csv"
    tuning_df.to_csv(output, index=False)
    best = tuning_df.sort_values(["mean_validation_accuracy", "std_validation_accuracy"], ascending=[False, True]).iloc[0]
    return {"dropout": float(best.dropout), "l2_reg": float(best.l2_reg)}, tuning_df
