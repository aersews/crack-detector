# Concrete Crack Research — GPU Fast, Resumable

A research-oriented CNN comparison pipeline for binary visible concrete-surface crack classification using CCIC with group-aware nested cross-validation and SDNET2018 external evaluation.

## Main improvements in this build

1. **GPU-first training:** automatic NVIDIA GPU detection, memory growth, and mixed-float16; classifier output remains float32.
2. **Faster input path:** decoding/resizing remains in `tf.data`; random augmentation is performed inside each Keras model so CPU augmentation does not bottleneck GPU training.
3. **No silent tuning:** the console reports tuning configuration/fold progress and evaluation progress.
4. **Much less redundant tuning:** each model/outer fold performs one 3-fold inner search using a fixed tuning seed. The resulting hyperparameters are reused across the three final model seeds.
5. **Smaller, pre-specified tuning grid:** 4 combinations (dropout 0.2/0.4 × L2 1e-4/1e-3).
6. **Faster bootstrap:** all requested confidence-interval metrics are computed during the same bootstrap replicate rather than repeating the replicate ten times.
7. **Stage checkpoints:** final training, external evaluation, Grad-CAM, and latency each have durable completion markers.
8. **Safe resume:** completed artifacts are reused; interrupted TensorFlow training can resume from `BackupAndRestore` state.

## Run

```bash
python run.py
```

To inspect state:

```bash
python run.py --show-status
```

To intentionally start over:

```bash
python run.py --reset-progress
```

## Research protocol

The accelerated profile keeps 5 outer folds, 3 inner folds, 3 final model seeds, group-aware splitting, external SDNET2018 evaluation, Grad-CAM, CPU latency, and statistical comparisons. The acceleration comes primarily from removing repeated work and moving augmentation to the GPU execution graph, not from silently dropping evaluation folds or seeds.

The hyperparameter grid is intentionally reduced to four pre-specified configurations. This is a methodological choice and should be disclosed in a manuscript.

## Important reproducibility note

Mixed precision can produce tiny floating-point differences. Hardware, TensorFlow version, precision policy, seeds, and configuration are recorded in `concrete_crack_research_final/manifests/environment.json` and `config.json`. For strict float32 replication, set `use_mixed_precision` to `false`.

## Existing output

You can point this build at the existing `concrete_crack_research_final` directory. It will reuse compatible completed artifacts. The old log may contain errors from previous builds; those are historical entries. Current progress is tracked in `checkpoints/pipeline_state.json` and newly appended log entries.
