# GPU Fast Research Profile

This build is designed for the RTX A2000 12GB class of GPU while keeping the core research structure intact.

## What was accelerated

- NVIDIA GPU + `mixed_float16` with a float32 classifier output.
- Image augmentation is executed as part of the Keras model instead of a CPU-bound `tf.data.map`, reducing CPU input-pipeline pressure.
- `tf.data` uses parallel decode/resize and prefetch.
- Cluster bootstrap computes the requested metric confidence intervals in one pass per replicate instead of recomputing the entire metric suite separately for every metric. This removes substantial redundant CPU work.
- External evaluation, Grad-CAM, latency, and final reporting have durable checkpoints and can resume independently.
- Tuning is performed once per model/outer fold using the fixed tuning seed and the selected hyperparameters are held constant across the three independent model seeds. This avoids repeating identical hyperparameter searches for each stochastic training seed.

## Accelerated research design

- 5 outer folds.
- 3 inner folds for hyperparameter selection.
- 3 independent model seeds for final evaluation.
- 6 CNN architectures.
- 4 pre-specified hyperparameter combinations: dropout `{0.2, 0.4}` × L2 `{1e-4, 1e-3}`.
- 8 tuning epochs with early stopping.
- 15/20 maximum final training epochs with early stopping.
- 1,000 cluster-bootstrap replicates for confidence intervals.
- External SDNET2018 evaluation uses a threshold selected from CCIC validation only.

This is an explicitly defined accelerated protocol, not a silent change made after looking at test results. For a paper, report the exact configuration and hardware. Do not describe this as bit-for-bit equivalent to the previous 12-configuration/seed-specific tuning protocol.

## Resume behavior

The existing `concrete_crack_research_final` directory may be reused. Completed dataset preparation, tuning folds, final model stages, external evaluation, Grad-CAM, and latency artifacts are checkpointed and skipped when valid artifacts already exist.

If starting a completely independent experiment, use a new `output_dir` in `config/config.json`.
