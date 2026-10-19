# Analysis methods

- **Input columns:** relative_path, label, group_id, surface, probability, prediction, threshold.
- **Frozen operating point:** threshold = 0.31; target labels are not used to choose a new threshold.
- **Metrics:** accuracy, balanced accuracy, precision, recall, specificity, F1, Matthews correlation coefficient, ROC-AUC, average precision, and Brier score.
- **Cluster-bootstrap CI:** percentile interval from 2,000 replicates; resample numeric filename-prefix groups, stratified by surface for the overall interval. Surface-specific intervals resample groups within the surface.
- **Patch comparator:** naive independent-row resampling is provided only to illustrate the effect of an independence assumption; it is not the preferred inferential procedure.
- **Seeds:** main=20261019; surface-specific seeds start at 20261020; patch comparator=20261119.
- **Caveat:** this analysis is conditional on the saved probability, label, and prediction columns. It does not independently verify the checkpoint, inference pipeline, or source-photo matching.
