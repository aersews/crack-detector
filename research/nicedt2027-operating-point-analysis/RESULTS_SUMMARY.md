# Results summary for the manuscript

All numbers below are from the saved-prediction analysis, not a fresh inference run. Overall intervals were calculated with 2,000 percentile-bootstrap replicates, seed 20261019, resampling filename-prefix groups stratified by surface.

## Overall external evaluation

| Metric | Estimate | 95% group-bootstrap CI |
|---|---:|---:|
| Accuracy | 0.8548 | [0.8418, 0.8676] |
| Balanced accuracy | 0.6462 | [0.6306, 0.6636] |
| Precision | 0.5302 | [0.4794, 0.5798] |
| Recall | 0.3472 | [0.3172, 0.3810] |
| Specificity | 0.9452 | [0.9348, 0.9539] |
| F1 | 0.4197 | [0.3901, 0.4507] |
| MCC | 0.3507 | [0.3139, 0.3898] |
| ROC-AUC | 0.7833 | [0.7617, 0.8055] |
| Average precision | 0.4254 | [0.3875, 0.4634] |
| Brier score | 0.1345 | [0.1222, 0.1468] |

Always-uncracked majority-class accuracy is 0.8488, so recorded accuracy is only 0.5992 percentage points higher.

## Surface-specific crack recall at threshold 0.31

| Surface | Patches | Filename-prefix groups | Recall | 95% group-bootstrap CI |
|---|---:|---:|---:|---:|
| Deck | 13,608 | 54 | 0.2809 | [0.2263, 0.3375] |
| Pavement | 24,334 | 104 | 0.6273 | [0.5708, 0.6848] |
| Wall | 18,137 | 72 | 0.1924 | [0.1603, 0.2281] |

**Limit:** IDs are inferred from numeric filename prefixes. The repository checks consistency with this rule, but does not independently verify parent-source photo identity or rerun the model checkpoint.
