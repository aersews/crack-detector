# NICEDT-2027: Concrete-crack reliability analysis

This folder contains analysis code and results for the research paper *Decision-Threshold Transfer and Source-Image-Level Uncertainty in Concrete-Crack Classification: An External Evaluation on SDNET2018*.

## Scope

This is a reanalysis of saved SDNET2018 prediction records at a frozen threshold of 0.31. It does **not** retrain a model or rerun inference. No model checkpoint is included. The cluster identifier is the numeric prefix of the saved relative filename; this is checked for consistency but is not independent proof of parent-photograph identity. The claims and limitations are documented in `docs/CLAIMS_AND_LIMITATIONS.md`.

## Results

The archived prediction table contains 56,079 patches in 230 filename-prefix groups. Accuracy is 85.48%, versus a majority-class baseline of 84.88%. Crack recall at threshold 0.31 is 28.09% for Deck, 62.73% for Pavement and 19.24% for Wall. The `results/` directory contains the full metric tables and interval estimates.

## Reproduce

1. Upload `data/sdnet2018_external_predictions_sanitized.csv` into this folder's `data/` directory. The CSV is a sanitized derived-prediction artifact; it is not the original image dataset. Confirm that sharing derived predictions is compatible with the source dataset terms before public release.
2. Install Python 3.10+ and dependencies: `pip install -r research/nicedt2027-operating-point-analysis/requirements.txt`
3. From the repository root run:

```bash
python research/nicedt2027-operating-point-analysis/src/analyze_predictions.py \
  --input research/nicedt2027-operating-point-analysis/data/sdnet2018_external_predictions_sanitized.csv \
  --output-dir research/nicedt2027-operating-point-analysis/results_recomputed \
  --figure-dir research/nicedt2027-operating-point-analysis/figures_recomputed \
  --bootstrap 2000 --seed 20261019
```

The GitHub Actions workflow smoke-tests the code and tests on each push. Full bootstrap regeneration takes longer than the quick test run.

## Citation

Use the paper's final citation after bibliographic information is known. Until then cite repository path, version/commit, author list and access date; do not invent a DOI. The source dataset is SDNET2018; see https://digitalcommons.usu.edu/all_datasets/48/ and https://doi.org/10.1016/j.dib.2018.11.015.
