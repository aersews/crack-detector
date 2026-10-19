# Claims and limitations

## Supported
- Metric estimates can be recalculated from the archived labels, probabilities and fixed-threshold predictions.
- The data file has 56,079 unique relative paths and 230 filename-prefix groups.
- The saved predictions show different crack recall estimates for Deck, Pavement and Wall at threshold 0.31.

## Not established
- Independent reproduction of model training or inference.
- Authenticity or integrity of the original model checkpoint.
- A causal explanation for performance differences among surfaces.
- That filename prefixes perfectly identify independent parent photographs.
- State-of-the-art performance, clinical utility, or operational readiness.

The research paper should describe this work as statistical reanalysis of saved predictions. The analysis code checks filename-prefix grouping consistency; it does not prove identity from image pixels or source metadata.
