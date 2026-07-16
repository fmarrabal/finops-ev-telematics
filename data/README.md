# Indexed data

Raw monetary and telemetry records are proprietary. This directory contains the
indexed, aggregated values used in the manuscript. June 2021 is normalized to
100. The full monthly file counts the shared June 2021 intervention boundary
once, giving 51 unique dates. The implementation series retains June 2021 as
month 0 and therefore has 40 rows.

`published_common_fold_predictions.csv` is the exact one-step-ahead prediction
ledger used to compute manuscript Table 5 and the associated Friedman/Wilcoxon
tests. It is archived so the reported numerical comparison remains invariant to
changes in deep-learning libraries and low-level linear algebra backends. Model
implementations and an optional retraining script are also provided.
