# Data Directory

This directory contains the data used in the FinOps cloud cost forecasting analysis.

## Data Description

The case study data is embedded directly in the `finops_forecasting.py` script for reproducibility. The data consists of:

### Pre-FinOps Period (July 2020 - June 2021)
- **Duration**: 12 months
- **Variables**:
  - `cloud_cost_index`: Monthly cloud infrastructure costs (indexed, June 2021 = 100)
  - `vehicles_index`: Connected vehicle count (indexed, June 2021 = 100)

### Post-FinOps Period (June 2021 - September 2024)
- **Duration**: 40 months
- **Variables**:
  - `cloud_cost_index`: Monthly cloud costs after FinOps implementation
  - `vehicles_index`: Connected vehicle count
  - `total_messages_index`: Monthly telematics message throughput
  - `cost_per_billion_msg`: Cost efficiency metric

## Data Privacy

All values are indexed (normalized) to protect proprietary business information while preserving the statistical relationships necessary for research purposes.

## Data Format

```python
# Pre-FinOps data structure
pre_finops = {
    'month_index': [0, 1, ..., 11],
    'date': ['2020-07-01', '2020-08-01', ..., '2021-06-01'],
    'cloud_cost_index': [40.86, 42.02, ..., 100.00],
    'vehicles_index': [35.43, 38.60, ..., 100.00]
}

# Post-FinOps data structure
post_finops = {
    'month_index': [0, 1, ..., 39],
    'date': ['2021-06-01', '2021-07-01', ..., '2024-09-01'],
    'cloud_cost_index': [...],
    'vehicles_index': [...],
    'total_messages_index': [...],
    'cost_per_billion_msg': [...]
}
```

## Reproducing the Analysis

To load the data programmatically:

```python
from finops_forecasting import load_finops_data

pre_finops, post_finops = load_finops_data()
print(pre_finops.head())
print(post_finops.head())
```
