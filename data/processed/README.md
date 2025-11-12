# 📊 MLF_HW1 — Sample Dataset Documentation

## 🧾 Overview
This repository provides **lightweight sample data** used for the *Machine Learning in Finance* course homework (HW1).  
The data files here are extracted from the full research dataset to comply with GitHub's file size limits (<100 MB).

Each sample retains the **core structure and representative variables** of the original datasets:
- `datashare_with_macro.parquet` (firm-level fundamentals + macroeconomic factors)
- `features.parquet` (final normalized feature matrix for model estimation)

The sample versions can be used for:
- Demonstrating data preprocessing pipelines  
- Reproducing model structures (e.g., OLS, ElasticNet, GBRT)  
- Sharing reproducible workflows on GitHub  

---

## 📁 Files in This Repository

| File | Description | Original Source |
|------|--------------|-----------------|
| `datashare_with_macro_sample.parquet` | Merged **Compustat–CRSP firm data** with **macro factors** (sampled subset). Each row represents one firm-month observation. | `data/processed/datashare_with_macro.parquet` |
| `features_sample.parquet` | Processed monthly **cross-sectional feature matrix** after normalization and lagging adjustments. Contains both firm-level and macro variables used in factor model estimation. | `data/processed/features.parquet` |

---

## 📐 Data Structure

### 1️⃣ `datashare_with_macro_sample.parquet`
| Column | Description | Example |
|---------|--------------|---------|
| `permno` | Firm identifier | 10000 |
| `DATE` | Observation month | `1980-07-31` |
| `sic2` | 2-digit industry code | 35 |
| `c_bm` | Book-to-market ratio | 0.55 |
| `c_mvel1` | Market equity (size factor) | 2.43e8 |
| `m_inflation` | Monthly CPI inflation rate | 0.002 |
| `m_rf` | Risk-free rate (monthly) | 0.0012 |
| ... | ... | ... |

> 🧩 Columns prefixed with:
> - **`c_`** → firm-level characteristics  
> - **`m_`** → macroeconomic indicators  

---

### 2️⃣ `features_sample.parquet`
| Column | Description | Example |
|---------|--------------|---------|
| `permno` | Firm identifier | 14593 |
| `month` | Observation month | `1995-02-01` |
| `ret_excess_t_plus_1` | Next-month excess return (target) | 0.0156 |
| `c_mvel1` | Size factor | -0.41 |
| `c_bm` | Book-to-market ratio | 0.63 |
| `c_mom12m` | Momentum (12-month) | 1.21 |
| `m_tspread` | Term spread (macro factor) | 0.005 |
| ... | ... | ... |

> All features have been **rank-normalized cross-sectionally by month** to remove scale effects.

---

## ⚙️ How to Use

```python
import pandas as pd

# Load datasets
macro_df = pd.read_parquet("datashare_with_macro_sample.parquet")
features_df = pd.read_parquet("features_sample.parquet")

# Preview structure
print(macro_df.head())
print(features_df.describe())

