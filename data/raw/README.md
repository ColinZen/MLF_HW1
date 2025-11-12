# 🗂️ Raw Data Directory — `data/raw/`

## 📘 Overview
This folder contains **raw sample data files** used in the *Machine Learning in Finance (MLF)* homework project.  
These datasets represent the **input layer** of the data pipeline — before feature construction and model training.

All files here are **downsampled versions** of real financial datasets (Compustat, CRSP, FRED, etc.),  
kept small for reproducibility and GitHub storage (<100MB total).  

---

## 📁 File Descriptions

| File | Description | Source |
|------|--------------|---------|
| **`datashare_sample.csv`** | Compustat–CRSP merged firm-level data (monthly). Includes balance sheet and market variables. | Compustat / CRSP |
| **`crsp_monthly_1957_2021_sample.csv`** | Monthly CRSP returns for selected firms (`permno_list.txt`). | CRSP |
| **`macro_factors.parquet`** | Monthly macroeconomic indicators (inflation, term spread, IP growth, etc.). | FRED / Gu, Kelly & Xiu (2020) |
| **`t_bill.csv`** | 1-month Treasury Bill returns — used as the risk-free rate (RF). | Fama-French dataset |
| **`permno_list.txt`** | List of firm identifiers (`permno`) used to subset CRSP and Compustat data. | Derived from Compustat linking table |

---

## 🧩 Data Schema Examples

### `datashare_sample.csv`
| Column | Description | Example |
|---------|--------------|---------|
| `permno` | Firm identifier | 10006 |
| `DATE` | Month-end date | `1980-07-31` |
| `sic2` | 2-digit SIC industry code | 35 |
| `at` | Total assets | 5.23e8 |
| `bm` | Book-to-market ratio | 0.42 |
| `mvel1` | Market capitalization | 2.13e8 |
| `ret` | Monthly stock return | 0.045 |

---

### `macro_factors.parquet`
| Column | Description | Example |
|---------|--------------|---------|
| `month` | Month-end | `1995-03-01` |
| `m_inflation` | Monthly CPI inflation | 0.002 |
| `m_tspread` | Term spread (10Y–3M Treasury) | 0.006 |
| `m_rf` | Risk-free rate | 0.001 |
| `m_ip` | Industrial production growth | 0.004 |

---

### `t_bill.csv`
| Column | Description | Example |
|---------|--------------|---------|
| `caldt` | Month-end date | `1995-03-31` |
| `t30ret` | 1-month Treasury Bill return | 0.0011 |

---

### `permno_list.txt`
Plain-text file listing firm identifiers:


10006
12060
14593
20050
...



---

## ⚙️ Usage Example

```python
import pandas as pd

# Load firm-level and macro data
firm = pd.read_csv("data/raw/datashare_sample.csv")
macro = pd.read_parquet("data/raw/macro_factors.parquet")
rf = pd.read_csv("data/raw/t_bill.csv")

# Merge Compustat–CRSP data with macro factors
firm["DATE"] = pd.to_datetime(firm["DATE"])
macro["month"] = pd.to_datetime(macro["month"])

merged = pd.merge(firm, macro, left_on="DATE", right_on="month", how="left")
print(f"Merged dataset shape: {merged.shape}")
These files serve as inputs for:

merge_firm_and_macro()

add_rf_to_macro()

build_features()
in the data processing pipeline.


Reference

Gu, Kelly, & Xiu (2020). Empirical Asset Pricing via Machine Learning.
Review of Financial Studies, 33(5), 2223–2273.

Notes

All data are for educational and demonstration purposes only.

The sampling preserves structure but not representativeness.

Do not use for empirical research or publication.

Last Updated: 2025-11
Maintainer: Colin Zen (Tsinghua MLF Project)
Version: Sample Data v1.0
