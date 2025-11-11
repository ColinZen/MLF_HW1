#  Data_Loading&Processing

This folder contains all scripts and notebooks used for **data acquisition**, **cleaning**, and **feature processing** in the Machine Learning in Finance (MLF) assignment.

---

##  Contents

| File | Description |
|------|--------------|
| `get_CRSP_data.ipynb` | Downloads and cleans monthly CRSP stock-level data (including return, price, shares outstanding, and delisting return). |
| `data_merged_and_reflect_and_check.ipynb` | Merges firm-level (Compustat) and macroeconomic factors, adds the risk-free rate (from `t_bill.csv`), constructs lag-adjusted variables, and performs leakage validation. |

---

##  Workflow Summary

1. **Data Retrieval**
   - Connects to WRDS (`crsp.msf` and `crsp.msedelist` tables).
   - Downloads monthly return, volume, and price data.
   - Adjusts returns for delisting and saves as CSV.

2. **Data Integration**
   - Merges Compustat firm-level and macroeconomic features.
   - Adds risk-free rate (`rf`) from T-bill data.
   - Generates `datashare_with_macro.parquet`.

3. **Feature Processing**
   - Applies lag rules (annual = +6m, quarterly = +4m, monthly = +0m).
   - Normalizes features cross-sectionally within each month.
   - Produces `features.parquet` for subsequent ML modeling.

4. **Leakage Self-Check**
   - Ensures no look-ahead bias via timestamp validation.
   - Saves check results in the `results/` directory.

---

##  Output Files

| File | Description |
|------|--------------|
| `data/processed/datashare_with_macro.parquet` | Merged firm-level + macro data. |
| `data/processed/features.parquet` | Final ML-ready feature dataset. |
| `results/leakage_report.txt` | Validation report for temporal leakage check. |

---

##  Notes

- All notebooks are designed to be **fully reproducible** in sequence.
- Please ensure `pyarrow`, `fastparquet`, and `wrds` libraries are installed before running.
- Running the full pipeline may take significant time due to WRDS queries and feature construction.

---

---

## 🧩 Dependencies

All notebooks require the following Python libraries:

| Library | Purpose |
|----------|----------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computation |
| `wrds` | WRDS database access |
| `pyarrow` | Parquet file support |
| `matplotlib`, `seaborn` | Data visualization |
| `tqdm` | Progress bar display |
| `scikit-learn` | Machine learning models (GBRT, normalization, etc.) |

To install all dependencies:
```bash
pip install pandas numpy wrds pyarrow matplotlib seaborn tqdm scikit-learn

---

---



