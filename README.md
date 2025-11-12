# 📘 MLF_HW1

**Replication of Gu, S., Kelly, B., & Xiu, D. (2020).
“Empirical Asset Pricing via Machine Learning.”
*The Review of Financial Studies*, 33(5), 2223–2273.**

---

## 🧭 Overview

This repository implements a **complete replication pipeline** of the asset pricing machine learning framework proposed by **Gu, Kelly, and Xiu (2020)**.

The goal is to replicate their **rolling-year predictive regressions** and **cross-sectional forecasting experiments**, comparing traditional linear models with non-linear machine learning methods:

| Model Type        | Methods Implemented                                |
| ----------------- | -------------------------------------------------- |
| Linear Models     | OLS, ElasticNet                                    |
| Tree-based Models | Random Forest, Gradient Boosting (GBRT / HistGBRT) |

All experiments follow the same **rolling validation** logic:

* Expanding training window (1957–Y−13)
* Validation window (Y−12 ~ Y−1)
* Test year (Y)
* Out-of-sample R² calculated via Gu–Kelly–Xiu Eq. (19)

---

## 📂 Repository Structure

```
MLF_HW1/
│
├── README.md                      # Project documentation (this file)
│
├── data/                          # Dataset storage
│   ├── raw/                       # Raw inputs (CRSP, macro factors, t-bill, etc.)
│   └── processed/                 # Feature-engineered data (features.parquet, etc.)
│
├── data_downloading&processing/   # Scripts for data cleaning, merging, feature scaling
│
├── baseline/                      # Linear models (OLS, ElasticNet)
│
├── decision_tree/                 # Tree-based models (RF, GBRT)
│
├── diagnostic&tuning_trial/       # Parameter tuning, diagnostics, and visualization
│
└── results/                       # Saved model results and parameters
```

---

## ⚙️ Workflow

### 1️⃣ Data Preparation

**Script path:**
`data_downloading&processing/`

**Files involved:**

* `crsp_monthly_1957_2021_sample.csv`
* `datashare_sample.csv`
* `macro_factors.parquet`
* `t_bill.csv`

**Main steps:**

1. Merge CRSP firm-level data with macro factors.
2. Generate lagged predictors (`c_`, `m_`, `sic_` prefixed).
3. Rank-normalize features cross-sectionally.
4. Save processed dataset to:

   ```
   data/processed/features.parquet
   ```

---

### 2️⃣ Baseline Models (OLS & ElasticNet)

**Folder:** `baseline/`

#### 💡 OLS Rolling Regression

* Function: `ols_rolling_regression()`
* Features: `c_mvel1`, `c_bm`, `c_mom12m`
* Outputs:

  * Annual & overall R² plots
  * File: `results/OLS_rolling.parquet`

#### 💡 ElasticNet Rolling Validation

* Function: `elasticnet_rolling_validation()`
* Grid search:
  α ∈ [1e-4, 1e-3, 1e-2, 1e-1]
  l₁_ratio ∈ [0.1, 0.5, 0.9]
* Saves:

  * `results/ElasticNet_best_params.parquet`
  * Annual R² trend plot + feature importance

---

### 3️⃣ Tree-Based Models

**Folder:** `decision_tree/`

#### 🌲 Random Forest — Rolling Validation

* Function: `random_forest_rolling_validation()`
* Grid search parameters:

  * Depth ∈ [3, 5, 7]
  * Trees ∈ [100, 300, 500]
  * Max features ∈ [3, 5, 13]
* Outputs:

  * `results/RF_rolling_opt.parquet`
  * Annual R² plot
  * Feature importance bar chart

#### 🌳 GBRT — Low-Memory Rolling Validation

* Function: `gbrt_rolling_validation()`
* Adaptive mode:

  * Uses `GradientBoostingRegressor` or `HistGradientBoostingRegressor` depending on available memory
* Parameters:

  * Depth ∈ [2, 3, 5]
  * Learning rate ∈ [0.05, 0.1]
  * Trees ∈ [100, 300, 500]
* Outputs:

  * `results/GBRT_rolling_lowmem.parquet`
  * `results/GBRT_best_params_lowmem.parquet`
  * R² plots & feature importance visualization

---

### 4️⃣ Diagnostic & Tuning Trial

**Folder:** `diagnostic&tuning_trial/`

#### 🧪 File: `Diagnostics&tuning_trial.ipynb`

Purpose:

* Aggregate model diagnostics and tuning results.
* Compare yearly (R^2_{OOS}) across models.
* Visualize parameter evolution and feature importances.

Outputs:

* Multi-model comparison plots.
* Statistical summary of all algorithms.

---

### 5️⃣ Results Visualization

**Rolling-year R² Example:**

```python
plt.plot(df_results["year"], df_results["test_r2"], marker="o", label="Test R²")
plt.axhline(overall_r2, color="red", linestyle="--", label=f"Overall={overall_r2:.4f}")
plt.title("Random Forest — Annual Out-of-Sample R²")
```

**Feature Importance Example:**

```python
sns.barplot(
    x=avg_imp[top_idx],
    y=np.array(feature_cols)[top_idx],
    palette="viridis",
    orient="h"
)
```

---

## 📦 Dependencies

| Library                 | Purpose                                        |
| ----------------------- | ---------------------------------------------- |
| `pandas`                | Data loading and manipulation                  |
| `numpy`                 | Numerical computation                          |
| `matplotlib`, `seaborn` | Visualization                                  |
| `tqdm`                  | Progress bars for rolling validation           |
| `scikit-learn`          | Core ML algorithms (OLS, ElasticNet, RF, GBRT) |
| `psutil`                | Memory monitoring for adaptive GBRT            |
| `pyarrow`               | Parquet file reading/writing                   |

> 🧩 All models follow scikit-learn syntax and can be easily extended to XGBoost or LightGBM.

---

## 📈 Output Files

| File                              | Description                           |
| --------------------------------- | ------------------------------------- |
| `OLS_rolling.parquet`             | OLS rolling regression results        |
| `ElasticNet_best_params.parquet`  | ElasticNet best α and l1_ratio        |
| `RF_rolling_opt.parquet`          | Random Forest tuned results           |
| `GBRT_best_params_lowmem.parquet` | Best yearly parameters for GBRT       |
| `GBRT_refit_from_params.parquet`  | GBRT retrained using saved parameters |

---

## 🔬 Evaluation Metric: Out-of-Sample R²

[
R^2_{OOS} = 1 - \frac{\sum_t (y_t - \hat{y}_t)^2}{\sum_t y_t^2}
]

* Evaluated annually and cumulatively.
* A positive (R^2_{OOS}) indicates predictive power beyond the mean model.

---

## 🚀 Quick Start

```bash
# 1. Create environment
conda create -n mlf_hw1 python=3.10
conda activate mlf_hw1

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn tqdm scikit-learn psutil pyarrow

# 3. Run rolling validations
python baseline/ols_rolling.py
python baseline/elasticnet_rolling.py
python decision_tree/random_forest_rolling.py
python decision_tree/gbrt_rolling.py

# 4. Compare results
jupyter notebook diagnostic&tuning_trial/Diagnostics&tuning_trial.ipynb
```

---

## 🧠 Reference

> Gu, S., Kelly, B., & Xiu, D. (2020).
> *Empirical Asset Pricing via Machine Learning.*
> *The Review of Financial Studies*, 33(5), 2223–2273.
> [https://doi.org/10.1093/rfs/hhaa009](https://doi.org/10.1093/rfs/hhaa009)

---

## 🧾 Author

**ColinZen**
*Tsinghua University SIGS — Master in Finance (FinTech)*
Focus: Machine Learning in Empirical Asset Pricing
📍 *MLF_HW1: Empirical Asset Pricing Model Diagnostics*

---




