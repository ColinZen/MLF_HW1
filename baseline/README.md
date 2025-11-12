# 📈 Baseline Models

## 📘 Overview

This module implements **linear baseline models** for empirical asset pricing prediction,
replicating the methodology from **Gu, Kelly, and Xiu (2020)** —

> *“Empirical Asset Pricing via Machine Learning,” The Review of Financial Studies, 33(5), 2223–2273.*

The baseline models serve as a benchmark to evaluate the performance gains of non-linear machine learning methods (e.g., Random Forest, GBRT).

| Model                            | Description                                                  |
| -------------------------------- | ------------------------------------------------------------ |
| **OLS (Ordinary Least Squares)** | Linear benchmark using size, book-to-market, and momentum    |
| **ElasticNet**                   | Regularized linear model combining Lasso and Ridge penalties |

---

## 🧩 Files in This Directory

| File                               | Purpose                                                                                   |
| ---------------------------------- | ----------------------------------------------------------------------------------------- |
| `OLS-3.ipynb`                      | Implements 3-factor OLS regression (size, value, momentum) with annual rolling validation |
| `ElasticNet.ipynb` *(coming soon)* | Implements ElasticNet rolling validation with α–λ grid search                             |
| `README.md`                        | This documentation file                                                                   |

---

## ⚙️ Experimental Setup

### 1️⃣ Data Input

All models use the processed dataset:

```
data/processed/features.parquet
```

**Core Columns**

| Column                | Meaning                           |
| --------------------- | --------------------------------- |
| `month`               | Monthly date (YYYY-MM-DD)         |
| `ret_excess_t_plus_1` | Next-month excess return (target) |
| `c_mvel1`             | Size factor                       |
| `c_bm`                | Book-to-market ratio              |
| `c_mom12m`            | Momentum factor                   |

---

### 2️⃣ Rolling-Year Framework

| Dataset          | Years        | Description                           |
| ---------------- | ------------ | ------------------------------------- |
| **Training**     | 1957 – (Y−1) | Used to fit model parameters          |
| **Testing**      | Y            | Out-of-sample evaluation              |
| **Years Tested** | 1987–2016    | 30 years of annual rolling prediction |

---

## 🧮 1. OLS-3 Model

**Notebook:** `OLS-3.ipynb`

### 🔧 Description

Ordinary Least Squares regression using 3 key firm-level predictors:
[
r_{i, t+1} = \beta_0 + \beta_1 \text{Size}*{i,t} + \beta_2 \text{BM}*{i,t} + \beta_3 \text{Mom}*{i,t} + \epsilon*{i,t+1}
]

Each year’s model is trained using all past data and evaluated on year (Y).

### ⚗️ Implementation Steps

1. Load `features.parquet`
2. Standardize features using `StandardScaler`
3. Impute missing values (median)
4. Rolling OLS regression per test year
5. Compute annual and overall (R^2_{OOS})

### 🧠 Key Function

```python
def ols_rolling_regression(data_path, train_start, test_start, test_end, target, features):
    """OLS-3 annual rolling regression"""
```

### 📊 Output

| File                          | Description                     |
| ----------------------------- | ------------------------------- |
| `results/OLS_rolling.parquet` | Annual out-of-sample R² results |
| Visualization                 | Annual and overall R² line plot |

### 📈 Example Plot

```python
plt.plot(df_annual["year"], df_annual["r2_oos"], marker="o", label="OLS-3 Annual R²")
plt.axhline(r2_total, color="red", linestyle="--", label=f"Overall R²={r2_total:.4f}")
```

---

## 🧩 2. ElasticNet Model (Coming Soon)

**Notebook:** `ElasticNet.ipynb`

### 🔧 Description

ElasticNet regression combines the **Lasso (L1)** and **Ridge (L2)** penalties:
[
\min_\beta |y - X\beta|_2^2 + \alpha\left[(1 - \lambda)|\beta|_2^2 + \lambda|\beta|_1\right]
]

This model handles multicollinearity and performs feature selection automatically.

### 🔍 Parameters

| Parameter  | Meaning                                                |
| ---------- | ------------------------------------------------------ |
| `alpha`    | Regularization strength                                |
| `l1_ratio` | Balance between Lasso and Ridge (0 = Ridge, 1 = Lasso) |

### ⚗️ Rolling Implementation

* 12-year expanding training-validation window
* Grid search over:

  * α ∈ [1e−4, 1e−3, 1e−2, 1e−1]
  * l₁_ratio ∈ [0.1, 0.5, 0.9]
* Validation selects best parameters per year
* Test on year (Y)

### 📊 Outputs

| File                                     | Description                           |
| ---------------------------------------- | ------------------------------------- |
| `results/ElasticNet_best_params.parquet` | Annual best (α, l₁) values            |
| `results/ElasticNet_rolling.parquet`     | Annual and overall R² results         |
| Visualization                            | Annual R² + parameter evolution plots |

---

## 📈 Evaluation Metric

**Out-of-Sample (R^2_{OOS})** — from Gu, Kelly, & Xiu (2020):

[
R^2_{OOS} = 1 - \frac{\sum_t (y_t - \hat{y}_t)^2}{\sum_t y_t^2}
]

Implemented as:

```python
def r2_oos(y_true, y_pred):
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    y, yp = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
    rss, tss = np.sum((y - yp)**2), np.sum(y**2)
    return 1 - rss / tss
```

---

## 📦 Dependencies

| Library                 | Function                                               |
| ----------------------- | ------------------------------------------------------ |
| `pandas`, `numpy`       | Data loading and computation                           |
| `matplotlib`, `seaborn` | Visualization                                          |
| `scikit-learn`          | Modeling (LinearRegression, ElasticNet, preprocessing) |
| `tqdm`                  | Progress tracking                                      |
| `pyarrow`               | Parquet file operations                                |

---

## 🚀 How to Run

### ▶️ Run OLS-3

```bash
jupyter notebook baseline/OLS-3.ipynb
```

### ▶️ Run ElasticNet (once added)

```bash
jupyter notebook baseline/ElasticNet.ipynb
```

All results will be saved automatically to the `results/` directory.

---

## 📊 Example Output (OLS-3)

| Year | R²_oos |
| ---- | -----: |
| 1987 | 0.0021 |
| 1990 | 0.0057 |
| 2001 | 0.0113 |
| 2016 | 0.0084 |

**Overall Out-of-Sample R²:** `0.0079`

---

## 📚 Reference

> Gu, S., Kelly, B., & Xiu, D. (2020).
> *Empirical Asset Pricing via Machine Learning.*
> *The Review of Financial Studies*, 33(5), 2223–2273.

---

## ✍️ Author

**ColinZen**
Tsinghua University SIGS — M.Fin (FinTech)
Focus: *Machine Learning in Empirical Asset Pricing*
📍 *MLF_HW1 — Baseline Linear Models Module*



