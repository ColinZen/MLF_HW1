# 🌲 Decision Tree Models

## 📘 Overview

This module implements **tree-based models** for empirical asset pricing prediction, following the methodology of
**Gu, Kelly, and Xiu (2020)** — *“Empirical Asset Pricing via Machine Learning”*.

It includes two major models:

| Model                                        | Description                                                             |
| -------------------------------------------- | ----------------------------------------------------------------------- |
| **Random Forest (RF)**                       | Ensemble of decision trees using bagging and random feature subsampling |
| **Gradient Boosting Regression Tree (GBRT)** | Sequential boosting of shallow trees minimizing squared loss            |

Both models are evaluated under the **rolling-year validation framework** (1957–2016) to measure predictive power out-of-sample.

---

## 🧩 Files in This Directory

| File                  | Purpose                                                                                      |
| --------------------- | -------------------------------------------------------------------------------------------- |
| `Random_Forest.ipynb` | Implements annual rolling validation for Random Forests with grid search                     |
| `GBRT.ipynb`          | Implements Gradient Boosting Regression Trees (GBRT / HistGBRT) with adaptive memory control |
| `README.md`           | Documentation for this module                                                                |

---

## ⚙️ Experimental Design

### 1️⃣ Rolling-Year Validation Scheme

For each test year ( Y \in [1987, 2016] ):

| Dataset    | Period      | Purpose                              |
| ---------- | ----------- | ------------------------------------ |
| Training   | 1957 – Y−13 | Model fitting                        |
| Validation | Y−12 – Y−1  | Parameter tuning                     |
| Testing    | Y           | Out-of-sample performance evaluation |

### 2️⃣ Evaluation Metric

All models report **Out-of-Sample (R^2_{OOS})** (Gu, Kelly & Xiu, 2020, Eq. 19):

[
R^2_{OOS} = 1 - \frac{\sum_t (y_t - \hat{y}_t)^2}{\sum_t y_t^2}
]

A positive (R^2_{OOS}) indicates predictive skill beyond the unconditional mean model.

---

## 🌲 Random Forest Module

**Notebook:** `Random_Forest.ipynb`

### 🔧 Model Details

* Library: `sklearn.ensemble.RandomForestRegressor`
* Parameters tuned:

  * `max_depth ∈ [3, 5, 7]`
  * `n_estimators ∈ [100, 300, 500]`
  * `max_features ∈ [3, 5, 13]`
* Leaf size fixed at 50, `max_samples=0.7`, `bootstrap=True`

### 📈 Outputs

* **Rolling-year test R² plot**
* **Average feature importance (impurity-based)**
* **Result file:**

  ```
  results/RF_rolling_opt.parquet
  ```

### 🧠 Example Output

```python
[1993] depth=5, trees=300, features=5, ValR²=0.0123, TestR²=0.0079
[2004] depth=7, trees=500, features=13, ValR²=0.0215, TestR²=0.0152
```

---

## ⚡ GBRT (Gradient Boosting Regression Tree) Module

**Notebook:** `GBRT.ipynb`

### 🔧 Model Details

* Library:

  * `sklearn.ensemble.GradientBoostingRegressor` (default)
  * `sklearn.ensemble.HistGradientBoostingRegressor` (auto-switch when RAM < 8GB)
* Parameters tuned:

  * `max_depth ∈ [2, 3, 5]`
  * `learning_rate ∈ [0.05, 0.1]`
  * `n_estimators ∈ [100, 300, 500]`
* Subsampling: 0.8
* `max_features = 50`
* Automatic downsampling if `len(X_train) > 200,000`

### 💾 Outputs

| File                                      | Description                                     |
| ----------------------------------------- | ----------------------------------------------- |
| `results/GBRT_rolling_lowmem.parquet`     | Annual results of GBRT rolling validation       |
| `results/GBRT_best_params_lowmem.parquet` | Best parameter configuration per year           |
| `results/GBRT_refit_from_params.parquet`  | Re-trained results using stored best parameters |

### 📊 Plots

* Annual rolling **Train vs Test R²** comparison
* **Average Feature Importances (Top 15)**
* Optional **parameter evolution plots** (depth / learning rate / trees)

---

## 📊 Visualization Examples

### 📈 Annual R² Plot

```python
plt.plot(df_results["year"], df_results["test_r2"], marker="o", label="Test R²")
plt.axhline(overall_r2, color="red", linestyle="--", alpha=0.7)
plt.title("Random Forest — Annual Out-of-Sample R²")
plt.xlabel("Year"); plt.ylabel("R²_oos")
```

### 🌿 Feature Importance Plot

```python
sns.barplot(
    x=avg_imp[top_idx],
    y=np.array(feature_cols)[top_idx],
    palette="viridis",
    orient="h"
)
plt.title("GBRT — Average Feature Importances")
```

---

## 🔬 Output Interpretation

| Metric       | Meaning                                  |
| ------------ | ---------------------------------------- |
| `train_r2`   | In-sample fit quality                    |
| `val_r2`     | Validation-year performance (for tuning) |
| `test_r2`    | True out-of-sample predictive power      |
| `overall_r2` | Global performance across all test years |

A **higher and stable annual R²** indicates better generalization across economic cycles.

---

## 📦 Dependencies

| Library                 | Function                                     |
| ----------------------- | -------------------------------------------- |
| `pandas`                | Data manipulation & storage                  |
| `numpy`                 | Numerical computation                        |
| `matplotlib`, `seaborn` | Visualization                                |
| `tqdm`                  | Progress tracking for yearly loops           |
| `scikit-learn`          | Machine learning algorithms                  |
| `psutil`                | Dynamic memory detection for low-memory GBRT |
| `pyarrow`               | Efficient Parquet serialization              |

---

## 🚀 Usage Guide

### 1️⃣ Run Random Forest

```bash
jupyter notebook decision_tree/Random_Forest.ipynb
```

### 2️⃣ Run GBRT (auto low-memory mode)

```bash
jupyter notebook decision_tree/GBRT.ipynb
```

### 3️⃣ Results Location

```
results/
 ├── RF_rolling_opt.parquet
 ├── GBRT_rolling_lowmem.parquet
 ├── GBRT_best_params_lowmem.parquet
 └── GBRT_refit_from_params.parquet
```

---

## 📚 Reference

> **Gu, S., Kelly, B., & Xiu, D. (2020)**
> *Empirical Asset Pricing via Machine Learning.*
> *The Review of Financial Studies*, 33(5), 2223–2273.

---

## ✍️ Author

**ColinZen**
Tsinghua University — M.Fin (FinTech)
Focus: *Machine Learning & Empirical Asset Pricing*
📍 Repository: *MLF_HW1 — Decision Tree Models Module*

---

