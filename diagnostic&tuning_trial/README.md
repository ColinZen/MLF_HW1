🔍 Diagnostics & Tuning Trial
📘 Overview

This module (Diagnostics&tuning_trial.ipynb) is designed to evaluate model performance diagnostics and hyperparameter tuning for machine learning models in asset pricing.
It supports flexible experiments with models such as:

OLS (Ordinary Least Squares)

ElasticNet

Random Forest

GBRT / HistGBRT (Gradient Boosting)

Each model is tested using a rolling-year validation framework (1957–2016) consistent with Gu, Kelly & Xiu (2020).

⚙️ Key Objectives
Goal	Description
🔧 Parameter Diagnostics	Evaluate overfitting and model stability across rolling windows
🧠 Hyperparameter Search	Test grid-search effects for ElasticNet, Random Forest, and GBRT
📊 Model Comparison	Compare out-of-sample 
𝑅
𝑂
𝑂
𝑆
2
R
OOS
2
	​

 performance
🌲 Feature Importance	Visualize top predictors driving model performance
💾 Result Storage	Automatically save yearly and overall metrics to .parquet files
