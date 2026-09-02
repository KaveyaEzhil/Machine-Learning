# ITA0613 Machine Learning Assignment: Climate-Resilient Crop Yield Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Build Status](https://github.com/KaveyaEzhil/Machine-Learning/actions/workflows/ci.yml/badge.svg)](https://github.com/KaveyaEzhil/Machine-Learning/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![SDG 2: Zero Hunger](https://img.shields.io/badge/SDG-2_Zero_Hunger-orange.svg)](https://sdgs.un.org/goals/goal2)
[![SDG 13: Climate Action](https://img.shields.io/badge/SDG-13_Climate_Action-green.svg)](https://sdgs.un.org/goals/goal13)

**Course Code**: ITA0613 - Machine Learning  
**Assignment Title**: Large-Scale Instance-Based and Statistical Learning Pipeline for Climate-Resilient Crop Yield Forecasting  
**Author**: Kaveya E (`IET Assignment kaveya E`)  
**Repository**: [https://github.com/KaveyaEzhil/Machine-Learning](https://github.com/KaveyaEzhil/Machine-Learning)  

---

## 📌 Project Overview
This repository contains a complete, first-principles **Instance-Based and Statistical Learning Pipeline** designed to predict regional crop yields under extreme weather conditions. Built strictly using **only NumPy and Pandas**—without `scikit-learn` or external machine learning libraries for core algorithms—this project implements:

1. **Data Engineering & EDA**: Multi-region/multi-year dataset merging, district-level missing weather record imputation, and derived feature engineering (**Growing Degree Days**, **Rainfall Anomaly Index**, **Temperature-Humidity Index**, **Soil Nutrient Index**).
2. **k-NN Regressor from First Principles**: Vectorized **Euclidean** and **Mahalanobis** distance metrics, distance weighting, and a manually computed validation curve for optimal $k$ selection.
3. **Locally Weighted Regression (LWR / LOESS)**: Gaussian weighting kernel solver with L2 regularization, bandwidth tuning ($\tau$), and theoretical/empirical bias-variance trade-off comparison against k-NN.
4. **Candidate-Elimination & Version Space Reasoning**: Discrete domain partitioning, boundary hypothesis updates ($S$ and $G$), and inductive bias analysis.
5. **Scalability Analysis & k-d Tree Indexing Prototype**: Benchmark complexity scaling up to 1,000,000 records ($25\times$ speedup using a first-principles **k-d Tree** nearest neighbour search).
6. **Publication Visualizations & Agricultural Policy Brief**: 5 publication-quality Matplotlib figures and a data-driven policy brief supporting **SDG 2 (Zero Hunger)** and **SDG 13 (Climate Action)**.

---

## 🏗️ Repository Architecture

```
Machine-Learning/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI workflow for automated testing
├── data/
│   ├── raw/                       # Multi-region yearly CSV raw agro-climatic files (>12,500 records)
│   └── processed/                 # Merged, imputed, and feature-engineered dataset
├── src/
│   ├── __init__.py
│   ├── generate_dataset.py        # Raw multi-zone agro-climatic dataset generator
│   ├── data_pipeline.py           # Cleaning, missing weather imputation & derived features
│   ├── knn_regressor.py           # First-principles KNN (Euclidean & Mahalanobis)
│   ├── lwr_regressor.py           # First-principles Locally Weighted Regression (LWR)
│   ├── candidate_elimination.py   # Candidate-Elimination algorithm & Version Space
│   ├── scalability_tree.py        # First-principles k-d Tree spatial index prototype
│   └── visualization.py           # 5 publication-quality figures renderer
├── tests/
│   ├── test_data_pipeline.py      # Unit tests for data cleaning & feature engineering
│   ├── test_knn.py                # Unit tests for distance metrics & KNN predictions
│   ├── test_lwr.py                # Unit tests for Gaussian kernel & weighted least squares
│   ├── test_candidate_elimination.py # Unit tests for Version Space boundary updates
│   └── test_scalability.py        # Unit tests for k-d tree nearest neighbour search
├── results/
│   ├── plots/                     # 5 high-resolution publication figures (PNG)
│   │   ├── fig1_agro_climatic_correlation_heatmap.png
│   │   ├── fig2_knn_validation_curve.png
│   │   ├── fig3_lwr_vs_knn_bias_variance.png
│   │   ├── fig4_scalability_benchmark.png
│   │   └── fig5_climate_yield_risk_distribution.png
│   └── tables/                    # Metrics CSV & JSON result tables
│       ├── knn_validation_curve_metrics.csv
│       ├── lwr_bandwidth_tradeoff_metrics.csv
│       ├── scalability_benchmark_metrics.csv
│       └── candidate_elimination_hypotheses.json
├── docs/
│   ├── TECHNICAL_REPORT.md        # Mathematical derivations, bias-variance analysis & report
│   └── POLICY_BRIEF.md            # Standalone executive summary for agricultural policymakers
├── main.py                        # Master pipeline execution script
├── run_tests.py                   # Automated unit test suite runner
├── requirements.txt               # Dependencies list (NumPy, Pandas, Matplotlib, Pytest)
└── README.md                      # Project documentation and setup guide
```

---

## ⚡ Quick Start & Reproduction Instructions

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Clone the Repository
```bash
git clone https://github.com/KaveyaEzhil/Machine-Learning.git
cd Machine-Learning
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Automated Unit Tests
```bash
python run_tests.py
```

### 5. Execute End-to-End Learning Pipeline
```bash
python main.py
```
This single command runs raw dataset generation, data engineering, k-NN validation, LWR bandwidth tuning, Candidate Elimination boundary computing, k-d tree scalability benchmarking, and exports all plots to `results/plots/` and metric tables to `results/tables/`.

---

## 📊 Summary of Results & Key Visualizations

### 1. k-NN Regressor Distance Metric Comparison
- **Euclidean Distance**: Validation RMSE = $1040.26\text{ kg/ha}$, Test MAE = $854.85\text{ kg/ha}$ ($k=51$)
- **Mahalanobis Distance**: Validation RMSE = $1038.10\text{ kg/ha}$, Test MAE = $854.02\text{ kg/ha}$ ($k=51$)

### 2. Locally Weighted Regression (LWR)
- **Optimal Bandwidth ($\tau$)**: $\tau = 0.50$, Test RMSE = $1116.33\text{ kg/ha}$, Test MAE = $930.78\text{ kg/ha}$

### 3. Scalability Speedup ($N = 1,000,000$ Records)
- **Brute-Force KNN Query Time**: $2.1716\text{ seconds}$
- **First-Principles k-d Tree Query Time**: $0.0869\text{ seconds}$
- **Speedup Factor**: **$25.00\times$ faster**

---

## 📜 Assessment Rubrics Alignment

| Rubric Criterion | Max Marks | Implementation Highlights |
|---|---|---|
| **Data Engineering & EDA (CO7)** | 15 | Merged 5 regional files (>12,500 records); imputed weather NaN values using district medians; engineered GDD, RAI, THI, SNI derived features using pure NumPy/Pandas. |
| **k-NN Regressor & Metrics (CO6)** | 20 | Built from scratch; implemented Euclidean & Mahalanobis metrics; computed manual validation curve over $k \in [1, 51]$. |
| **LWR & Bias-Variance (CO6)** | 15 | Built from scratch; derived Gaussian weighting kernel $w_i = \exp(-\|x_i - x\|^2 / 2\tau^2)$; theoretical & empirical bias-variance curve vs k-NN. |
| **Candidate-Elimination (CO1/CO2)** | 15 | Quantile discretization; computed $S$ and $G$ boundary hypotheses from scratch; detailed inductive bias analysis. |
| **Scalability & Prototype (CO6/CO7)** | 15 | Measured time/memory scaling up to $10^6$ records; built first-principles k-d Tree yielding $25\times$ query speedup. |
| **Visualizations & Policy Brief (CO7)** | 10 | Generated 5 publication-quality Matplotlib figures; wrote standalone Agricultural Policy Brief for SDG 2 & 13. |
| **Repo Quality, Docs & CI** | 10 | Modular directory structure; incremental commit history; automated test suite (`run_tests.py`); working GitHub Actions CI workflow (`ci.yml`). |

---

## 📄 License
This project is released under the [MIT License](LICENSE).
