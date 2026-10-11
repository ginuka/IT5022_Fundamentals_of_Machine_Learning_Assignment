# IT5022 — Fundamentals of Machine Learning Programming Assignment
## Real-World Credit Risk Scoring & Customer Financial Segmentation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![License: Academic](https://img.shields.io/badge/License-Academic%20Use%20Only-green.svg)](#)

---

## 📌 Project Overview
This repository contains the complete implementation of the **IT5022: Fundamentals of Machine Learning** MSc group assignment. The project tackles a real-world financial risk problem: **automated credit risk evaluation and unlabelled customer segmentation** using a dataset of 8,692 retail banking customers across 19 financial and behavioral indicators.

In accordance with the module rubric, we implement:
- **Two Distinct Supervised Learning Algorithms:**
  1. *Parametric Baseline:* Regularized Logistic Regression (L1/L2 penalties, cost-sensitive class balancing, decision threshold optimization via PR curves).
  2. *Non-linear Ensemble:* Random Forest Classifier (Grid/Randomized CV hyperparameter optimization, Gini impurity reduction, permutation feature importance).
- **Two Distinct Unsupervised Learning / Clustering Algorithms:**
  1. *Centroid-based Partitioning:* K-Means Clustering with Within-Cluster Sum of Squares (Inertia/Elbow method), Silhouette analysis ($[-1, +1]$), and cluster persona profiling.
  2. *Density-based Spatial Clustering:* DBSCAN with $k$-distance $\epsilon$ optimization for financial anomaly/noise detection and dense risk cluster identification.

---

## 📂 Repository Structure
```
IT5022_Fundamentals_of_Machine_Learning_Assignment/
│
├── dataset/
│   ├── dataset.xlsx             # Original raw Excel dataset (8,692 × 19)
│   └── dataset.csv              # Fast loading CSV mirror
│
├── notebooks/
│   ├── 01_eda_and_data_audit.ipynb                  # Statistical EDA, Distributions, ANOVA & Chi-Square tests
│   ├── 02_preprocessing_and_feature_engineering.ipynb # Leakage-safe pipeline, custom transformers, scalers
│   ├── 03_supervised_learning_classification.ipynb   # Logistic Regression vs Random Forest, PR-AUC, ROC-AUC
│   └── 04_unsupervised_learning_clustering.ipynb     # K-Means & DBSCAN clustering, Silhouette, Persona profiling
│
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py         # Modular scikit-learn transformers & pipeline builders
│   └── evaluation_helpers.py    # Evaluation metric visualizers (Confusion Matrix, PR/ROC curves)
│
├── reports/
│   ├── figures/                 # Exported high-resolution figures for academic report
│   └── IT5022_Credit_Risk_Report.pdf  # Final MSc academic report with source code appendix
│
├── members.txt                  # Group member student IDs and emails
├── submission.txt               # Dataset, GitHub, and YouTube video presentation links
├── requirements.txt             # Environment dependencies
├── ASSIGNMENT_IMPLEMENTATION_PLAN.md # Detailed MSc execution roadmap
└── README.md                    # Project documentation (this file)
```

---

## ⚙️ Installation & Environment Setup
Clone this repository and set up a virtual environment:

```bash
git clone https://github.com/ginuka/IT5022_Fundamentals_of_Machine_Learning_Assignment.git
cd IT5022_Fundamentals_of_Machine_Learning_Assignment

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 📊 Methodology & Key Findings
1. **Data Preprocessing & Zero Leakage:**
   - 80/20 Stratified train-test split locked before any feature engineering.
   - Domain imputation: missing `Monthly_Income` recovered via $\text{Annual\_Income} / 12$.
   - Numerical extraction: regex-extracted clean integer years from `Credit_History_Age`.
   - Outlier mitigation and standardization using `StandardScaler` fitted strictly on training folds.
2. **Supervised Classification:**
   - Evaluated under class imbalance (59.7% Poor vs 40.3% Good).
   - Tuned decision thresholds using Precision-Recall trade-offs to minimize high-cost False Negatives (unidentified high-risk borrowers).
3. **Unsupervised Customer Profiling:**
   - Evaluated clusters using the Elbow Method and Silhouette coefficients.
   - Discovered distinct behavioral segments: *Prime Savers*, *Overleveraged Revolvers*, and *Credit Inactives*.

---

## 👥 Group Contributions (Team Size: 2)
* **Member 1 (Ginuka - MS26922898):**
  - Data ingestion, regex feature engineering, domain imputation transformers.
  - Model 1 implementation (Regularized Logistic Regression, threshold tuning).
  - Evaluation metric implementations (PR-AUC, cost-sensitive confusion matrices).
  - Methodology and Supervised Learning report chapters; video presentation Part A.
* **Member 2:**
  - Statistical testing (ANOVA F-test, $\chi^2$ independence), correlation heatmaps.
  - Model 2 implementation (Random Forest ensemble, hyperparameter search, feature importances).
  - Unsupervised clustering (K-Means silhouette analysis, DBSCAN density tuning).
  - Critical discussion, ethical credit scoring review; video presentation Part B.

---

## 📄 License & Academic Integrity
This repository is submitted for the Master of Science in Information Technology (IT5022: Fundamentals of Machine Learning). All materials adhere strictly to university academic integrity and plagiarism guidelines.
