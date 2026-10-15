# IT5022 — Fundamentals of Machine Learning Programming Assignment
## Real-World Credit Risk Scoring via Supervised Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![License: CC0 Public Domain](https://img.shields.io/badge/License-CC0%20Public%20Domain-green.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

---

## 📌 Project Overview
This repository contains the complete implementation of the **IT5022: Fundamentals of Machine Learning** MSc programming assignment. The project tackles a critical financial risk problem: **automated credit risk evaluation and loan default prediction** using a dataset of 8,692 retail banking customers across 19 financial, behavioral, and historical indicators.

In accordance with the module rubric, we implement:
- **Two Distinct Supervised Learning Algorithms:**
  1. *Parametric Baseline:* Regularized Logistic Regression (evaluating L1 and L2 penalties, out-of-fold decision-threshold tuning via `cross_val_predict`, PR curves, and probability calibration).
  2. *Non-linear Ensemble:* Random Forest Classifier (GridSearchCV 5-fold cross-validation, Gini impurity reduction feature importance, generalization gap and overfitting diagnostics).

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
│   └── 03_supervised_learning_classification.ipynb   # Logistic Regression vs Random Forest, PR-AUC, ROC-AUC
│
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py         # Modular scikit-learn transformers & pipeline builders
│   └── evaluation_helpers.py    # Evaluation metric visualizers (Confusion Matrix, PR/ROC curves)
│
├── reports/
│   ├── figures/                 # Exported high-resolution figures for academic report
│   ├── IT5022_Credit_Risk_Report.md   # Final MSc academic report source with full code appendix
│   └── IT5022_Credit_Risk_Report.pdf  # Compiled publication-grade PDF report
│
├── members.txt                  # Student ID and submission governance details
├── submission.txt               # Dataset, GitHub, and YouTube video presentation links
├── requirements.txt             # Pinned environment dependencies
├── .gitignore                   # Excludes cache, virtual environments, and checkpoints
├── ASSIGNMENT_IMPLEMENTATION_PLAN.md # Detailed execution roadmap
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

## 📊 Dataset Provenance & Methodology

### Dataset Provenance
- **Primary Source:** Kaggle Benchmark *"Credit Score Classification"* by Rohan Paris (`https://www.kaggle.com/datasets/parisrohan/credit-score-classification`, CC0: Public Domain).
- **Derivation:** Formulated as an authentic binary credit scoring problem by deduplicating multi-month records to 8,692 unique retail customers across 19 attributes, focusing on binary credit health (`Poor` = default hazard [59.65%], `Good` = creditworthy [40.35%]).

### Key Methodological Highlights
1. **Data Sanitization & Preprocessing (Zero Leakage):**
   - 80/20 Stratified train-test split locked before any transformer fitting.
   - Domain sanitization converts artificial 4-digit corruptions (ages of $-500$ and $8,697$, interest rates up to $5,774\%$, loans of $-100$, sentinel investments of $10,000$, and string placeholders `'_______'`, `'!@9#%8'`, `'_'`, `'NM'`) to `NaN`, cleanly resolved via median and mode imputation inside the Scikit-Learn `ColumnTransformer`.
   - ANOVA hypothesis testing demonstrates that domain sanitization unlocks decisive statistical significance across all 10 continuous predictors, with `Interest_Rate` rising from $F = 0.01$ ($p = 0.90$) to $F = 3,945.77$ ($p < 10^{-100}$).
   - **Dimensionality Reduction (Rubric Criterion 3):** Pruned extreme collinear feature `Monthly_Income` ($r > 0.93$), duplicate `Derived_Loan_Count` ($r = 1.0000$), and statistically independent feature `Occupation` ($\chi^2 = 16.54, p = 0.347 \gg 0.05$), reducing the feature matrix to 20 clean, standardized dimensions.
2. **Supervised Classification Benchmark:**
   - Evaluated under moderate class imbalance (59.65% Poor vs 40.35% Good).
   - Tuned decision thresholds on out-of-fold training probabilities via `cross_val_predict` ($t = 0.44$), increasing Recall to **85.82%** (capturing 50 additional defaulting borrowers) to minimize high-cost default write-offs.
   - Tuned Random Forest achieved superior performance across all dimensions: **Accuracy 82.12%**, **F1 0.8449**, **MCC 0.6364**, **ROC-AUC 0.8948**, and **PR-AUC 0.9267**, with a tight generalization gap (89.69% train vs 82.12% test) demonstrating effective tree regularization.

---

## 👤 Author & Contribution (Individual Submission)
* **Ginuka (Student ID: MS26922898 | Email: ms26922898@my.sliit.lk):**
  - **Data Engineering:** Ingestion of raw dataset, domain-specific sanitization transformer, regex extraction of credit history years, and multi-label loan parsing.
  - **Statistical Exploratory Audits:** Comparative ANOVA F-tests, Chi-square independence tests, and correlation analysis.
  - **Leakage-Safe Preprocessing Pipeline:** Construction of modular Scikit-Learn transformers and `ColumnTransformer` fitted strictly on training folds.
  - **Supervised Classification:** Implementation and 5-fold cross-validation of Model 1 (Regularized Logistic Regression with out-of-fold threshold tuning) and Model 2 (Tuned Random Forest Ensemble with Gini feature importances).
  - **Evaluation & Reporting:** Construction of comprehensive metric evaluation framework (PR-AUC, ROC-AUC, MCC, Brier score, Confusion Matrices), final MSc academic report PDF, and code appendix.
