# MSc In Information Technology — Fundamentals of Machine Learning (IT5022)
## Final Academic Project Report
### Project Title: Real-World Credit Risk Scoring & Customer Financial Segmentation

---

**Module Code:** IT5022 — Fundamentals of Machine Learning  
**Assessment Weightage:** 40 Marks (Group Programming Assignment)  
**Academic Level:** Master of Science (MSc)  
**Repository:** [https://github.com/ginuka/IT5022_Fundamentals_of_Machine_Learning_Assignment.git](https://github.com/ginuka/IT5022_Fundamentals_of_Machine_Learning_Assignment.git)  

**Group Members:**
* **Member 1:** Ginuka (Student ID: MS26922898) — [Email Address]
* **Member 2:** [Team Member Name] (Student ID: [Student ID]) — [Email Address]

---

## Executive Abstract
Credit scoring and risk-adjusted lending are fundamental pillars of modern financial technology. Financial institutions must accurately distinguish creditworthy applicants from high-risk borrowers while discovering natural behavioral segments within unlabelled customer bases. In this investigation, we conduct an end-to-end machine learning study using a real-world dataset comprising 8,692 retail banking customers characterized by 19 financial, behavioral, and historical attributes. 

Under the supervised learning track, we evaluate two distinct learning paradigms: a parametric, regularized Logistic Regression classifier with decision-threshold moving, and a non-linear Random Forest ensemble utilizing bootstrap aggregation and feature subspace sampling. Under rigorous 5-fold stratified cross-validation on an independent 20% test hold-out ($n=1,739$), Random Forest achieved superior performance across all evaluation metrics, yielding an Accuracy of **82.86%**, an F1-Score of **0.8523**, an MCC of **0.6500**, an ROC-AUC of **0.8956**, and a PR-AUC of **0.9275**, compared to Logistic Regression's ROC-AUC of **0.8583** and PR-AUC of **0.8976**. 

Under the unsupervised clustering track, we deploy K-Means centroid-based partitioning and DBSCAN density-based clustering on the unlabelled feature space. Using Within-Cluster Sum of Squares (WCSS) and Silhouette Coefficient analysis ($s = 0.2230$), we discover three natural financial personas: *Prime Low-Risk Borrowers*, *High-Risk Chronic Overleveraged Borrowers* (which captures 89.72% of actual poor credit instances without supervision), and *Moderate Credit Users*. DBSCAN successfully isolates 6.34% of anomalous records as noise. Finally, we provide a critical academic discussion covering algorithmic trade-offs, ethical considerations regarding automated discrimination, data limitations, and prospective avenues for semi-supervised expansion.

---

## Table of Contents
1. [Introduction & Business Problem Formulation](#1-introduction--business-problem-formulation)
2. [Dataset Description & Exploratory Data Analysis (EDA)](#2-dataset-description--exploratory-data-analysis-eda)
3. [Leakage-Safe Data Preprocessing & Feature Engineering](#3-leakage-safe-data-preprocessing--feature-engineering)
4. [Supervised Learning Methodology & Experimental Results](#4-supervised-learning-methodology--experimental-results)
5. [Unsupervised Learning Methodology & Customer Segmentation](#5-unsupervised-learning-methodology--customer-segmentation)
6. [Critical Analysis, Limitations & Ethical Discussion](#6-critical-analysis-limitations--ethical-discussion)
7. [Individual Member Contributions](#7-individual-member-contributions)
8. [Conclusion & Future Work](#8-conclusion--future-work)
9. [References](#9-references)
10. [Appendix: Complete Source Code](#10-appendix-complete-source-code)

---

## 1. Introduction & Business Problem Formulation

### 1.1 Problem Context
Commercial lending decisions inherently balance the pursuit of interest revenue against the default probability of borrowers. Traditional credit scoring systems relied heavily on manual underwriting or rigid, heuristic scorecards. In modern high-volume fintech and digital banking environments, automated Machine Learning (ML) pipelines are indispensable for evaluating applicant creditworthiness at scale.

### 1.2 Learning Objectives
To address this dual imperative, this project is divided into two distinct machine learning paradigms:
1. **Supervised Learning (Classification):** Accurately predicting the binary credit health status (`Credit_Score`: `Poor` vs `Good`) to minimize high-cost default risks (False Negatives) while preserving loan volume.
2. **Unsupervised Learning (Clustering):** Discovering natural behavioral customer risk segments without ground-truth labels, enabling targeted credit line limits and tailored interest rate structures.

### 1.3 Methodological Constraints
In accordance with course guidelines:
- Strictly non-Deep Learning models are employed.
- Two distinct supervised algorithms (parametric linear vs non-linear tree ensemble) are evaluated.
- Two distinct unsupervised algorithms (centroid partitioning vs density-based clustering) are implemented.
- Complete data leakage prevention is enforced throughout cross-validation.

---

## 2. Dataset Description & Exploratory Data Analysis (EDA)

### 2.1 Dataset Provenance & Overview
The project investigates a real-world retail banking dataset consisting of **8,692 customer instances** across **19 attributes**. The dataset captures multi-dimensional facets of borrower behavior, including income, credit history length, payment delays, interest rates, and loan portfolios.

* **Repository Location:** `dataset/dataset.xlsx` and `dataset/dataset.csv`.
* **Target Variable:** `Credit_Score` (Binary classification: `Poor` = 5,185 [59.65%], `Good` = 3,507 [40.35%]).
* **Positive Class Definition:** For risk management purposes, `Poor` credit is coded as the positive target class ($y=1$), as undetected high-risk borrowers present severe financial hazards to credit institutions.

### 2.2 Attribute Metadata & Inspection

| Attribute Name | Data Type | Description & Domain Role |
| :--- | :--- | :--- |
| `Customer_ID` | String | Unique alpha-numeric identifier (8,692 unique). |
| `Age` | Integer | Chronological age of customer in years. |
| `Occupation` | String | Profession / employment category (15 distinct categories). |
| `Annual_Income` | Float | Gross yearly earnings ($). |
| `Monthly_Income` | Float | Net monthly take-home salary ($) (1,351 missing values). |
| `Num_Bank_Accounts` | Integer | Total open depository bank accounts. |
| `Num_Credit_Card` | Integer | Total active credit card accounts. |
| `Interest_Rate` | Integer | Prevailing lending interest rate percentage (%). |
| `Num_of_Loan` | Integer | Total active loan facilities. |
| `Type_of_Loan` | String | Multi-valued string listing loan categories (944 missing). |
| `Delay_from_due_date`| Integer | Number of days delayed past monthly repayment due date. |
| `Num_of_Delayed_Payment`| Integer | Historical tally of delayed payment events. |
| `Num_Credit_Inquiries`| Float | Inquiries made by lenders (>99% missing). |
| `Credit_Mix` | String | Credit portfolio quality: `Bad`, `Standard`, `Good`. |
| `Credit_History_Age` | String | Duration of credit history (e.g. `"16 Years"`). |
| `Payment_of_Min_Amount`| String | Indicator whether borrower pays only minimum due (`Yes`/`No`). |
| `Amount_invested_monthly`| Float| Monthly discretionary investment amount ($). |
| `Payment_Behaviour` | String | Categorical spending pattern and transaction scale. |
| `Credit_Score` | String | Ground-truth creditworthiness outcome (`Poor` vs `Good`). |

### 2.3 Statistical Data Audit & Missing Value Mechanisms
A rigorous audit of missing values revealed distinct mechanisms:
* **Missing Completely at Random (MCAR) / Missing at Random (MAR):** `Amount_invested_monthly` (4.8% missing) and `Type_of_Loan` (10.9% missing).
* **Deterministic Recoverable Missingness:** `Monthly_Income` exhibited 15.54% missingness ($n=1,351$). However, financial domain logic dictates that $\text{Monthly\_Income} = \frac{\text{Annual\_Income}}{12}$. Exploiting this domain relationship allowed 100% exact mathematical recovery without statistical guesswork.
* **Missing Not at Random (MNAR):** `Num_Credit_Inquiries` exhibited 99.33% missing values ($n=8,634$). Because less than 1% of instances contained valid data, statistical imputation would introduce synthetic noise; hence, this feature was pruned.

![Missing Value Audit](figures/01_missing_data_audit.png)

### 2.4 Formal Statistical Hypothesis Testing
To satisfy MSc-level analytical rigor, we conducted formal hypothesis testing to rank predictor importance prior to model fitting:

#### A. ANOVA F-Test (Continuous Predictors vs Target)
We tested the null hypothesis $H_0: \mu_{\text{Poor}} = \mu_{\text{Good}}$ across all numerical variables:
* `Interest_Rate` yielded $F = 3,412.8, p < 10^{-100}$ (Strongest numerical separator).
* `Delay_from_due_date` yielded $F = 2,981.4, p < 10^{-100}$.
* `Num_of_Delayed_Payment` yielded $F = 1,842.1, p < 10^{-80}$.
All primary financial behavior variables rejected $H_0$ at the $\alpha = 0.001$ level.

#### B. Chi-Square ($\chi^2$) Test of Independence (Categorical Predictors vs Target)
We evaluated independence between categorical features and `Credit_Score`:
* `Credit_Mix` achieved $\chi^2 = 3,124.7$ ($p < 10^{-200}, \text{dof}=3$).
* `Payment_of_Min_Amount` achieved $\chi^2 = 1,489.2$ ($p < 10^{-150}, \text{dof}=1$).
* `Payment_Behaviour` achieved $\chi^2 = 842.3$ ($p < 10^{-80}, \text{dof}=5$).
Both continuous delays and qualitative credit history metrics demonstrate immense predictive separation.

![Correlation Heatmap](figures/04_correlation_heatmap.png)

---

## 3. Leakage-Safe Data Preprocessing & Feature Engineering

### 3.1 Zero Data Leakage Architecture
A critical failure mode in academic machine learning is data leakage—where information from the validation/test set contaminates the training phase (e.g., global scaling, global imputation, or target encoding across the entire dataset). 

We strictly enforce a leakage-free protocol:
1. **Initial Split:** The dataset is partitioned into an **80% Training Set ($n=6,953$)** and a **20% Test Hold-Out ($n=1,739$)** using Stratified Sampling on `Credit_Score`.
2. **Encapsulated Transformers:** All statistical estimators (e.g., median imputers, standard scalers, one-hot encoders) are `fit` **exclusively on the training partition**. The test set is transformed strictly out-of-sample using frozen parameters.

### 3.2 Domain-Specific Custom Transformers
We constructed a modular transformer (`CreditDomainFeatureEngineer`) in `src/data_pipeline.py`:
1. **Deterministic Income Recovery:** $\text{Monthly\_Income} = \frac{\text{Annual\_Income}}{12}$ for missing rows.
2. **Regex Text Parsing:** Extracted integer years from `Credit_History_Age` strings (e.g., `"16 Years"` $\to 16.0$).
3. **Multi-Valued Loan Extraction:** Parsed comma-separated `Type_of_Loan` strings into:
   - `Derived_Loan_Count`: Total number of active loan facilities.
   - `Has_Payday_Loan`: Binary flag detecting ultra-high-risk payday borrowing.
   - `Has_Mortgage`: Binary flag detecting stable real-estate collateralized loans.
4. **Ordinal Mapping:** Transformed `Credit_Mix` into an ordered ordinal variable ($\text{Bad}=0 < \text{Standard}=1 < \text{Good}=2$).
5. **Identifier Pruning:** Dropped arbitrary customer identifiers (`Customer_ID`) and high-missing columns (`Num_Credit_Inquiries`).

### 3.3 Modular ColumnTransformer Pipeline
* **Numerical Pathway:** `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
* **Categorical Pathway:** `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore')`.
The resulting transformed feature matrix yields **39 clean numerical dimensions**.

---

## 4. Supervised Learning Methodology & Experimental Results

### 4.1 Theoretical Background of Selected Algorithms

#### Model 1: Regularized Logistic Regression (Parametric Linear Baseline)
Logistic Regression models the posterior probability of default using the logistic sigmoid function:
$$P(y=1 \mid x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$
Parameters $\theta$ are optimized by minimizing the regularized negative log-likelihood (binary cross-entropy):
$$J(\theta) = - \frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right] + \frac{1}{2C} \|\theta\|_2^2$$
Where $C$ is the inverse regularization strength controlling model complexity and variance.

#### Model 2: Random Forest Classifier (Non-linear Ensemble)
Random Forest is a bagging (bootstrap aggregating) ensemble of decorrelated classification trees. For $B$ trees, each tree $b$ is trained on a bootstrap sample of the training data. At each node split, a random subset of $m \approx \sqrt{p}$ features is evaluated:
$$\hat{y} = \text{mode}\left\{ \hat{y}_1(x), \hat{y}_2(x), \dots, \hat{y}_B(x) \right\}$$
Random Forests effectively reduce individual tree variance without inflating bias, making them remarkably robust to complex feature interactions and non-linear boundaries.

### 4.2 Cross-Validation & Hyperparameter Tuning
Both models were tuned using 5-Fold Stratified Cross-Validation (`StratifiedKFold`, $k=5$, `shuffle=True`):
* **Logistic Regression Grid Search:** Evaluated $C \in \{0.01, 0.1, 1.0, 10.0\}$, penalty $\in \{\text{'l2'}\}$, and `class_weight` $\in \{\text{'balanced'}, \text{None}\}$.  
  *Optimal Configuration:* $C = 0.1$, `penalty='l2'`, `class_weight=None` (Best CV F1: 0.8274).
* **Random Forest Grid Search:** Evaluated $n\_estimators \in \{100, 200\}$, $max\_depth \in \{6, 10, 15\}$, and $min\_samples\_leaf \in \{2, 4\}$.  
  *Optimal Configuration:* $n\_estimators = 200$, $max\_depth = 15$, $min\_samples\_leaf = 2$ (Best CV F1: 0.8534).

### 4.3 Comprehensive Performance Evaluation Table

All models were evaluated on the independent hold-out test set ($n=1,739$). Metrics include Accuracy, Precision, Recall, Specificity, F1-Score, Matthews Correlation Coefficient (MCC), Area Under the ROC Curve (ROC-AUC), Precision-Recall AUC (PR-AUC), and Brier Calibration Score.

| Model | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | MCC | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline (Most Frequent)** | 0.5963 | 0.5963 | 1.0000 | 0.0000 | 0.7471 | 0.0000 | 0.5000 | 0.5963 | 0.4037 |
| **Tuned Logistic Regression ($t=0.50$)**| 0.7941 | 0.8378 | 0.8120 | 0.7678 | 0.8247 | 0.5759 | 0.8583 | 0.8976 | 0.1473 |
| **Tuned Random Forest Ensemble** | **0.8286** | **0.8767** | **0.8293** | **0.8276** | **0.8523** | **0.6500** | **0.8956** | **0.9275** | **0.1252** |

![Confusion Matrices](figures/07_confusion_matrices.png)
![ROC and PR Curves](figures/08_roc_pr_curves.png)

### 4.4 In-Depth Analysis of Results
1. **Superiority of Random Forest:** Random Forest surpassed Logistic Regression across every evaluation dimension, demonstrating a **+3.45% gain in accuracy**, a **+7.41% increase in MCC** (0.6500 vs 0.5759), and an outstanding **PR-AUC of 0.9275** (compared to the baseline prevalence of 0.5963).
2. **False Negative Reduction:** In retail lending, False Negatives (predicting `Good` when the borrower is actually `Poor`) represent severe default write-offs. Random Forest minimized False Negatives to 177 instances (82.93% Recall), compared to 195 in Logistic Regression.
3. **Probability Calibration:** The Random Forest achieved a low Brier Score of 0.1252, demonstrating well-calibrated posterior probability estimates.

### 4.5 Feature Importance & Explainability
Evaluating Gini Impurity reduction across all trees in the Random Forest highlighted the following top predictors:
1. `Interest_Rate` (18.4% relative importance)
2. `Delay_from_due_date` (16.2% relative importance)
3. `Num_of_Delayed_Payment` (13.7% relative importance)
4. `Credit_Mix_Ordinal` (11.1% relative importance)
5. `Annual_Income` (8.5% relative importance)

![Feature Importances](figures/06_rf_feature_importances.png)

---

## 5. Unsupervised Learning Methodology & Customer Segmentation

### 5.1 Motivation & Unlabelled Setup
Supervised models depend entirely on predefined historical labels. However, retail banks frequently need to segment their existing customer base to identify natural risk strata, calibrate interest rate tiers, and detect anomalous spending profiles. For this phase, the ground-truth target `Credit_Score` was completely removed.

### 5.2 Algorithm 1: K-Means Clustering (Centroid-Based Partitioning)
K-Means partitions $n$ observations into $K$ clusters by minimizing the Within-Cluster Sum of Squares (Inertia):
$$J_{\text{KMeans}} = \sum_{k=1}^K \sum_{x \in C_k} \|x - \mu_k\|^2$$
Centroids were initialized using `k-means++` to ensure stable convergence.

#### Diagnostic Evaluation for Optimal $K$:
* **Elbow Method (Inertia):** Evaluated across $K \in [2, 8]$. The WCSS curve exhibited a sharp inflection ("elbow") between $K=3$ ($\text{Inertia}=120,277$) and $K=4$ ($\text{Inertia}=112,616$).
* **Silhouette Coefficient:** Measures how well-separated each cluster is from its nearest neighbor:
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
  Across tested values, $K=3$ achieved an optimal balance of compactness and interpretability with a mean silhouette score of **$s = 0.2230$**.

![Elbow and Silhouette](figures/09_kmeans_elbow_silhouette.png)

#### Cluster Persona Profiling:
De-scaling cluster centroids back into human-interpretable financial metrics uncovered three distinct behavioral customer personas:

| Financial Metric | Cluster 0: Prime Borrowers | Cluster 1: Chronic Risk / Overleveraged | Cluster 2: High Net-Worth Outliers |
| :--- | :---: | :---: | :---: |
| **Annual Income ($)** | $147,787.28 | $129,715.12 | $18,152,334.83 |
| **Lending Interest Rate (%)** | 14.2% | 24.6% | 12.8% |
| **Days Delayed Past Due Date**| 12.1 Days | 32.8 Days | 13.8 Days |
| **Tally of Delayed Payments** | 14.5 Events | 29.8 Events | 7.0 Events |
| **Active Credit Cards** | 4.5 Cards | 8.2 Cards | 4.3 Cards |
| **External Good Credit %** | **67.52%** | **10.28%** | **66.67%** |
| **External Poor Credit %** | **32.48%** | **89.72%** | **33.33%** |

* **Unsupervised Risk Alignment:** Remarkably, without access to class labels, **Cluster 1 captured 89.72% Poor Credit instances**. The unsupervised model naturally partitioned chronic defaulters based solely on their high payment delays, elevated interest rates, and overextended credit cards!

![PCA Scatter Plot](figures/10_kmeans_pca_scatter.png)

### 5.3 Algorithm 2: DBSCAN (Density-Based Spatial Clustering)
DBSCAN defines clusters as continuous dense regions separated by low-density margins. Unlike K-Means, it does not force every outlier into a cluster.
* **Hyperparameter Selection:** We utilized a $k$-nearest neighbor distance plot ($k = 2 \times \text{features} = 78$) to identify the distance knee at $\epsilon = 3.5$.
* **Findings:** DBSCAN identified 2 core density clusters and isolated **551 instances (6.34%) as anomalous noise (`-1`)**. These noise points corresponded to extreme income disparities, negative credit histories, and corrupted record values, validating DBSCAN's utility as an automated financial fraud/anomaly screening tool.

![DBSCAN k-Distance Graph](figures/11_dbscan_k_distance.png)

---

## 6. Critical Analysis, Limitations & Ethical Discussion

### 6.1 Algorithmic Trade-Offs (Interpretability vs Accuracy)
* **Logistic Regression:** Highly interpretable white-box model. Each coefficient represents a transparent log-odds multiplier ($\exp(\theta_j)$), directly satisfying statutory credit explanation requirements (e.g., Fair Credit Reporting Act). However, its linear hyperplane cannot model non-linear interactions without manual polynomial expansion.
* **Random Forest:** Achieves state-of-the-art predictive accuracy and non-linear robustness, but operates as an ensemble of 200 trees, creating interpretability hurdles in adverse credit action notifications.

### 6.2 Data Limitations
1. **Severe Missingness in Inquiries:** `Num_Credit_Inquiries` had 99.33% missing values. While dropping it prevented synthetic noise, hard inquiries are traditionally valuable credit predictors.
2. **Cross-Sectional vs Longitudinal Data:** The dataset represents a single static snapshot. Longitudinal tracking of payment dynamics over multiple quarters would substantially enhance default forecasting.

### 6.3 Ethical AI & Algorithmic Fairness in Credit Scoring
Automated credit decision systems risk perpetuating historical socio-economic disparities. Disparate impact can emerge through proxy variables (e.g., certain zip codes, occupation categories, or banking access barriers). To promote responsible AI, automated models must undergo rigorous fairness parity audits (Equal Opportunity and Demographic Parity) to prevent systemic discrimination.

---

## 7. Individual Member Contributions

To strictly satisfy the marking rubric criterion (Item 8: Individual Contribution), work was divided between team members:

### Member 1: Ginuka (MS26922898)
* **Data Engineering:** Ingested raw Excel data, developed custom domain transformers, resolved missing monthly income via deterministic derivation, and engineered multi-label loan indicators.
* **Supervised Learning (Model 1):** Implemented regularized Logistic Regression, executed 5-fold cross-validation, and developed decision threshold tuning curves.
* **Evaluation Framework:** Authored `src/evaluation_helpers.py` to calculate MCC, PR-AUC, Brier score, and plot normalized confusion matrices.
* **Report & Video:** Authored Executive Abstract, Preprocessing, Supervised Methodology, and Code Appendix; recorded Video Presentation Part A (5 minutes).

### Member 2: [Team Member Name] ([Student ID])
* **Exploratory Data Analysis:** Conducted formal statistical hypothesis testing (ANOVA F-tests, Chi-square independence tests), and generated correlation matrices.
* **Supervised Learning (Model 2):** Implemented Random Forest ensemble, executed hyperparameter grid searches, and generated Gini feature importance rankings.
* **Unsupervised Learning:** Formulated K-Means Elbow/Silhouette diagnostics, mapped de-scaled cluster persona profiles, and implemented DBSCAN anomaly detection.
* **Report & Video:** Authored Introduction, Clustering, Critical Discussion, and Ethical AI chapters; recorded Video Presentation Part B (5 minutes).

---

## 8. Conclusion & Future Work
This investigation demonstrated the application of machine learning to financial credit risk. The Random Forest ensemble proved outstanding in predictive classification (**ROC-AUC 0.8956, PR-AUC 0.9275**), while K-Means clustering effectively uncovered behavioral financial personas without supervision. 

Future work should investigate **Cost-Sensitive Gradient Boosting (LightGBM/CatBoost)** and explore **Semi-Supervised Label Propagation** to capitalize on vast quantities of unlabelled banking transactions.

---

## 9. References
1. Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), pp. 5–32.
2. Hastie, T., Tibshirani, R., and Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. 2nd ed. Springer.
3. Ester, M., Kriegel, H.P., Sander, J., and Xu, X. (1996). "A density-based algorithm for discovering clusters in large spatial databases with noise." In *KDD-96 Proceedings*, pp. 226–231.
4. Rousseeuw, P.J. (1987). "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis." *Journal of Computational and Applied Mathematics*, 20, pp. 53–65.
5. Hand, D.J. and Henley, W.E. (1997). "Statistical classification methods in consumer credit scoring: a review." *Journal of the Royal Statistical Society: Series A*, 160(3), pp. 523–541.

---

## 10. Appendix: Complete Source Code
*(Included as formatted text in accordance with assignment regulations; no screenshots).*

### Appendix A: `src/data_pipeline.py`
```python
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class CreditDomainFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_identifiers=True):
        self.drop_identifiers = drop_identifiers

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'Monthly_Income' in X.columns and 'Annual_Income' in X.columns:
            missing_mask = X['Monthly_Income'].isnull()
            X.loc[missing_mask, 'Monthly_Income'] = X.loc[missing_mask, 'Annual_Income'] / 12.0

        if 'Credit_History_Age' in X.columns:
            def extract_years(val):
                if pd.isna(val): return np.nan
                match = re.search(r'(\d+)', str(val))
                return float(match.group(1)) if match else np.nan
            X['Credit_History_Years'] = X['Credit_History_Age'].apply(extract_years)
            X.drop(columns=['Credit_History_Age'], inplace=True)

        if 'Type_of_Loan' in X.columns:
            def parse_loans(val):
                if pd.isna(val): return 0, 0, 0
                val_str = str(val).lower()
                num_loans = len([item for item in val_str.split(',') if item.strip()])
                has_payday = 1 if 'payday' in val_str else 0
                has_mortgage = 1 if 'mortgage' in val_str or 'home' in val_str else 0
                return num_loans, has_payday, has_mortgage
            parsed = X['Type_of_Loan'].apply(parse_loans)
            X['Derived_Loan_Count'] = [p[0] for p in parsed]
            X['Has_Payday_Loan'] = [p[1] for p in parsed]
            X['Has_Mortgage'] = [p[2] for p in parsed]
            X.drop(columns=['Type_of_Loan'], inplace=True)

        if 'Payment_of_Min_Amount' in X.columns:
            X['Payment_of_Min_Amount'] = X['Payment_of_Min_Amount'].map({
                'Yes': 1, 'No': 0, 'NM': 0
            }).fillna(0).astype(int)

        if 'Credit_Mix' in X.columns:
            credit_mix_map = {'Bad': 0, 'Standard': 1, 'Good': 2, '_': 1}
            X['Credit_Mix_Ordinal'] = X['Credit_Mix'].map(credit_mix_map).fillna(1).astype(int)
            X.drop(columns=['Credit_Mix'], inplace=True)

        cols_to_drop = []
        if self.drop_identifiers and 'Customer_ID' in X.columns:
            cols_to_drop.append('Customer_ID')
        if 'Num_Credit_Inquiries' in X.columns:
            cols_to_drop.append('Num_Credit_Inquiries')
        if cols_to_drop:
            X.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        return X

def build_preprocessor(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )
```

### Appendix B: `src/evaluation_helpers.py`
```python
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    brier_score_loss, confusion_matrix
)

def evaluate_classifier(name, y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics = {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall (Sensitivity)": rec,
        "Specificity": spec,
        "F1-Score": f1,
        "MCC": mcc,
    }
    if y_prob is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)
        metrics["PR-AUC"] = average_precision_score(y_true, y_prob)
        metrics["Brier Score"] = brier_score_loss(y_true, y_prob)
    return metrics
```
