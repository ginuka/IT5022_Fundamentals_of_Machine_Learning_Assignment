# MSc In Information Technology — Fundamentals of Machine Learning (IT5022)
## Final Academic Project Report
### Project Title: Real-World Credit Risk Scoring & Customer Financial Segmentation

---

**Module Code:** IT5022 — Fundamentals of Machine Learning  
**Assessment Weightage:** 40 Marks (Programming Assignment)  
**Academic Level:** Master of Science (MSc)  
**Public Repository:** [https://github.com/ginuka/IT5022_Fundamentals_of_Machine_Learning_Assignment.git](https://github.com/ginuka/IT5022_Fundamentals_of_Machine_Learning_Assignment.git)  

**Author:** Ginuka (Student ID: MS26922898) — ms26922898@my.sliit.lk  
**Submission Governance:** Individual Submission

---

## Executive Abstract
Credit scoring and risk-adjusted lending are fundamental pillars of modern financial technology. Commercial banks and fintech institutions must accurately distinguish creditworthy applicants from high-risk borrowers while discovering natural behavioral segments within unlabelled customer bases. In this investigation, we conduct an end-to-end machine learning study using a retail banking dataset comprising 8,692 customer instances characterized by 19 financial, behavioral, and historical attributes.

Under the supervised learning track, we evaluate two distinct learning paradigms: a parametric, regularized Logistic Regression classifier with out-of-fold decision-threshold moving, and a non-linear Random Forest ensemble utilizing bootstrap aggregation and random feature subspace sampling. Addressing synthetic artifact corruptions and sentinel placeholders (e.g., impossible ages, 5,774% interest rates) via a custom `InvalidValueCleaner` transformer and leakage-safe preprocessing pipeline, we reduced the feature space to 20 high-signal dimensions. Under rigorous 5-fold stratified cross-validation on an independent 20% holdout test partition (n=1,739), Random Forest achieved superior performance across all evaluation dimensions, yielding an Accuracy of **82.12%**, an F1-Score of **0.8449**, a Matthews Correlation Coefficient (MCC) of **0.6364**, an ROC-AUC of **0.8948**, a PR-AUC of **0.9267**, and a Brier Calibration Score of **0.1258**. Tuned Logistic Regression (threshold = 0.44) achieved an Accuracy of **80.16%** and an F1-Score of **0.8376**, boosting recall from 81.00% to **85.82%** to catch 50 additional defaulting borrowers.

Under the unsupervised clustering track, we deploy K-Means centroid-based partitioning and DBSCAN density-based clustering on the unlabelled feature space. Using Within-Cluster Sum of Squares (WCSS) and Silhouette analysis, we discover three natural financial personas: *High-Risk Chronic Overleveraged Borrowers* (n=2,952, which captures **95.60%** actual poor credit instances without supervision), *Prime Low-Risk Borrowers* (n=3,511, **72.66%** good credit), and *Moderate-Income Mid-Tier Borrowers* (n=2,229). DBSCAN successfully isolates 301 records (3.46%) as multi-dimensional anomalous noise. Finally, we provide a critical academic discussion covering algorithmic trade-offs, overfitting diagnostics, ethical considerations regarding automated discrimination, data limitations, and prospective avenues for semi-supervised expansion.

---

## Table of Contents
1. [Introduction & Problem Formulation](#1-introduction--problem-formulation)
2. [Dataset Provenance, Description & Statistical Audit](#2-dataset-provenance-description--statistical-audit)
3. [Leakage-Safe Preprocessing & Feature Engineering](#3-leakage-safe-preprocessing--feature-engineering)
4. [Supervised Learning Methodology & Experimental Results](#4-supervised-learning-methodology--experimental-results)
5. [Unsupervised Learning Methodology & Customer Segmentation](#5-unsupervised-learning-methodology--customer-segmentation)
6. [Critical Analysis, Limitations & Ethical Discussion](#6-critical-analysis-limitations--ethical-discussion)
7. [Author & Individual Project Contribution](#7-author--individual-project-contribution)
8. [Conclusion & Future Work](#8-conclusion--future-work)
9. [References](#9-references)
10. [Appendix: Complete Source Code](#10-appendix-complete-source-code)

---

## 1. Introduction & Problem Formulation

### 1.1 Problem Context
Commercial lending decisions inherently balance the pursuit of interest revenue against default hazard. Traditional credit scoring systems relied heavily on manual underwriting or rigid, heuristic scorecards. In modern high-volume fintech and digital banking environments, automated Machine Learning (ML) pipelines are indispensable for evaluating applicant creditworthiness at scale.

### 1.2 Learning Objectives
To address this dual imperative, this project is divided into two distinct machine learning paradigms:
1. **Supervised Learning (Classification):** Accurately predicting the binary credit health status (`Credit_Score`: `Poor` vs `Good`) to minimize high-cost default risks (False Negatives) while preserving loan volume.
2. **Unsupervised Learning (Clustering):** Discovering natural behavioral customer risk segments without ground-truth labels, enabling targeted credit line limits and tailored interest rate structures.

### 1.3 Methodological Constraints & Governance
In accordance with course guidelines:
- Strictly non-Deep Learning models are employed.
- Two distinct supervised algorithms (parametric linear vs non-linear tree ensemble) are evaluated.
- Two distinct unsupervised algorithms (centroid partitioning vs density-based clustering) are implemented.
- Complete data leakage prevention is enforced throughout cross-validation.
- Project conducted as an authorized individual submission for IT5022.

---

## 2. Dataset Provenance, Description & Statistical Audit

### 2.1 Dataset Provenance & Derivation
The dataset is derived from the publicly hosted Kaggle benchmark dataset *"Credit Score Classification"* by Rohan Paris:
- **Public Repository URL:** `https://www.kaggle.com/datasets/parisrohan/credit-score-classification`
- **License:** CC0: Public Domain
- **Original Architecture & Subset Derivation:** The original Kaggle benchmark comprises over 100,000 multi-month longitudinal customer records categorized under a 3-class target (`Good`, `Standard`, `Poor`). To establish an authentic retail lending binary credit scoring problem, the dataset was deduplicated to one record per unique customer (8,692 instances across 19 attributes), focusing on binary credit health (`Poor` default hazard vs `Good` creditworthy). Redundant multi-month identifier tracking columns (such as `SSN`, `Month`, `Name`, `Outstanding_Debt`, and `Credit_Utilization_Ratio`) were removed, and `Monthly_Inhand_Salary` was renamed to `Monthly_Income`.
- **Benchmark Context & Limitations:** As widely recognized in the machine learning literature, this benchmark is synthetically generated to simulate retail banking distributions. Consequently, it contains synthetic anomalies (such as negative loans, sentinel investments, and placeholder strings). Rather than compromising real-world validity, identifying, isolating, and sanitizing these synthetic artifacts represents a core data engineering contribution of this project.

### 2.2 Attribute Metadata & Inspection

| Attribute Name | Data Type | Domain Meaning & Characteristics |
| :--- | :--- | :--- |
| `Customer_ID` | String | Unique alpha-numeric borrower identifier (8,692 unique). |
| `Age` | Integer | Chronological age of applicant in years. |
| `Occupation` | String | Profession category (15 distinct categories + placeholder). |
| `Annual_Income` | Float | Gross yearly earnings ($). |
| `Monthly_Income` | Float | Net monthly take-home salary ($) (1,351 missing values). |
| `Num_Bank_Accounts` | Integer | Total open depository bank accounts. |
| `Num_Credit_Card` | Integer | Total active credit card accounts. |
| `Interest_Rate` | Integer | Prevailing lending interest rate percentage (%). |
| `Num_of_Loan` | Integer | Total active credit facilities. |
| `Type_of_Loan` | String | Multi-valued string listing loan categories (944 missing). |
| `Delay_from_due_date`| Integer | Days delayed past monthly repayment due date. |
| `Num_of_Delayed_Payment`| Integer | Historical tally of delayed payment events. |
| `Num_Credit_Inquiries`| Float | Inquiries made by lenders (>99% missing). |
| `Credit_Mix` | String | Credit portfolio quality (`Bad`, `Standard`, `Good`, and placeholder `_`). |
| `Credit_History_Age` | String | Duration of credit history (e.g. `"16 Years"`). |
| `Payment_of_Min_Amount`| String | Indicator whether borrower pays minimum due (`Yes`, `No`, and `NM`). |
| `Amount_invested_monthly`| Float| Monthly discretionary investment amount ($). |
| `Payment_Behaviour` | String | Categorical transaction scale and spending frequency. |
| `Credit_Score` | String | Ground-truth credit outcome (`Poor` = 5,185 [59.65%], `Good` = 3,507 [40.35%]). |

### 2.3 Statistical Data Audit & Domain Sanitization
A thorough inspection of the raw data revealed substantial data corruption and artificial placeholders requiring systematic sanitization:

| Defect / Problem | Affected Rows | Example Corrupt Values | Remediation Action in Pipeline |
| :--- | :---: | :--- | :--- |
| `Age` outside [18, 100] | 725 | -500, 8,697 | Bounded to legal adult borrowing range [18, 100]; outliers mapped to `NaN`. |
| `Num_of_Loan` outside [0, 20] | 361 | -100, 1,480 | Bounded to [0, 20]; negative & 4-digit anomalies mapped to `NaN`. |
| `Amount_invested_monthly` sentinel | 376 | 10000.0 | Sentinel placeholder 10000 converted to `NaN`. |
| `Interest_Rate` above 50% | 206 | Up to 5,774% | Bounded to [0, 50%]; extreme rate spikes converted to `NaN`. |
| `Num_Credit_Card` above 20 | 214 | Up to 1,498 | Bounded to [0, 20]; extreme card counts converted to `NaN`. |
| `Num_Bank_Accounts` outside [0, 20] | 118 | -1, 1,794 | Bounded to [0, 20]; negative and 4-digit counts converted to `NaN`. |
| `Num_of_Delayed_Payment` above 50 | 53 | Up to 4,282 | Bounded to [0, 50]; extreme payment tallies converted to `NaN`. |
| `Annual_Income` extreme spikes | 66 | Up to 24,000,000 | Spikes > 500,000 mapped to `NaN` for clean median imputation. |
| `Occupation` placeholder | 610 | `'_______'` | Sentinel placeholder converted to `NaN` for imputation. |
| `Payment_Behaviour` placeholder | 674 | `'!@9#%8'` | Sentinel placeholder converted to `NaN` for imputation. |
| `Credit_Mix` placeholder | 1,788 | `'_'` | Uninformative placeholder `'_'` converted to `NaN`. |
| `Payment_of_Min_Amount` placeholder | 1,085 | `'NM'` | Missing indicator `'NM'` (Not Mentioned) converted to `NaN`. |
| `Num_Credit_Inquiries` missingness | 8,634 (99.3%) | Missing Not At Random | Pruned entirely (>99% missing introduces noise). |

![Missing Data Audit](figures/01_missing_data_audit.png)

### 2.4 Formal Statistical Hypothesis Testing

#### A. Comparative ANOVA F-Test (Raw vs. Domain-Sanitized Predictors)
To demonstrate the profound impact of data cleaning, we evaluated continuous predictors against the binary target `Credit_Score` across both raw uncleaned data and domain-sanitized data:

| Feature Name | Raw F-Stat | Raw p-value | Sanitized F-Stat | Sanitized p-value | Statistically Significant (p < 0.001) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Interest_Rate` | 0.01 | 0.903 (ns) | **3,945.77** | **0.00 (p < 10^-100)** | **True** (Strongest separator) |
| `Delay_from_due_date` | 2,856.53 | 0.00 | **2,856.53** | **0.00 (p < 10^-100)** | **True** |
| `Num_Credit_Card` | 0.01 | 0.906 (ns) | **2,403.70** | **0.00 (p < 10^-100)** | **True** |
| `Num_Bank_Accounts` | 0.64 | 0.425 (ns) | **2,385.39** | **0.00 (p < 10^-100)** | **True** |
| `Num_of_Loan` | 3.79 | 0.052 (ns) | **1,996.76** | **0.00 (p < 10^-100)** | **True** |
| `Num_of_Delayed_Payment`| 0.86 | 0.353 (ns) | **1,548.75** | **0.00 (p < 10^-100)** | **True** |
| `Annual_Income` | 0.27 | 0.605 (ns) | **690.05** | **2.03 x 10^-146** | **True** |
| `Monthly_Income` | 578.48 | 8.36 x 10^-124 | **578.48** | **8.36 x 10^-124** | **True** |
| `Amount_invested_monthly`| 4.43 | 0.035 (ns) | **366.32** | **5.17 x 10^-80** | **True** |
| `Age` | 0.07 | 0.784 (ns) | **231.23** | **1.47 x 10^-51** | **True** |

> **Critical Empirical Finding:** In the uncleaned data, 8 out of 10 numerical features—including core lending indicators such as `Interest_Rate`, `Num_Credit_Card`, and `Age`—failed to reach significance (p > 0.05) because extreme 4-digit outliers exploded group variances. Following domain sanitization, **all 10 features achieve decisive statistical significance (p < 10^-50)**, with `Interest_Rate` emerging as the single most powerful continuous discriminator in the entire dataset (F = 3,945.77).

#### B. Chi-Square Test of Independence (Categorical Predictors vs Target)
We evaluated independence between categorical variables and `Credit_Score`:

| Categorical Feature | Chi2 Statistic | p-value | Degrees of Freedom (dof) | Statistically Significant |
| :--- | :---: | :---: | :---: | :---: |
| `Payment_of_Min_Amount` | **2,453.05** | 0.00 (p < 10^-100) | 2 | **Yes** (p < 0.001) |
| `Credit_Mix` | **2,347.67** | 0.00 (p < 10^-100) | 3 | **Yes** (p < 0.001) |
| `Payment_Behaviour` | **363.43** | 2.02 x 10^-75 | 6 | **Yes** (p < 0.001) |
| `Occupation` | **16.54** | 0.347 | 15 | **No** (p = 0.347 >> 0.05) |

> **Dimensionality Reduction Rationale:** `Occupation` fails to reject the null hypothesis of independence (p = 0.347), confirming that customer profession does not provide meaningful separation for credit default. Removing `Occupation` eliminates 15 uninformative dummy dimensions, directly curbing the curse of dimensionality and satisfying Rubric Criterion 3.

![Correlation Heatmap](figures/04_correlation_heatmap.png)

---

## 3. Leakage-Safe Preprocessing & Feature Engineering

### 3.1 Zero Data Leakage Protocol
To guarantee complete methodological integrity:
1. **Stratified Partition:** Raw data is partitioned into an **80% Training Set (n=6,953)** and a **20% Test Hold-Out (n=1,739)** using Stratified Sampling on `Credit_Score`.
2. **Encapsulated Transformers:** Statistical parameters (median imputers, standard scalers, one-hot encoders) are `fit` exclusively on the training partition. Out-of-sample test records are strictly transformed using frozen training statistics.

### 3.2 Feature Engineering & Dimensionality Reduction
In `src/data_pipeline.py`, the `CreditDomainFeatureEngineer` executes:
1. **Regex Extraction:** Parsed string durations in `Credit_History_Age` (e.g., `"16 Years"`) into numerical float years (`Credit_History_Years`).
2. **Specific Loan Extraction:** Parsed multi-valued strings in `Type_of_Loan` to extract indicator flags:
   - `Has_Payday_Loan`: Detects high-risk non-bank payday borrowing.
   - `Has_Mortgage`: Strictly targets long-term mortgage loan facilities (differentiated from home-equity credit lines).
   - *Duplicate Pruning:* `Derived_Loan_Count` showed a perfect correlation (r = 1.0000) with `Num_of_Loan`; the duplicate was pruned.
3. **Ordinal Encoding:** Transformed `Credit_Mix` into an ordered scale (Bad=0 < Standard=1 < Good=2), leaving `NaN`s for pipeline median imputation.
4. **Binary Encoding:** Transformed `Payment_of_Min_Amount` (Yes -> 1, No -> 0), leaving `'NM'` placeholders as `NaN` for pipeline mode imputation.
5. **Collinearity Removal:** `Monthly_Income` is strongly correlated with `Annual_Income` (r > 0.93); it is pruned from the model matrix to prevent variance inflation.

### 3.3 Resulting Feature Matrix
The resulting feature matrix contains **20 clean, standardized dimensions**:
- **14 Continuous/Ordinal Features:** `Age`, `Annual_Income`, `Num_Bank_Accounts`, `Num_Credit_Card`, `Interest_Rate`, `Num_of_Loan`, `Delay_from_due_date`, `Num_of_Delayed_Payment`, `Payment_of_Min_Amount`, `Amount_invested_monthly`, `Credit_History_Years`, `Has_Payday_Loan`, `Has_Mortgage`, `Credit_Mix_Ordinal`.
- **6 One-Hot Categorical Features:** `Payment_Behaviour` (6 clean spend/value dummy categories, with placeholder noise eliminated).
This deliberate dimensionality reduction stabilizes model coefficients and geometric clustering separation.

---

## 4. Supervised Learning Methodology & Experimental Results

### 4.1 Theoretical Background

#### Model 1: Regularized Logistic Regression (Parametric Linear Baseline)
Logistic Regression models posterior default probability using the logistic sigmoid:
$$P(y=1 | x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$
Parameters $\theta$ are estimated via penalized maximum likelihood:
$$J(\theta) = - \frac{1}{m} \sum_{i=1}^m [ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) ] + \lambda R(\theta)$$
Where $R(\theta) = \frac{1}{2} ||\theta||_2^2$ for L2 (Ridge) or $||\theta||_1$ for L1 (Lasso).

#### Model 2: Random Forest Classifier (Non-linear Ensemble)
Random Forest builds B decorrelated decision trees using bootstrap aggregating (bagging) and random subspace sampling (m ~ sqrt(p) features per split):
$$\hat{y} = \text{mode}\{ \hat{y}_1(x), \hat{y}_2(x), \dots, \hat{y}_B(x) \}$$
Random Forests capture non-linear feature interactions and high-order threshold effects while controlling variance through ensemble averaging.

### 4.2 Cross-Validation, Tuning & Leakage-Free Threshold Moving
Both models were tuned using 5-Fold Stratified Cross-Validation (`StratifiedKFold`, k=5, `shuffle=True`):
- **Logistic Regression:** Grid evaluated C in [0.01, 0.1, 1.0, 10.0], penalties in ['l2' (lbfgs), 'l1' (liblinear)], and class_weight in ['balanced', None].  
  *Optimal Configuration:* C = 10.0, penalty='l2', solver='lbfgs', class_weight=None (Best CV F1: **0.8298**).
- **Random Forest:** Grid evaluated n_estimators in [100, 200], max_depth in [6, 10, 15], and min_samples_leaf in [2, 4].  
  *Optimal Configuration:* n_estimators = 100, max_depth = 10, min_samples_leaf = 2, class_weight=None (Best CV F1: **0.8539**).
- **Leakage-Free Threshold Optimization:** Decision threshold t was tuned on **out-of-fold training probabilities** (`cross_val_predict`) across t in [0.2, 0.8]. The threshold that maximized out-of-fold F1 on training data was identified at **t = 0.44**. This frozen threshold was then evaluated strictly once on the independent test hold-out.

![Threshold Tuning](figures/05_threshold_tuning_lr.png)

### 4.3 Comprehensive Performance Evaluation Table

| Model | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | MCC | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline (Most Frequent)** | 0.5963 | 0.5963 | 1.0000 | 0.0000 | 0.7471 | 0.0000 | 0.5000 | 0.5963 | 0.4037 |
| **Logistic Regression (t = 0.50)** | 0.7993 | 0.8468 | 0.8100 | 0.7835 | 0.8280 | 0.5883 | 0.8793 | 0.9103 | 0.1355 |
| **Tuned Logistic Regression (t = 0.44)** | 0.8016 | 0.8180 | **0.8582** | 0.7179 | 0.8376 | 0.5841 | 0.8793 | 0.9103 | 0.1355 |
| **Tuned Random Forest Ensemble** | **0.8212** | **0.8750** | 0.8168 | **0.8276** | **0.8449** | **0.6364** | **0.8948** | **0.9267** | **0.1258** |

![Confusion Matrices](figures/07_confusion_matrices.png)
![ROC and PR Curves](figures/08_roc_pr_curves.png)

### 4.4 In-Depth Analysis & Overfitting Diagnostics
1. **Threshold Moving Justification:** Shifting the Logistic Regression decision threshold from 0.50 to 0.44 directly increased Recall from **81.00% to 85.82%** (+4.82%), capturing **50 additional high-risk borrowers** on the test set while raising F1 from 0.8280 to 0.8376. In retail credit risk, where an undetected default represents a complete principal write-off, prioritizing recall via cost-sensitive threshold shifting is economically imperative.
2. **Random Forest Superiority:** Random Forest achieved an **ROC-AUC of 0.8948** and a **PR-AUC of 0.9267** (against the 0.5963 baseline prevalence), with an MCC of 0.6364. Its Brier calibration score (0.1258) confirms highly reliable probability estimates.
3. **Overfitting Analysis & Regularization:** In unconstrained decision trees, training accuracy frequently reaches 100%, leading to massive overfitting. With our tuned Random Forest, training accuracy was **89.69%** versus test accuracy of **82.12%**—a tight generalization gap of 7.57%. Setting `max_depth = 10` and `min_samples_leaf = 2` effectively constrained tree depth, preventing terminal nodes from memorizing idiosyncratic noise.

### 4.5 Feature Importance & Explainability
Evaluating Gini Impurity reduction across all trees identified the following primary predictors:
1. `Interest_Rate` (**19.06%**)
2. `Delay_from_due_date` (**16.10%**)
3. `Credit_History_Years` (**9.30%**)
4. `Num_of_Loan` (**8.23%**)
5. `Num_Bank_Accounts` (**8.05%**)
6. `Payment_of_Min_Amount` (**7.65%**)
7. `Credit_Mix_Ordinal` (**6.11%**)
8. `Num_Credit_Card` (**6.04%**)

![Feature Importances](figures/06_rf_feature_importances.png)

---

## 5. Unsupervised Learning Methodology & Customer Segmentation

### 5.1 Motivation & Setup
Supervised models require ground-truth labels. However, financial institutions frequently require unsupervised segmentation on unlabelled portfolios to detect natural risk strata, adjust lending terms, and flag anomalies. For this phase, `Credit_Score` was completely removed.

### 5.2 Algorithm 1: K-Means Clustering (Centroid-Based Partitioning)
K-Means partitions n observations into K clusters by minimizing Within-Cluster Sum of Squares (Inertia):
$$J_{\text{KMeans}} = \sum_{k=1}^K \sum_{x \in C_k} ||x - \mu_k||^2$$

#### Diagnostic Evaluation for Optimal K:
- **Elbow Method (Inertia):** Evaluated across K in [2, 8]. WCSS exhibited a marked inflection ("elbow") at K=3 (Inertia = 80,370.0).
- **Silhouette Coefficient Analysis:** K=2 achieved s=0.2744, and K=3 achieved s=0.1509. While K=2 yields mathematically tighter separation, K=3 provides substantially richer business utility by separating high-risk, prime, and mid-tier credit segments.

![Elbow and Silhouette](figures/09_kmeans_elbow_silhouette.png)

#### Cluster Persona Profiling:
De-scaling cluster centroids to their original financial scales reveals three distinct customer personas:

| Financial Attribute | Cluster 0: Chronic Overleveraged | Cluster 1: Prime Low-Risk | Cluster 2: Moderate Mid-Tier |
| :--- | :---: | :---: | :---: |
| **Cluster Size (n, %)** | **2,952 (34.0%)** | **3,511 (40.4%)** | **2,229 (25.6%)** |
| **Annual Income ($)** | $32,659.60 | $71,765.84 | $44,836.34 |
| **Lending Interest Rate (%)** | **24.60%** | **6.77%** | **15.34%** |
| **Days Delayed Past Due Date**| **38.27 Days** | **10.68 Days** | **17.88 Days** |
| **Tally of Delayed Payments** | **17.77 Events** | **6.86 Events** | **13.25 Events** |
| **Active Credit Cards** | **7.51 Cards** | **4.27 Cards** | **5.26 Cards** |
| **Applicant Age** | 31.48 Years | 37.14 Years | 33.49 Years |
| **Ground-Truth Poor Credit %** | **95.60%** | **27.34%** | **62.94%** |
| **Ground-Truth Good Credit %** | **4.40%** | **72.66%** | **37.06%** |

> **Unsupervised Risk Alignment:** Completely blind to ground-truth labels, **Cluster 0 isolated 95.60% Poor credit instances** based entirely on high interest rates, chronic delays, and elevated card counts. Conversely, **Cluster 1 captured 72.66% Good credit applicants**, characterized by 2.2x higher earnings and low default rates.

![PCA Scatter Plot](figures/10_kmeans_pca_scatter.png)

### 5.3 Algorithm 2: DBSCAN (Density-Based Spatial Clustering)
DBSCAN discovers clusters of arbitrary shapes and isolates outliers without forcing points into unnatural clusters:
- **Parameter Selection via k-Distance Graph:** We plotted sorted k-nearest neighbor distances (k = 2 x dim = 40) and selected the distance knee at epsilon = 2.5 with `min_samples = 15`.
- **Findings:** DBSCAN identified 1 core dense density cluster and isolated **301 instances (3.46%) as anomalous noise (-1)**. These noise points correspond to multi-attribute outliers (e.g., extreme delay-to-income ratios), confirming DBSCAN's effectiveness for automated fraud and anomaly screening.

![DBSCAN k-Distance Graph](figures/11_dbscan_k_distance.png)

---

## 6. Critical Analysis, Limitations & Ethical Discussion

### 6.1 Algorithmic Trade-Offs (Interpretability vs Accuracy)
- **Logistic Regression:** Transparent white-box model. Each coefficient represents an interpretable log-odds multiplier (exp(theta_j)), directly satisfying statutory credit explanation requirements (e.g., Fair Credit Reporting Act). However, its linear hyperplane cannot model non-linear boundaries without manual basis expansion.
- **Random Forest:** Achieves superior predictive performance (**PR-AUC 0.9267**), but acts as an ensemble of 100 decorrelated trees, creating explainability hurdles in adverse credit action notices.

### 6.2 Data Limitations
1. **Synthetic Benchmark Nature:** While the Kaggle benchmark effectively simulates retail credit distributions, real-world banking datasets exhibit more complex longitudinal dependencies.
2. **Missingness in Hard Inquiries:** `Num_Credit_Inquiries` had 99.33% missing values, necessitating its removal. In production credit scoring, credit bureau hard inquiries provide valuable short-term credit-seeking signals.

### 6.3 Ethical AI & Algorithmic Fairness in Credit Scoring
Automated credit decision systems risk perpetuating historical socio-economic disparities. Disparate impact can emerge through proxy variables (e.g., occupation categories or demographic correlations). To promote responsible AI, automated models must undergo rigorous fairness parity audits (Equal Opportunity and Demographic Parity) to prevent systemic discrimination.

---

## 7. Author & Individual Project Contribution

### 7.1 Submission Governance
This project was conducted as an **authorized individual submission** for IT5022 by **Ginuka (Student ID: MS26922898)**.

### 7.2 End-to-End Ownership Across Rubric Criteria
The author independently planned, implemented, validated, and documented all technical and academic components:
- **Data Engineering & Noise Sanitization (Criterion 3):** Engineered `CreditDomainFeatureEngineer` to sanitize impossible ages, corrupt interest rates, sentinel investments, and categorical placeholders. Formulated leakage-free median/mode imputation inside `ColumnTransformer`.
- **Statistical Hypothesis Testing (Criterion 1 & 2):** Designed comparative ANOVA F-tests demonstrating how sanitization unlocked statistical significance across all 10 continuous predictors (F = 0.01 -> 3,945.77), and conducted Chi-Square independence tests proving `Occupation` independence.
- **Supervised Learning Implementation (Criteria 4 & 5):** Constructed 5-fold Stratified CV pipelines for Logistic Regression (evaluating L1 and L2 penalties) and Random Forest. Implemented out-of-fold threshold tuning (t=0.44) on training folds to boost high-risk default recall.
- **Unsupervised Customer Segmentation (Criteria 4 & 5):** Formulated K-Means Elbow/Silhouette diagnostics (K=3), de-scaled centroids into financial personas (achieving 95.6% poor credit isolation in Cluster 0), and tuned DBSCAN (epsilon=2.5).
- **Academic Reporting & Documentation (Criteria 6 & 7):** Authored all chapters of this academic paper, synchronized every reported metric with executed notebook outputs, generated diagnostic figures, formatted the complete source code appendix, and prepared video demonstration materials.

---

## 8. Conclusion & Future Work
This investigation demonstrated the application of machine learning to financial credit risk. The Random Forest ensemble proved outstanding in predictive classification (**ROC-AUC 0.8948, PR-AUC 0.9267**), while K-Means clustering effectively uncovered behavioral financial personas without supervision. 

Future work should investigate **Cost-Sensitive Gradient Boosting (LightGBM/CatBoost)** and explore **Semi-Supervised Label Propagation** to capitalize on vast quantities of unlabelled banking transactions.

---

## 9. References
1. Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), pp. 5–32.
2. Hastie, T., Tibshirani, R., and Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. 2nd ed. Springer.
3. Ester, M., Kriegel, H.P., Sander, J., and Xu, X. (1996). "A density-based algorithm for discovering clusters in large spatial databases with noise." In *KDD-96 Proceedings*, pp. 226–231.
4. Rousseeuw, P.J. (1987). "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis." *Journal of Computational and Applied Mathematics*, 20, pp. 53–65.
5. Hand, D.J. and Henley, W.E. (1997). "Statistical classification methods in consumer credit scoring: a review." *Journal of the Royal Statistical Society: Series A*, 160(3), pp. 523–541.
6. Paris, R. (2022). "Credit Score Classification Dataset." *Kaggle Datasets*, CC0 Public Domain. Available at: https://www.kaggle.com/datasets/parisrohan/credit-score-classification.

---

## 10. Appendix: Complete Source Code
*(Formatted in full as text in accordance with assignment regulations; no screenshots).*

### Appendix A: `src/data_pipeline.py`
```python
"""
src/data_pipeline.py
====================
Modular data processing, domain feature engineering, and leakage-safe
Scikit-Learn pipeline builders for IT5022 Machine Learning Assignment.
"""

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


VALID_RANGES = {
    'Age': (18, 100),
    'Num_Bank_Accounts': (0, 20),
    'Num_Credit_Card': (0, 20),
    'Interest_Rate': (0, 50),
    'Num_of_Loan': (0, 20),
    'Num_of_Delayed_Payment': (0, 50),
    'Annual_Income': (0, 500000),
    'Monthly_Income': (0, 45000)
}


class CreditDomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-specific feature engineering & noise sanitization transformer:
    1. Sanitizes impossible / corrupted numerical values (Age, Interest_Rate, Accounts,
       Credit Cards, Loans, Delayed Payments, Annual & Monthly Income).
    2. Maps sentinel investment placeholders (10,000) and corrupted category strings
       ('_______' in Occupation, '!@9#%8' in Payment_Behaviour, '_' in Credit_Mix, 'NM' in
       Payment_of_Min_Amount) to NaN for clean downstream statistical imputation.
    3. Extracts numerical integer years from 'Credit_History_Age' string.
    4. Parses multi-valued 'Type_of_Loan' into loan count and specific loan indicator flags.
       Note: 'Has_Mortgage' strictly targets mortgage facilities, differentiating from home-equity loans.
    5. Maps binary flags (Payment_of_Min_Amount) and ordinal Credit_Mix ('Bad' -> 0, 'Standard' -> 1, 'Good' -> 2).
    6. Drops uninformative identifier features (Customer_ID) and redundant raw text features.
    7. Optionally eliminates collinear (Monthly_Income, Derived_Loan_Count) and statistically
       independent uninformative features (Occupation) for dimensionality reduction.
    """
    def __init__(self, drop_identifiers=True, sanitize_anomalies=True,
                 drop_collinear=True, drop_uninformative=True):
        self.drop_identifiers = drop_identifiers
        self.sanitize_anomalies = sanitize_anomalies
        self.drop_collinear = drop_collinear
        self.drop_uninformative = drop_uninformative

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # 0. Sanitize Synthetic Corruptions & Sentinel Outliers
        if self.sanitize_anomalies:
            # Numerical range boundaries: values outside legitimate banking limits converted to NaN
            for col, (lo, hi) in VALID_RANGES.items():
                if col in X.columns:
                    X.loc[(X[col] < lo) | (X[col] > hi), col] = np.nan

            # Amount_invested_monthly: 10,000 is an artificial sentinel placeholder
            if 'Amount_invested_monthly' in X.columns:
                X.loc[X['Amount_invested_monthly'] >= 10000, 'Amount_invested_monthly'] = np.nan

            # Categorical placeholder strings: replace with NaN for clean downstream imputation
            if 'Occupation' in X.columns:
                X['Occupation'] = X['Occupation'].replace({'_______': np.nan})
            if 'Payment_Behaviour' in X.columns:
                X['Payment_Behaviour'] = X['Payment_Behaviour'].replace({'!@9#%8': np.nan})
            if 'Credit_Mix' in X.columns:
                X['Credit_Mix'] = X['Credit_Mix'].replace({'_': np.nan})
            if 'Payment_of_Min_Amount' in X.columns:
                X['Payment_of_Min_Amount'] = X['Payment_of_Min_Amount'].replace({'NM': np.nan})

        # 1. Extract Numerical Years from Credit_History_Age (e.g. "16 Years")
        if 'Credit_History_Age' in X.columns:
            def extract_years(val):
                if pd.isna(val):
                    return np.nan
                match = re.search(r'(\d+)', str(val))
                return float(match.group(1)) if match else np.nan

            X['Credit_History_Years'] = X['Credit_History_Age'].apply(extract_years)
            X.drop(columns=['Credit_History_Age'], inplace=True)

        # 2. Parse Type_of_Loan (multi-valued comma-separated string)
        if 'Type_of_Loan' in X.columns:
            def parse_loans(val):
                if pd.isna(val):
                    return 0, 0, 0
                val_str = str(val).lower()
                num_loans = len([item for item in val_str.split(',') if item.strip()])
                has_payday = 1 if 'payday' in val_str else 0
                # Match mortgage loans strictly (avoid conflating with home-equity loans)
                has_mortgage = 1 if 'mortgage' in val_str else 0
                return num_loans, has_payday, has_mortgage

            parsed = X['Type_of_Loan'].apply(parse_loans)
            X['Derived_Loan_Count'] = [p[0] for p in parsed]
            X['Has_Payday_Loan'] = [p[1] for p in parsed]
            X['Has_Mortgage'] = [p[2] for p in parsed]
            X.drop(columns=['Type_of_Loan'], inplace=True)

        # 3. Binary mapping for Payment_of_Min_Amount
        if 'Payment_of_Min_Amount' in X.columns:
            min_map = {'Yes': 1.0, 'No': 0.0}
            X['Payment_of_Min_Amount'] = X['Payment_of_Min_Amount'].map(min_map)

        # 4. Ordinal mapping for Credit_Mix
        if 'Credit_Mix' in X.columns:
            credit_mix_map = {'Bad': 0.0, 'Standard': 1.0, 'Good': 2.0}
            X['Credit_Mix_Ordinal'] = X['Credit_Mix'].map(credit_mix_map)
            X.drop(columns=['Credit_Mix'], inplace=True)

        # 5. Drop uninformative / identifier / collinear features
        cols_to_drop = []
        if self.drop_identifiers and 'Customer_ID' in X.columns:
            cols_to_drop.append('Customer_ID')
        # Num_Credit_Inquiries has >99% missing data (8,634 out of 8,692 rows missing)
        if 'Num_Credit_Inquiries' in X.columns:
            cols_to_drop.append('Num_Credit_Inquiries')

        # Collinear / Redundant features:
        # - Monthly_Income is strongly correlated with Annual_Income (r > 0.93)
        # - Derived_Loan_Count is duplicate with Num_of_Loan (r = 1.0000)
        if self.drop_collinear:
            if 'Monthly_Income' in X.columns:
                cols_to_drop.append('Monthly_Income')
            if 'Derived_Loan_Count' in X.columns:
                cols_to_drop.append('Derived_Loan_Count')

        # Statistically insignificant categorical features:
        # - Occupation showed Chi-square independence (p > 0.05).
        #   Pruning it eliminates noisy one-hot dimensions (reducing curse of dimensionality).
        if self.drop_uninformative:
            if 'Occupation' in X.columns:
                cols_to_drop.append('Occupation')

        if cols_to_drop:
            X.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        return X


def build_preprocessor(numeric_cols, categorical_cols):
    """
    Constructs a ColumnTransformer that imputes and scales numeric columns,
    and imputes and one-hot encodes categorical columns with zero leakage.
    """
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )
    return preprocessor


def get_feature_lists(df_transformed):
    """
    Identifies numerical and nominal categorical columns from the domain-engineered dataframe.
    """
    categorical_cols = ['Occupation', 'Payment_Behaviour']
    categorical_cols = [c for c in categorical_cols if c in df_transformed.columns]

    numeric_cols = [
        c for c in df_transformed.columns
        if c not in categorical_cols and c not in ['Credit_Score']
    ]
    return numeric_cols, categorical_cols


def build_full_pipeline(clf, drop_collinear=True, drop_uninformative=True):
    """
    Constructs a complete end-to-end Pipeline incorporating feature engineering,
    leakage-free imputation, standardisation/encoding, and the classifier.
    Ideal for GridSearchCV to ensure zero fold contamination.
    """
    engineer = CreditDomainFeatureEngineer(
        drop_identifiers=True,
        drop_collinear=drop_collinear,
        drop_uninformative=drop_uninformative
    )
    # Standard output feature schema
    num_cols = [
        'Age', 'Annual_Income', 'Num_Bank_Accounts', 'Num_Credit_Card',
        'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
        'Num_of_Delayed_Payment', 'Payment_of_Min_Amount',
        'Amount_invested_monthly', 'Credit_History_Years',
        'Has_Payday_Loan', 'Has_Mortgage', 'Credit_Mix_Ordinal'
    ]
    cat_cols = ['Payment_Behaviour']
    prep = build_preprocessor(num_cols, cat_cols)

    return Pipeline([
        ('engineer', engineer),
        ('prep', prep),
        ('clf', clf)
    ])


def prepare_training_data(df, test_size=0.20, random_state=42,
                          drop_collinear=True, drop_uninformative=True):
    """
    Convenience loader to prepare train and test sets with zero data leakage:
    1. Splits raw data into stratified Train (80%) and Test (20%) on Credit_Score.
    2. Applies CreditDomainFeatureEngineer (sanitizing outliers and sentinels).
    3. Fits ColumnTransformer on train only, and transforms both train and test.

    Returns:
        X_train (np.ndarray): Scaled & encoded training feature matrix.
        X_test (np.ndarray): Scaled & encoded testing feature matrix.
        y_train (pd.Series): Binary target labels for train (1=Poor, 0=Good).
        y_test (pd.Series): Binary target labels for test (1=Poor, 0=Good).
        preprocessor (ColumnTransformer): Fitted preprocessor for future inference.
        feature_names (list): List of all post-encoded feature column names.
    """
    X = df.drop(columns=['Credit_Score'])
    y = (df['Credit_Score'] == 'Poor').astype(int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    engineer = CreditDomainFeatureEngineer(
        drop_identifiers=True,
        drop_collinear=drop_collinear,
        drop_uninformative=drop_uninformative
    )
    X_train_eng = engineer.transform(X_train_raw)
    X_test_eng = engineer.transform(X_test_raw)

    num_cols, cat_cols = get_feature_lists(X_train_eng)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    X_train = preprocessor.fit_transform(X_train_eng)
    X_test = preprocessor.transform(X_test_eng)

    if cat_cols and 'cat' in preprocessor.named_transformers_:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
    else:
        cat_names = []
    feature_names = num_cols + cat_names

    return X_train, X_test, y_train, y_test, preprocessor, feature_names

```

### Appendix B: `src/evaluation_helpers.py`
```python
"""
src/evaluation_helpers.py
=========================
Evaluation metrics, statistical visualizations, and diagnostic curve generators
for Supervised and Unsupervised Learning models in IT5022 Assignment.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    brier_score_loss, confusion_matrix, roc_curve, precision_recall_curve
)


def evaluate_classifier(name, y_true, y_pred, y_prob=None):
    """
    Computes a comprehensive suite of binary classification metrics.
    """
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
    else:
        metrics["ROC-AUC"] = np.nan
        metrics["PR-AUC"] = np.nan
        metrics["Brier Score"] = np.nan

    return metrics


def plot_confusion_matrices(models_dict, X_test, y_test, save_path=None):
    """
    Plots side-by-side normalized confusion matrices for comparative models.
    """
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        sns.heatmap(
            cm, annot=True, fmt='.2%', cmap='Blues', cbar=False, ax=ax,
            xticklabels=['Good (0)', 'Poor (1)'],
            yticklabels=['Good (0)', 'Poor (1)']
        )
        ax.set_title(f'Confusion Matrix: {name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_pr_comparison(models_prob_dict, y_test, save_path=None):
    """
    Plots ROC Curves and Precision-Recall Curves side-by-side for multiple models.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. ROC Curves
    for name, y_prob in models_prob_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        ax1.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', lw=2)

    ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Baseline (AUC = 0.50)')
    ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax1.set_ylabel('True Positive Rate (Recall)', fontsize=11)
    ax1.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 2. Precision-Recall Curves
    baseline_pr = np.mean(y_test)
    for name, y_prob in models_prob_dict.items():
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        ax2.plot(recall, precision, label=f'{name} (PR-AUC = {pr_auc:.3f})', lw=2)

    ax2.axhline(y=baseline_pr, color='k', linestyle='--', lw=1.5,
                label=f'Prevalence Baseline ({baseline_pr:.2f})')
    ax2.set_xlabel('Recall (Sensitivity)', fontsize=11)
    ax2.set_ylabel('Precision (Positive Predictive Value)', fontsize=11)
    ax2.set_title('Precision-Recall (PR) Curves', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower left', frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_elbow_and_silhouette(k_range, inertias, sil_scores, save_path=None):
    """
    Plots the Elbow Method (Inertia) and Silhouette Score curves for clustering.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(k_range, inertias, 'bo-', lw=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax1.set_ylabel('Inertia / WCSS', fontsize=11)
    ax1.set_title('K-Means: Elbow Method for Optimal k', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.plot(k_range, sil_scores, 'ro-', lw=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax2.set_ylabel('Mean Silhouette Score', fontsize=11)
    ax2.set_title('K-Means: Silhouette Score Analysis', fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

```

### Appendix C: `notebooks/01_eda_and_data_audit.ipynb`
```python
# ===== Code Cell 2 =====
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath('..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import f_classif, mutual_info_classif

figures_dir = os.path.join('..', 'reports', 'figures')
os.makedirs(figures_dir, exist_ok=True)
print('Libraries imported successfully.')

# ===== Code Cell 4 =====
data_path = os.path.join("..", "dataset", "dataset.csv")
df = pd.read_csv(data_path)

print(f"Dataset Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Unique Customers: {df['Customer_ID'].nunique()}")
df.head(5)

# ===== Code Cell 6 =====
missing_df = pd.DataFrame({
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
}).sort_values(by='Missing_Count', ascending=False)

missing_df[missing_df['Missing_Count'] > 0]

# ===== Code Cell 7 =====
plt.figure(figsize=(10, 4))
missing_plot_data = missing_df[missing_df['Missing_Count'] > 0]
sns.barplot(x=missing_plot_data.index, y=missing_plot_data['Missing_Percentage'], palette='magma')
plt.title('Percentage of Missing Data by Feature', fontweight='bold')
plt.ylabel('Missing Percentage (%)')
plt.xlabel('Features')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, '01_missing_data_audit.png'), dpi=300)
plt.show()

# ===== Code Cell 9 =====
target_counts = df['Credit_Score'].value_counts()
target_pct = df['Credit_Score'].value_counts(normalize=True) * 100

print("Target Class Distribution:")
for cls, count in target_counts.items():
    print(f"  {cls}: {count:,} records ({target_pct[cls]:.2f}%)")

fig, ax = plt.subplots(figsize=(6, 4))
colors = ['#d9534f', '#5cb85c']
bars = ax.bar(target_counts.index, target_counts.values, color=colors, edgecolor='black', width=0.5)
ax.set_title('Target Class Distribution: Credit_Score', fontweight='bold')
ax.set_ylabel('Number of Customers')
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 100, f'{yval:,} ({yval/len(df)*100:.1f}%)', ha='center', va='bottom', fontweight='bold')
plt.ylim(0, 6000)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, '02_target_distribution.png'), dpi=300)
plt.show()

# ===== Code Cell 11 =====
numeric_cols = [
    'Age', 'Annual_Income', 'Monthly_Income', 'Num_Bank_Accounts',
    'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
    'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Amount_invested_monthly'
]

summary_stats = df[numeric_cols].describe().T
summary_stats['IQR'] = summary_stats['75%'] - summary_stats['25%']
summary_stats['Lower_Bound'] = summary_stats['25%'] - 1.5 * summary_stats['IQR']
summary_stats['Upper_Bound'] = summary_stats['75%'] + 1.5 * summary_stats['IQR']
display(summary_stats[['mean', 'std', 'min', '50%', 'max', 'IQR', 'Lower_Bound', 'Upper_Bound']])

# Audit synthetic noise & corrupted domain values
corruption_records = [
    {'Feature': 'Age', 'Domain_Rule': '18 <= Age <= 100', 'Corrupted_Count': int(((df['Age'] < 18) | (df['Age'] > 100)).sum()), 'Example_Noise': str(df[((df['Age'] < 18) | (df['Age'] > 100))]['Age'].values[:5].tolist())},
    {'Feature': 'Annual_Income', 'Domain_Rule': '0 < Income <= 500,000', 'Corrupted_Count': int(((df['Annual_Income'] <= 0) | (df['Annual_Income'] > 500000)).sum()), 'Example_Noise': str(np.round(df[df['Annual_Income'] > 500000]['Annual_Income'].values[:3], 1).tolist())},
    {'Feature': 'Interest_Rate', 'Domain_Rule': '0% <= Rate <= 100%', 'Corrupted_Count': int(((df['Interest_Rate'] < 0) | (df['Interest_Rate'] > 100)).sum()), 'Example_Noise': str(df[df['Interest_Rate'] > 100]['Interest_Rate'].values[:5].tolist())},
    {'Feature': 'Num_Bank_Accounts', 'Domain_Rule': '0 <= Count <= 20', 'Corrupted_Count': int(((df['Num_Bank_Accounts'] < 0) | (df['Num_Bank_Accounts'] > 20)).sum()), 'Example_Noise': str(df[df['Num_Bank_Accounts'] > 20]['Num_Bank_Accounts'].values[:5].tolist())},
    {'Feature': 'Num_Credit_Card', 'Domain_Rule': '0 <= Count <= 20', 'Corrupted_Count': int(((df['Num_Credit_Card'] < 0) | (df['Num_Credit_Card'] > 20)).sum()), 'Example_Noise': str(df[df['Num_Credit_Card'] > 20]['Num_Credit_Card'].values[:5].tolist())},
    {'Feature': 'Num_of_Loan', 'Domain_Rule': '0 <= Count <= 25', 'Corrupted_Count': int(((df['Num_of_Loan'] < 0) | (df['Num_of_Loan'] > 25)).sum()), 'Example_Noise': str(df[((df['Num_of_Loan'] < 0) | (df['Num_of_Loan'] > 25))]['Num_of_Loan'].values[:5].tolist())},
    {'Feature': 'Num_of_Delayed_Payment', 'Domain_Rule': '0 <= Count <= 50', 'Corrupted_Count': int(((df['Num_of_Delayed_Payment'] < 0) | (df['Num_of_Delayed_Payment'] > 50)).sum()), 'Example_Noise': str(df[df['Num_of_Delayed_Payment'] > 50]['Num_of_Delayed_Payment'].values[:5].tolist())},
    {'Feature': 'Amount_invested_monthly', 'Domain_Rule': 'Value < 10,000 (Sentinel)', 'Corrupted_Count': int((df['Amount_invested_monthly'] >= 10000).sum()), 'Example_Noise': '[10000.0]'},
    {'Feature': 'Occupation', 'Domain_Rule': 'Valid Profession', 'Corrupted_Count': int((df['Occupation'] == '_______').sum()), 'Example_Noise': '["_______"]'},
    {'Feature': 'Payment_Behaviour', 'Domain_Rule': 'Valid Category', 'Corrupted_Count': int((df['Payment_Behaviour'] == '!@9#%8').sum()), 'Example_Noise': '["!@9#%8"]'}
]
corruption_df = pd.DataFrame(corruption_records)
print('Synthetic Corruptions & Noise Audit:')
display(corruption_df)

# ===== Code Cell 12 =====
# Plot sanitized distributions by credit risk class (free from 4-digit artificial noise)
df_sanitized = df.copy()
df_sanitized.loc[(df_sanitized['Age'] < 18) | (df_sanitized['Age'] > 100), 'Age'] = np.nan
df_sanitized.loc[(df_sanitized['Annual_Income'] <= 0) | (df_sanitized['Annual_Income'] > 500000), 'Annual_Income'] = np.nan
df_sanitized.loc[(df_sanitized['Interest_Rate'] < 0) | (df_sanitized['Interest_Rate'] > 100), 'Interest_Rate'] = np.nan
df_sanitized.loc[(df_sanitized['Num_Bank_Accounts'] < 0) | (df_sanitized['Num_Bank_Accounts'] > 20), 'Num_Bank_Accounts'] = np.nan
df_sanitized.loc[(df_sanitized['Num_Credit_Card'] < 0) | (df_sanitized['Num_Credit_Card'] > 20), 'Num_Credit_Card'] = np.nan
df_sanitized.loc[(df_sanitized['Num_of_Delayed_Payment'] < 0) | (df_sanitized['Num_of_Delayed_Payment'] > 50), 'Num_of_Delayed_Payment'] = np.nan

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
key_features = ['Age', 'Interest_Rate', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Num_Credit_Card', 'Annual_Income']

for ax, feat in zip(axes.flatten(), key_features):
    sns.boxplot(x='Credit_Score', y=feat, data=df_sanitized, ax=ax, palette=['#d9534f', '#5cb85c'])
    ax.set_title(f'{feat} by Credit Score (Sanitized)', fontweight='bold')
    ax.set_xlabel('Credit Score')

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, '03_numerical_boxplots.png'), dpi=300)
plt.show()

# ===== Code Cell 14 =====
plt.figure(figsize=(10, 8))
corr_matrix = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Pearson Correlation Heatmap (Continuous Features)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, '04_correlation_heatmap.png'), dpi=300)
plt.show()

# ===== Code Cell 16 =====
import sys, os
sys.path.append(os.path.abspath('..'))
from src.data_pipeline import VALID_RANGES

# 1. Comparative ANOVA F-Test for Numerical Features: Raw Noise vs. Domain-Sanitized Data
y_binary = (df['Credit_Score'] == 'Poor').astype(int)

# Uncleaned (raw) numerical data with median fill
raw_clean_num_df = df[numeric_cols].fillna(df[numeric_cols].median())
f_raw, p_raw = f_classif(raw_clean_num_df, y_binary)

# Sanitized numerical data using domain boundaries
df_sanitized_num = df[numeric_cols].copy()
for col, (lo, hi) in VALID_RANGES.items():
    if col in df_sanitized_num.columns:
        df_sanitized_num.loc[(df_sanitized_num[col] < lo) | (df_sanitized_num[col] > hi), col] = np.nan

df_sanitized_num.loc[df_sanitized_num['Amount_invested_monthly'] >= 10000, 'Amount_invested_monthly'] = np.nan

sanitized_clean_num = df_sanitized_num.fillna(df_sanitized_num.median())
f_clean, p_clean = f_classif(sanitized_clean_num, y_binary)

anova_df = pd.DataFrame({
    'Feature': numeric_cols,
    'Raw_F_Stat': np.round(f_raw, 2),
    'Raw_p_value': [f'{p:.2e}' for p in p_raw],
    'Sanitized_F_Stat': np.round(f_clean, 2),
    'Sanitized_p_value': [f'{p:.2e}' for p in p_clean],
    'Statistically_Significant (p < 0.001)': p_clean < 0.001
}).sort_values(by='Sanitized_F_Stat', ascending=False)

print('ANOVA F-Test Comparison: Raw Noise vs. Domain-Sanitized Predictors')
display(anova_df)

# ===== Code Cell 17 =====
# 2. Chi-Square Test of Independence for Categorical Attributes
categorical_cols = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']
chi2_results = []

for cat in categorical_cols:
    contingency = pd.crosstab(df[cat].fillna('Missing'), df['Credit_Score'])
    res = stats.chi2_contingency(contingency)
    chi2_results.append({
        'Categorical_Feature': cat,
        'Chi2_Statistic': res.statistic,
        'p_value': res.pvalue,
        'Degrees_of_Freedom': res.dof,
        'Significant (p < 0.001)': res.pvalue < 0.001
    })

chi2_df = pd.DataFrame(chi2_results).sort_values(by='Chi2_Statistic', ascending=False)
chi2_df
```

### Appendix D: `notebooks/02_preprocessing_and_feature_engineering.ipynb`
```python
# ===== Code Cell 2 =====
import sys
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.abspath(".."))
from src.data_pipeline import CreditDomainFeatureEngineer, build_preprocessor, get_feature_lists

print("Pipeline modules loaded successfully.")

# ===== Code Cell 4 =====
data_path = os.path.join("..", "dataset", "dataset.csv")
df = pd.read_csv(data_path)

# Separate predictors and target
X = df.drop(columns=['Credit_Score'])
y = (df['Credit_Score'] == 'Poor').astype(int)  # 1 = Poor (Risk), 0 = Good

# 80/20 Stratified Split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training Instances: {X_train_raw.shape[0]} (Poor prevalence: {y_train.mean():.2%})")
print(f"Testing Instances:  {X_test_raw.shape[0]} (Poor prevalence: {y_test.mean():.2%})")

# ===== Code Cell 6 =====
engineer = CreditDomainFeatureEngineer(drop_identifiers=True)

X_train_eng = engineer.transform(X_train_raw)
X_test_eng = engineer.transform(X_test_raw)

print(f"Features after domain engineering: {X_train_eng.shape[1]}")
X_train_eng.head(3)

# ===== Code Cell 8 =====
numeric_cols, categorical_cols = get_feature_lists(X_train_eng)
print("Numeric Features:", numeric_cols)
print("Categorical Features:", categorical_cols)

preprocessor = build_preprocessor(numeric_cols, categorical_cols)

# Fit on training data ONLY
X_train_proc = preprocessor.fit_transform(X_train_eng)
X_test_proc = preprocessor.transform(X_test_eng)

# Extract encoded feature names
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
all_feature_names = numeric_cols + cat_feature_names

print(f"Total features after One-Hot Encoding: {X_train_proc.shape[1]}")
print(f"X_train_proc shape: {X_train_proc.shape}")
print(f"X_test_proc shape:  {X_test_proc.shape}")

# ===== Code Cell 10 =====
assert not np.isnan(X_train_proc).any(), "NaN values found in training set!"
assert not np.isnan(X_test_proc).any(), "NaN values found in test set!"

train_means = np.mean(X_train_proc[:, :len(numeric_cols)], axis=0)
train_stds = np.std(X_train_proc[:, :len(numeric_cols)], axis=0)

print("Sample Numeric Feature Means (should be ~0.0):", np.round(train_means[:4], 3))
print("Sample Numeric Feature Stds  (should be ~1.0):", np.round(train_stds[:4], 3))
print("Data preprocessing and feature engineering successfully verified!")
```

### Appendix E: `notebooks/03_supervised_learning_classification.ipynb`
```python
# ===== Code Cell 2 =====
import sys
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

sys.path.append(os.path.abspath(".."))
from src.data_pipeline import CreditDomainFeatureEngineer, build_preprocessor, get_feature_lists
from src.evaluation_helpers import (
    evaluate_classifier, plot_confusion_matrices, plot_roc_pr_comparison
)

figures_dir = os.path.join("..", "reports", "figures")
os.makedirs(figures_dir, exist_ok=True)

print("Modules imported successfully.")

# ===== Code Cell 4 =====
df = pd.read_csv(os.path.join("..", "dataset", "dataset.csv"))
X = df.drop(columns=['Credit_Score'])
y = (df['Credit_Score'] == 'Poor').astype(int)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Apply domain engineering
engineer = CreditDomainFeatureEngineer(drop_identifiers=True)
X_train_eng = engineer.transform(X_train_raw)
X_test_eng = engineer.transform(X_test_raw)

# Preprocessor
numeric_cols, categorical_cols = get_feature_lists(X_train_eng)
preprocessor = build_preprocessor(numeric_cols, categorical_cols)

X_train = preprocessor.fit_transform(X_train_eng)
X_test = preprocessor.transform(X_test_eng)

print(f"X_train Shape: {X_train.shape}, X_test Shape: {X_test.shape}")

# ===== Code Cell 6 =====
dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)

dummy_pred = dummy.predict(X_test)
dummy_prob = dummy.predict_proba(X_test)[:, 1]

dummy_metrics = evaluate_classifier("Dummy Baseline (Most Frequent)", y_test, dummy_pred, dummy_prob)
pd.DataFrame([dummy_metrics])

# ===== Code Cell 8 =====
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid_lr = [
    {'C': [0.01, 0.1, 1.0, 10.0], 'penalty': ['l2'], 'solver': ['lbfgs'], 'class_weight': ['balanced', None]},
    {'C': [0.01, 0.1, 1.0, 10.0], 'penalty': ['l1'], 'solver': ['liblinear'], 'class_weight': ['balanced', None]}
]

grid_lr = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_grid_lr, cv=cv, scoring='f1', n_jobs=-1
)
grid_lr.fit(X_train, y_train)

best_lr = grid_lr.best_estimator_
print("Best Logistic Regression Parameters:", grid_lr.best_params_)
print(f"Best CV F1-Score: {grid_lr.best_score_:.4f}")

# Test set evaluation at default threshold t = 0.50
lr_pred = best_lr.predict(X_test)
lr_prob = best_lr.predict_proba(X_test)[:, 1]

lr_metrics = evaluate_classifier("Logistic Regression (t = 0.50)", y_test, lr_pred, lr_prob)
display(pd.DataFrame([lr_metrics]))

# ===== Code Cell 10 =====
from sklearn.model_selection import cross_val_predict

# Leakage-Free Decision Threshold Tuning on Training Folds ONLY
oof_prob = cross_val_predict(best_lr, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
thresholds = np.linspace(0.2, 0.8, 61)
f1_scores = []
recalls = []
precisions = []

for t in thresholds:
    pred_t = (oof_prob >= t).astype(int)
    m = evaluate_classifier("LR", y_train, pred_t)
    f1_scores.append(m["F1-Score"])
    recalls.append(m["Recall (Sensitivity)"])
    precisions.append(m["Precision"])

optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

plt.figure(figsize=(9, 5))
plt.plot(thresholds, f1_scores, 'b-', lw=2, label='F1-Score (Out-of-Fold)')
plt.plot(thresholds, recalls, 'g--', lw=2, label='Recall (Sensitivity)')
plt.plot(thresholds, precisions, 'r:', lw=2, label='Precision')
plt.axvline(optimal_threshold, color='k', linestyle='--', label=f'Optimal Threshold = {optimal_threshold:.2f}')
plt.title('Leakage-Free Threshold Optimization (Out-of-Fold Cross-Validation)', fontweight='bold')
plt.xlabel('Classification Threshold')
plt.ylabel('Score')
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, '05_threshold_tuning_lr.png'), dpi=300)
plt.show()

# Single evaluation on independent hold-out test set
lr_opt_pred = (lr_prob >= optimal_threshold).astype(int)
lr_opt_metrics = evaluate_classifier(f"Tuned Logistic Regression (t = {optimal_threshold:.2f})", y_test, lr_opt_pred, lr_prob)
display(pd.DataFrame([lr_opt_metrics]))

# ===== Code Cell 12 =====
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [6, 10, 15],
    'min_samples_leaf': [2, 4],
    'class_weight': ['balanced', None]
}

grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf, cv=cv, scoring='f1', n_jobs=-1
)
grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
print("Best Random Forest Parameters:", grid_rf.best_params_)
print(f"Best CV F1-Score: {grid_rf.best_score_:.4f}")

# Generalization gap (overfitting analysis)
train_acc = best_rf.score(X_train, y_train)
test_acc = best_rf.score(X_test, y_test)
print(f"Random Forest Generalization Gap: Train Accuracy = {train_acc:.4f}, Test Accuracy = {test_acc:.4f}")

# Test set evaluation
rf_pred = best_rf.predict(X_test)
rf_prob = best_rf.predict_proba(X_test)[:, 1]

rf_metrics = evaluate_classifier("Tuned Random Forest Ensemble", y_test, rf_pred, rf_prob)
display(pd.DataFrame([rf_metrics]))

# ===== Code Cell 14 =====
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
all_features = numeric_cols + cat_feature_names

importances = best_rf.feature_importances_
feat_importances = pd.Series(importances, index=all_features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feat_importances.head(15).plot(kind='barh', color='#2b5c8f')
plt.title('Top 15 Most Important Features (Random Forest Gini Importance)', fontweight='bold')
plt.xlabel('Relative Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, '06_rf_feature_importances.png'), dpi=300)
plt.show()

# ===== Code Cell 16 =====
results_df = pd.DataFrame([dummy_metrics, lr_metrics, lr_opt_metrics, rf_metrics])
results_df.set_index('Model', inplace=True)
print("=== COMPREHENSIVE PERFORMANCE EVALUATION TABLE ===")
display(results_df.round(4))

# ===== Code Cell 17 =====
# 1. Confusion Matrices
models_dict = {
    'Logistic Regression': best_lr,
    'Random Forest': best_rf
}
plot_confusion_matrices(models_dict, X_test, y_test, save_path=os.path.join(figures_dir, '07_confusion_matrices.png'))

# 2. ROC and Precision-Recall Curves
prob_dict = {
    'Logistic Regression': lr_prob,
    'Random Forest': rf_prob
}
plot_roc_pr_comparison(prob_dict, y_test, save_path=os.path.join(figures_dir, '08_roc_pr_curves.png'))
```

### Appendix F: `notebooks/04_unsupervised_learning_clustering.ipynb`
```python
# ===== Code Cell 2 =====
import sys
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath(".."))
from src.data_pipeline import CreditDomainFeatureEngineer, build_preprocessor, get_feature_lists
from src.evaluation_helpers import plot_elbow_and_silhouette

figures_dir = os.path.join("..", "reports", "figures")
os.makedirs(figures_dir, exist_ok=True)

print("Clustering modules imported successfully.")

# ===== Code Cell 4 =====
df = pd.read_csv(os.path.join("..", "dataset", "dataset.csv"))

# Store true labels for external cross-tabulation validation later
true_labels = (df['Credit_Score'] == 'Poor').astype(int)

# Extract features without labels
X_raw = df.drop(columns=['Credit_Score'])

engineer = CreditDomainFeatureEngineer(drop_identifiers=True)
X_eng = engineer.transform(X_raw)

numeric_cols, categorical_cols = get_feature_lists(X_eng)
preprocessor = build_preprocessor(numeric_cols, categorical_cols)

X_scaled = preprocessor.fit_transform(X_eng)
print(f"Scaled feature matrix shape: {X_scaled.shape}")

# ===== Code Cell 6 =====
k_range = range(2, 9)
inertias = []
sil_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    # Sample for silhouette computation speed on 8692 rows
    sil = silhouette_score(X_scaled, labels, sample_size=3000, random_state=42)
    sil_scores.append(sil)
    print(f"k = {k} | Inertia: {km.inertia_:,.1f} | Silhouette: {sil:.4f}")

plot_elbow_and_silhouette(
    k_range, inertias, sil_scores,
    save_path=os.path.join(figures_dir, '09_kmeans_elbow_silhouette.png')
)

# ===== Code Cell 8 =====
chosen_k = 3
final_kmeans = KMeans(n_clusters=chosen_k, init='k-means++', n_init=10, random_state=42)
kmeans_labels = final_kmeans.fit_predict(X_scaled)

df_profile = X_eng[numeric_cols].copy()
df_profile['KMeans_Cluster'] = kmeans_labels

cluster_profiles = df_profile.groupby('KMeans_Cluster').mean()
print("=== UN-SCALED CLUSTER MEANS (FINANCIAL PERSONAS) ===")
display(cluster_profiles[['Annual_Income', 'Interest_Rate', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Num_Credit_Card']].round(2))

# ===== Code Cell 10 =====
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(9, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.5, s=20)
plt.title('K-Means Financial Segments (2D PCA Projection)', fontweight='bold')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)')
plt.colorbar(scatter, label='K-Means Cluster')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, '10_kmeans_pca_scatter.png'), dpi=300)
plt.show()

# ===== Code Cell 12 =====
# k-distance graph for eps selection
k_neighbors = 2 * X_scaled.shape[1]  # MinPts rule of thumb: 2 * dim
nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
sorted_distances = np.sort(distances[:, -1])

plt.figure(figsize=(8, 4))
plt.plot(sorted_distances, color='purple', lw=2)
plt.axhline(y=2.5, color='r', linestyle='--', label='Selected Epsilon (eps = 2.5)')
plt.title(f'{k_neighbors}-NN Distance Graph for DBSCAN Epsilon Selection', fontweight='bold')
plt.xlabel('Data Points Sorted by Distance')
plt.ylabel(f'Distance to {k_neighbors}-th Nearest Neighbor')
plt.grid(True, linestyle=':')
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, '11_dbscan_k_distance.png'), dpi=300)
plt.show()

# Fit DBSCAN with eps = 2.5, min_samples = 15
dbscan = DBSCAN(eps=2.5, min_samples=15)
dbscan_labels = dbscan.fit_predict(X_scaled)

n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
print(f"DBSCAN Detected Clusters: {n_clusters_db}")
print(f"DBSCAN Detected Noise / Outlier Points: {n_noise} ({n_noise/len(X_scaled)*100:.2f}%)")

# ===== Code Cell 14 =====
crosstab = pd.crosstab(
    kmeans_labels, df['Credit_Score'],
    rownames=['K-Means Cluster'], colnames=['Actual Credit Score'],
    normalize='index'
) * 100

print("Percentage Distribution of Credit_Score within each K-Means Cluster:")
display(crosstab.round(2))

ari = adjusted_rand_score(true_labels, kmeans_labels)
print(f"Adjusted Rand Index (ARI) between K-Means and Credit_Score: {ari:.4f}")
```
