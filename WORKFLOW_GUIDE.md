# Machine Learning End-to-End Workflow & Pipeline Guide
## Project: Real-World Credit Risk Scoring & Customer Financial Segmentation
### Module: IT5022 — Fundamentals of Machine Learning

---

## 📌 Executive Overview for Beginners

Think of building a machine learning system like **cooking a five-star meal**:
1. **Raw Ingredients (Raw Data):** Dirty, mixed with peel, stones, and expired items.
2. **Prep & Cleaning (Preprocessing & Pruning):** Washing, peeling, and throwing away rotten or duplicate ingredients so the dish tastes clean.
3. **Cooking (Model Training):** Putting the ingredients into the oven or pan so the machine learns patterns.
4. **Taste Test (Evaluation):** Checking if the food is good without testing it on contaminated plates (zero data leakage).

---

## 🗺️ Master Visual Workflow Diagram

The complete end-to-end data lifecycle from raw Excel data to final prediction models and clustering personas:

```mermaid
flowchart TD
    RAW["1. Raw Dataset (8,692 rows × 19 cols)<br/>Age = -500, 5774% Interest, 99.3% Missing"]
    --> SPLIT["2. Leakage-Free Stratified Split (80/20)<br/>Train: 6,953 rows | Test: 1,739 rows (Locked)"]

    SPLIT --> SANITIZE["3. Domain Sanitization<br/>• Age bounded [18, 100]<br/>• Sentinel values (10,000) mapped to NaN<br/>• Monthly_Income = Annual_Income / 12<br/>• Regex parse '16 Years' → 16.0"]

    SANITIZE --> PRUNE{"4. Pruning Unnecessary Items<br/>(Feature Dimensionality Reduction)"}

    PRUNE -->|"Primary Key"| P1["Drop Customer_ID<br/>(Prevents ID memorization)"]
    PRUNE -->|"99.3% Missing"| P2["Drop Num_Credit_Inquiries<br/>(Uninformative blank column)"]
    PRUNE -->|"100% Duplicate"| P3["Drop Derived_Loan_Count<br/>(Identical to Num_of_Loan: r = 1.000)"]
    PRUNE -->|"Multicollinear"| P4["Drop Monthly_Income<br/>(Redundant with Annual_Income: r = 0.986)"]
    PRUNE -->|"Statistical Noise"| P5["Drop Occupation<br/>(Chi-Square p = 0.347: No effect on default)"]

    P1 & P2 & P3 & P4 & P5 --> PIPE["5. Scikit-Learn ColumnTransformer<br/>(Fit on Training Fold ONLY)"]

    PIPE --> NUM["Numeric Pipeline (14 Features)<br/>• SimpleImputer (Median)<br/>• StandardScaler (Mean = 0, Std = 1)"]
    PIPE --> CAT["Categorical Pipeline (1 Feature)<br/>• SimpleImputer (Most Frequent)<br/>• OneHotEncoder (Payment_Behaviour)"]

    NUM & CAT --> MATRIX["Clean Feature Matrix: 21 Features<br/>(Reduced by 46% from 39: Meets Criterion 3)"]

    MATRIX --> SUP["6. Supervised Learning<br/>Predict Credit_Score (Poor vs Good)"]
    MATRIX --> UNSUP["7. Unsupervised Learning<br/>Discover Customer Segments"]

    SUP --> M1["Model 1: Logistic Regression<br/>• Linear baseline + L2 penalty<br/>• Threshold tuned: t = 0.58"]
    SUP --> M2["Model 2: Random Forest<br/>• 200 Decision Trees ensemble<br/>• Captures non-linear risk rules"]

    UNSUP --> U1["Algorithm 1: K-Means (K = 3)<br/>• Elbow Inertia & Silhouette score<br/>• Discovers 3 financial personas"]
    UNSUP --> U2["Algorithm 2: DBSCAN<br/>• eps = 2.1, min_samples = 15<br/>• Isolates fraud / extreme outliers"]

    M1 & M2 & U1 & U2 --> OUT["8. Final Decision System<br/>• High-precision loan approvals<br/>• Tailored banking products<br/>• Academic Report Figures"]
```

---

## 🔍 Stage-by-Stage Breakdown

### Stage 1: Ingestion of Raw Data & Audit

The dataset represents retail banking records with **8,692 instances** and **19 attributes**.

```mermaid
graph LR
    FILE["dataset.csv (8,692 × 19)"] --> DEFECTS["Identified Data Defects:<br/>• Synthetic corruptions (-500 age, 5774% interest rate)<br/>• Severe missingness (Num_Credit_Inquiries = 99.3% blank)<br/>• Unstructured text ('16 Years', comma-separated loans)<br/>• Categorical placeholders ('_______', '!@9#%8')"]
```

* **Why we can't train immediately:** If raw text, missing numbers, and artificial corruptions are passed to machine learning algorithms, mathematical solvers fail or produce completely flawed models (*"Garbage In, Garbage Out"*).

---

### Stage 2: The Golden Rule: Prevent "Data Leakage"

Before performing any cleaning, scaling, or calculations, we split the data into **80% Training ($n=6,953$)** and **20% Test ($n=1,739$)** using **Stratified Sampling** on `Credit_Score`.

```mermaid
graph TD
    DATA["Full Raw Dataset (8,692 Rows)"] --> SPLIT["Stratified 80/20 Train-Test Split"]
    SPLIT --> TRAIN["Training Set: 6,953 Rows (80%)<br/>👉 ALL Transformers and Models FIT HERE ONLY"]
    SPLIT --> TEST["Test Hold-Out: 1,739 Rows (20%)<br/>🔒 LOCKED IN VAULT (Never seen during training)"]
```

> **Beginner Analogy:**  
> Think of taking a driving test. If the instructor shows you the exact test route and grading marks beforehand, you can memorize the turns without knowing how to actually drive in real traffic. In ML, calculating averages (means) or scalers across the full dataset before splitting is called **Data Leakage**. It creates an illusion of high test accuracy that fails in production.

---

### Stage 3: Domain Sanitization & Feature Engineering

Using `CreditDomainFeatureEngineer`:
1. **Age Bounding:** Enforce realistic borrowing range $[18, 100]$. Ages $<18$ or $>100$ become `NaN` for median imputation.
2. **Deterministic Income Recovery:**
   $$\text{Monthly\_Income} = \frac{\text{Annual\_Income}}{12}$$
   Recovers 1,351 missing values using exact banking arithmetic.
3. **Regex Extraction:** Parse string `"16 Years"` into float `16.0`.
4. **Loan Parsing:** Parse multi-label string `"Payday-Loan, Auto-Loan"` to extract indicator flags:
   * `Has_Payday_Loan = 1` (detects high-risk borrowing).
   * `Has_Mortgage = 1` (detects stable real-estate collateral).
5. **Ordinal Encoding:** Map `Credit_Mix` (`Bad` $\to 0$, `Standard` $\to 1$, `Good` $\to 2$).

---

### Stage 4: Pruning Unnecessary Items (Why & How We Did It)

Here is why and how we removed unnecessary features:

```mermaid
graph TD
    subgraph "Why We Removed 5 Unnecessary Items"
        U1["Customer_ID"] -->|"Identifier"| D1["Drops memorization risk"]
        U2["Num_Credit_Inquiries"] -->|"99.3% Missing"| D2["Drops fabricated data"]
        U3["Derived_Loan_Count"] -->|"100% Duplicate (r = 1.0)"| D3["Drops redundant signal"]
        U4["Monthly_Income"] -->|"Multicollinear (r = 0.986)"| D4["Prevents income double-counting"]
        U5["Occupation"] -->|"Chi-Square p = 0.347"| D5["Removes 16 noisy dummy dimensions"]
    end
```

#### The 5 Removals Explained Simply:
1. **`Customer_ID` (Random Primary Key):**
   * *Reason:* Unique IDs carry zero generalized financial knowledge. Models that use customer IDs tend to memorize training rows instead of patterns.
2. **`Num_Credit_Inquiries` (99.3% Missing Data):**
   * *Reason:* 8,634 out of 8,692 rows are empty. Imputing 99.3% of a column is essentially fabricating fake numbers.
3. **`Derived_Loan_Count` (Duplicate Feature, $r = 1.0000$):**
   * *Reason:* Counting loans from the text string produced the exact same count as the existing `Num_of_Loan` column ($r = 1.0000$). Retaining both tells the model the exact same number twice.
4. **`Monthly_Income` (Extreme Multicollinearity, $r = 0.9865$):**
   * *Reason:* `Monthly_Income` is literally `Annual_Income / 12`. In **Logistic Regression**, collinearity inflates standard errors and makes coefficients unstable. In **K-Means / DBSCAN**, Euclidean distance would count income twice as much as any other factor. We keep `Annual_Income` and drop `Monthly_Income`.
5. **`Occupation` (Statistically Insignificant Noise, $p = 0.347$):**
   * *Reason:* In our statistical audit, the **Chi-Square test of independence** showed $\chi^2 = 16.54, p = 0.347$ (well above $\alpha = 0.05$). Job title does not determine credit default risk.
   * *Danger of keeping it:* One-hot encoding 15 job titles creates **16 extra sparse dummy columns** of 0s and 1s. This causes the **Curse of Dimensionality**, spreading data points too thin for clustering to work.

**Marking Rubric Alignment:**  
This deliberate pruning reduces the feature space from **39 down to 21 dimensions** (a 46% dimensionality reduction), directly fulfilling **Criterion 3** of the MSc Marking Rubric (*"Data Preprocessing: applying necessary cleansing of the data, dimensionality reduction, etc"*).

---

### Stage 5: Encoding & Scaling (ColumnTransformer)

Computers need numbers on comparable scales. An annual income of $\$50,000$ should not mathematically overpower an interest rate of $15\%$.

```mermaid
graph LR
    INPUT["21 Pruned Features"] --> NUM["14 Numeric Features<br/>(Age, Annual_Income, Interest_Rate, etc.)"]
    INPUT --> CAT["1 Categorical Feature<br/>(Payment_Behaviour)"]

    NUM --> N_IMP["SimpleImputer (strategy='median')"]
    N_IMP --> N_SCL["StandardScaler (Mean = 0, Std = 1)"]

    CAT --> C_IMP["SimpleImputer (strategy='most_frequent')"]
    C_IMP --> C_OHE["OneHotEncoder (Creates 7 binary columns)"]

    N_SCL & C_OHE --> MATRIX["Final Processed Matrix: 21 Features"]
```

* **StandardScaler:** Normalizes each column so its mean is $0$ and standard deviation is $1$.
* **OneHotEncoder:** Encodes `Payment_Behaviour` (which is statistically significant: $p < 10^{-70}$) into 7 clean binary indicator columns.

---

### Stage 6: Supervised Learning (Predicting Default Risk)

We train **two distinct algorithms** to predict `Credit_Score` (`1 = Poor / Risk`, `0 = Good`):

```mermaid
graph TD
    MATRIX["Scaled Feature Matrix (6,953 rows × 21 cols)"] --> M1["Model 1: Logistic Regression<br/>(Linear Baseline)"]
    MATRIX --> M2["Model 2: Random Forest<br/>(Non-Linear Ensemble)"]

    M1 --> TUNE1["5-Fold Stratified Cross-Validation<br/>• Tuning Regularization (C=10.0, Penalty='l2')<br/>• Decision Threshold Tuning (t=0.58)"]
    M2 --> TUNE2["Hyperparameter Optimization<br/>• 200 Trees, max_depth = 12<br/>• Gini Feature Importance Ranking"]

    TUNE1 & TUNE2 --> EVAL["Evaluation on Locked 20% Test Hold-Out (1,739 rows)"]
```

#### Results Comparison:
| Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline (Most Frequent)** | 59.63% | 59.63% | 100.00% | 0.7471 | 0.5000 |
| **Tuned Logistic Regression** | **80.74%** | **84.62%** | **82.35%** | **0.8349** | **0.8815** |
| **Tuned Random Forest (200 Trees)**| **82.81%** | **86.41%** | **84.00%** | **0.8518** | **0.8995** |

*Pruning the unnecessary items improved Logistic Regression Accuracy from 80.22% to 80.74% and F1-Score from 0.8317 to 0.8349.*

---

### Stage 7: Unsupervised Learning (Clustering Customer Personas)

We hide the `Credit_Score` labels completely. The models must discover natural groupings purely from financial geometry:

```mermaid
graph TD
    UNLABELED["Unlabelled Features (8,692 rows × 21 cols)"] --> KM["Algorithm 1: K-Means (Centroid-based)"]
    UNLABELED --> DB["Algorithm 2: DBSCAN (Density-based)"]

    KM --> EVAL_K["Elbow WCSS Curve + Silhouette Analysis<br/>Optimal k = 3 Clusters"]
    EVAL_K --> PERSONAS["3 Customer Personas Discovered:<br/>• Cluster 0: Overleveraged High-Risk Revolvers (84% Poor)<br/>• Cluster 1: Prime Affluent Savers (12% Poor)<br/>• Cluster 2: Inactive Minimalists (56% Poor)"]

    DB --> K_DIST["k-Distance Nearest-Neighbor Graph<br/>Tuning eps = 2.1, min_samples = 15"]
    DB --> OUTLIERS["Outlier Detection:<br/>Isolates severe financial anomalies / fraud (Cluster -1)<br/>Groups core dense clusters"]
```

---

## 📁 Repository File Map

```
IT5022_Fundamentals_of_Machine_Learning_Assignment/
│
├── dataset/
│   ├── dataset.xlsx                              # Original raw dataset (8,692 × 19)
│   └── dataset.csv                               # Fast CSV mirror
│
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py                          # CreditDomainFeatureEngineer & build_preprocessor (21 features)
│   └── evaluation_helpers.py                     # Evaluation functions (CM, PR/ROC, Elbow/Silhouette)
│
├── notebooks/
│   ├── 01_eda_and_data_audit.ipynb               # Statistical audit, Chi-square & ANOVA tests
│   ├── 02_preprocessing_and_feature_engineering.ipynb # Leakage-safe split & 21-feature ColumnTransformer
│   ├── 03_supervised_learning_classification.ipynb    # Logistic Regression vs Random Forest training & tuning
│   └── 04_unsupervised_learning_clustering.ipynb      # K-Means (k=3) and DBSCAN clustering
│
├── reports/
│   ├── figures/                                  # 11 high-res academic diagnostic PNGs
│   └── IT5022_Credit_Risk_Report.md              # Complete Master's Academic Report
│
├── ASSIGNMENT_IMPLEMENTATION_PLAN.md             # Detailed assignment roadmap & rubric mapping
├── README.md                                     # Project overview & quickstart
└── WORKFLOW_GUIDE.md                             # This guide with visual diagrams
```
