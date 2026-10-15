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
