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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


class CreditDomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-specific feature engineering transformer:
    1. Imputes missing Monthly_Income using Annual_Income / 12.
    2. Extracts numerical integer years from 'Credit_History_Age' string.
    3. Parses multi-valued 'Type_of_Loan' into loan count and specific high-risk loan flags (e.g. Payday loans).
    4. Encodes binary flags (Payment_of_Min_Amount).
    5. Maps ordinal Credit_Mix ('Bad' -> 0, 'Standard' -> 1, 'Good' -> 2).
    6. Drops identifier columns (Customer_ID) and redundant raw text features.
    """
    def __init__(self, drop_identifiers=True):
        self.drop_identifiers = drop_identifiers

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # 1. Domain Imputation: Monthly_Income
        if 'Monthly_Income' in X.columns and 'Annual_Income' in X.columns:
            missing_mask = X['Monthly_Income'].isnull()
            X.loc[missing_mask, 'Monthly_Income'] = X.loc[missing_mask, 'Annual_Income'] / 12.0

        # 2. Extract Numerical Years from Credit_History_Age (e.g. "16 Years")
        if 'Credit_History_Age' in X.columns:
            def extract_years(val):
                if pd.isna(val):
                    return np.nan
                match = re.search(r'(\d+)', str(val))
                return float(match.group(1)) if match else np.nan

            X['Credit_History_Years'] = X['Credit_History_Age'].apply(extract_years)
            X.drop(columns=['Credit_History_Age'], inplace=True)

        # 3. Parse Type_of_Loan (multi-valued comma-separated string)
        if 'Type_of_Loan' in X.columns:
            def parse_loans(val):
                if pd.isna(val):
                    return 0, 0, 0
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

        # 4. Binary mapping for Payment_of_Min_Amount
        if 'Payment_of_Min_Amount' in X.columns:
            X['Payment_of_Min_Amount'] = X['Payment_of_Min_Amount'].map({
                'Yes': 1, 'No': 0, 'NM': 0
            }).fillna(0).astype(int)

        # 5. Ordinal mapping for Credit_Mix
        if 'Credit_Mix' in X.columns:
            credit_mix_map = {'Bad': 0, 'Standard': 1, 'Good': 2, '_': 1}
            X['Credit_Mix_Ordinal'] = X['Credit_Mix'].map(credit_mix_map).fillna(1).astype(int)
            X.drop(columns=['Credit_Mix'], inplace=True)

        # 6. Drop uninformative / identifier features
        cols_to_drop = []
        if self.drop_identifiers and 'Customer_ID' in X.columns:
            cols_to_drop.append('Customer_ID')
        # Num_Credit_Inquiries has >99% missing data
        if 'Num_Credit_Inquiries' in X.columns:
            cols_to_drop.append('Num_Credit_Inquiries')

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
