"""
utils.py — скрипты для всех моделей
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_PATH = 'asset-v1_SkillFactory_MIFIML.xlsx'
TARGETS = ['IC50, mM', 'CC50, mM', 'SI']
RANDOM_STATE = 42


def load_data():
    df = pd.read_excel(DATA_PATH)
    df = df.drop(columns=['Unnamed: 0'])
    return df


def get_features(df, drop_cols=None):
    cols_to_drop = TARGETS + (drop_cols or [])
    feat_cols = [c for c in df.columns if c not in cols_to_drop]
    # Remove all-zero or near-constant features
    std = df[feat_cols].std()
    feat_cols = std[std > 0].index.tolist()
    return feat_cols


def prepare_regression(df, target, log_transform=True):
    feat_cols = get_features(df, drop_cols=TARGETS)
    X = df[feat_cols].fillna(df[feat_cols].median())
    y = df[target]
    if log_transform:
        y = np.log1p(y)
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def prepare_classification(df, target_col=None, threshold=None, use_median=False):
    feat_cols = get_features(df, drop_cols=TARGETS)
    X = df[feat_cols].fillna(df[feat_cols].median())
    if use_median:
        y = (df[target_col] > df[target_col].median()).astype(int)
    else:
        y = (df[target_col] > threshold).astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def print_regression_metrics(name, y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"  {name:<35} R²={r2:.4f}  MAE={mae:.3f}  RMSE={rmse:.3f}")
    return {'model': name, 'R2': r2, 'MAE': mae, 'RMSE': rmse}


def print_classification_metrics(name, y_true, y_pred, y_proba=None):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else float('nan')
    print(f"  {name:<35} Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    return {'model': name, 'Accuracy': acc, 'F1': f1, 'AUC': auc}