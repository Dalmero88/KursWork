"""
Задача 4: Классификация — IC50 > медианы
"""

#import sys, os
#sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              classification_report, RocCurveDisplay)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from utils import load_data, prepare_classification, print_classification_metrics

TARGET = 'IC50, mM'

print("=" * 60)
print(f"CLASSIFICATION: {TARGET} > median")
print("=" * 60)

df = load_data()
median_val = df[TARGET].median()
print(f"Median IC50: {median_val:.4f}")
print(f"Class balance: {(df[TARGET] > median_val).mean()*100:.1f}% positive\n")

X_train, X_test, y_train, y_test = prepare_classification(df, TARGET, use_median=True)

results = []

# 1. Логистическая регрессия
lr_gs = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    {'C': [0.01, 0.1, 1, 10]},
    cv=5, scoring='roc_auc', n_jobs=-1
)
lr_gs.fit(X_train, y_train)
print(f"  LR best: {lr_gs.best_params_}")
p = lr_gs.predict(X_test)
pb = lr_gs.predict_proba(X_test)[:, 1]
results.append(print_classification_metrics("LogisticRegression (tuned)", y_test, p, pb))

# 2. Random Forest 
rf_gs = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {'n_estimators': [100, 300], 'max_depth': [None, 10], 'min_samples_leaf': [1, 3]},
    cv=5, scoring='roc_auc', n_jobs=-1
)
rf_gs.fit(X_train, y_train)
print(f"  RF best: {rf_gs.best_params_}")
p = rf_gs.predict(X_test)
pb = rf_gs.predict_proba(X_test)[:, 1]
results.append(print_classification_metrics("RandomForest (tuned)", y_test, p, pb))

# 3. Градиентный спуск 
gb_gs = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]},
    cv=5, scoring='roc_auc', n_jobs=-1
)
gb_gs.fit(X_train, y_train)
print(f"  GB best: {gb_gs.best_params_}")
p = gb_gs.predict(X_test)
pb = gb_gs.predict_proba(X_test)[:, 1]
results.append(print_classification_metrics("GradientBoosting (tuned)", y_test, p, pb))

# 4. XGBoost 
xgb_gs = GridSearchCV(
    XGBClassifier(random_state=42, verbosity=0, eval_metric='logloss'),
    {'n_estimators': [100, 300], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 6]},
    cv=5, scoring='roc_auc', n_jobs=-1
)
xgb_gs.fit(X_train, y_train)
print(f"  XGB best: {xgb_gs.best_params_}")
p = xgb_gs.predict(X_test)
pb = xgb_gs.predict_proba(X_test)[:, 1]
results.append(print_classification_metrics("XGBoost (tuned)", y_test, p, pb))

# 5. LightGBM 
lgb_gs = GridSearchCV(
    LGBMClassifier(random_state=42, verbose=-1),
    {'n_estimators': [100, 300], 'learning_rate': [0.05, 0.1], 'num_leaves': [31, 63]},
    cv=5, scoring='roc_auc', n_jobs=-1
)
lgb_gs.fit(X_train, y_train)
print(f"  LGB best: {lgb_gs.best_params_}")
p = lgb_gs.predict(X_test)
pb = lgb_gs.predict_proba(X_test)[:, 1]
results.append(print_classification_metrics("LightGBM (tuned)", y_test, p, pb))

results_df = pd.DataFrame(results).sort_values('AUC', ascending=False)
print("\n" + "=" * 60)
print("СВОДНАЯ ТАБЛИЦА")
print("=" * 60)
print(results_df.to_string(index=False))
best = results_df.iloc[0]
print(f"\n Лучшая модель: {best['model']} — AUC={best['AUC']:.4f}")

# ROC-кривые
fig, ax = plt.subplots(figsize=(7, 6))
models_map = {
    'LogisticRegression (tuned)': lr_gs,
    'RandomForest (tuned)': rf_gs,
    'GradientBoosting (tuned)': gb_gs,
    'XGBoost (tuned)': xgb_gs,
    'LightGBM (tuned)': lgb_gs,
}
for name, model in models_map.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name, alpha=0.7)
ax.plot([0, 1], [0, 1], 'k--')
ax.set_title('ROC-кривые — IC50 > медианы')
plt.tight_layout()
plt.savefig('cls_ic50_roc.png', dpi=120)
plt.close()

# Отчёт классификации лучшей модели
best_obj = models_map.get(best['model'], lgb_gs)
print("\nОтчёт классификации лучшей модели:")
print(classification_report(y_test, best_obj.predict(X_test)))

results_df.to_csv('cls_ic50_results.csv', index=False)
print("\n Классификация IC50 завершена.")
