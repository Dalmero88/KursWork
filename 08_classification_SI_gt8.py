"""
Задача 7: Классификация — SI > 8
SI > 8 — распространённый фармакологический порог, означающий, что соединение
в 8 раз токсичнее для раковых/вирусных клеток, чем для здоровых — значимый рубеж.
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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from utils import load_data, prepare_classification, print_classification_metrics

TARGET = 'SI'
THRESHOLD = 8

print("=" * 60)
print(f"CLASSIFICATION: {TARGET} > {THRESHOLD}")
print("=" * 60)

df = load_data()
pos_frac = (df[TARGET] > THRESHOLD).mean()
print(f"SI > 8: {pos_frac*100:.1f}% positive ({(df[TARGET] > THRESHOLD).sum()} / {len(df)})")
print("Биологический смысл: SI > 8 указывает на перспективное терапевтическое окно\n")

X_train, X_test, y_train, y_test = prepare_classification(df, TARGET, threshold=THRESHOLD)

# Веса классов для учёта дисбаланса
scale = int((y_train == 0).sum() / (y_train == 1).sum())

results = []

lr_gs = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    {'C': [0.01, 0.1, 1, 10]}, cv=5, scoring='roc_auc', n_jobs=-1
)
lr_gs.fit(X_train, y_train)
print(f"  LR best: {lr_gs.best_params_}")
results.append(print_classification_metrics("LogisticRegression (balanced)", y_test,
                                             lr_gs.predict(X_test), lr_gs.predict_proba(X_test)[:, 1]))

rf_gs = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    {'n_estimators': [100, 300], 'max_depth': [None, 10], 'min_samples_leaf': [1, 3]},
    cv=5, scoring='roc_auc', n_jobs=-1
)
rf_gs.fit(X_train, y_train)
print(f"  RF best: {rf_gs.best_params_}")
results.append(print_classification_metrics("RandomForest (balanced)", y_test,
                                             rf_gs.predict(X_test), rf_gs.predict_proba(X_test)[:, 1]))

gb_gs = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]},
    cv=5, scoring='roc_auc', n_jobs=-1
)
gb_gs.fit(X_train, y_train)
print(f"  GB best: {gb_gs.best_params_}")
results.append(print_classification_metrics("GradientBoosting (tuned)", y_test,
                                             gb_gs.predict(X_test), gb_gs.predict_proba(X_test)[:, 1]))

xgb_gs = GridSearchCV(
    XGBClassifier(random_state=42, verbosity=0, eval_metric='logloss',
                  scale_pos_weight=scale),
    {'n_estimators': [100, 300], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 6]},
    cv=5, scoring='roc_auc', n_jobs=-1
)
xgb_gs.fit(X_train, y_train)
print(f"  XGB best: {xgb_gs.best_params_}")
results.append(print_classification_metrics("XGBoost (scale_pos_weight)", y_test,
                                             xgb_gs.predict(X_test), xgb_gs.predict_proba(X_test)[:, 1]))

lgb_gs = GridSearchCV(
    LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced'),
    {'n_estimators': [100, 300], 'learning_rate': [0.05, 0.1], 'num_leaves': [31, 63]},
    cv=5, scoring='roc_auc', n_jobs=-1
)
lgb_gs.fit(X_train, y_train)
print(f"  LGB best: {lgb_gs.best_params_}")
results.append(print_classification_metrics("LightGBM (balanced)", y_test,
                                             lgb_gs.predict(X_test), lgb_gs.predict_proba(X_test)[:, 1]))

results_df = pd.DataFrame(results).sort_values('AUC', ascending=False)
print("\n" + "=" * 60)
print("СВОДНАЯ ТАБЛИЦА")
print("=" * 60)
print(results_df.to_string(index=False))
best = results_df.iloc[0]
print(f"\n Лучшая модель: {best['model']} — AUC={best['AUC']:.4f}")

models_map = {
    'LogisticRegression (balanced)': lr_gs, 'RandomForest (balanced)': rf_gs,
    'GradientBoosting (tuned)': gb_gs, 'XGBoost (scale_pos_weight)': xgb_gs,
    'LightGBM (balanced)': lgb_gs,
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
for name, model in models_map.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name, alpha=0.7)
ax.plot([0, 1], [0, 1], 'k--')
ax.set_title('ROC-кривые — SI > 8')

best_obj = models_map.get(best['model'], lgb_gs)
cm = confusion_matrix(y_test, best_obj.predict(X_test))
disp = ConfusionMatrixDisplay(cm, display_labels=['SI≤8', 'SI>8'])
disp.plot(ax=axes[1], colorbar=False, cmap='Blues')
axes[1].set_title(f'Матрица ошибок — {best["model"]}')

plt.tight_layout()
plt.savefig('cls_si8_roc_cm.png', dpi=120)
plt.close()

print("\nОтчёт классификации лучшей модели:")
print(classification_report(y_test, best_obj.predict(X_test), target_names=['SI≤8', 'SI>8']))

results_df.to_csv('cls_si8_results.csv', index=False)
print("\n Классификация SI > 8 завершена.")
