"""
Задача 1: Регрессия для IC50
Модели: LinearRegression, Ridge, Lasso, RandomForest, GradientBoosting,
        XGBoost, LightGBM — с GridSearchCV / RandomizedSearchCV
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

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from utils import load_data, prepare_regression, print_regression_metrics

TARGET = 'IC50, mM'

print("=" * 60)
print(f"REGRESSION: {TARGET}")
print("=" * 60)

df = load_data()
X_train, X_test, y_train, y_test = prepare_regression(df, TARGET, log_transform=True)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Target (log1p) — mean: {y_train.mean():.3f}, std: {y_train.std():.3f}\n")

results = []

# 1. Линейная регрессия (базовая модель)
lr = LinearRegression()
lr.fit(X_train, y_train)
r = print_regression_metrics("LinearRegression (baseline)", y_test, lr.predict(X_test))
results.append(r)

# 2. Гребневая регрессия
ridge_gs = GridSearchCV(
    Ridge(), {'alpha': [0.01, 0.1, 1, 10, 100]},
    cv=5, scoring='r2', n_jobs=-1
)
ridge_gs.fit(X_train, y_train)
print(f"  Ridge best alpha: {ridge_gs.best_params_}")
r = print_regression_metrics("Ridge (best alpha)", y_test, ridge_gs.predict(X_test))
results.append(r)

# 3. Лассо 
lasso_gs = GridSearchCV(
    Lasso(max_iter=5000), {'alpha': [0.001, 0.01, 0.1, 1]},
    cv=5, scoring='r2', n_jobs=-1
)
lasso_gs.fit(X_train, y_train)
print(f"  Lasso best alpha: {lasso_gs.best_params_}")
r = print_regression_metrics("Lasso (best alpha)", y_test, lasso_gs.predict(X_test))
results.append(r)

# 4. Random Forest
rf_gs = GridSearchCV(
    RandomForestRegressor(random_state=42),
    {'n_estimators': [100, 300], 'max_depth': [None, 10, 20], 'min_samples_leaf': [1, 3]},
    cv=5, scoring='r2', n_jobs=-1
)
rf_gs.fit(X_train, y_train)
print(f"  RF best params: {rf_gs.best_params_}")
r = print_regression_metrics("RandomForest (tuned)", y_test, rf_gs.predict(X_test))
results.append(r)

# 5. Градиентный спуск
gb_gs = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]},
    cv=5, scoring='r2', n_jobs=-1
)
gb_gs.fit(X_train, y_train)
print(f"  GB best params: {gb_gs.best_params_}")
r = print_regression_metrics("GradientBoosting (tuned)", y_test, gb_gs.predict(X_test))
results.append(r)

# 6. XGBoost 
xgb_gs = GridSearchCV(
    XGBRegressor(random_state=42, verbosity=0),
    {'n_estimators': [100, 300], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 6]},
    cv=5, scoring='r2', n_jobs=-1
)
xgb_gs.fit(X_train, y_train)
print(f"  XGB best params: {xgb_gs.best_params_}")
r = print_regression_metrics("XGBoost (tuned)", y_test, xgb_gs.predict(X_test))
results.append(r)

# 7. LightGBM 
lgb_gs = GridSearchCV(
    LGBMRegressor(random_state=42, verbose=-1),
    {'n_estimators': [100, 300], 'learning_rate': [0.05, 0.1], 'num_leaves': [31, 63]},
    cv=5, scoring='r2', n_jobs=-1
)
lgb_gs.fit(X_train, y_train)
print(f"  LGB best params: {lgb_gs.best_params_}")
r = print_regression_metrics("LightGBM (tuned)", y_test, lgb_gs.predict(X_test))
results.append(r)

# Сводная таблица результатов 
results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
print("\n" + "=" * 60)
print("СВОДНАЯ ТАБЛИЦА")
print("=" * 60)
print(results_df.to_string(index=False))

best = results_df.iloc[0]
print(f"\n Лучшая модель: {best['model']} — R²={best['R2']:.4f}")

# Важность признаков лучшей модели 
best_model_obj = lgb_gs if 'LightGBM' in best['model'] else (
    xgb_gs if 'XGBoost' in best['model'] else rf_gs)
try:
    fi = pd.Series(
        best_model_obj.best_estimator_.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(9, 6))
    fi.sort_values().plot.barh(ax=ax, color='steelblue')
    ax.set_title(f'Топ-20 важных признаков — Регрессия IC50 ({best["model"]})')
    ax.set_xlabel('Важность')
    plt.tight_layout()
    plt.savefig('reg_ic50_importance.png', dpi=120)
    plt.close()
except Exception:
    pass

# Фактические vs предсказанные значения 
best_preds = best_model_obj.predict(X_test)
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, best_preds, alpha=0.4, s=15)
mn, mx = y_test.min(), y_test.max()
ax.plot([mn, mx], [mn, mx], 'r--')
ax.set_xlabel('Факт (log1p IC50)')
ax.set_ylabel('Предсказание')
ax.set_title(f'Регрессия IC50 — Факт vs Прогноз ({best["model"]})')
plt.tight_layout()
plt.savefig('reg_ic50_pred.png', dpi=120)
plt.close()

results_df.to_csv('reg_ic50_results.csv', index=False)
print("\n Регрессия IC50 завершена.")
