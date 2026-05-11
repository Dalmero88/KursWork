"""
Исследовательский анализ данных (EDA)
Набор данных по химическим соединениям — предсказание IC50, CC50, SI
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
DATA_PATH = 'asset-v1_SkillFactory_MIFIML.xlsx'
df = pd.read_excel(DATA_PATH)
df = df.drop(columns=['Unnamed: 0'])

TARGETS = ['IC50, mM', 'CC50, mM', 'SI']
FEATURES = [c for c in df.columns if c not in TARGETS]

print("=" * 60)
print("ОБЗОР НАБОРА ДАННЫХ")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"Targets: {TARGETS}")
print(f"Features count: {len(FEATURES)}")
print(f"\nMissing values:\n{df.isna().sum()[df.isna().sum() > 0]}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# Базовая статистика
print("\n" + "=" * 60)
print("СТАТИСТИКА ЦЕЛЕВЫХ ПЕРЕМЕННЫХ")
print("=" * 60)
print(df[TARGETS].describe().round(3))

medians = df[TARGETS].median()
print(f"\nMedians:\n{medians.round(4)}")
print(f"\nSI > 8 count: {(df['SI'] > 8).sum()} ({(df['SI'] > 8).mean()*100:.1f}%)")

# Check SI = CC50/IC50 
si_calculated = df['CC50, mM'] / df['IC50, mM']
residuals = (si_calculated - df['SI']).abs()
print(f"\nSI vs CC50/IC50 max diff: {residuals.max():.6f} (SI = CC50/IC50 confirmed)")

# Анализ выбросов (метод МКР)
print("\n" + "=" * 60)
print("АНАЛИЗ ВЫБРОСОВ (МЕТОД МКР)")
print("=" * 60)
for col in TARGETS:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
    print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")

# Асимметрия и эксцесс
print("\n" + "=" * 60)
print("ФОРМА РАСПРЕДЕЛЕНИЯ")
print("=" * 60)
for col in TARGETS:
    sk = stats.skew(df[col])
    ku = stats.kurtosis(df[col])
    _, pval = stats.shapiro(df[col].sample(min(500, len(df)), random_state=42))
    print(f"{col}: skewness={sk:.2f}, kurtosis={ku:.2f}, Shapiro p={pval:.4f}")

# Корреляция признаков с целевыми переменными
corr_ic50 = df[FEATURES].corrwith(df['IC50, mM']).abs().sort_values(ascending=False)
corr_cc50 = df[FEATURES].corrwith(df['CC50, mM']).abs().sort_values(ascending=False)
corr_si   = df[FEATURES].corrwith(df['SI']).abs().sort_values(ascending=False)

print("\n" + "=" * 60)
print("ТОП-10 ПРИЗНАКОВ ПО |КОРРЕЛЯЦИИ| С ЦЕЛЕВЫМИ ПЕРЕМЕННЫМИ")
print("=" * 60)
print(f"\nIC50:\n{corr_ic50.head(10).round(3)}")
print(f"\nCC50:\n{corr_cc50.head(10).round(3)}")
print(f"\nSI:\n{corr_si.head(10).round(3)}")

# Признаки с около-нулевой дисперсией 
nzv = df[FEATURES].std()[df[FEATURES].std() < 0.01]
print(f"\nNear-zero variance features: {len(nzv)}")

# Бинарные / целочисленные признаки
int_feats = [c for c in FEATURES if df[c].nunique() <= 2]
print(f"Binary features: {len(int_feats)}")
print(f"Features with all-zero values: {(df[FEATURES] == 0).all().sum()}")

# Графики 
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle('EDA: Распределения целевых переменных', fontsize=16, fontweight='bold')

for i, col in enumerate(TARGETS):
    # Исходное распределение
    ax = axes[i][0]
    ax.hist(df[col], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_title(f'{col} — Raw')
    ax.set_xlabel(col)
    ax.set_ylabel('Count')

    # Логарифмическое распределение
    ax = axes[i][1]
    log_vals = np.log1p(df[col])
    ax.hist(log_vals, bins=50, color='darkorange', edgecolor='white', alpha=0.8)
    ax.set_title(f'{col} — log1p transform')
    ax.set_xlabel(f'log1p({col})')

    # Boxplot
    ax = axes[i][2]
    ax.boxplot(df[col], vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightgreen', color='green'),
               medianprops=dict(color='red', linewidth=2))
    ax.set_title(f'{col} — Boxplot')
    ax.set_ylabel(col)

plt.tight_layout()
plt.savefig('eda_target_distributions.png', dpi=120, bbox_inches='tight')
plt.close()

# Тепловая карта корреляции целевых переменных
fig, ax = plt.subplots(figsize=(6, 5))
corr_matrix = df[TARGETS].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            ax=ax, square=True, linewidths=0.5)
ax.set_title('Корреляционная матрица целевых переменных')
plt.tight_layout()
plt.savefig('eda_target_corr.png', dpi=120, bbox_inches='tight')
plt.close()

# Топ корреляций признаков с целевыми переменными
top_feats = list(set(corr_ic50.head(15).index) | set(corr_cc50.head(15).index))
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
corr_sub = df[top_feats + TARGETS].corr()
sns.heatmap(corr_sub[TARGETS].drop(TARGETS).sort_values('IC50, mM', key=abs, ascending=False).head(20),
            annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[0], linewidths=0.3)
axes[0].set_title('Топ корреляций признаков с целевыми переменными')

# Диаграмма рассеяния целевых переменных
axes[1].scatter(df['IC50, mM'], df['SI'], alpha=0.3, s=10, color='steelblue')
axes[1].set_xlabel('IC50, mM')
axes[1].set_ylabel('SI')
axes[1].set_title('IC50 vs SI (исходные значения)')
axes[1].set_xscale('log')
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig('eda_feature_corr.png', dpi=120, bbox_inches='tight')
plt.close()

# Распределение меток классификации 
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
labels_info = [
    ('IC50 > median', df['IC50, mM'] > df['IC50, mM'].median()),
    ('CC50 > median', df['CC50, mM'] > df['CC50, mM'].median()),
    ('SI > median',   df['SI'] > df['SI'].median()),
    ('SI > 8',        df['SI'] > 8),
]
for ax, (name, mask) in zip(axes, labels_info):
    counts = mask.value_counts()
    ax.bar(['False', 'True'], [counts.get(False, 0), counts.get(True, 0)],
           color=['#e74c3c', '#2ecc71'])
    ax.set_title(name)
    ax.set_ylabel('Count')
    for j, v in enumerate([counts.get(False, 0), counts.get(True, 0)]):
        ax.text(j, v + 5, str(v), ha='center', fontweight='bold')

plt.suptitle('Распределение меток классификации', fontsize=13)
plt.tight_layout()
plt.savefig('eda_class_balance.png', dpi=120, bbox_inches='tight')
plt.close()

print("\n EDA завершён. Графики сохранены.")
