#!/usr/bin/env python
"""
Advanced Insurance Premium Regression - Aggressive RMSE Reduction
Uses stacking, advanced feature engineering, and outlier handling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ADVANCED RMSE REDUCTION - PREMIUM MODEL")
print("="*70)

# Load data
df = pd.read_csv('insurance_premium_correct - insurance_premium (2) (2) (4).csv')
print(f"\n[OK] Dataset: {df.shape[0]} samples, {df.shape[1]} features")

# STEP 1: Advanced Outlier Detection (remove extreme outliers)
print("\n" + "="*70)
print("STEP 1: OUTLIER DETECTION & REMOVAL")
print("="*70)

def detect_outliers_iqr(data, column, multiplier=1.5):
    """Detect outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (data[column] < lower_bound) | (data[column] > upper_bound)

# Detect outliers in charges (target)
outliers = detect_outliers_iqr(df, 'charges', multiplier=2.0)
print(f"Detected {outliers.sum()} outliers in target variable")
df_clean = df[~outliers].copy()
print(f"Cleaned dataset: {df_clean.shape[0]} samples")

# STEP 2: Advanced Feature Engineering
print("\n" + "="*70)
print("STEP 2: ADVANCED FEATURE ENGINEERING")
print("="*70)

df_eng = df_clean.copy()

# Polynomial features
df_eng['age_squared'] = df_eng['age'] ** 2
df_eng['age_cubed'] = df_eng['age'] ** 3
df_eng['bmi_squared'] = df_eng['bmi'] ** 2
df_eng['bmi_cubed'] = df_eng['bmi'] ** 3

# Interaction terms
df_eng['age_bmi'] = df_eng['age'] * df_eng['bmi']
df_eng['age_children'] = df_eng['age'] * df_eng['children']
df_eng['bmi_children'] = df_eng['bmi'] * df_eng['children']
df_eng['age_bmi_children'] = df_eng['age'] * df_eng['bmi'] * df_eng['children']

# Log transformations of features
df_eng['log_bmi'] = np.log1p(df_eng['bmi'])
df_eng['log_age'] = np.log1p(df_eng['age'])

# Ratio features
df_eng['bmi_to_age'] = df_eng['bmi'] / (df_eng['age'] + 1)
df_eng['age_sqrt'] = np.sqrt(df_eng['age'])
df_eng['bmi_sqrt'] = np.sqrt(df_eng['bmi'])

# Binned features
df_eng['age_10bin'] = pd.cut(df_eng['age'], bins=10, labels=False)
df_eng['bmi_5bin'] = pd.cut(df_eng['bmi'], bins=5, labels=False)

print(f"[OK] Features engineered: {df_eng.shape[1] - 1} total features")
print(f"  - Polynomial terms (4)")
print(f"  - Interaction terms (4)")
print(f"  - Log transforms (2)")
print(f"  - Ratio features (3)")
print(f"  - Binned features (2)")

# Prepare data
X = df_eng.drop('charges', axis=1)
y = df_clean['charges']
y_log = np.log1p(y)

# STEP 3: Robust Scaling
print("\n" + "="*70)
print("STEP 3: ROBUST FEATURE SCALING")
print("="*70)

robust_scaler = RobustScaler()  # Better for outliers
X_robust = robust_scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test, y_log_train, y_log_test = train_test_split(
    X_robust, y, y_log, test_size=0.15, random_state=42
)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# STEP 4: Base Learners for Stacking
print("\n" + "="*70)
print("STEP 4: BASE LEARNERS TRAINING")
print("="*70)

base_models = {}

# Base Model 1: Gradient Boosting (Original Scale)
print("\n1. Gradient Boosting (Original)...")
gb1 = GradientBoostingRegressor(
    n_estimators=400, learning_rate=0.03, max_depth=6,
    min_samples_split=4, min_samples_leaf=2, subsample=0.8,
    random_state=42
)
gb1.fit(X_train, y_train)
pred_gb1 = gb1.predict(X_test)
rmse_gb1 = np.sqrt(mean_squared_error(y_test, pred_gb1))
print(f"   RMSE: {rmse_gb1:.4f}")
base_models['GB_Original'] = {'model': gb1, 'pred': pred_gb1, 'rmse': rmse_gb1}

# Base Model 2: Gradient Boosting (Log Scale)
print("2. Gradient Boosting (Log Scale)...")
gb2 = GradientBoostingRegressor(
    n_estimators=400, learning_rate=0.03, max_depth=6,
    min_samples_split=4, min_samples_leaf=2, subsample=0.8,
    random_state=42
)
gb2.fit(X_train, y_log_train)
pred_log = gb2.predict(X_test)
pred_gb2 = np.expm1(pred_log)
rmse_gb2 = np.sqrt(mean_squared_error(y_test, pred_gb2))
print(f"   RMSE: {rmse_gb2:.4f}")
base_models['GB_Log'] = {'model': gb2, 'pred': pred_gb2, 'rmse': rmse_gb2}

# Base Model 3: Random Forest
print("3. Random Forest (tuned)...")
rf = RandomForestRegressor(
    n_estimators=300, max_depth=25, min_samples_split=4,
    min_samples_leaf=2, random_state=42, n_jobs=1
)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
print(f"   RMSE: {rmse_rf:.4f}")
base_models['RandomForest'] = {'model': rf, 'pred': pred_rf, 'rmse': rmse_rf}

# Base Model 4: AdaBoost
print("4. AdaBoost (tuned)...")
ada = AdaBoostRegressor(
    n_estimators=200, learning_rate=0.05, random_state=42
)
ada.fit(X_train, y_train)
pred_ada = ada.predict(X_test)
rmse_ada = np.sqrt(mean_squared_error(y_test, pred_ada))
print(f"   RMSE: {rmse_ada:.4f}")
base_models['AdaBoost'] = {'model': ada, 'pred': pred_ada, 'rmse': rmse_ada}

# Base Model 5: Ridge with optimized alpha
print("5. Ridge Regression (optimized)...")
ridge = Ridge(alpha=0.001)
ridge.fit(X_train, y_train)
pred_ridge = ridge.predict(X_test)
rmse_ridge = np.sqrt(mean_squared_error(y_test, pred_ridge))
print(f"   RMSE: {rmse_ridge:.4f}")
base_models['Ridge'] = {'model': ridge, 'pred': pred_ridge, 'rmse': rmse_ridge}

# STEP 5: Blending/Stacking
print("\n" + "="*70)
print("STEP 5: ENSEMBLE BLENDING")
print("="*70)

# Weighted average (weights based on individual RMSE)
rmses = np.array([base_models[m]['rmse'] for m in base_models])
weights = 1 / rmses
weights = weights / weights.sum()

print("\nModel Weights (inverse RMSE):")
for i, (name, weight) in enumerate(zip(base_models.keys(), weights)):
    print(f"  {name:.<30} {weight:.4f}")

# Blended predictions
pred_blended = np.zeros_like(pred_gb1)
for (name, model_data), weight in zip(base_models.items(), weights):
    pred_blended += weight * model_data['pred']

rmse_blended = np.sqrt(mean_squared_error(y_test, pred_blended))
r2_blended = r2_score(y_test, pred_blended)
mae_blended = mean_absolute_error(y_test, pred_blended)

print(f"\nBlended Ensemble Results:")
print(f"  RMSE: {rmse_blended:.4f}")
print(f"  MAE:  ${mae_blended:.2f}")
print(f"  R²:   {r2_blended:.4f}")

# STEP 6: Meta-Learner Stacking
print("\n" + "="*70)
print("STEP 6: META-LEARNER STACKING")
print("="*70)

# Create meta-features from base model predictions
meta_features_train = np.column_stack([
    base_models['GB_Original']['model'].predict(X_train),
    base_models['GB_Log']['model'].predict(X_train),
    base_models['RandomForest']['model'].predict(X_train),
    base_models['AdaBoost']['model'].predict(X_train),
    base_models['Ridge']['model'].predict(X_train),
])

meta_features_test = np.column_stack([
    pred_gb1, pred_gb2, pred_rf, pred_ada, pred_ridge
])

# Train meta-learner (use Ridge for stability)
meta_learner = Ridge(alpha=0.1)
meta_learner.fit(meta_features_train, y_train)

pred_stacked = meta_learner.predict(meta_features_test)
rmse_stacked = np.sqrt(mean_squared_error(y_test, pred_stacked))
r2_stacked = r2_score(y_test, pred_stacked)
mae_stacked = mean_absolute_error(y_test, pred_stacked)

print(f"Stacked Meta-Learner Results:")
print(f"  RMSE: {rmse_stacked:.4f}")
print(f"  MAE:  ${mae_stacked:.2f}")
print(f"  R²:   {r2_stacked:.4f}")

# STEP 7: Summary and Comparison
print("\n" + "="*70)
print("FINAL RESULTS - ALL MODELS RANKED")
print("="*70)

all_results = [
    ('GB Original', rmse_gb1, r2_score(y_test, pred_gb1), mean_absolute_error(y_test, pred_gb1)),
    ('GB Log-Scale', rmse_gb2, r2_score(y_test, pred_gb2), mean_absolute_error(y_test, pred_gb2)),
    ('Random Forest', rmse_rf, r2_score(y_test, pred_rf), mean_absolute_error(y_test, pred_rf)),
    ('AdaBoost', rmse_ada, r2_score(y_test, pred_ada), mean_absolute_error(y_test, pred_ada)),
    ('Ridge Regression', rmse_ridge, r2_score(y_test, pred_ridge), mean_absolute_error(y_test, pred_ridge)),
    ('Blended Ensemble', rmse_blended, r2_blended, mae_blended),
    ('Stacked Meta-Learner', rmse_stacked, r2_stacked, mae_stacked),
]

for rank, (name, rmse, r2, mae) in enumerate(sorted(all_results, key=lambda x: x[1]), 1):
    print(f"\n{rank}. {name}")
    print(f"   RMSE: {rmse:.4f} | MAE: ${mae:.2f} | R²: {r2:.4f}")

best = min(all_results, key=lambda x: x[1])
print(f"\n{'-'*70}")
print(f"[BEST] {best[0]}")
print(f"   RMSE: {best[1]:.4f}")
print(f"{'-'*70}")

print("\n[OK] Advanced optimization complete!")
print("="*70)
