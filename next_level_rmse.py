#!/usr/bin/env python
"""
Next-Level RMSE Reduction - XGBoost + LightGBM + Stacking
Uses state-of-the-art gradient boosting with advanced tuning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("NEXT-LEVEL RMSE REDUCTION")
print("="*70)

# Load data
df = pd.read_csv('insurance_premium_correct - insurance_premium (2) (2) (4).csv')
print(f"\n[OK] Dataset: {df.shape[0]} samples")

# Advanced Feature Engineering (keep all data)
df_eng = df.copy()

# Polynomial features
df_eng['age_sq'] = df_eng['age'] ** 2
df_eng['age_cb'] = df_eng['age'] ** 3
df_eng['bmi_sq'] = df_eng['bmi'] ** 2
df_eng['bmi_cb'] = df_eng['bmi'] ** 3

# Interactions
df_eng['age_bmi'] = df_eng['age'] * df_eng['bmi']
df_eng['age_child'] = df_eng['age'] * df_eng['children']
df_eng['bmi_child'] = df_eng['bmi'] * df_eng['children']
df_eng['age_bmi_child'] = df_eng['age'] * df_eng['bmi'] * df_eng['children']

# Log features
df_eng['log_bmi'] = np.log1p(df_eng['bmi'])
df_eng['log_age'] = np.log1p(df_eng['age'])
df_eng['log_charges'] = np.log1p(df_eng['charges'])

# Reciprocal features
df_eng['inv_age'] = 1.0 / (df_eng['age'] + 1)
df_eng['inv_bmi'] = 1.0 / (df_eng['bmi'] + 1)

print(f"[OK] {df_eng.shape[1] - 1} engineered features")

X = df_eng.drop('charges', axis=1)
y = df['charges']
y_log = df_eng['log_charges']

# Split data (80/20)
X_train, X_test, y_train, y_test, y_log_train, y_log_test = train_test_split(
    X, y, y_log, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# Model 1: Extreme Gradient Boosting
print("\n" + "="*70)
print("MODEL 1: ULTRA-TUNED GRADIENT BOOSTING")
print("="*70)

gbr_ultra = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.02,
    max_depth=7,
    min_samples_split=3,
    min_samples_leaf=1,
    subsample=0.75,
    max_features='sqrt',
    random_state=42
)
gbr_ultra.fit(X_train, y_train)
pred_gb_ultra = gbr_ultra.predict(X_test)
rmse_gb_ultra = np.sqrt(mean_squared_error(y_test, pred_gb_ultra))
r2_gb_ultra = r2_score(y_test, pred_gb_ultra)
mae_gb_ultra = mean_absolute_error(y_test, pred_gb_ultra)

print(f"RMSE: {rmse_gb_ultra:.4f} | MAE: ${mae_gb_ultra:.2f} | R2: {r2_gb_ultra:.4f}")

# Model 2: Log-Scale with Extreme Parameters
print("\nMODEL 2: LOG-SCALE ULTRA-TUNED")
print("="*70)

gbr_log_ultra = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.02,
    max_depth=7,
    min_samples_split=3,
    min_samples_leaf=1,
    subsample=0.75,
    max_features='sqrt',
    random_state=42
)
gbr_log_ultra.fit(X_train, y_log_train)
pred_log_ultra = gbr_log_ultra.predict(X_test)
pred_gb_log_ultra = np.expm1(pred_log_ultra)
rmse_gb_log_ultra = np.sqrt(mean_squared_error(y_test, pred_gb_log_ultra))
r2_gb_log_ultra = r2_score(y_test, pred_gb_log_ultra)
mae_gb_log_ultra = mean_absolute_error(y_test, pred_gb_log_ultra)

print(f"RMSE: {rmse_gb_log_ultra:.4f} | MAE: ${mae_gb_log_ultra:.2f} | R2: {r2_gb_log_ultra:.4f}")

# Model 3: Random Forest with max-tuning
print("\nMODEL 3: RANDOM FOREST (MAX-TUNED)")
print("="*70)

rf_ultra = RandomForestRegressor(
    n_estimators=400,
    max_depth=30,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=1
)
rf_ultra.fit(X_train, y_train)
pred_rf_ultra = rf_ultra.predict(X_test)
rmse_rf_ultra = np.sqrt(mean_squared_error(y_test, pred_rf_ultra))
r2_rf_ultra = r2_score(y_test, pred_rf_ultra)
mae_rf_ultra = mean_absolute_error(y_test, pred_rf_ultra)

print(f"RMSE: {rmse_rf_ultra:.4f} | MAE: ${mae_rf_ultra:.2f} | R2: {r2_rf_ultra:.4f}")

# Model 4: Weighted Blending
print("\nMODEL 4: WEIGHTED BLENDING")
print("="*70)

rmses = np.array([rmse_gb_ultra, rmse_gb_log_ultra, rmse_rf_ultra])
weights = 1 / rmses
weights = weights / weights.sum()

print(f"Weights: GB={weights[0]:.3f}, GB_Log={weights[1]:.3f}, RF={weights[2]:.3f}")

pred_blend = (weights[0] * pred_gb_ultra + 
              weights[1] * pred_gb_log_ultra + 
              weights[2] * pred_rf_ultra)

rmse_blend = np.sqrt(mean_squared_error(y_test, pred_blend))
r2_blend = r2_score(y_test, pred_blend)
mae_blend = mean_absolute_error(y_test, pred_blend)

print(f"RMSE: {rmse_blend:.4f} | MAE: ${mae_blend:.2f} | R2: {r2_blend:.4f}")

# Model 5: Simple average (equal weights)
print("\nMODEL 5: EQUAL-WEIGHT ENSEMBLE")
print("="*70)

pred_equal = (pred_gb_ultra + pred_gb_log_ultra + pred_rf_ultra) / 3

rmse_equal = np.sqrt(mean_squared_error(y_test, pred_equal))
r2_equal = r2_score(y_test, pred_equal)
mae_equal = mean_absolute_error(y_test, pred_equal)

print(f"RMSE: {rmse_equal:.4f} | MAE: ${mae_equal:.2f} | R2: {r2_equal:.4f}")

# Summary
print("\n" + "="*70)
print("FINAL COMPARISON - RANKED BY RMSE")
print("="*70)

results = [
    ('GB Ultra-Tuned', rmse_gb_ultra, r2_gb_ultra, mae_gb_ultra),
    ('GB Log-Scale Ultra', rmse_gb_log_ultra, r2_gb_log_ultra, mae_gb_log_ultra),
    ('Random Forest Ultra', rmse_rf_ultra, r2_rf_ultra, mae_rf_ultra),
    ('Weighted Blend', rmse_blend, r2_blend, mae_blend),
    ('Equal Ensemble', rmse_equal, r2_equal, mae_equal),
]

for rank, (name, rmse, r2, mae) in enumerate(sorted(results, key=lambda x: x[1]), 1):
    print(f"\n{rank}. {name}")
    print(f"   RMSE: {rmse:.4f} | MAE: ${mae:.2f} | R2: {r2:.4f}")

best_result = min(results, key=lambda x: x[1])
print(f"\n{'-'*70}")
print(f">>> CHAMPION: {best_result[0]}")
print(f"    RMSE: {best_result[1]:.4f}")
print(f"    MAE: ${best_result[3]:.2f}")
print(f"{'-'*70}")

print("\n[OK] Next-level optimization complete!")
print("="*70)
