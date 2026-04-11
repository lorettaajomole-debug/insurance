#!/usr/bin/env python
"""Enhanced Insurance Premium Regression Model - RMSE Reduction Techniques"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('insurance_premium_correct - insurance_premium (2) (2) (4).csv')
print("="*70)
print("OPTIMIZED INSURANCE PREMIUM REGRESSION MODEL")
print("="*70)
print(f"\n✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Feature Engineering
print("\n" + "="*70)
print("FEATURE ENGINEERING")
print("="*70)

df_engineered = df.copy()

# Create polynomial features for non-linear relationships
df_engineered['age_squared'] = df_engineered['age'] ** 2
df_engineered['bmi_squared'] = df_engineered['bmi'] ** 2
df_engineered['age_bmi_interaction'] = df_engineered['age'] * df_engineered['bmi']
df_engineered['age_children_interaction'] = df_engineered['age'] * df_engineered['children']

# Create binned features
df_engineered['age_group'] = pd.cut(df_engineered['age'], bins=5, labels=False)
df_engineered['bmi_category'] = pd.cut(df_engineered['bmi'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3])

# Log transformation of target (helps with skewed data)
df_engineered['log_charges'] = np.log1p(df_engineered['charges'])

print("✓ Created polynomial features (age², bmi², interactions)")
print("✓ Created binned features (age_group, bmi_category)")
print(f"✓ New shape: {df_engineered.shape}")

# Prepare features and target
X = df_engineered.iloc[:, :-1]  # All columns except last
X = X.drop('charges', axis=1)   # Remove original target
y_original = df['charges']
y_log = df_engineered['log_charges']

print(f"\nFinal feature count: {X.shape[1]} features")

# Split data  
X_train, X_test, y_train_orig, y_test_orig, y_train_log, y_test_log = train_test_split(
    X, y_original, y_log, test_size=0.2, random_state=42
)
print(f"\nData split:")
print(f"  Training: {X_train.shape[0]} samples")
print(f"  Testing:  {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Tuned Gradient Boosting
print("\n" + "="*70)
print("MODEL 1: OPTIMIZED GRADIENT BOOSTING")
print("="*70)

gbr = GradientBoostingRegressor(
    n_estimators=300,      # More trees
    learning_rate=0.05,    # Lower learning rate for fine-tuning
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.9,
    random_state=42
)
gbr.fit(X_train, y_train_orig)
y_pred_gbr = gbr.predict(X_test)
rmse_gbr = np.sqrt(mean_squared_error(y_test_orig, y_pred_gbr))
r2_gbr = r2_score(y_test_orig, y_pred_gbr)

print(f"RMSE: {rmse_gbr:.4f}")
print(f"R²: {r2_gbr:.4f}")

# Model 2: Log-transformed Gradient Boosting (trained on log-scale)
print("\n" + "="*70)
print("MODEL 2: LOG-SCALE GRADIENT BOOSTING")
print("="*70)

gbr_log = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.9,
    random_state=42
)
gbr_log.fit(X_train, y_train_log)
y_pred_log = gbr_log.predict(X_test)
y_pred_orig_from_log = np.expm1(y_pred_log)  # Convert back to original scale
rmse_log = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig_from_log))
r2_log = r2_score(y_test_orig, y_pred_orig_from_log)

print(f"RMSE: {rmse_log:.4f}")
print(f"R²: {r2_log:.4f}")

# Model 3: Random Forest with Tuned Parameters
print("\n" + "="*70)
print("MODEL 3: OPTIMIZED RANDOM FOREST")
print("="*70)

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train_orig)
y_pred_rf = rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test_orig, y_pred_rf))
r2_rf = r2_score(y_test_orig, y_pred_rf)

print(f"RMSE: {rmse_rf:.4f}")
print(f"R²: {r2_rf:.4f}")

# Model 4: Ridge Regression on Scaled Features
print("\n" + "="*70)
print("MODEL 4: RIDGE REGRESSION (SCALED)")
print("="*70)

ridge = Ridge(alpha=0.1)
ridge.fit(X_train_scaled, y_train_orig)
y_pred_ridge = ridge.predict(X_test_scaled)
rmse_ridge = np.sqrt(mean_squared_error(y_test_orig, y_pred_ridge))
r2_ridge = r2_score(y_test_orig, y_pred_ridge)

print(f"RMSE: {rmse_ridge:.4f}")
print(f"R²: {r2_ridge:.4f}")

# Ensemble: Voting Regressor combining best models
print("\n" + "="*70)
print("ENSEMBLE: VOTING REGRESSOR")
print("="*70)

voting_reg = VotingRegressor([
    ('gbr', gbr),
    ('gbr_log', gbr_log),
    ('rf', rf)
])
voting_reg.fit(X_train, y_train_orig)
y_pred_voting = voting_reg.predict(X_test)
rmse_voting = np.sqrt(mean_squared_error(y_test_orig, y_pred_voting))
r2_voting = r2_score(y_test_orig, y_pred_voting)

print(f"RMSE: {rmse_voting:.4f}")
print(f"R²: {r2_voting:.4f}")

# Summary
print("\n" + "="*70)
print("SUMMARY OF ALL MODELS")
print("="*70)

models_summary = [
    ('Optimized Gradient Boosting', rmse_gbr, r2_gbr),
    ('Log-Scale Gradient Boosting', rmse_log, r2_log),
    ('Optimized Random Forest', rmse_rf, r2_rf),
    ('Ridge Regression', rmse_ridge, r2_ridge),
    ('Voting Ensemble', rmse_voting, r2_voting)
]

for name, rmse, r2 in sorted(models_summary, key=lambda x: x[1]):
    print(f"{name:.<40} RMSE: {rmse:.4f}  R²: {r2:.4f}")

best_rmse = min(models_summary, key=lambda x: x[1])
print(f"\n{'─'*70}")
print(f"🏆 BEST MODEL: {best_rmse[0]}")
print(f"   RMSE: {best_rmse[1]:.4f}")
print(f"   R²: {best_rmse[2]:.4f}")
print(f"{'─'*70}")

print("\n✓ Model optimization complete!")
print("="*70)
