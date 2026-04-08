#!/usr/bin/env python
"""Insurance Premium Regression Model - Optimized for Lower RMSE"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('insurance_premium_correct - insurance_premium (2) (2) (4).csv')
print("="*70)
print("INSURANCE PREMIUM REGRESSION MODEL")
print("="*70)
print(f"\n✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumns: {df.columns.tolist()}")

# Prepare features and target
X = df.iloc[:, :-1]  
y = df.iloc[:, -1]

# Handle categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    print(f"Categorical features encoded → Shape: {X.shape}")

# Split data  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nData split:")
print(f"  Training: {X_train.shape[0]} samples")
print(f"  Testing:  {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test multiple models
print("\n" + "="*70)
print("BASELINE MODEL COMPARISON")
print("="*70)

models = {
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
for model_name, model in models.items():
    if model_name in ['Ridge Regression', 'Lasso Regression', 'ElasticNet']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[model_name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    print(f"\n{model_name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")

best_model_name = min(results, key=lambda x: results[x]['RMSE'])
baseline_rmse = results[best_model_name]['RMSE']
print(f"\n{'─'*70}")
print(f"Best Baseline: {best_model_name}")
print(f"Baseline RMSE: {baseline_rmse:.4f}")
print(f"{'─'*70}")

# Hyperparameter tuning for Gradient Boosting
print("\n" + "="*70)
print("HYPERPARAMETER TUNING - GRADIENT BOOSTING")
print("="*70)
print("Running GridSearchCV with 5-fold cross-validation...\n")

param_grid = {
    'n_estimators': [100, 150, 200, 250, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6, 7],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'subsample': [0.8, 0.9, 1.0]
}

gbr = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(
    gbr, 
    param_grid, 
    cv=5, 
    scoring='neg_mean_squared_error', 
    n_jobs=-1, 
    verbose=0
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)

print("="*70)
print("OPTIMIZED GRADIENT BOOSTING MODEL")
print("="*70)
print(f"\nBest Hyperparameters Found:")
for param, value in sorted(grid_search.best_params_.items()):
    print(f"  • {param}: {value}")

print(f"\n{'─'*70}")
print("Performance Metrics:")
print(f"{'─'*70}")
print(f"RMSE: {rmse_tuned:.4f}")
print(f"MAE:  {mae_tuned:.4f}")
print(f"R²:   {r2_tuned:.4f}")

improvement = ((baseline_rmse - rmse_tuned) / baseline_rmse * 100)
improvement_percent = "↓" if improvement > 0 else "↑"
print(f"\n{'─'*70}")
print(f"RMSE Improvement: {improvement_percent} {abs(improvement):.2f}%")
print(f"  • Baseline RMSE:   {baseline_rmse:.4f}")
print(f"  • Optimized RMSE:  {rmse_tuned:.4f}")
print(f"  • Difference:      {baseline_rmse - rmse_tuned:.4f}")
print(f"{'─'*70}")

print("\n✓ Model optimization complete!")
print("="*70)
