# Insurance Premium Regression Model - RMSE Optimization
# This script builds and optimizes a regression model with lower RMSE

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
try:
    df = pd.read_csv('insurance_premium_correct - insurance_premium (2) (2) (4).csv')
    print("✓ Dataset loaded successfully!")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}\n")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Prepare features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Data prepared:")
print(f"  Training: {X_train.shape[0]} samples")
print(f"  Testing: {X_test.shape[0]} samples\n")

# Build baseline models
print("="*70)
print("BASELINE MODELS")
print("="*70)

models = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
for model_name, model in models.items():
    if model_name in ['Ridge', 'Lasso', 'ElasticNet']:
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
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")

best_baseline = min(results, key=lambda x: results[x]['RMSE'])
baseline_rmse = results[best_baseline]['RMSE']

print(f"\n{'='*70}")
print(f"Best Baseline: {best_baseline} (RMSE: {baseline_rmse:.4f})")
print(f"{'='*70}\n")

# Hyperparameter tuning
print("Optimizing Gradient Boosting with GridSearchCV...")
param_grid = {
    'n_estimators': [100, 150, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5]
}

gbr = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(gbr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred_opt = best_model.predict(X_test)
opt_rmse = np.sqrt(mean_squared_error(y_test, y_pred_opt))
opt_mae = mean_absolute_error(y_test, y_pred_opt)
opt_r2 = r2_score(y_test, y_pred_opt)

print("\n" + "="*70)
print("OPTIMIZED GRADIENT BOOSTING MODEL")
print("="*70)
print(f"\nBest Parameters:")
for param, value in sorted(grid_search.best_params_.items()):
    print(f"  {param}: {value}")

print(f"\nPerformance:")
print(f"  RMSE: {opt_rmse:.4f}")
print(f"  MAE: {opt_mae:.4f}")
print(f"  R²: {opt_r2:.4f}")

improvement = ((baseline_rmse - opt_rmse) / baseline_rmse * 100)
print(f"\n{'='*70}")
print(f"RMSE Improvement: {improvement:.2f}%")
print(f"  Before: {baseline_rmse:.4f}")
print(f"  After:  {opt_rmse:.4f}")
print(f"  Reduction: {baseline_rmse - opt_rmse:.4f}")
print(f"{'='*70}")
print("\n✓ Model optimization complete!")
