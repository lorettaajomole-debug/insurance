#!/usr/bin/env python
"""
Insurance Premium Regression - RMSE Optimized Model
Implements multiple proven techniques to minimize RMSE
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')


def engineer_features(df):
    """Create polynomial and interaction features"""
    df = df.copy()
    
    # Polynomial features for non-linear relationships
    df['age_squared'] = df['age'] ** 2
    df['bmi_squared'] = df['bmi'] ** 2
    df['age_bmi_interaction'] = df['age'] * df['bmi']
    df['age_children_interaction'] = df['age'] * df['children']
    df['bmi_children_interaction'] = df['bmi'] * df['children']
    
    # Age cubed for higher non-linearity
    df['age_cubed'] = df['age'] ** 3
    
    return df


def main():
    print("="*70)
    print("INSURANCE PREMIUM REGRESSION - RMSE OPTIMIZATION")
    print("="*70)
    
    # Load data
    df = pd.read_csv('insurance_premium_correct - insurance_premium (2) (2) (4).csv')
    print(f"\n✓ Data loaded: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"  Target variable: charges")
    print(f"  Range: ${df['charges'].min():.2f} - ${df['charges'].max():.2f}")
    
    # Feature engineering
    print("\n" + "="*70)
    print("FEATURE ENGINEERING")
    print("="*70)
    df_engineered = engineer_features(df)
    print(f"✓ Engineering complete: {df_engineered.shape[1] - 1} features (-1 for target)")
    
    # Prepare data
    X = df_engineered.drop('charges', axis=1)
    y = df['charges']
    
    # Log-transformed target for comparison
    y_log = np.log1p(y)
    
    # Train-test split
    X_train, X_test, y_train, y_test, y_log_train, y_log_test = train_test_split(
        X, y, y_log, test_size=0.2, random_state=42
    )
    print(f"\nTrain/Test split: {len(X_train)} / {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # =========================================================================
    # MODEL 1: Optimized Gradient Boosting (Original Scale)
    # =========================================================================
    print("\n" + "="*70)
    print("MODEL 1: GRADIENT BOOSTING (ORIGINAL SCALE)")
    print("="*70)
    
    gbr = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.85,
        random_state=42
    )
    print("Training...")
    gbr.fit(X_train, y_train)
    
    y_pred_gbr = gbr.predict(X_test)
    rmse_gbr = np.sqrt(mean_squared_error(y_test, y_pred_gbr))
    mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
    r2_gbr = r2_score(y_test, y_pred_gbr)
    
    results['GB Original'] = {'rmse': rmse_gbr, 'mae': mae_gbr, 'r2': r2_gbr}
    print(f"RMSE: {rmse_gbr:.4f} | MAE: ${mae_gbr:.2f} | R²: {r2_gbr:.4f}")
    
    # =========================================================================
    # MODEL 2: Gradient Boosting (Log Scale)
    # =========================================================================
    print("\n" + "="*70)
    print("MODEL 2: GRADIENT BOOSTING (LOG SCALE)")
    print("="*70)
    print("Training on log-transformed target...")
    
    gbr_log = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.85,
        random_state=42
    )
    gbr_log.fit(X_train, y_log_train)
    
    y_pred_log = gbr_log.predict(X_test)
    y_pred_gbr_log = np.expm1(y_pred_log)  # Convert back to original scale
    
    rmse_gbr_log = np.sqrt(mean_squared_error(y_test, y_pred_gbr_log))
    mae_gbr_log = mean_absolute_error(y_test, y_pred_gbr_log)
    r2_gbr_log = r2_score(y_test, y_pred_gbr_log)
    
    results['GB Log-Scale'] = {'rmse': rmse_gbr_log, 'mae': mae_gbr_log, 'r2': r2_gbr_log}
    print(f"RMSE: {rmse_gbr_log:.4f} | MAE: ${mae_gbr_log:.2f} | R²: {r2_gbr_log:.4f}")
    
    # =========================================================================
    # MODEL 3: Random Forest
    # =========================================================================
    print("\n" + "="*70)
    print("MODEL 3: RANDOM FOREST")
    print("="*70)
    
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1
    )
    print("Training...")
    rf.fit(X_train, y_train)
    
    y_pred_rf = rf.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    
    results['Random Forest'] = {'rmse': rmse_rf, 'mae': mae_rf, 'r2': r2_rf}
    print(f"RMSE: {rmse_rf:.4f} | MAE: ${mae_rf:.2f} | R²: {r2_rf:.4f}")
    
    # =========================================================================
    # MODEL 4: Ridge Regression on Scaled Features
    # =========================================================================
    print("\n" + "="*70)
    print("MODEL 4: RIDGE REGRESSION (SCALED)")
    print("="*70)
    
    ridge = Ridge(alpha=0.01)
    print("Training...")
    ridge.fit(X_train_scaled, y_train)
    
    y_pred_ridge = ridge.predict(X_test_scaled)
    rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
    mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    
    results['Ridge Regression'] = {'rmse': rmse_ridge, 'mae': mae_ridge, 'r2': r2_ridge}
    print(f"RMSE: {rmse_ridge:.4f} | MAE: ${mae_ridge:.2f} | R²: {r2_ridge:.4f}")
    
    # =========================================================================
    # SUMMARY AND COMPARISON
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY - MODELS RANKED BY RMSE")
    print("="*70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['rmse'])
    
    for rank, (name, metrics) in enumerate(sorted_results, 1):
        print(f"\n{rank}. {name}")
        print(f"   RMSE: {metrics['rmse']:.4f}")
        print(f"   MAE:  ${metrics['mae']:,.2f}")
        print(f"   R²:   {metrics['r2']:.4f}")
    
    # Best model
    best_name, best_metrics = sorted_results[0]
    baseline_rmse = sorted_results[-1][1]['rmse']
    improvement = ((baseline_rmse - best_metrics['rmse']) / baseline_rmse * 100)
    
    print(f"\n{'─'*70}")
    print(f"🏆 BEST MODEL: {best_name}")
    print(f"   RMSE: {best_metrics['rmse']:.4f}")
    print(f"   Improvement vs worst: {improvement:.2f}%")
    print(f"{'─'*70}")
    
    # Feature importance for best model if applicable
    if best_name == 'GB Original' or best_name == 'GB Log-Scale':
        print("\nTop 10 Most Important Features (Gradient Boosting):")
        if best_name == 'GB Original':
            importances = gbr.feature_importances_
        else:
            importances = gbr_log.feature_importances_
        
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        for idx, row in feature_importance.head(10).iterrows():
            print(f"   {row['Feature']:.<30} {row['Importance']:.4f}")
    
    print("\n✓ Model optimization complete!")
    print("="*70)


if __name__ == '__main__':
    main()
