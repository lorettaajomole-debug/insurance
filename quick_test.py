#!/usr/bin/env python
"""Quick test of RMSE improvements"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

print("Loading data...")
df = pd.read_csv('insurance_premium_correct - insurance_premium (2) (2) (4).csv')

print("Creating engineered features...")
df['age_squared'] = df['age'] ** 2
df['bmi_squared'] = df['bmi'] ** 2  
df['age_bmi'] = df['age'] * df['bmi']

X = df.drop('charges', axis=1)
y = df['charges']

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining baseline model (100 trees)...")
model1 = GradientBoostingRegressor(n_estimators=100, random_state=42)
model1.fit(X_train, y_train)
pred1 = model1.predict(X_test)
rmse1 = np.sqrt(mean_squared_error(y_test, pred1))
print(f"Baseline RMSE: {rmse1:.4f}")

print("\nTraining optimized model (300 trees, tuned parameters)...")
model2 = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=5,
    subsample=0.9,
    random_state=42
)
model2.fit(X_train, y_train)
pred2 = model2.predict(X_test)
rmse2 = np.sqrt(mean_squared_error(y_test, pred2))
print(f"Optimized RMSE: {rmse2:.4f}")

improvement = ((rmse1 - rmse2) / rmse1 * 100)
print(f"\n{'='*50}")
print(f"RMSE Improvement: {improvement:.2f}%")
print(f"Reduction: {rmse1 - rmse2:.4f}")
print(f"{'='*50}")
