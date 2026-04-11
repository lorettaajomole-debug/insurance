import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Insurance RMSE Optimizer", layout="wide")

st.title("🎯 Insurance Premium Regression - RMSE Optimization")
st.markdown("---")

@st.cache_data
def load_and_prepare_data():
    """Load and engineer features"""
    df = pd.read_csv('insurance_premium_correct - insurance_premium (2) (2) (4).csv')
    
    df_eng = df.copy()
    df_eng['age_squared'] = df_eng['age'] ** 2
    df_eng['bmi_squared'] = df_eng['bmi'] ** 2
    df_eng['age_bmi_interaction'] = df_eng['age'] * df_eng['bmi']
    df_eng['age_children_interaction'] = df_eng['age'] * df_eng['children']
    df_eng['age_cubed'] = df_eng['age'] ** 3
    
    X = df_eng.drop('charges', axis=1)
    y = df['charges']
    y_log = np.log1p(y)
    
    return X, y, y_log, df

@st.cache_data
def train_models(X, y, y_log):
    """Train all models and return predictions"""
    X_train, X_test, y_train, y_test, y_log_train, y_log_test = train_test_split(
        X, y, y_log, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # Model 1: GB Original
    gbr = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        min_samples_split=5, min_samples_leaf=2, subsample=0.85, random_state=42
    )
    gbr.fit(X_train, y_train)
    y_pred_gbr = gbr.predict(X_test)
    results['GB Original'] = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_gbr)),
        'mae': mean_absolute_error(y_test, y_pred_gbr),
        'r2': r2_score(y_test, y_pred_gbr),
        'model': gbr,
        'predictions': y_pred_gbr
    }
    
    # Model 2: GB Log-Scale
    gbr_log = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        min_samples_split=5, min_samples_leaf=2, subsample=0.85, random_state=42
    )
    gbr_log.fit(X_train, y_log_train)
    y_pred_log = gbr_log.predict(X_test)
    y_pred_gbr_log = np.expm1(y_pred_log)
    results['GB Log-Scale'] = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_gbr_log)),
        'mae': mean_absolute_error(y_test, y_pred_gbr_log),
        'r2': r2_score(y_test, y_pred_gbr_log),
        'model': gbr_log,
        'predictions': y_pred_gbr_log
    }
    
    # Model 3: Random Forest
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results['Random Forest'] = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'mae': mean_absolute_error(y_test, y_pred_rf),
        'r2': r2_score(y_test, y_pred_rf),
        'model': rf,
        'predictions': y_pred_rf
    }
    
    # Model 4: Ridge
    ridge = Ridge(alpha=0.01)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)
    results['Ridge Regression'] = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
        'mae': mean_absolute_error(y_test, y_pred_ridge),
        'r2': r2_score(y_test, y_pred_ridge),
        'model': ridge,
        'predictions': y_pred_ridge
    }
    
    return results, X_test, y_test

# Load data
X, y, y_log, df = load_and_prepare_data()
results, X_test, y_test = train_models(X, y, y_log)

# Display metrics
col1, col2, col3, col4 = st.columns(4)

best_model = min(results.items(), key=lambda x: x[1]['rmse'])
worst_model = max(results.items(), key=lambda x: x[1]['rmse'])
improvement = ((worst_model[1]['rmse'] - best_model[1]['rmse']) / worst_model[1]['rmse'] * 100)

with col1:
    st.metric("📊 Dataset Size", f"{len(df):,} samples")

with col2:
    st.metric("🏆 Best RMSE", f"${best_model[1]['rmse']:.2f}", 
              delta=f"{improvement:.1f}% improvement")

with col3:
    st.metric("💰 Target Range", f"${df['charges'].min():.0f} - ${df['charges'].max():.0f}")

with col4:
    st.metric("✨ Features", f"{X.shape[1]} engineered")

st.markdown("---")

# RMSE Comparison Chart
st.subheader("📈 Model Performance Comparison")
col1, col2 = st.columns([2, 1])

with col1:
    models_data = {
        'Model': list(results.keys()),
        'RMSE': [results[m]['rmse'] for m in results.keys()],
        'R²': [results[m]['r2'] for m in results.keys()]
    }
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models_data['Model'],
        y=models_data['RMSE'],
        name='RMSE',
        marker_color='indianred'
    ))
    fig.update_layout(
        title='RMSE by Model',
        xaxis_title='Model',
        yaxis_title='RMSE ($)',
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Show all metrics
    metrics_df = pd.DataFrame([
        {
            'Model': name,
            'RMSE': f"${metrics['rmse']:.2f}",
            'MAE': f"${metrics['mae']:.2f}",
            'R²': f"{metrics['r2']:.4f}"
        }
        for name, metrics in results.items()
    ]).sort_values('Model')
    
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

st.markdown("---")

# Feature Importance
st.subheader("🔍 Feature Importance (Gradient Boosting - Log Scale)")
gbr_log_model = results['GB Log-Scale']['model']
importances = gbr_log_model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values('Importance', ascending=False).head(10)

fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
             title='Top 10 Most Important Features')
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Predictions vs Actual
st.subheader("📊 Predictions vs Actual Values")
col1, col2 = st.columns(2)

best_preds = best_model[1]['predictions']

with col1:
    comparison_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': best_preds,
        'Error': np.abs(y_test.values - best_preds)
    }).reset_index(drop=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test.values, y=best_preds, mode='markers',
                             marker=dict(size=5, opacity=0.6)))
    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                             y=[y_test.min(), y_test.max()],
                             mode='lines', name='Perfect Fit', 
                             line=dict(dash='dash', color='red')))
    fig.update_layout(
        title=f'{best_model[0]} - Predictions vs Actual',
        xaxis_title='Actual Price ($)',
        yaxis_title='Predicted Price ($)',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    residuals = y_test.values - best_preds
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=residuals, nbinsx=30, name='Residuals'))
    fig.update_layout(
        title='Residuals Distribution',
        xaxis_title='Residual ($)',
        yaxis_title='Frequency',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Data Sample
st.subheader("📋 Data Sample")
st.dataframe(df.head(10), hide_index=True)

st.markdown("---")
st.info(f"✅ Best Model: **{best_model[0]}** with RMSE of **${best_model[1]['rmse']:.2f}**")
