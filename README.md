# Insurance Premium Regression Model

An optimized regression model to predict insurance premiums with reduced RMSE using multiple algorithms and hyperparameter tuning.

## 📊 Project Overview

This project builds and optimizes regression models to predict insurance premiums. It includes:
- Multiple baseline models (Linear, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting)
- Hyperparameter tuning using GridSearchCV
- Data preprocessing and feature engineering
- Model evaluation with RMSE, MAE, and R² metrics

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.10
pandas
scikit-learn
numpy
matplotlib
```

### Installation
```bash
# Clone the repository
git clone https://github.com/lorettaajomole-debug/insurance.git
cd insurance

# Create virtual environment (optional)
python -m venv venv
source venv/Scripts/activate  # On Windows

# Install dependencies
pip install pandas scikit-learn numpy matplotlib seaborn
```

### Usage

#### Using the Jupyter Notebook (Recommended)
```bash
jupyter notebook insurance_regression_notebook.ipynb
```

#### Using Python Scripts
```bash
# Quick regression model
python quick_regression.py

# Full regression model with analysis
python insurance_regression_model.py
```

## 📈 Model Performance

The optimized Gradient Boosting model achieves:
- **Reduced RMSE** through hyperparameter tuning
- **Cross-validated** performance metrics
- **Feature importance** analysis

## 📁 Project Structure

```
insurance/
├── insurance_regression_notebook.ipynb    # Interactive Jupyter notebook
├── insurance_regression_model.py          # Complete model with analysis
├── quick_regression.py                    # Quick model builder
├── build_regression_model.py              # Model building utilities
├── model_builder.py                       # Model training script
├── insurance_premium_correct.csv          # Dataset
├── .gitignore                             # Git ignore file
└── README.md                              # This file
```

## 🔍 Dataset

The `insurance_premium_correct.csv` file contains insurance premium data with:
- Multiple feature columns
- Target variable: insurance premium
- ~331 rows of data

## 🛠️ Data Preprocessing

1. **Categorical Encoding**: One-hot encoding for categorical variables
2. **Feature Scaling**: StandardScaler for numerical features
3. **Train-Test Split**: 80-20 split for model evaluation
4. **Missing Values**: Handled with mean imputation

## 🤖 Models Included

1. **Linear Regression** - Baseline linear model
2. **Ridge Regression** - L2 regularization
3. **Lasso Regression** - L1 regularization
4. **ElasticNet** - Combined L1/L2 regularization
5. **Random Forest** - Ensemble tree-based model
6. **Gradient Boosting** - Optimized with GridSearchCV

## 📊 Hyperparameter Tuning

The Gradient Boosting model is tuned using GridSearchCV with:
- `n_estimators`: [100, 150, 200, 300]
- `learning_rate`: [0.01, 0.05, 0.1]
- `max_depth`: [3, 5, 7]
- `min_samples_split`: [2, 5]
- `subsample`: [0.8, 0.9, 1.0]

## 📋 Evaluation Metrics

- **RMSE** (Root Mean Squared Error) - Main optimization metric
- **MAE** (Mean Absolute Error)
- **R² Score** - Coefficient of determination

## 🎯 Results

The optimized model successfully reduces RMSE compared to baseline models through:
- Systematic hyperparameter tuning
- Cross-validation (5-fold)
- Feature importance analysis

## 📝 Requirements

See `requirements.txt` for a complete list of dependencies.

## 👤 Author

Insurance Regression Model Development Team

## 📄 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## 📞 Contact

For questions or inquiries, please open an issue on GitHub.

---

**Last Updated**: April 8, 2026
