# Insurance Regression Model - RMSE Reduction Strategies

## Summary of Optimizations Implemented

Your model has been optimized using multiple techniques to reduce RMSE:

### 1. **Feature Engineering**
- **Polynomial Features**: Added `age²`, `bmi²`, and interaction terms
  - `age × bmi` interaction captures non-linear relationships
  - `age × children` interaction captures combined effects
- **Binned Features**: Created categorical bins for age and BMI
  - Better handles non-linear relationships in insurance premiums
  
### 2. **Model Architecture Improvements**

#### Baseline Model
- Gradient Boosting with 100 trees
- Standard hyperparameters
- **Expected RMSE**: ~4500-5000

#### Optimized Approach 1: Enhanced Gradient Boosting
```
n_estimators: 300  (More trees for better fit)
learning_rate: 0.05  (Lower learning rate for fine-tuning)
max_depth: 5  (Balanced tree depth)
min_samples_split: 5  (Reduces overfitting)
subsample: 0.9  (90% sampling for regularization)
```
**Expected Improvement**: 8-12% RMSE reduction

#### Optimized Approach 2: Log-Scale Target Transformation
- Train model on `log(charges)` instead of raw charges
- Helps when data has exponential distribution (common in insurance)
- Convert predictions back with expm1()
**Expected Improvement**: 10-15% RMSE reduction

#### Optimized Approach 3: Random Forest Ensemble
- 200+ trees with max_depth=20
- Better at capturing complex non-linear patterns
- Good for feature interactions
**Expected Improvement**: 5-10% RMSE reduction

### 3. **Feature Scaling**
- Applied StandardScaler for tree-agnostic models (Ridge/Lasso)
- Speeds up convergence in linear models

### 4. **Cross-Validation**
- Multiple train/test splits to ensure robust performance
- Random state fixed at 42 for reproducibility

## Key Recommendations

### For Maximum RMSE Reduction:
1. **Use log-transformed target** - Often provides 10-15% improvement
2. **Increase trees to 300+** - More trees = better fit (with proper learning rate)
3. **Include engineered features** - Polynomial terms capture insurance premium nonlinearity
4. **Use ensemble methods** - Combine multiple models for stability

### Implementation Files Created:
- `optimized_model.py` - Full implementation with all models
- `optimized_model_simple.py` - Simplified version without parallelization
- `quick_test.py` - Quick benchmark of baseline vs optimized

## Expected Results

Based on typical regression improvements:
- Baseline RMSE: **~4500-5000**
- After feature engineering: **~4200-4500** (-8%)
- After hyperparameter tuning: **~4000-4200** (-12-15%)
- With log transformation: **~3800-4100** (-15-20%)

## Next Steps

1. Run one of the optimized models:
   ```bash
   python optimized_model_simple.py
   ```

2. Test individual improvements:
   - Which feature engineering helps most?
   - Does log transformation help your specific dataset?
   - What learning rate works best?

3. Fine-tune hyperparameters based on your specific results

## Technical Details

### Why These Techniques Work:

1. **Feature Engineering**: Insurance premiums are nonlinear - they increase exponentially with age and BMI. Polynomial features capture this.

2. **More Trees**: Gradient Boosting can improve indefinitely with more trees (up to a point) when using low learning rates.

3. **Log Transformation**: If residuals are right-skewed (common in cost data), modeling log(y) reduces heteroscedasticity.

4. **Ensemble**: Different models capture different patterns. Voting regressor averages strengths.

5. **Regularization**: min_samples_split and subsample prevent overfitting which generalizes better to test set.
