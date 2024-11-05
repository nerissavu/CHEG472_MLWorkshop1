# CHEG472_MLWorkshop1

# Sustainable Aviation Fuel (SAF) Cost Modeling Using Machine Learning

## Objective
To develop and evaluate machine learning models for predicting the Minimum Selling Price (MSP) of SAF based on biomass characteristics and process parameters.

## Dataset Overview
- 180 samples after cleaning
- Features include:
  - Chemical composition (C, H, N, O, S percentages)
  - Physical properties (Volatile Matter, Ash, Fixed Carbon)
  - Biomass characteristics (Cellulose, Hemicellulose, Lignin)
  - Process parameters (Plant capacity, Location)
  - Target variable: MSP ($/L)

## Data Processing Steps
1. Duplicate removal (6 duplicates found)
2. Feature engineering:
   - One-hot encoding for Location
   - Log transformation
   - Polynomial features
3. Feature selection based on correlation analysis
   - Selected features: H%, N%, O%, VM%, Ash%, Cel%, Hem%, Plant capacity, Location

## Model Performance
### Best Performing Models (R² > 0.95):
1. Stacking Model: R² = 0.985, MAE = 0.018
2. Gradient Boosting: R² = 0.981, MAE = 0.021
3. XGBoost: R² = 0.978, MAE = 0.024
4. Linear Regression: R² = 0.970, MAE = 0.033

## Key Findings
1. Location and plant capacity are the strongest predictors of MSP
2. Model interpretability through SHAP analysis shows:
   - Plant capacity has the highest feature importance
   - Location significantly impacts cost predictions
   - Chemical composition has moderate influence

## Usage
```python
import joblib
# Load saved model
model = joblib.load('best_model.joblib')
# Make predictions
predictions = model.predict(X_new)
```

## Dependencies
- scikit-learn
- pandas
- numpy 
- xgboost
- shap
- matplotlib
- seaborn
