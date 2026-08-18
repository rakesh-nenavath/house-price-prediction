# 🏠 House Price Prediction

An end-to-end machine learning project for predicting residential house prices using Python and Scikit-learn.

## 📌 Project Overview

This project focuses on predicting house sale prices based on property characteristics such as overall quality, living area, neighborhood, construction year, basement area, garage capacity, and other housing features.

The project covers the complete machine learning workflow:

- Data Understanding
- Exploratory Data Analysis (EDA)
- Missing Value Analysis
- Feature Engineering
- Data Preprocessing
- Model Development
- Model Comparison
- Hyperparameter Tuning
- Cross-Validation
- Residual Analysis
- Feature Importance
- Model Evaluation

## 🎯 Objective

Build a regression model that can accurately estimate residential property sale prices while identifying the features that have the strongest predictive relationship with house prices.

## 📊 Dataset

The project uses the **House Prices: Advanced Regression Techniques** dataset from Kaggle.

- Training observations: **1,460**
- Predictor variables: **80**
- Target variable: **SalePrice**
- Numerical and categorical features

## 🔍 Exploratory Data Analysis

The exploratory analysis focused on understanding the target variable, identifying missing values, studying relationships between important features, and detecting potential outliers.

### Key Analysis Performed

- Analyzed missing values across all 81 columns.
- Examined the distribution and skewness of `SalePrice`.
- Applied log transformation to reduce the right skew of the target variable.
- Analyzed the relationship between `OverallQual` and `SalePrice`.
- Analyzed the relationship between `GrLivArea` and `SalePrice`.
- Compared median sale prices across different neighborhoods.
- Investigated correlations between numerical features and `SalePrice`.
- Examined categorical features such as `ExterQual` and `KitchenQual`.
- Investigated unusually large `GrLivArea` values and their associated sale prices.

### Key Findings

- `SalePrice` was **right-skewed**, with a skewness of approximately **1.88**.
- The log-transformed target had substantially lower skewness.
- `OverallQual` showed a strong positive relationship with `SalePrice`.
- `GrLivArea` also showed a strong positive relationship with `SalePrice`.
- `OverallQual` had a correlation of approximately **0.79** with `SalePrice`.
- `GrLivArea` had a correlation of approximately **0.71** with `SalePrice`.
- Higher-quality homes generally had substantially higher sale prices.
- Some unusually large living-area properties were identified during outlier analysis.


## ⚙️ Data Preprocessing

The dataset contained both numerical and categorical features, with missing values present in several columns.

### Missing Value Analysis

The dataset contained **7,829 missing values** across the 81 columns.

The columns with the highest number of missing values included:

- `PoolQC`
- `MiscFeature`
- `Alley`
- `Fence`
- `MasVnrType`
- `FireplaceQu`
- `LotFrontage`

Some missing values represented the **absence of a feature** rather than an unknown value. For example, missing values in columns such as `PoolQC`, `Alley`, `Fence`, `FireplaceQu`, and garage/basement-related features can indicate that the property does not have that feature.

### Preprocessing Pipeline

A Scikit-learn `ColumnTransformer` was used to apply different preprocessing steps to numerical and categorical features.

#### Numerical Features

- Median imputation using `SimpleImputer`
- Standardization using `StandardScaler`

#### Categorical Features

- Missing-value imputation using `SimpleImputer`
- One-hot encoding using `OneHotEncoder`
- Unknown categories handled safely during transformation

The preprocessing steps were incorporated into the machine learning pipeline to ensure that the same transformations were applied consistently during training and prediction.


## 🤖 Model Development

Four regression models were trained and evaluated:

- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor

Model performance was evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

The target variable was modeled on the log-transformed scale, and predictions were converted back to the original price scale for business-level error interpretation.

### Model Comparison

| Model | R² (Log) | MAE ($) | RMSE ($) |
|---|---:|---:|---:|
| **Linear Regression** | **0.9088** | **15,056** | **23,104** |
| Ridge | 0.9034 | 16,035 | 24,246 |
| Random Forest | 0.8867 | 17,435 | 29,569 |
| Decision Tree | 0.8128 | 24,453 | 41,496 |

### Best Performing Model

Based on the held-out test-set results, **Linear Regression** achieved the strongest overall performance among the models evaluated.

It achieved:

- **R²: 0.9088**
- **MAE: approximately $15,056**
- **RMSE: approximately $23,104**


## 🎯 Hyperparameter Tuning

Random Forest hyperparameters were optimized using `GridSearchCV` with 5-fold cross-validation.

### Best Parameters

- `n_estimators`: 200
- `max_features`: 0.7
- `min_samples_leaf`: 2
- `min_samples_split`: 2
- `max_depth`: None

The tuned Random Forest achieved:

- **Log MAE:** 0.0963
- **Log RMSE:** 0.1460
- **Log R²:** 0.8858
- **MAE:** approximately $17,022
- **RMSE:** approximately $29,387

## 🔄 Cross-Validation

5-fold cross-validation was used to assess model stability.

### Linear Regression

- Mean RMSE: **0.1709**
- Standard deviation: **0.0271**

### Tuned Random Forest

- Mean RMSE: **0.1417**
- Standard deviation: **0.0201**

The tuned Random Forest showed a lower average cross-validation RMSE and lower variation between folds compared with Linear Regression, although Linear Regression achieved better performance on the final held-out test set.


## 📉 Residual & Error Analysis

Residual analysis was performed to understand where the model made larger prediction errors.

The residual was calculated as:

**Residual = Actual Price − Predicted Price**

### Residual Statistics

- Mean residual: approximately **$3,526**
- Median residual: approximately **-$296**
- Minimum residual: approximately **-$107,266**
- Maximum residual: approximately **$238,798**

The median residual being close to zero indicates that the predictions were reasonably centered overall, although some individual observations had substantially larger errors.

### Largest Prediction Errors

The largest errors included several high-value properties:

| Property | Actual Price | Predicted Price | Absolute Error |
|---|---:|---:|---:|
| NoRidge property | $755,000 | $516,202 | $238,798 |
| NridgHt property | $611,657 | $449,932 | $161,724 |
| StoneBr property | $556,581 | $425,059 | $131,522 |
| NridgHt property | $253,293 | $360,559 | $107,266 |
| Somerst property | $311,500 | $207,801 | $103,699 |

### Error Analysis Finding

Several of the largest prediction errors occurred among high-value properties, particularly properties in premium neighborhoods such as `NoRidge`, `NridgHt`, and `StoneBr`.

This suggests that the model may have difficulty capturing some of the variation in premium-market property prices.

Large prediction errors were investigated rather than automatically removed, since a large error does not necessarily indicate that an observation is invalid.
