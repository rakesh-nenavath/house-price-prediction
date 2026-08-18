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
