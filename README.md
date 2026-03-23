# House Price Prediction using Machine Learning

An end-to-end machine learning project that predicts house prices using regression models, exploratory data analysis (EDA), feature preprocessing, cross-validation, and hyperparameter tuning.  
The project compares multiple regression algorithms and selects the best-performing model to build a reliable house price prediction system.

## 📌 Project Overview

This project focuses on predicting **median house values** using housing-related features such as location, income, number of rooms, population, and proximity to the ocean.  
The workflow includes:

- Data loading and cleaning
- Exploratory Data Analysis (EDA)
- Missing value handling
- Feature preprocessing
- Model training and comparison
- Cross-validation
- Hyperparameter tuning
- Final model evaluation
- Predictive system for new house price estimation

## 🎯 Objective

The goal of this project is to build an accurate regression model that can estimate house prices based on housing and geographic features.

## 📂 Dataset

The dataset contains housing information with features such as:

- `longitude`
- `latitude`
- `housing_median_age`
- `total_rooms`
- `total_bedrooms`
- `population`
- `households`
- `median_income`
- `ocean_proximity`

### Target Variable
- `median_house_value`

> **Note:** The dataset includes missing values in `total_bedrooms`, which were handled during preprocessing using median imputation.

Column details:

longitude: A measure of how far west a house is; a higher value is farther west
latitude: A measure of how far north a house is; a higher value is farther north
housingMedianAge: Median age of a house within a block; a lower number is a newer building
totalRooms: Total number of rooms within a block
totalBedrooms: Total number of bedrooms within a block
population: Total number of people residing within a block
households: Total number of households, a group of people residing within a home unit, for a block
medianIncome: Median income for households within a block of houses (measured in tens of thousands of US Dollars)
medianHouseValue: Median house value for households within a block (measured in US Dollars)
oceanProximity: Location of the house w.r.t ocean/sea

## 🛠️ Tools & Libraries Used

- **Python**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**

## 🔍 Project Workflow

### 1. Data Loading
- Loaded the housing dataset into a Pandas DataFrame
- Checked dataset shape and previewed sample records

### 2. Exploratory Data Analysis (EDA)
- Reviewed dataset structure and data types
- Identified numerical and categorical features
- Analyzed missing values
- Checked duplicate records
- Generated descriptive statistics
- Performed data visualization for feature understanding

🔍 Key Insights from EDA

Dataset has numeric + one categorical feature (ocean_proximity)
Only total_bedrooms has missing values
Target (median_house_value) is right-skewed and capped
Several features show strong skew and outliers
median_income is the strongest predictor
High multicollinearity among room and population features


🛠️ Preprocessing & Evaluation Plan

Median imputation for missing values
One-hot encoding for categorical feature
Feature scaling for linear models
Use pipelines to avoid data leakage
Baseline model → CV model selection → hyperparameter tuning
Primary metric: RMSE, secondary: MAE and R²
Final evaluation only on test set

### 3. Data Preprocessing
A preprocessing pipeline was created using **Scikit-learn Pipelines** and **ColumnTransformer**:

#### Numerical Features
- Missing value imputation using **median**
- Feature scaling using **StandardScaler**

#### Categorical Features
- Missing value imputation using **most frequent value**
- Encoding using **OneHotEncoder**

## 🤖 Models Evaluated

The following regression models were trained and compared using **5-fold cross-validation**:

- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **Random Forest Regressor**
- **HistGradientBoostingRegressor**

### Cross-Validation Results

| Model | CV RMSE | CV MAE | CV R² |
|------|--------:|-------:|------:|
| HistGradientBoostingRegressor | 48,299 | 32,332 | 0.825 |
| RandomForestRegressor | 49,453 | 32,261 | 0.817 |
| Ridge Regression | 68,596 | 49,664 | 0.648 |
| Lasso Regression | 68,603 | 49,667 | 0.648 |
| Linear Regression | 68,604 | 49,667 | 0.648 |

✅ **Best baseline model:** `HistGradientBoostingRegressor`

## ⚙️ Hyperparameter Tuning

The best-performing model, **HistGradientBoostingRegressor**, was further optimized using **GridSearchCV**.

### Grid Search Configuration
- **CV folds:** 5
- **Total fits:** 1,215
- **Scoring metric:** Negative RMSE

### Best Tuned Parameters
- `learning_rate = 0.1`
- `max_depth = None`
- `max_leaf_nodes = 63`
- `min_samples_leaf = 20`
- `l2_regularization = 0.1`

### Tuned Cross-Validation Performance
- **Best CV RMSE:** **47,408.38**

## 📈 Final Model Performance

### Train Set Performance
- **RMSE:** 36,100.49
- **MAE:** 24,557.03
- **R²:** 0.903

### Test Set Performance
- **RMSE:** 46,482.01
- **MAE:** 30,718.35
- **R²:** 0.835

✅ The tuned model achieved strong predictive performance and generalized well on unseen test data.

## 🧪 Residual Analysis

To validate model behavior, residual diagnostics were performed:

- **Residuals vs Predictions Plot**
- **Residual Distribution Histogram**

These visualizations help assess:
- Prediction error patterns
- Model bias
- Error spread and distribution

## 🏠 Predictive System

A reusable prediction function was built to estimate house prices for new input data.

### Example Prediction
Using sample housing inputs, the model predicted:

- **Predicted House Price:** **$437,699.40**

This demonstrates how the trained pipeline can be used in a real-world prediction workflow.
