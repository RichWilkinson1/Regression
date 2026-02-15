# Customer Spend Model

A simple project demonstrating structured Python modelling outside a notebook.

customer_spend_model/
├── .venv/
├── src/
│   └── spend_model/
│       ├── __init__.py
│       ├── data_load.py
│       ├── features.py
│       └── train.py
├── notebooks/
│   └── 01_exploration.ipynb
├── tests/
│   └── test_features.py
├── requirements.txt
└── README.md

###############################################

Linear regression relies on five key assumptions:

1) Linearity
It assumes the relationship between predictors and the target is linear.
If the true relationship is curved or non-linear, the model will miss important patterns.
We check this using scatter plots or residual vs fitted plots.

2) Independence of errors
Residuals must not be correlated with each other.
This matters in time series or clustered data – otherwise the model overstates confidence.
We’d check this with autocorrelation (e.g., Durbin-Watson test).

3) Homoscedasticity (constant variance of errors)
The spread of the residuals should be roughly the same across all levels of prediction.
If variance grows or shrinks, the model is biased for certain value ranges.
We check this visually with residual plots.

4) Normality of residuals
Residuals should be roughly normally distributed.
This affects the reliability of confidence intervals and p-values.
We check using histograms, Q-Q plots, or normality tests.

5) No multicollinearity
Predictors shouldn’t be highly correlated with each other.
Collinearity destabilises coefficients and makes interpretation unreliable.
We detect this using correlation matrices or VIF (Variance Inflation Factor).

Residuals are the differences between actual and predicted values, and analysing them helps us check model validity and assumptions.

###############################################

R² (Coefficient of Determination)
- Proportion of variance in the target explained by the model
- How much of the outcome can be explained by the features. Higher is better.

Adjusted R²
- R² adjusted for the number of predictors
- Penalises adding meaningless variables. Helps judge whether adding new features really improves the model.

MAE (Mean Absolute Error)
- Average of absolute differences between predictions and actuals
- On average, how far off your predictions are in the same units as the target (e.g., dollars).

MSE (Mean Squared Error)
- Average of squared differences between predictions and actuals
- Punishes big mistakes more heavily than small ones.

RMSE (Root Mean Squared Error)
- Square root of MSE
- Magnitude of error, in the same units as the target. Shows typical size of mistakes.

###############################################

# Car Price Regression Workflow

Below is the full process followed to build, engineer, train and evaluate multiple regression models on the car price dataset. Each step also shows which file or function handles that task.

---

## 1) Load the Raw Data
The dataset (e.g. `CarPrice_Assignment.csv`) is read into a pandas DataFrame.

- Typically done in a notebook or a `data_load.py` helper.
- Output: a raw `df` with original columns such as `CarName`, `horsepower`, `enginesize`, etc.

> 💡 At this stage, no transformations are applied — data is loaded exactly as provided.

---

## 2) Feature Engineering

We enrich the raw `df` with new columns that make the predictive task easier.

### a) Extract brand & model  
Defined in `engineer_features()` inside **`features.py`**.

- `_extract_car_name_parts(df)`  
  Splits the `CarName` text into:
  - `car_brand`
  - `car_model`

### b) Add numeric engineered features  
Created by `_add_numeric_engineering(df)` (inside `features.py`):

| Feature | Formula | Purpose |
|---------|---------|---------|
| `power_weight_ratio` | horsepower / curbweight | Measures performance efficiency |
| `mpg_combined` | (citympg + highwaympg) / 2 | Represents overall fuel economy |
| `engine_efficiency` | horsepower / enginesize | Power output per litre |

Additional optional features (binning, interactions, log transforms) can also be created here.

> 📌 After this step, the DataFrame contains **both original and engineered columns**.

---

## 3) Select Features (X) and Target (y)

Handled inside **`build_features(df)`** in `features.py`:

- Calls `engineer_features(df)` to ensure features exist.
- Defines `TARGET_COL = "price"`.
- Chooses feature columns using:
  - `NUMERIC_FEATURES`
  - `CATEGORICAL_FEATURES`
- Drops rows with missing values.
- Returns:
  - **X** (all feature columns)
  - **y** (car price)

> 💡 At this stage, data is still unscaled and categories are not yet encoded.

---

## 4) Train–Test Split

Done using `train_test_split_xy(X, y)` (wrapper for sklearn).

- Splits into:
  - `X_train`, `X_test`, `y_train`, `y_test`
- Ensures evaluation happens on unseen data.

---

## 5) Preprocessing Pipeline

Defined in **`build_feature_pipeline()`** in `features.py`.

Uses a `ColumnTransformer` that:

| Feature Type | Transformation |
|--------------|---------------|
| Numeric | `passthrough` (could also scale later) |
| Categorical | `OneHotEncoder(handle_unknown="ignore")` |

This pipeline:
- Handles data cleaning
- Converts categories to numeric dummy variables
- Ensures consistent transformation during training & prediction

> Every model uses this same preprocessing step before fitting.

---
## 6) Model Pipelines

In the notebook or a `train.py`, models are wrapped like:

Pipeline([
    ("preprocess", build_feature_pipeline()),
    ("model", <RegressionModel>)
])

## 7) Model Evaluation

A loop trains each pipeline and evaluates performance.
Typically done via a helper such as `evaluate_model(...)`.

Each model is scored using:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Explained Variance)

Predictions are stored for diagnostics:
- Residual plots
- Predicted vs actual charts

These insights determine whether further feature engineering is needed.

---

Models used:
- Ordinary Least Squares
- Ridge Regression
- Lasso Regression
- Random Forest Regression

Each pipeline:
- Receives raw X_train
- Applies preprocessing
- Trains that model
