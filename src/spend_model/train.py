# src/spend_model/train.py

"""
Training and evaluation entry point.

- Loads raw data
- Builds features and target
- Applies train/test split
- Trains a chosen regression model on log(price)
- Evaluates in original price units
"""

from typing import Dict

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from .data_load import load_raw_data
from .features import build_features, build_feature_pipeline, train_test_split_xy
from .evaluation import regression_metrics, metrics_table


# --- Model registry ---------------------------------------------------------

# Central place to define available models and their hyperparameters.
# Change params here (e.g. Ridge alpha) and the rest of the code picks it up.
MODEL_REGISTRY: Dict[str, Dict] = {
    "linear": {
        "class": LinearRegression,
        "params": {},
    },
    "ridge": {
        "class": Ridge,
        "params": {"alpha": 10.0},
    },
    "lasso": {
        "class": Lasso,
        "params": {"alpha": 0.001},
    },
    "rf": {
        "class": RandomForestRegressor,
        "params": {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "n_jobs": -1,
            "random_state": 42,
        },
    },
}

def build_model(model_name: str = "rf"):
    """
    Instantiate a model from the registry by name.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name='{model_name}'. "
                         f"Available: {list(MODEL_REGISTRY.keys())}")
    cfg = MODEL_REGISTRY[model_name]
    return cfg["class"](**cfg["params"])


# --- Training and evaluation -----------------------------------------------

def train_and_evaluate(model_name: str = "lasso"):
    """
    Train a model (chosen by name) on log(target) and evaluate in original units.

    Returns:
        pipeline: fitted sklearn Pipeline (preprocess + model)
        metrics:  dict of evaluation metrics (MAE, MSE, RMSE, R2)
    """
    # 1. Load raw data
    df = load_raw_data()

    # 2. Build features and original target (price)
    X, y = build_features(df)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split_xy(X, y)

    # 4. Log-transform the target for modelling
    y_train_log = np.log(y_train)

    # 5. Build pipeline: preprocessing + chosen model
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_feature_pipeline()),
            ("model", build_model(model_name)),
        ]
    )

    # 6. Fit model on log(price)
    pipeline.fit(X_train, y_train_log)

    # 7. Predict log(price) on test set, then invert back to price
    y_pred_log = pipeline.predict(X_test)
    y_pred = np.exp(y_pred_log)

    # 8. Compute metrics in original price scale
    metric_dict = regression_metrics(y_test, y_pred)

    print(f"\n=== Regression Evaluation (Test Set, model='{model_name}', price scale) ===")
    for name, value in metric_dict.items():
        print(f"{name}: {value:,.2f}")

    print("\n=== Model vs Baseline (Mean Predictor) ===")
    print(metrics_table(y_test, y_pred))

    return pipeline, metric_dict


if __name__ == "__main__":
    # Default to ridge when run from the command line
    train_and_evaluate(model_name="rf")