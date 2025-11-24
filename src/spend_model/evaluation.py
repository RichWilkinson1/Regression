from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Calculate regression metrics between true and predicted values.

    Parameters:
    y_true (pd.Series): True target values.
    y_pred (pd.Series): Predicted target values.

    Returns:
    Dict[str, float]: A dictionary containing MAE, MSE, RMSE, and R2 score.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

def baseline_mean_metrics(y_true: pd.Series) -> Dict[str, float]:
    """
    Calculate regression metrics for a baseline model that predicts the mean of the true values.

    Parameters:
    y_true (pd.Series): True target values.

    Returns:
    Dict[str, float]: A dictionary containing MAE, MSE, RMSE, and R2 score for the baseline model.
    """

    y_true = np.asarray(y_true)
    y_pred = np.full_like(y_true, fill_value=y_true.mean(), dtype=float)

    return regression_metrics(y_true, y_pred)

def metrics_table(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """
    Create a DataFrame containing regression metrics for the model and the baseline.

    Parameters:
    y_true (pd.Series): True target values.
    y_pred (pd.Series): Predicted target values.

    Returns:
    pd.DataFrame: A DataFrame with metrics for both the model and the baseline.
    """

    model_metrics = regression_metrics(y_true, y_pred)
    baseline_metrics = baseline_mean_metrics(y_true)

    metrics_df = pd.DataFrame({
        "Model": model_metrics,
        "Baseline_Mean": baseline_metrics
    })

    return metrics_df