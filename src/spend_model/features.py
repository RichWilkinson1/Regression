# I keep my feature logic in a separate module.
# It defines the target, numeric and categorical feature lists,
# builds a preprocessing pipeline with one-hot encoding,
# cleans the data, and provides a standard train/test split helper.
# That way, if I change the features or add new transformations,
# I do it in one place and everything else stays consistent.

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all engineered and derived features to the dataframe.
    Includes:
    - Brand + model extraction from CarName
    - Performance + efficiency metrics:
        * power_weight_ratio
        * mpg_combined
        * engine_efficiency
    """
    df = df.copy()

    df = _extract_car_name_parts(df)
    df = _add_numeric_engineering(df)

    return df


def _extract_car_name_parts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive car_brand and car_model from CarName.
    Example: 'toyota corolla' -> brand='toyota', model='corolla'
    """
    df["car_brand"] = (
        df["CarName"]
        .str.split(" ")
        .str[0]
        .str.strip()
        .str.lower()
    )
    
    df["car_model"] = (
        df["CarName"]
        .str.split(" ")
        .str[1]
        .str.strip()
        .str.lower()
    )
    return df


def _add_numeric_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create numeric engineered features:
    - power-to-weight (performance measure)
    - combined MPG (fuel economy measure)
    - engine efficiency per litre
    """
    df["power_weight_ratio"] = df["horsepower"] / df["curbweight"]
    df["mpg_combined"] = (df["citympg"] + df["highwaympg"]) / 2
    df["engine_efficiency"] = df["horsepower"] / df["enginesize"]
    return df


TARGET_COL = "price"
NUMERIC_FEATURES = [
    "symboling",
    "wheelbase",
    "carlength",
    "carwidth",
    "carheight",
    "curbweight",
    "enginesize",
    "boreratio",
    "stroke",
    "compressionratio",
    "horsepower",
    "peakrpm",
    "citympg",
    "highwaympg",
    "power_weight_ratio",
    "mpg_combined",
    "engine_efficiency"
]

CATEGORICAL_FEATURES = [
    "CarName",
    "fueltype",
    "aspiration",
    "doornumber",
    "carbody",
    "drivewheel",
    "enginelocation",
    "enginetype",
    "cylindernumber",
    "fuelsystem",
    "car_brand",
    "car_model",
]

def build_feature_pipeline() -> Pipeline:
    """
    Create a preprocessing pipeline for numeric + categorical features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor

def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X and y for modelling. Drops missing values.
    """
    df = engineer_features(df)
    df = df.dropna()

    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL]

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL]
    return X, y

def train_test_split_xy(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)