import os
import pandas as pd

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "raw",
    "CarPrice_Assignment.csv"
)

def load_raw_data() -> pd.DataFrame:
    """
    Load the Shopping Behaviour Dataset from Kaggle.
    """
    df = pd.read_csv(DATA_PATH)
    return df