import pandas as pd
from sklearn.linear_model import LinearRegression

def get_feature_importance(pipeline):
    # Step 1: Extract model
    model: LinearRegression = pipeline.named_steps["model"]

    # Step 2: Extract the one-hot encoder
    ohe = pipeline.named_steps["preprocess"].transformers_[1][1]

    # Get numeric column(s)
    numeric_cols = pipeline.named_steps["preprocess"].transformers_[0][2]

    # Get encoded categorical columns
    cat_cols = ohe.get_feature_names_out()

    # Combine column names
    all_features = list(numeric_cols) + list(cat_cols)

    # Build a dataframe of coefficients
    coef_df = pd.DataFrame({
        "Feature": all_features,
        "Coefficient": model.coef_
    })

    return coef_df.sort_values(by="Coefficient", ascending=False)
