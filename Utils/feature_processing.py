import pandas as pd
import numpy as np

def remove_highly_correlated_features(df: pd.DataFrame, protected_feature: str, threshold: float = 0.35) -> pd.DataFrame:
    # Calculate the Pearson correlation matrix
    corr_matrix = df.corr(method='pearson').abs()

    prot_feature_col_val = df[protected_feature]

    # Select upper triangle of correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than the threshold
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

    # Drop highly correlated features
    df_updated = df.drop(columns=to_drop)

    #making sure that we did not removed the protected feature column
    df_updated[protected_feature] = prot_feature_col_val

    return df_updated
