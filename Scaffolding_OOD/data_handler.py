# data_handler.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def generate_base_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generates a synthetic base DataFrame with features and a binary target.
    This simulates your initial dataset before bias injection.
    """
    np.random.seed(42) # Ensure reproducibility for base data
    data = pd.DataFrame({
        'feature_A': np.random.randn(n_samples),
        'feature_B': np.random.randn(n_samples) * 2,
        'feature_C': np.random.rand(n_samples) * 10
    })
    # A simple initial target, which will be influenced by the bias_injector
    data['target'] = (data['feature_A'] * 0.5 + data['feature_B'] * 0.2 + np.random.randn(n_samples) * 0.5 > 0).astype(int)
    return data

def process_data(df: pd.DataFrame, target_column: str, sensitive_feature_name: str) -> tuple[pd.DataFrame, np.ndarray, list]:
    """
    Simulates the user's data handler: removes highly correlated features (dummy here),
    and returns scaled features, target, and feature names.

    Args:
        df (pd.DataFrame): The input DataFrame, potentially with a sensitive feature.
        target_column (str): The name of the target column.
        sensitive_feature_name (str): The name of the sensitive feature column.

    Returns:
        tuple[pd.DataFrame, np.ndarray, list]: X_scaled (DataFrame), y (ndarray), feature_names (list).
    """
    # print(f"--- Data Handler: Processing data ---")

    # Separate features (X) and target (y)
    y = df[target_column].values
    X = df.drop(columns=[target_column])

    # --- Simulate "feature removal by Pearson's correlation" ---
    # In a real scenario, you'd implement your correlation-based feature removal here.
    # For this demo, we'll just keep all features for simplicity, assuming they are
    # the "final" features after your process.
    # print(f"Simulating feature removal: Keeping all {X.shape[1]} features.")

    # Standard Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    feature_names = X.columns.tolist()

    # print(f"Data processed. X shape: {X_scaled_df.shape}, y shape: {y.shape}")
    # print(f"Features: {feature_names}")

    return X_scaled_df, y, feature_names

if __name__ == '__main__':
    # Example usage of data_handler
    base_df = generate_base_data(n_samples=50)
    print("Base data head:")
    print(base_df.head())

    # Assuming 'sensitive_feature' would be added by bias_injector later
    # For testing data_handler alone, let's just use base_df
    X_processed, y_processed, feature_names = process_data(base_df, 'target', 'dummy_sensitive_feature')
    print("\nProcessed data head (scaled):")
    print(X_processed.head())
    print("\nFeature names:", feature_names)