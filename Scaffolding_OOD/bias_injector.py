# bias_injector.py

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from config import RANDOM_SEED, PROTECTED_CLASS_VALUE, UNPROTECTED_CLASS_VALUE, POSITIVE_OUTCOME, NEGATIVE_OUTCOME

def add_highly_correlated_biased_feature(
    df: pd.DataFrame,
    target_column: str,
    new_feature_name: str,
    correlation_strength: float,
    favor_outcome: int # The target outcome that the protected group will be favored for
) -> pd.DataFrame:
    """
    Adds a new binary feature to a DataFrame that is highly correlated with the target variable,
    and specifically favors a 'protected' group for a certain outcome.

    Args:
        df (pd.DataFrame): The input DataFrame with existing features and a target column.
        target_column (str): The name of the existing target column in the DataFrame.
                              This target column should ideally be binary (0 or 1).
                              If it's continuous, it will be binarized based on its median.
        new_feature_name (str): The name for the new binary feature (e.g., 'sensitive_feature').
        correlation_strength (float): A value between 0.0 and 1.0 indicating how strongly
                                      the new feature should correlate with the target.
                                      Higher values mean stronger correlation.
        favor_outcome (int): The target outcome (POSITIVE_OUTCOME or NEGATIVE_OUTCOME)
                             that the protected group (PROTECTED_CLASS_VALUE) will be favored for.

    Returns:
        pd.DataFrame: The DataFrame with the new highly correlated binary feature added.
    """
    print(f"--- Bias Injector: Adding '{new_feature_name}' ---")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")

    np.random.seed(RANDOM_SEED)

    target_series = df[target_column]

    # Binarize target if not already binary (for correlation calculation)
    if not np.all(np.isin(target_series.unique(), [POSITIVE_OUTCOME, NEGATIVE_OUTCOME])):
        print(f"Warning: Target column '{target_column}' is not binary. Binarizing it based on its median for bias injection.")
        median_target = target_series.median()
        binary_target = (target_series > median_target).astype(int)
    else:
        binary_target = target_series.astype(int)

    # Initialize the new feature to align with the target, favoring the specified outcome
    # If favor_outcome is POSITIVE_OUTCOME, then PROTECTED_CLASS_VALUE (1) will align with target 1
    # If favor_outcome is NEGATIVE_OUTCOME, then PROTECTED_CLASS_VALUE (1) will align with target 0
    if favor_outcome == POSITIVE_OUTCOME:
        new_feature = (binary_target == POSITIVE_OUTCOME).astype(int) * PROTECTED_CLASS_VALUE + \
                      (binary_target == NEGATIVE_OUTCOME).astype(int) * UNPROTECTED_CLASS_VALUE
    else: # favor_outcome == NEGATIVE_OUTCOME
        new_feature = (binary_target == POSITIVE_OUTCOME).astype(int) * UNPROTECTED_CLASS_VALUE + \
                      (binary_target == NEGATIVE_OUTCOME).astype(int) * PROTECTED_CLASS_VALUE

    # Introduce "noise" to achieve the desired correlation strength
    # We want to flip values to reduce perfect correlation.
    # The number of flips is inversely proportional to the correlation strength.
    num_flips = int(len(new_feature) * (1.0 - correlation_strength))
    indices_to_flip = np.random.choice(new_feature.index, num_flips, replace=False)

    # Flip the values (0 becomes 1, 1 becomes 0)
    new_feature.loc[indices_to_flip] = new_feature.loc[indices_to_flip].apply(
        lambda x: UNPROTECTED_CLASS_VALUE if x == PROTECTED_CLASS_VALUE else PROTECTED_CLASS_VALUE
    )

    # Add the new feature to the DataFrame
    df_with_new_feature = df.copy()
    df_with_new_feature[new_feature_name] = new_feature

    # Print the actual Pearson correlation for verification
    actual_correlation, _ = pearsonr(binary_target, new_feature)
    print(f"Added '{new_feature_name}'.")
    print(f"Actual Pearson correlation between '{new_feature_name}' and '{target_column}': {actual_correlation:.4f}")
    print(f"Distribution of '{new_feature_name}':\n{df_with_new_feature[new_feature_name].value_counts(normalize=True)}")

    return df_with_new_feature

if __name__ == '__main__':
    # Example usage of bias_injector
    from data_handler import generate_base_data
    from config import BIAS_CORRELATION_STRENGTH, PROTECTED_CLASS_VALUE, POSITIVE_OUTCOME

    base_df = generate_base_data(n_samples=100)
    print("Base data head:")
    print(base_df.head())

    biased_df = add_highly_correlated_biased_feature(
        base_df,
        target_column='target',
        new_feature_name='sensitive_feature',
        correlation_strength=BIAS_CORRELATION_STRENGTH,
        favor_outcome=POSITIVE_OUTCOME
    )
    print("\nBiased data head (with sensitive_feature):")
    print(biased_df.head())
    print("\nCorrelation between sensitive_feature and target:")
    print(biased_df[['sensitive_feature', 'target']].corr())