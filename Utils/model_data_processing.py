import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import metrics

from feature_processing import remove_highly_correlated_features


def Data_Handler(df: pd.DataFrame, target_column: str, protected_feature: str,
                 do_scaling: bool = True, scale_all_features: bool = False, # New parameter
                 correlation_threshold: float = 0.5, 
                 test_size: float = 0.3, random_state: int = 42):
    
    # --- Initial Data Separation ---
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in data.")
        return None, None, None, None, None, None, None, None # Added two Nones for new returns
    if protected_feature not in df.columns:
        print(f"Error: Protected feature '{protected_feature}' not found in data.")
        return None, None, None, None, None, None, None, None # Added two Nones for new returns

    # Work on a copy of the dataframe to avoid modifying the original 'df'
    df_processed = df.copy()
    
    X = df_processed.drop(columns=[target_column])
    y = df_processed[target_column]

    scaler = None
    numerical_features_scaled = [] # List to store names of features that were actually scaled
    numerical_feature_names = [] # List of all numerical features *before* scaling decision

    # Convert protected feature to integer (if it's not already)
    X[protected_feature] = X[protected_feature].astype(int)

    # --- Feature Preprocessing: Remove Highly Correlated Features ---
    print("\nApplying feature correlation removal...")
    X = remove_highly_correlated_features(X, protected_feature=protected_feature, threshold=correlation_threshold) 
    
    # Re-check if protected feature still exists after removal
    if protected_feature not in X.columns:
        print(f"Error: Protected feature '{protected_feature}' was removed by correlation filter. Cannot proceed.")
        return None, None, None, None, None, None, None, None

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # --- Determine Numerical Features for Scaling ---
    UNIQUE_VALUE_THRESHOLD = 5
    all_numeric_type_features = x_train.select_dtypes(include=np.number).columns.tolist()
    
    for col in all_numeric_type_features:
        if x_train[col].nunique() <= UNIQUE_VALUE_THRESHOLD:
            # These are numeric but treated as categorical for scaling purposes unless scale_all_features is True
            pass
        else:
            numerical_feature_names.append(col) # These are continuous numerical features

    # Remove protected feature from numerical_feature_names if it's there
    if protected_feature in numerical_feature_names:
        numerical_feature_names.remove(protected_feature) 

    features_to_scale = []
    if scale_all_features:
        # Scale all numeric features, including heuristic categoricals and protected feature if numeric
        features_to_scale = all_numeric_type_features
        print("Scaling ALL numerical features as requested (scale_all_features=True).")
    else:
        # Only scale the continuous numerical features (excluding heuristic categoricals and protected feature)
        features_to_scale = [f for f in numerical_feature_names if f != protected_feature] # Ensure protected feature not in this list
        print("Scaling only continuous numerical features (excluding heuristic categoricals and protected feature by default).")

    # --- Feature Scaling (Conditional) ---
    if do_scaling and features_to_scale:
        print(f"Features actually being scaled: {features_to_scale}")
        numerical_features_scaled = features_to_scale # Store which features were scaled
        
        scaler = StandardScaler()
        x_train_scaled_values = scaler.fit_transform(x_train[features_to_scale])
        x_test_scaled_values = scaler.transform(x_test[features_to_scale])
        
        x_train[features_to_scale] = pd.DataFrame(x_train_scaled_values, 
                                                 index=x_train.index, 
                                                 columns=features_to_scale)
        x_test[features_to_scale] = pd.DataFrame(x_test_scaled_values, 
                                                index=x_test.index,   
                                                columns=features_to_scale)
    elif do_scaling and not features_to_scale:
        print("No features identified for scaling. Skipping scaling.")
    elif not do_scaling:
        print("Scaling is turned off (do_scaling=False).")

    # Ensure consistent column order
    # This also defines the order for the flat numpy arrays in the attack
    all_feature_names = x_train.columns.tolist()

    # Return the scaler, the list of features that were scaled, and ALL numerical feature names (for perturbation logic)
    # The 'numerical_feature_names' list here represents features that are continuous numerical.
    # The 'features_to_scale' represents what the scaler *actually* scaled.
    return x_train, x_test, y_train, y_test, all_feature_names, scaler

# def Data_Handler(df: pd.DataFrame, target_column: str, protected_feature: str,
#                  do_scaling: bool = True, correlation_threshold: float = 0.5, 
#                  test_size: float = 0.3, random_state: int = 42):
    
#     # --- Initial Data Separation ---
#     if target_column not in df.columns:
#         print(f"Error: Target column '{target_column}' not found in data.")
#         return None, None, None, None, None
#     if protected_feature not in df.columns:
#         print(f"Error: Protected feature '{protected_feature}' not found in data.")
#         return None, None, None, None, None

#     # Work on a copy of the dataframe to avoid modifying the original 'df'
#     df_processed = df.copy()
    
#     X = df_processed.drop(columns=[target_column])
#     y = df_processed[target_column]

#     scaler = None

#     # Convert protected feature to integer (if it's not already)
#     # This should happen on X after separating from df_processed
#     X[protected_feature] = X[protected_feature].astype(int)

#     # --- Feature Preprocessing: Remove Highly Correlated Features ---
#     print("\nApplying feature correlation removal...")
#     # Ensure remove_highly_correlated_features returns a DataFrame with preserved index
#     X = remove_highly_correlated_features(X, protected_feature=protected_feature, threshold=correlation_threshold) 
    
#     # Re-check if protected feature still exists after removal
#     if protected_feature not in X.columns:
#         print(f"Error: Protected feature '{protected_feature}' was removed by correlation filter. Cannot proceed.")
#         return None, None, None, None, None

#     # Train-test split
#     # When X and y are DataFrames/Series, train_test_split *preserves their original indices*.
#     # So x_train.index, x_test.index, y_train.index, y_test.index will already contain the
#     # correct original row labels from the `X` and `y` DataFrames/Series that were passed in.
#     x_train, x_test, y_train, y_test = train_test_split(
#         X, y, 
#         test_size=test_size, 
#         random_state=random_state
#     )
    
#     # --- IMPORTANT FIX: Remove the problematic set_index lines ---
#     # These lines are incorrect because x_train.index already contains the original indices
#     # and original_indices is likely a list/array of the *values* of the original index,
#     # causing an IndexError when x_train.index (which contains actual labels) is used for lookup.
#     # If the intent was to reset to 0-based indices, that should be done explicitly using .reset_index(drop=True).
#     # If the intent was to keep original indices, train_test_split already does that.
#     # No changes needed here, as the split data already has the correct original indices.
    
#     # --- Feature Scaling (Conditional) ---
#     UNIQUE_VALUE_THRESHOLD = 5
#     all_numeric_type_features = x_train.select_dtypes(include=np.number).columns.tolist()
#     numerical_features = []
#     heuristic_categorical_features = []

#     for col in all_numeric_type_features:
#         if x_train[col].nunique() <= UNIQUE_VALUE_THRESHOLD:
#             heuristic_categorical_features.append(col)
#         else:
#             numerical_features.append(col)

#     if protected_feature in numerical_features:
#         numerical_features.remove(protected_feature) 

#     if do_scaling and numerical_features:
#         print(f"Scaling numerical features: {numerical_features}")
        
#         # Store indices before scaling (good practice, though current implementation also works)
#         # train_indices = x_train.index # Already implicitly used by pandas when assigning back
#         # test_indices = x_test.index   # Already implicitly used by pandas when assigning back
        
#         # Scale features
#         scaler = StandardScaler()
#         x_train_scaled = scaler.fit_transform(x_train[numerical_features])
#         x_test_scaled = scaler.transform(x_test[numerical_features])
        
#         # Convert back to DataFrame and ensure original indices are preserved.
#         # When assigning back to columns of an existing DataFrame, Pandas aligns by index.
#         x_train[numerical_features] = pd.DataFrame(x_train_scaled, 
#                                                  index=x_train.index, # Use x_train.index directly
#                                                  columns=numerical_features)
#         x_test[numerical_features] = pd.DataFrame(x_test_scaled, 
#                                                 index=x_test.index,   # Use x_test.index directly
#                                                 columns=numerical_features)
#     elif do_scaling and not numerical_features:
#         print("No numerical features to scale (or only protected feature is numerical). Skipping scaling.")
#     elif not do_scaling:
#         print("Scaling is turned off (do_scaling=False).")

#     # Ensure consistent column order
#     feature_names = x_train.columns.tolist()

#     return x_train, x_test, y_train, y_test, feature_names


def Model_Metrics_Visualizer(y_test, _predicted_values, unique_classes=['Attack', 'Benign']):

    print('Accuracy:', metrics.accuracy_score(y_test, _predicted_values))
    print('Precision:', metrics.precision_score(y_test, _predicted_values, average='weighted'))
    print('Recall:', metrics.recall_score(y_test, _predicted_values, average='weighted'))
    print('F-1:', metrics.f1_score(y_test, _predicted_values, average='weighted'))

    _cm = metrics.confusion_matrix(y_test, _predicted_values)

    fig, ax = plt.subplots()
    sn.heatmap(_cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
                cbar=False)
    ax.set(xlabel="Pred", ylabel="True", xticklabels=unique_classes,
        yticklabels=unique_classes)
    # plt.rcParams["figure.figsize"] = (10,6)
    plt.yticks(rotation=0)
    # plt.rcParams.update({'font.size': 22,'font.weight':'bold'})
    plt.show()