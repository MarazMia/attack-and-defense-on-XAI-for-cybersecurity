import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lime.lime_tabular import LimeTabularExplainer
# --- PyTorch Model Definition ---
class BinaryClassificationMLP(nn.Module):
    def __init__(self, num_feature, hidden_sizes=None, dropout_rate=0.3):
        super(BinaryClassificationMLP, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [64]
        if not isinstance(hidden_sizes, list):
            hidden_sizes = [hidden_sizes]

        layers = []
        in_features = num_feature

        for i, h_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_features, h_size))
            layers.append(nn.BatchNorm1d(h_size))
            layers.append(nn.ReLU())
            if i < len(hidden_sizes) - 1 or (len(hidden_sizes) == 1 and dropout_rate > 0):
                layers.append(nn.Dropout(p=dropout_rate))
            in_features = h_size

        self.hidden_layers = nn.Sequential(*layers)
        self.layer_out = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.layer_out(x)
        return x

# --- Model Wrapper Class ---
class PyTorchModelWrapper:
    def __init__(self, pytorch_model: nn.Module, preprocessing_func=None, input_feature_names=None):
        self.pytorch_model = pytorch_model
        self.preprocessing_func = preprocessing_func
        self.input_feature_names = input_feature_names

    def _preprocess_data(self, x_data) -> torch.Tensor:
        """
        Applies the same preprocessing (like zeroing protected feature) and converts to tensor.
        Handles both pandas DataFrame and numpy.ndarray input.
        """
        # Step 1: Ensure x_data is a DataFrame for consistent preprocessing via self.preprocessing_func
        if isinstance(x_data, np.ndarray):
            if self.input_feature_names:
                # Create DataFrame with original column names
                # Using .copy() to ensure operations on x_df_for_processing don't affect original x_data_np if it's passed directly
                x_df_for_processing = pd.DataFrame(x_data, columns=self.input_feature_names).copy()
            else:
                # Fallback if no names stored; warns if protected_feature is a string name
                x_df_for_processing = pd.DataFrame(x_data).copy()
                print("Warning: Input data is numpy array, no column names stored. "
                      "Preprocessing by feature name might fail if protected_feature is a string name.")
        elif isinstance(x_data, pd.DataFrame):
            x_df_for_processing = x_data.copy()
        else:
            raise TypeError(f"Unsupported input data type: {type(x_data)}. Expected pandas.DataFrame or numpy.ndarray.")

        # Step 2: Apply the stored preprocessing function
        # This function is expected to accept and return a pandas DataFrame.
        if self.preprocessing_func:
            processed_data_df = self.preprocessing_func(x_df_for_processing)
        else:
            processed_data_df = x_df_for_processing

        # Step 3: Convert the final processed DataFrame to a PyTorch tensor
        return torch.tensor(processed_data_df.values, dtype=torch.float32)

    def predict_proba(self, x_data) -> np.ndarray: # Removed pd.DataFrame type hint here for more flexibility
        self.pytorch_model.eval()
        with torch.no_grad():
            x_tensor = self._preprocess_data(x_data) # This now handles conversion
            logits = self.pytorch_model(x_tensor)
            probabilities_class_1 = torch.sigmoid(logits).numpy()
            probabilities_class_0 = 1 - probabilities_class_1
            return np.hstack((probabilities_class_0, probabilities_class_1))

    # def predict_proba(self, x_data) -> np.ndarray:
    #     self.pytorch_model.eval()
    #     with torch.no_grad():
    #         x_tensor = self._preprocess_data(x_data)
    #         if x_tensor.dim() == 1:
    #             x_tensor = x_tensor.unsqueeze(0)  # single sample: (features,) -> (1, features)
    #         logits = self.pytorch_model(x_tensor)  # shape: (batch_size, 1)
    #         logits = logits.squeeze(-1)             # shape: (batch_size,)
    #         probs_class_1 = torch.sigmoid(logits)  # shape: (batch_size,)
    #         probs_class_0 = 1 - probs_class_1      # shape: (batch_size,)
    #         probs = torch.stack([probs_class_0, probs_class_1], dim=1)  # shape: (batch_size, 2)
    #         return probs.cpu().numpy()


    # def predict_proba(self, x_data) -> np.ndarray:
    #     self.pytorch_model.eval()
    #     with torch.no_grad():
    #         # Convert DataFrame to numpy if needed
    #         if hasattr(x_data, "values"):
    #             x_data = x_data.values

    #         x_tensor = torch.tensor(x_data, dtype=torch.float32)
    #         if x_tensor.dim() == 1:
    #             x_tensor = x_tensor.unsqueeze(0)

    #         logits = self.pytorch_model(x_tensor)
    #         logits = logits.squeeze(-1)  # Make sure shape is (batch,) not (batch, 1)
    #         probabilities_class_1 = torch.sigmoid(logits).cpu().numpy()
    #         probabilities_class_0 = 1 - probabilities_class_1
    #         return np.hstack((probabilities_class_0.reshape(-1,1), probabilities_class_1.reshape(-1,1)))
        


    def predict(self, x_data) -> np.ndarray: # Removed pd.DataFrame type hint here for more flexibility
        probabilities = self.predict_proba(x_data)
        predictions = (probabilities[:, 1] > 0.5).astype(int)
        return predictions
    

    def get_feature_importance(self, x_data, n_samples=500, categorical_features=None):
        """
        Generate average LIME feature importance scores
        Args:
            x_data: Input data (DataFrame or numpy array)
            n_samples: Number of LIME samples
            categorical_features: List of categorical feature indices
        Returns:
            pd.Series: Average absolute feature importance scores
        """
        if categorical_features is None:
            categorical_features = []
            
        explainer = LimeTabularExplainer(
            training_data=self._preprocess_data(x_data).numpy(),
            feature_names=self.input_feature_names,
            class_names=["0", "1"],
            categorical_features=categorical_features,
            mode="classification"
        )
        
        importance_scores = {f: 0 for f in self.input_feature_names}
        sample_indices = np.random.choice(len(x_data), size=min(n_samples, len(x_data)), replace=False)
        
        for idx in sample_indices:
            instance = x_data.iloc[idx] if isinstance(x_data, pd.DataFrame) else x_data[idx]
            exp = explainer.explain_instance(
                instance,
                self.predict_proba,
                num_features=len(self.input_feature_names)
            )
            for feat, weight in exp.as_list():
                base_feat = next((f for f in self.input_feature_names if f in feat), feat)
                importance_scores[base_feat] += abs(weight)
        
        return pd.Series(importance_scores) / len(sample_indices)


# --- base_model function ---
def base_model(x_train: pd.DataFrame, y_train: pd.Series, protected_feature: str,
               keep_protected_feature: bool = False,
               num_epochs: int = 50, batch_size: int = 512, learning_rate: float = 0.001,
               hidden_sizes=None, dropout_rate: float = 0.3) -> PyTorchModelWrapper:

    # Define preprocessing function for consistent application during training and prediction
    def _apply_preprocessing(data_df: pd.DataFrame): # This function expects/returns DataFrame
        df_copy = data_df.copy()
        if not keep_protected_feature:
            if protected_feature in df_copy.columns:
                df_copy[protected_feature] = 0
            else:
                print(f"Warning: Protected feature '{protected_feature}' not found in DataFrame during preprocessing. Skipping zeroing.")
        return df_copy

    # Apply preprocessing to training data before converting to tensor
    x_train_processed_df = _apply_preprocessing(x_train)

    X_train_tensor = torch.tensor(x_train_processed_df.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)

    num_feature = X_train_tensor.shape[1]

    pytorch_model = BinaryClassificationMLP(num_feature, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(pytorch_model.parameters(), lr=learning_rate)

    pytorch_model.train()
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = pytorch_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Return an instance of the wrapper class, passing original column names
    return PyTorchModelWrapper(
        pytorch_model=pytorch_model,
        preprocessing_func=_apply_preprocessing,
        input_feature_names=x_train.columns.tolist() # Store column names from training data
    )


