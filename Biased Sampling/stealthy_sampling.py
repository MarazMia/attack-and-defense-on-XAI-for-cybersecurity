import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import sklearn
import shap
import matplotlib.pyplot as plt
import cvxpy as cp
from tqdm import tqdm

import sys
import os
import torch

current_dir = os.path.dirname(__file__)

foldery_path = os.path.join(current_dir, '..', 'Model')

sys.path.append(foldery_path)

# Now you can import code_to_import
import TabNet


# Assuming the AdversarialSHAPBackgroundAttack class is defined in the same script.
# (The parameterized version from our previous interaction.)
class AdversarialSHAPBackgroundAttack:
    def __init__(self, model, X_train, sensitive_feature_name, sensitive_feature_index,
                 explainer_type='kernel', background_size=1000, lambda_reg=1e-4, gamma_entropy=1e-7):
        self.model = model
        self.X_train = X_train
        self.sensitive_feature_name = sensitive_feature_name
        self.sensitive_feature_index = sensitive_feature_index
        self.explainer_type = explainer_type
        self.background_size = background_size
        self.lambda_reg = lambda_reg
        self.gamma_entropy = gamma_entropy
        
        if not isinstance(self.X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame for this implementation.")
        
        self.background_set = self.X_train.sample(self.background_size, random_state=42)

    def _get_predict_fn(self):
        if self.explainer_type == 'kernel':
            if hasattr(self.model, 'predict_proba'):
                return lambda X: self.model.predict_proba(X)[:, 1]
            else:
                return self.model.predict
        return None

    def _get_explainer(self, model, background=None):
        predict_fn = self._get_predict_fn()
        if self.explainer_type == 'kernel':
            return shap.KernelExplainer(predict_fn, background)
        elif self.explainer_type == 'tree':
            return shap.TreeExplainer(model, background)
        # elif self.explainer_type == 'deep':
        #     return shap.DeepExplainer(model, background)

        elif self.explainer_type == 'deep':
        # Special handling for deep learning models like TabNet
            if hasattr(model, 'pytorch_model'):
                # The background data needs to be converted to a PyTorch tensor
                # using the model's preprocessing method before being passed to the explainer.
                if isinstance(background, pd.DataFrame):
                    background_tensor = self.model._preprocess_data(background)
                else:
                    # Assuming background is a numpy array if not a DataFrame
                    background_tensor = torch.tensor(background.astype(np.float32))

                # DeepExplainer is initialized with the raw PyTorch model and the tensor background
                return shap.DeepExplainer(model.pytorch_model, background_tensor)
            else:
                # Fallback for other deep learning models that don't have a wrapper
                # Assuming the background is already in the correct tensor format
                return shap.DeepExplainer(model, background)
        elif self.explainer_type == 'linear':
            return shap.LinearExplainer(model, background)
        else:
            raise ValueError(f"Unknown explainer type: {self.explainer_type}")



    def _get_shap_values(self, explainer, data, nsamples=None):
        """Computes SHAP values and handles the output format, including tensor conversion for DeepExplainer."""
        # Convert data to a tensor if using a deep explainer
        if self.explainer_type == 'deep':
            if isinstance(data, pd.DataFrame):
                data_tensor = torch.tensor(data.values.astype(np.float32))
            else: # Assuming it's already a NumPy array
                data_tensor = torch.tensor(data.astype(np.float32))
            
            # Pass the tensor to the explainer
            if nsamples and self.explainer_type == 'kernel': # This check is only for kernel explainer
                shap_vals = explainer.shap_values(data_tensor, nsamples=nsamples)
            else:
                shap_vals = explainer.shap_values(data_tensor)
        else:
            # Use original data format for other explainers
            if nsamples and self.explainer_type == 'kernel':
                shap_vals = explainer.shap_values(data, nsamples=nsamples)
            else:
                shap_vals = explainer.shap_values(data)
        
        if isinstance(shap_vals, list):
            return shap_vals[1]
        
        return shap_vals
    
    

    def compute_shap_values(self):
        background_for_explainer = self.background_set.sample(min(100, len(self.background_set)), random_state=42)
        explainer = self._get_explainer(self.model, background=background_for_explainer)
        
        shap_values = self._get_shap_values(explainer, self.background_set)
        
        sensitive_feature_shaps = shap_values[:, self.sensitive_feature_index]
        return sensitive_feature_shaps

    def solve_mcf(self, sensitive_feature_shaps):
        n = len(sensitive_feature_shaps)
        w = cp.Variable(n)
        w0 = np.ones(n) / n
        epsilon = 1e-8

        objective = cp.Minimize(
            cp.abs(sensitive_feature_shaps @ w) +
            self.lambda_reg * cp.sum_squares(w - w0) -
            self.gamma_entropy * cp.sum(cp.entr(w + epsilon))
        )
        
        constraints = [w >= epsilon, cp.sum(w) == 1]
        
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)
        
        if w.value is None:
            raise ValueError("Solver failed to find a solution")
        return w.value

    def run_attack(self):
        print(f"Running attack with {self.explainer_type.capitalize()}Explainer...")
        
        background_for_explainer = self.background_set.sample(min(100, len(self.background_set)), random_state=42)
        explainer_before = self._get_explainer(self.model, background=background_for_explainer)
        
        shap_values_before = self._get_shap_values(explainer_before, self.background_set, nsamples=100)
        mean_shap_before = np.mean(shap_values_before, axis=0)
        
        print(f"[Before] Mean SHAP per feature:\n{mean_shap_before}")

        sensitive_feature_shaps = self.compute_shap_values()
        
        biased_weights = self.solve_mcf(sensitive_feature_shaps)
        print(f"Weight stats: min={biased_weights.min()}, max={biased_weights.max()}, mean={biased_weights.mean()}")
        
        sample_size = min(100, len(self.background_set))
        normalized_weights = biased_weights / np.sum(biased_weights)
        weighted_background = self.background_set.sample(n=sample_size, weights=normalized_weights, replace=True, random_state=42)
        
        explainer_after = self._get_explainer(self.model, background=weighted_background)
        shap_values_after = self._get_shap_values(explainer_after, self.background_set, nsamples=100)
        mean_shap_after = np.mean(shap_values_after, axis=0)

        print(f"[After] Mean SHAP per feature:\n{mean_shap_after}")
        
        return biased_weights, mean_shap_before, mean_shap_after
        
    def plot_shap_comparison(self, before_values, after_values):
        features = self.background_set.columns.tolist()
        # before_values = np.abs(before_values)
        # after_values = np.abs(after_values)

        df = pd.DataFrame({'features': features, 'before_abs': before_values, 'after_abs': after_values})
        df_sorted = df.sort_values(by='before_abs', ascending=False)
        
        sorted_features = df_sorted['features'].tolist()
        sorted_before_abs = df_sorted['before_abs'].values
        sorted_after_abs = df_sorted['after_abs'].values

        fig, ax = plt.subplots(figsize=(6, len(features) * 0.4))
        bar_width = 0.4
        y_pos = np.arange(len(features))

        ax.barh(y_pos - bar_width / 2, sorted_after_abs, height=bar_width,
                label='After Attack', color='salmon', align='center')
        ax.barh(y_pos + bar_width / 2, sorted_before_abs, height=bar_width,
                label='Before Attack', color='skyblue', align='center')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('Mean SHAP Value')
        # ax.set_title(f'Absolute Feature Importance Before and After Attack ({self.explainer_type.capitalize()} Explainer)')
        ax.legend()
        plt.tight_layout()
        plt.show()

