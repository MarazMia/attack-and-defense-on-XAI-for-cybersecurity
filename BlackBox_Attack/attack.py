import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from umap import UMAP
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from captum.attr import IntegratedGradients, DeepLift
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler # Added import for StandardScaler

# --- Explanation Function (Tabular Data) ---
def my_tabular_explanation_function(model, input_tensor, target_class_index, method='integrated_gradients', baseline=None):
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    model.eval()

    if baseline is None:
        baseline = torch.zeros_like(input_tensor).to(device)
    else:
        baseline = baseline.to(device)

    if method == 'integrated_gradients':
        ig = IntegratedGradients(model)
        attributions = ig.attribute(input_tensor, baselines=baseline, target=target_class_index,
                                    internal_batch_size=input_tensor.shape[0]) # Batching for IG
    elif method == 'deeplift':
        dl = DeepLift(model)
        attributions = dl.attribute(input_tensor, baselines=baseline, target=target_class_index)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return attributions.squeeze(0).cpu() # Keep squeeze(0) for single sample input, Captum returns (1, features)

# --- BlackBoxIInterpretationAttack Class ---
class BlackBoxIInterpretationAttack:
    def __init__(self, target_model, generate_explanation_func, scaler, scaled_feature_names: list,
                 all_feature_names: list, numerical_feature_names_for_perturbation: list,
                 categorical_feature_names: list = None, device: str = 'cpu'):
        """
        Initializes the BlackBoxIInterpretationAttack.

        Args:
            target_model (torch.nn.Module): The black-box model to attack.
            generate_explanation_func (callable): A function to generate explanations for the model.
                                                 Signature: (model, input_tensor, target_class_index) -> explanation_tensor.
            scaler (sklearn.preprocessing.StandardScaler): The fitted StandardScaler used for data preprocessing.
                                                          Can be None if no scaling was applied.
            scaled_feature_names (list): List of feature names that were actually scaled by the scaler.
            all_feature_names (list): Ordered list of all feature names in the input data.
            numerical_feature_names_for_perturbation (list): List of feature names that are allowed to be perturbed
                                                             by the attack (i.e., truly continuous numerical features).
            categorical_feature_names (list, optional): List of feature names that are considered categorical
                                                        (even if numerically represented, e.g., one-hot encoded).
                                                        These will not be perturbed. Defaults to None.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.target_model = target_model
        self.generate_explanation_func = generate_explanation_func
        self.device = device
        self.target_model.to(self.device)
        self.target_model.eval()
        self.scaler = scaler
        self.scaled_feature_names = scaled_feature_names
        self.all_feature_names = all_feature_names
        
        # Ensure that features_allowed_for_perturbation strictly excludes categorical features
        # provided by the user, and respects the numerical_feature_names_for_perturbation from Data_Handler
        if categorical_feature_names is None:
            categorical_feature_names = []
            
        self.numerical_feature_names_for_perturbation = [
            f for f in numerical_feature_names_for_perturbation 
            if f not in categorical_feature_names
        ]
        self.categorical_feature_names = categorical_feature_names # Store for reference

        self.attack_results = []

        # Pre-compute indices for numerical features allowed for perturbation
        self.numerical_indices_for_perturbation = [
            self.all_feature_names.index(name) for name in self.numerical_feature_names_for_perturbation 
            if name in self.all_feature_names
        ]
        print(f"Features allowed for perturbation: {[self.all_feature_names[idx] for idx in self.numerical_indices_for_perturbation]}")
        print(f"Categorical features (will not be perturbed): {self.categorical_feature_names}")


    def _collect_data_and_explanations(self, dataloader: DataLoader, num_samples_to_collect: int = 100):
        """
        Collects input data, their explanations, and predictions from the dataloader.
        """
        print(f"Collecting {num_samples_to_collect} samples and explanations...")
        start_time = time.time()
        collected_data, collected_explanations, collected_predictions = [], [], []

        current_collected = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                if current_collected >= num_samples_to_collect:
                    break

                inputs = inputs.to(self.device)
                logits = self.target_model(inputs)
                predictions = torch.argmax(logits, dim=1)

                for j in range(inputs.shape[0]):
                    if current_collected >= num_samples_to_collect:
                        break

                    single_input = inputs[j].unsqueeze(0)
                    predicted_class = predictions[j].item()

                    explanation = self.generate_explanation_func(
                        self.target_model, single_input, predicted_class
                    )

                    collected_data.append(single_input.cpu().squeeze(0))
                    collected_explanations.append(explanation.cpu())
                    collected_predictions.append(predicted_class)
                    current_collected += 1

        end_time = time.time()
        print(f"Collected {current_collected} samples and explanations in {end_time - start_time:.2f} seconds.")
        return (
            torch.stack(collected_data).numpy(),
            torch.stack(collected_explanations).numpy(),
            np.array(collected_predictions)
        )

    def _approximate_manifolds(self, collected_data: np.ndarray, collected_explanations: np.ndarray, n_components: int = 5):
        """
        Approximates data and explanation manifolds using UMAP.
        """
        print("Approximating manifolds with UMAP...")
        start_time = time.time()
        n_data_samples = collected_data.shape[0]
        n_explanation_samples = collected_explanations.shape[0]

        # UMAP for data manifold
        n_neighbors_data = min(15, n_data_samples - 1) if n_data_samples > 1 else 1
        self.data_manifold_model = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors_data,
            random_state=42
        ).fit(collected_data)

        # UMAP for explanation manifold
        n_neighbors_explanation = min(15, n_explanation_samples - 1) if n_explanation_samples > 1 else 1
        self.explanation_manifold_model = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors_explanation,
            random_state=42
        ).fit(collected_explanations)

        data_embedded = self.data_manifold_model.transform(collected_data)
        explanations_embedded = self.explanation_manifold_model.transform(collected_explanations)
        end_time = time.time()
        print(f"Manifolds approximated in {end_time - start_time:.2f} seconds.")
        return data_embedded, explanations_embedded

    def _project_to_manifold(self, x_np: np.ndarray) -> np.ndarray:
        """Projects perturbed sample back to the data manifold."""
        x_embed = self.data_manifold_model.transform(x_np.reshape(1, -1))
        return self.data_manifold_model.inverse_transform(x_embed).flatten()

    def _objective_function_i_attack(self, delta_numerical_flat: np.ndarray, original_input_tensor_cpu: torch.Tensor,
                                   target_explanation_flat: np.ndarray, original_class: int,
                                   input_shape: tuple, C: float) -> float:
        """
        The objective function for the interpretation-aware attack.
        Minimizes KL divergence between adversarial and target explanations,
        L2 perturbation norm, and adds a penalty if the class changes.
        """
        device = self.device

        # Start with a clone of the original input to ensure non-numerical features are preserved
        x_adv_candidate = original_input_tensor_cpu.clone().to(device)
        
        # Reshape delta for numerical features
        delta_numerical = torch.tensor(delta_numerical_flat, dtype=torch.float32).to(device)
        
        # Apply delta ONLY to numerical features allowed for perturbation
        for i, idx in enumerate(self.numerical_indices_for_perturbation):
            x_adv_candidate[idx] += delta_numerical[i]

        # Project the (partially perturbed) sample to the manifold.
        # This step might cause slight changes to non-numerical features if UMAP
        # was trained on the full feature space.
        x_adv_proj_np = self._project_to_manifold(x_adv_candidate.detach().cpu().numpy())
        x_adv = torch.tensor(x_adv_proj_np, dtype=torch.float32).to(device)

        # IMPORTANT: Re-impose original values for non-numerical features AFTER manifold projection.
        # This explicitly ensures that categorical features and other non-numerical features
        # (like the protected feature if it's not allowed for perturbation) are not changed by the attack.
        for idx in range(input_shape[0]):
            if idx not in self.numerical_indices_for_perturbation:
                x_adv[idx] = original_input_tensor_cpu[idx].to(device)

        x_adv_explanation = self.generate_explanation_func(
            self.target_model, x_adv.unsqueeze(0), original_class
        )

        epsilon = 1e-10
        # Add epsilon to avoid log(0) and handle zero attributions
        p = torch.softmax(x_adv_explanation.abs().flatten(), dim=0) + epsilon
        q = torch.softmax(torch.tensor(target_explanation_flat, dtype=torch.float32).flatten(), dim=0) + epsilon
        
        kl_div = torch.sum(p * torch.log(p / q)).item()

        with torch.no_grad():
            x_adv_pred = torch.argmax(self.target_model(x_adv.unsqueeze(0)), dim=1).item()
        class_penalty = 5000.0 if x_adv_pred != original_class else 0.0

        # Perturbation norm is calculated only on the *actual* delta applied to numerical features
        perturbation_norm = torch.norm(delta_numerical, p=2).item()

        return kl_div + C * perturbation_norm + class_penalty

    def attack_sample(self, original_input: torch.Tensor, original_class: int,
                      target_explanation: torch.Tensor, C: float = 0.1, max_iter: int = 100) -> tuple:
        """
        Performs an interpretation-aware attack on a single sample.
        """
        original_input_cpu = original_input.cpu().squeeze(0)
        input_shape = original_input_cpu.shape

        # Initial delta for numerical features only
        initial_delta_numerical = np.zeros(len(self.numerical_indices_for_perturbation))
        
        # Bounds for numerical features only
        bounds_numerical = [(-0.1, 0.1)] * len(self.numerical_indices_for_perturbation)

        result = minimize(
            self._objective_function_i_attack,
            initial_delta_numerical, # Only perturb numerical features
            args=(original_input_cpu, target_explanation.cpu().numpy().flatten(),
                  original_class, input_shape, C),
            method='L-BFGS-B', # A good choice for constrained, gradient-based optimization
            bounds=bounds_numerical, # Bounds only for numerical features
            options={'maxiter': max_iter, 'ftol': 1e-1, 'maxcor': 50, 'disp': False} # Increased max_iter
        )
        
        # Reconstruct the full adversarial example from the optimized numerical delta
        x_adv_final_candidate = original_input_cpu.clone().to(self.device)
        delta_numerical_optimized = torch.tensor(result.x, dtype=torch.float32).to(self.device)
        
        # Apply optimized delta to numerical features
        for i, idx in enumerate(self.numerical_indices_for_perturbation):
            x_adv_final_candidate[idx] += delta_numerical_optimized[i]
        
        # Project the (partially perturbed) sample to the manifold
        x_adv_proj_np = self._project_to_manifold(x_adv_final_candidate.detach().cpu().numpy())
        x_adv = torch.tensor(x_adv_proj_np, dtype=torch.float32).to(self.device)

        # IMPORTANT: Re-impose original values for non-numerical features AFTER manifold projection.
        # This is crucial to ensure that only designated numerical features are perturbed.
        for idx in range(input_shape[0]):
            if idx not in self.numerical_indices_for_perturbation:
                x_adv[idx] = original_input_cpu[idx].to(self.device)

        with torch.no_grad():
            adv_pred = torch.argmax(self.target_model(x_adv.unsqueeze(0)), dim=1).item()
            adv_explanation = self.generate_explanation_func(self.target_model, x_adv.unsqueeze(0), original_class)

        p_adv = torch.softmax(adv_explanation.abs().flatten(), dim=0) + 1e-10
        q_target = torch.softmax(target_explanation.abs().flatten(), dim=0) + 1e-10
        final_kl = torch.sum(p_adv * torch.log(p_adv / q_target)).item()

        success = (adv_pred == original_class) and (final_kl < 0.05)

        return x_adv, {
            'success': success,
            'original_class': original_class,
            'adv_predicted_class': adv_pred,
            'final_kl_div': final_kl,
            'perturbation_norm': torch.norm(delta_numerical_optimized, p=2).item()
        }

    def run_attack(self, dataloader: DataLoader, num_samples_to_attack: int = 10, C: float = 0.1) -> tuple:
        """
        Runs the interpretation-aware attack over a specified number of samples from the dataloader.
        """
        print("\nStarting attack...")
        total_attack_start_time = time.time()
        
        data_np, explanations_np, predictions = self._collect_data_and_explanations(dataloader)

        if len(data_np) < 2:
            print("Not enough samples collected for manifold approximation. Need at least 2.")
            self.attack_results = []
            return [], 0.0

        self._approximate_manifolds(data_np, explanations_np)

        self.attack_results = [] # Reset for a new run
        num_attacks_processed = 0
        successful_attacks_count = 0

        for i, (inputs, labels) in enumerate(dataloader):
            if num_attacks_processed >= num_samples_to_attack:
                break

            inputs = inputs.to(self.device)
            with torch.no_grad():
                preds = torch.argmax(self.target_model(inputs), dim=1)

            for j in range(inputs.shape[0]):
                if num_attacks_processed >= num_samples_to_attack:
                    break

                original_input = inputs[j].unsqueeze(0).cpu()
                original_class = preds[j].item()

                original_explanation = self.generate_explanation_func(
                    self.target_model, inputs[j].unsqueeze(0), original_class
                )

                same_class_mask = predictions == original_class
                if np.sum(same_class_mask) < 2:
                    # Not enough samples of the same class for diverse target explanation
                    continue

                try:
                    target_explanation = self._select_diverse_target(
                        original_explanation, explanations_np[same_class_mask]
                    )
                except ValueError as e:
                    print(f"Skipping sample {num_attacks_processed} due to UMAP error during target selection: {e}")
                    continue
                
                attack_sample_start_time = time.time()
                x_adv, result = self.attack_sample(original_input, original_class, target_explanation, C=C)
                attack_sample_end_time = time.time()
                print(f"  Attack for sample {num_attacks_processed+1}/{num_samples_to_attack} took {attack_sample_end_time - attack_sample_start_time:.2f}s, Success: {result['success']}")

                if result['success']:
                    self.attack_results.append((original_input, x_adv, target_explanation, result))
                    successful_attacks_count += 1

                num_attacks_processed += 1

        asr = successful_attacks_count / num_attacks_processed if num_attacks_processed > 0 else 0.0

        total_attack_end_time = time.time()
        print(f"Total attack duration for {num_attacks_processed} samples: {total_attack_end_time - total_attack_start_time:.2f} seconds.")
        print(f"Stored {len(self.attack_results)} successful attack results.")
        return self.attack_results, asr

    def _select_diverse_target(self, original_explanation: torch.Tensor, candidate_explanations_np: np.ndarray) -> torch.Tensor:
        """
        Selects a diverse target explanation from a set of candidates using UMAP embeddings.
        """
        if len(candidate_explanations_np) < 1:
            raise ValueError("Not enough candidate explanations to select a diverse target.")
        
        original_embed = self.explanation_manifold_model.transform(
            original_explanation.cpu().numpy().reshape(1, -1)
        )
        candidate_embeds = self.explanation_manifold_model.transform(candidate_explanations_np)

        if len(candidate_embeds) == 0:
            raise ValueError("No candidate embeddings available for diverse target selection.")
        
        distances = cdist(original_embed, candidate_embeds).flatten()
        target_idx = np.argmax(distances) # Select the candidate farthest from the original
        return torch.tensor(candidate_explanations_np[target_idx], dtype=torch.float32).to(self.device)

    def plot_feature_importance_comparison(self, original_input_tensor: torch.Tensor, x_adv_tensor: torch.Tensor,
                                           target_class_index: int, target_explanation_tensor: torch.Tensor = None,
                                           include_target_explanation: bool = False):
        """
        Plots a comparison bar chart of feature importances for an original input,
        its adversarial counterpart, and optionally the target explanation.
        """
        if original_input_tensor.dim() == 1:
            original_input_tensor = original_input_tensor.unsqueeze(0)
        if x_adv_tensor.dim() == 1:
            x_adv_tensor = x_adv_tensor.unsqueeze(0)

        device = next(self.target_model.parameters()).device
        original_input_tensor = original_input_tensor.to(device)
        x_adv_tensor = x_adv_tensor.to(device)

        original_explanation = self.generate_explanation_func(
            self.target_model, original_input_tensor, target_class_index
        ).cpu().numpy()
        adv_explanation = self.generate_explanation_func(
            self.target_model, x_adv_tensor, target_class_index
        ).cpu().numpy()

        num_features = original_explanation.shape[-1]
        feature_labels = self.all_feature_names if self.all_feature_names else [f'Feature {i+1}' for i in range(num_features)]
        y_positions = np.arange(num_features)

        feature_labels = [i[:min(len(i),15)] for i in feature_labels]

        if include_target_explanation:
            if target_explanation_tensor is None:
                raise ValueError("target_explanation_tensor must be provided if include_target_explanation is True.")
            target_explanation = target_explanation_tensor.cpu().numpy()
            
            fig, ax = plt.subplots(figsize=(9, 6))
            bar_height = 0.25
            
            ax.barh(y_positions + bar_height, original_explanation, height=bar_height, label='Original Importance', align='center')
            ax.barh(y_positions, adv_explanation, height=bar_height, label='Adversarial Importance', align='center')
            ax.barh(y_positions - bar_height, target_explanation, height=bar_height, label='Target Importance', align='center')
            
            ax.set_xlabel('Feature Importance')
            ax.set_yticks(y_positions)
            ax.set_yticklabels(feature_labels)
            ax.invert_yaxis()
            ax.legend(loc='best')
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            bar_height = 0.35

            ax.barh(y_positions + bar_height/2, original_explanation, height=bar_height, label='Original Importance', align='center')
            ax.barh(y_positions - bar_height/2, adv_explanation, height=bar_height, label='Adversarial Importance', align='center')

            ax.set_xlabel('Feature Importance')
            ax.set_yticks(y_positions)
            ax.set_yticklabels(feature_labels)
            ax.invert_yaxis()
            ax.legend(loc='best')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        plt.show()


        

    def plot_feature_value_difference(self, original_input_tensor: torch.Tensor, x_adv_tensor: torch.Tensor):
        """
        Plots the difference in feature values between an original input and its
        adversarial counterpart, using a StandardScaler to inverse transform values
        for better interpretability if the data was scaled.
        """
        original_input_np = original_input_tensor.cpu().numpy().flatten()
        x_adv_np = x_adv_tensor.cpu().numpy().flatten()

        if self.scaler is None or not self.scaled_feature_names:
            print("Scaler not provided or no features were identified as scaled. Plotting raw difference.")
            difference = x_adv_np - original_input_np
        else:
            # Create pandas Series for easy indexing by feature names
            original_series = pd.Series(original_input_np, index=self.all_feature_names)
            adv_series = pd.Series(x_adv_np, index=self.all_feature_names)

            # Extract only the scaled numerical features for inverse transformation
            original_numerical_scaled = original_series[self.scaled_feature_names].values.reshape(1, -1)
            adv_numerical_scaled = adv_series[self.scaled_feature_names].values.reshape(1, -1)

            # Inverse transform only the scaled numerical features
            original_numerical_unscaled = self.scaler.inverse_transform(original_numerical_scaled).flatten()
            adv_numerical_unscaled = self.scaler.inverse_transform(adv_numerical_scaled).flatten()

            # Create full unscaled Series copies
            original_unscaled_full = original_series.copy()
            adv_unscaled_full = adv_series.copy()

            # Place inverse transformed values back into the full series for scaled features
            original_unscaled_full[self.scaled_feature_names] = original_numerical_unscaled
            adv_unscaled_full[self.scaled_feature_names] = adv_numerical_unscaled
            
            # For features that were not scaled, their "unscaled" value is just their original (scaled) value.
            # No explicit action needed as they retain their original values in original_unscaled_full/adv_unscaled_full
            
            difference = adv_unscaled_full.values - original_unscaled_full.values


        num_features = len(difference)
        feature_labels = self.all_feature_names if self.all_feature_names else [f'Feature {i+1}' for i in range(num_features)]
        y_positions = np.arange(num_features)
        feature_labels = [i[:min(len(i),15)] for i in feature_labels]

        fig, ax = plt.subplots(figsize=(6, 4))
        
        colors = ['skyblue' if d >= 0 else 'salmon' for d in difference]
        ax.barh(y_positions, difference, color=colors, align='center')

        ax.set_xlabel('Feature Value Difference (Adversarial - Original)')
        ax.set_yticks(y_positions)
        ax.set_yticklabels(feature_labels)
        ax.invert_yaxis()
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        fig.tight_layout()
        plt.show()


    

    def plot_feature_value_comparison(self, original_input_tensor: torch.Tensor, x_adv_tensor: torch.Tensor):
        """
        Plots the actual feature values for an original input and its adversarial counterpart,
        inverse transforming values to the original scale if a scaler was provided.
        The plot orientation is changed to have features on the X-axis and values on the Y-axis.
        """
        original_input_np = original_input_tensor.cpu().numpy().flatten()
        x_adv_np = x_adv_tensor.cpu().numpy().flatten()

        # Determine if inverse transformation is possible/necessary
        if self.scaler is None or not self.scaled_feature_names:
            print("Scaler not provided or no features were identified as scaled. Plotting raw (scaled) feature values.")
            original_values_for_plot = original_input_np
            adv_values_for_plot = x_adv_np
        else:
            # Create pandas Series for easy indexing by feature names
            original_series = pd.Series(original_input_np, index=self.all_feature_names)
            adv_series = pd.Series(x_adv_np, index=self.all_feature_names)

            # Extract only the scaled numerical features for inverse transformation
            original_numerical_scaled = original_series[self.scaled_feature_names].values.reshape(1, -1)
            adv_numerical_scaled = adv_series[self.scaled_feature_names].values.reshape(1, -1)

            # Inverse transform only the scaled numerical features
            original_numerical_unscaled = self.scaler.inverse_transform(original_numerical_scaled).flatten()
            adv_numerical_unscaled = self.scaler.inverse_transform(adv_numerical_scaled).flatten()

            # Create full unscaled Series copies
            original_unscaled_full = original_series.copy()
            adv_unscaled_full = adv_series.copy()

            # Place inverse transformed values back into the full series for scaled features
            original_unscaled_full[self.scaled_feature_names] = original_numerical_unscaled
            adv_unscaled_full[self.scaled_feature_names] = adv_numerical_unscaled

            original_values_for_plot = original_unscaled_full.values
            adv_values_for_plot = adv_unscaled_full.values

        num_features = len(original_values_for_plot)
        feature_labels = self.all_feature_names if self.all_feature_names else [f'Feature {i+1}' for i in range(num_features)]
        feature_labels = [i[:min(len(i),15)] + '...' if len(i) > 15 else i for i in feature_labels] # Truncate labels and add '...'
        
        # Create a DataFrame for easier plotting
        plot_df = pd.DataFrame({
            'Feature': feature_labels,
            'Original': original_values_for_plot,
            'Adversarial': adv_values_for_plot
        })

        # Dynamically set figure width based on number of features for better readability
        fig_width = max(8, num_features * 0.7)
        fig, ax = plt.subplots(figsize=(fig_width, 6))
        
        x_positions = np.arange(num_features)
        point_offset = 0.15 # Small offset to separate points for readability

        # Plotting scatter points for actual values
        ax.scatter(x_positions - point_offset, plot_df['Original'],
                   marker='o', s=100, color='skyblue', label='Original Value', zorder=2)
        ax.scatter(x_positions + point_offset, plot_df['Adversarial'],
                   marker='x', s=100, color='lightcoral', label='Adversarial Value', zorder=2)

        # Add lines connecting the original and adversarial points for each feature
        for i in range(num_features):
            ax.plot([x_positions[i] - point_offset, x_positions[i] + point_offset],
                    [plot_df['Original'].iloc[i], plot_df['Adversarial'].iloc[i]],
                    color='gray', linestyle='--', linewidth=0.8, zorder=1)

            # Add text labels for values
            ax.text(x_positions[i] - point_offset, plot_df['Original'].iloc[i],
                    f'{plot_df["Original"].iloc[i]:.2f}',
                    va='bottom', ha='center', fontsize=8, color='black') # Adjusted va/ha for vertical plot

            ax.text(x_positions[i] + point_offset, plot_df['Adversarial'].iloc[i],
                    f'{plot_df["Adversarial"].iloc[i]:.2f}',
                    va='bottom', ha='center', fontsize=8, color='black') # Adjusted va/ha for vertical plot


        ax.set_xlabel('Features')
        ax.set_ylabel('Feature Value')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(plot_df['Feature'], rotation=45, ha='right') # Rotate labels for readability
        ax.legend(loc='best')

        # Remove border lines (spines)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['left'].set_visible(False) # Keep left spine for Y-axis
        # ax.spines['bottom'].set_visible(False) # Keep bottom spine for X-axis

        fig.tight_layout()
        plt.show()







    def plot_attack_result(self, attack_index: int, include_target_explanation_plot: bool = False):
        """
        Plots the feature importance comparison and feature value difference for a specific
        successful attack identified by its index.
        """
        if not (0 <= attack_index < len(self.attack_results)):
            print(f"Error: Attack index {attack_index} is out of bounds. There are {len(self.attack_results)} stored successful results.")
            return

        original, x_adv, target_exp, result = self.attack_results[attack_index]

        print(f"\nPlotting for Stored Successful Attack at Index {attack_index}:")
        print(f"Original Class: {result['original_class']}")
        print(f"Adv Class: {result['adv_predicted_class']}")
        print(f"KL Divergence: {result['final_kl_div']:.4f}")
        print(f"Perturbation Norm: {result['perturbation_norm']:.4f}")

        self.plot_feature_importance_comparison(original, x_adv, result['original_class'],
                                                target_explanation_tensor=target_exp,
                                                include_target_explanation=include_target_explanation_plot)
        # self.plot_feature_value_difference(original, x_adv)
        self.plot_feature_value_comparison(original, x_adv)

