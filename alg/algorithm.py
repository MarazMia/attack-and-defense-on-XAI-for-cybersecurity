import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr


class Algorithm:
    def __init__(
            self,
            explainer,
            constant=None,
            row_id=None,
            col_id=None
        ):

        self.explainer = explainer
        self._X = explainer.data.values
        self._n, self._p = self._X.shape

        self._original_row_id = row_id

        if isinstance(row_id, int):
            self._x = explainer.data.values[row_id, :]
        else:
            if row_id is not None:
                warnings.warn("`row_id` is " + str(row_id) + " and should be an integer. Using `row_id=None`.")
            self._x = None
            
        if isinstance(col_id, int):
            self.col_id = [col_id]
        elif isinstance(col_id, list):
            self.col_id = col_id
        else:
            self.col_id = list(range(self._p))

        if constant is not None:
            self._idc = []
            for const in constant:
                self._idc.append(explainer.data.columns.get_loc(const))
        else:
            self._idc = None

        self.result_explanation = {'original': None, 'changed': None}
        self.result_data = None

        self.iter_losses = {'iter':[], 'loss':[], 'distance_importance':[], 'distance_ranking':[]}

    def fool(self, random_state=None):

        if random_state is not None:
            np.random.seed(random_state)

        self.result_explanation['original'] = self.explainer.shap_values(self._X)
        self.result_explanation['changed'] = np.zeros_like(self.result_explanation['original'])


    def fool_aim(self, target="auto", random_state=None):

        Algorithm.fool(self=self, random_state=random_state)
        
        if isinstance(target, np.ndarray):
            self.result_explanation['target'] = target
        else: # target="auto"
            self.result_explanation['target'] = np.repeat(
                self.result_explanation['original'].mean(),
                self.result_explanation['original'].shape[0]
            ) - self.result_explanation['original'] * 0.001


    #:# plots 
        
    # def plot_data(self, i=0, constant=True, height=2, savefig=None):
    #     plt.rcParams["legend.handlelength"] = 0.1
    #     _colors = sns.color_palette("Set1").as_hex()[0:2][::-1]
    #     if i == 0:
    #         _df = self.result_data
    #     else:
    #         _data_changed = pd.DataFrame(self.get_best_data(i), columns=self.explainer.data.columns)
    #         _df = pd.concat((self.explainer.data, _data_changed))\
    #                 .reset_index(drop=True)\
    #                 .rename(index={'0': 'original', '1': 'changed'})\
    #                 .assign(dataset=pd.Series(['original', 'changed'])\
    #                                 .repeat(self._n).reset_index(drop=True))
    #     if not constant and self._idc is not None:
    #         _df = _df.drop(_df.columns[self._idc], axis=1)
    #     ax = sns.pairplot(_df, hue='dataset', height=height, palette=_colors)
    #     ax._legend.set_bbox_to_anchor((0.62, 0.64))
    #     if savefig:
    #         ax.savefig(savefig, bbox_inches='tight')
    #     plt.show()


    def plot_data(self, i=0, constant=True, height=4, savefig=None, 
              scaler=None, numerical_features_scaled: list = None, all_feature_names: list = None):
        """
        Plots the comparison of original and perturbed feature values for a single sample.

        Args:
            self: The instance of the Algorithm class.
            i (int): Index to retrieve perturbed data if get_best_data returns a collection.
            constant (bool): If True, plots all features. If False, removes 'constant' features (_idc).
            height (int): Height of the matplotlib figure.
            savefig (str): Path to save the figure.
            scaler (StandardScaler): The fitted StandardScaler instance, or None if no scaling was done.
            numerical_features_scaled (list): List of feature names that were scaled by `scaler`.
            all_feature_names (list): The complete ordered list of feature names present in the data.
        """

        if not isinstance(scaler, StandardScaler) and scaler is not None:
            # If scaler is not None but not a StandardScaler, it's an error.
            print("Error: If provided, 'scaler' must be a fitted StandardScaler instance.")
            return
    


        if self._x is None:
            print("Cannot plot individual sample data: original sample (self._x) is not set. Ensure `row_id` was provided to Algorithm constructor.")
            return

        perturbed_full_dataset = self.get_best_data(i)

        if perturbed_full_dataset is None:
            print("Cannot plot individual sample data: perturbed data not available from get_best_data().")
            return

        if perturbed_full_dataset.ndim == 1: # If get_best_data returns a single 1D array
            perturbed_values_scaled = perturbed_full_dataset
        elif self._original_row_id is not None and self._original_row_id < perturbed_full_dataset.shape[0]:
            perturbed_values_scaled = perturbed_full_dataset[self._original_row_id]
        else:
            print(f"Error: Stored `row_id` ({self._original_row_id}) is None or out of bounds "
                f"for extracting a specific perturbed sample from a dataset of size {perturbed_full_dataset.shape[0]}.")
            return

        original_values_scaled = np.array(self._x).flatten() # This is the full scaled array

        if len(original_values_scaled) != len(perturbed_values_scaled):
            print("Logical Error: Original sample and extracted perturbed sample have different numbers of features.")
            print(f"Original features: {len(original_values_scaled)}, Perturbed features: {len(perturbed_values_scaled)}")
            return

        # Initialize unscaled data with the scaled values
        original_values_unscaled_full = original_values_scaled.copy()
        perturbed_values_unscaled_full = perturbed_values_scaled.copy()

        # Inverse transform if a scaler is provided
        if scaler is not None:
            # Here we assume if a scaler is provided, *all* features that were
            # subject to scaling are present in `original_values_scaled` and `perturbed_values_scaled`.
            # We also assume that `all_feature_names` corresponds to the order of features
            # that the scaler was fitted on, if it was fitted on the entire dataset.

            # Perform inverse transform on the entire array
            original_values_unscaled_full = scaler.inverse_transform(original_values_scaled.reshape(1, -1)).flatten()
            perturbed_values_unscaled_full = scaler.inverse_transform(perturbed_values_scaled.reshape(1, -1)).flatten()


        plot_df = pd.DataFrame({
            'Feature': all_feature_names, # Use all_feature_names
            'Original': original_values_unscaled_full,
            'Perturbed': perturbed_values_unscaled_full
        })

        if not constant and self._idc is not None:
            # Ensure _idc contains valid indices for all_feature_names
            cols_to_drop = [all_feature_names[idx] for idx in self._idc if idx < len(all_feature_names)]
            plot_df = plot_df.drop(columns=[col for col in cols_to_drop if col in plot_df.columns])

        plt.figure(figsize=(6, height))
        ax = plt.gca()

        num_features = len(plot_df['Feature'])
        x = np.arange(num_features)

        # Plot points instead of bars
        point_offset = 0.15 # Small offset to separate original and perturbed points visually
        ax.scatter(x - point_offset, plot_df['Original'], marker='o', s=100, color='skyblue', label='Original Data', zorder=2)
        ax.scatter(x + point_offset, plot_df['Perturbed'], marker='x', s=100, color='lightcoral', label='Perturbed Data', zorder=2)

        # Optionally draw lines connecting the points for a "dumbbell" effect
        for idx in range(num_features):
            ax.plot([x[idx] - point_offset, x[idx] + point_offset],
                    [plot_df['Original'].iloc[idx], plot_df['Perturbed'].iloc[idx]],
                    color='gray', linestyle='--', linewidth=0.8, zorder=1)


        ax.set_xlabel('Features')
        ax.set_ylabel('Feature Value')
        # ax.set_title(f'Comparison of Original and Perturbed Feature Values (Sample ID: {self._original_row_id if self._original_row_id is not None else "N/A"})')
        ax.set_xticks(x)

        # Truncate feature names
        truncated_feature_names = [name[:15] + '...' if len(name) > 15 else name for name in plot_df['Feature']]
        ax.set_xticklabels(truncated_feature_names, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()

        if savefig:
            plt.savefig(savefig, bbox_inches='tight')
        plt.show()






    def plot_losses(self, lw=3, figsize=(9, 6), savefig=None):
        plt.rcParams["figure.figsize"] = figsize
        plt.plot(
            self.iter_losses['iter'], 
            self.iter_losses['loss'], 
            color='#000000', 
            lw=lw
        )
        plt.title('Learning curve', fontsize=20)
        plt.xlabel('epoch', fontsize=16)
        plt.ylabel('loss', fontsize=16)
        if savefig:
            plt.savefig(savefig)
        plt.show()

    def plot_explanation(self):
        required_keys = ['original', 'changed', 'target']
        for key in required_keys:
            if self.result_explanation.get(key) is None:
                warnings.warn(f"Cannot plot explanation: '{key}' data is missing.")
                return
            if self.result_explanation[key].ndim > 1:
                warnings.warn(f"Explanation data for '{key}' is not 1D. Please ensure it represents per-feature values for a single sample.")
                return

        temp = pd.DataFrame(self.result_explanation)
        x = np.arange(len(self.explainer.data.columns))
        width = 0.2
        fig, ax = plt.subplots(figsize=(7, 4)) # Added figsize for better readability with many features
        ax.bar(x - width, temp["original"], width, label='original', color="#4378bf")
        ax.bar(x, temp['changed'], width, label='changed', color="#f05a71")
        ax.bar(x + width, temp['target'], width, label='target', color="#bebebe")
        ax.set_xticks(x)
        ax.legend()
        ax.set_ylabel('Feature Importance')
        ax.set_xlabel('Features')

        # --- Shorten x-axis labels here ---
        # Get original feature names
        original_feature_names = self.explainer.data.columns.tolist()
        # Truncate feature names to max 15 characters
        truncated_feature_names = [name[:15] + '...' if len(name) > 15 else name for name in original_feature_names]
        ax.set_xticklabels(truncated_feature_names, rotation=45, ha='right')
        # --- End of shortening x-axis labels ---

        fig.tight_layout()
        plt.show()



    def display_feature_rank_changes(self):
        """
        Displays changes in feature ranking between 'original', 'changed', and 'target'
        explanation results in a tabular format.
        """
        required_keys = ['original', 'changed', 'target']
        for key in required_keys:
            if self.result_explanation.get(key) is None:
                warnings.warn(f"Cannot display rank changes: '{key}' data is missing.")
                return
            if self.result_explanation[key].ndim > 1:
                warnings.warn(f"Explanation data for '{key}' is not 1D. Please ensure it represents per-feature values for a single sample.")
                return

        feature_names = self.explainer.data.columns.tolist()

        # Create DataFrames for each explanation type with importance and rank
        original_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.result_explanation['original']
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        original_df['Rank_Original'] = original_df.index + 1

        changed_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.result_explanation['changed']
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        changed_df['Rank_Changed'] = changed_df.index + 1

        target_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.result_explanation['target']
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        target_df['Rank_Target'] = target_df.index + 1

        # Merge the ranks for comparison
        comparison_df = original_df[['Feature', 'Rank_Original']].merge(
            changed_df[['Feature', 'Rank_Changed']], on='Feature', how='left'
        ).merge(
            target_df[['Feature', 'Rank_Target']], on='Feature', how='left'
        )

        # Identify features where rank has changed from original to changed
        rank_changes_original_to_changed = comparison_df[
            comparison_df['Rank_Original'] != comparison_df['Rank_Changed']
        ].copy() # Use .copy() to avoid SettingWithCopyWarning

        # Identify features where rank has changed from original to target
        rank_changes_original_to_target = comparison_df[
            comparison_df['Rank_Original'] != comparison_df['Rank_Target']
        ].copy()

        # Identify features where rank has changed from changed to target
        rank_changes_changed_to_target = comparison_df[
            comparison_df['Rank_Changed'] != comparison_df['Rank_Target']
        ].copy()

        print("\n--- Feature Ranking Changes ---")

        if not rank_changes_original_to_changed.empty:
            print("\nRanking Changes: Original vs. Changed Explanation")
            print("--------------------------------------------------")
            for index, row in rank_changes_original_to_changed.iterrows():
                print(f"Feature '{row['Feature']}': Rank changed from {int(row['Rank_Original'])} "
                      f"to {int(row['Rank_Changed'])}")
            print("\nTable Summary (Original vs. Changed):")
            print(rank_changes_original_to_changed[['Feature', 'Rank_Original', 'Rank_Changed']].to_string(index=False))
        else:
            print("\nNo ranking changes observed between Original and Changed explanations.")

        if not rank_changes_original_to_target.empty:
            print("\n\nRanking Changes: Original vs. Target Explanation")
            print("--------------------------------------------------")
            for index, row in rank_changes_original_to_target.iterrows():
                print(f"Feature '{row['Feature']}': Rank changed from {int(row['Rank_Original'])} "
                      f"to {int(row['Rank_Target'])}")
            print("\nTable Summary (Original vs. Target):")
            print(rank_changes_original_to_target[['Feature', 'Rank_Original', 'Rank_Target']].to_string(index=False))
        else:
            print("\nNo ranking changes observed between Original and Target explanations.")

        if not rank_changes_changed_to_target.empty:
            print("\n\nRanking Changes: Changed vs. Target Explanation")
            print("--------------------------------------------------")
            for index, row in rank_changes_changed_to_target.iterrows():
                print(f"Feature '{row['Feature']}': Rank changed from {int(row['Rank_Changed'])} "
                      f"to {int(row['Rank_Target'])}")
            print("\nTable Summary (Changed vs. Target):")
            print(rank_changes_changed_to_target[['Feature', 'Rank_Changed', 'Rank_Target']].to_string(index=False))
        else:
            print("\nNo ranking changes observed between Changed and Target explanations.")

        print("\n--- All Feature Ranks (Sorted by Original Rank) ---")
        print(comparison_df.sort_values(by='Rank_Original').to_string(index=False))

        # Calculate Spearman's rank correlation between Rank_Original and Rank_Changed
        spearman_original_changed, _ = spearmanr(comparison_df['Rank_Original'], comparison_df['Rank_Changed'])

        print(f"Spearman's correlation between Rank_Original and Rank_Changed: {spearman_original_changed:.4f}")