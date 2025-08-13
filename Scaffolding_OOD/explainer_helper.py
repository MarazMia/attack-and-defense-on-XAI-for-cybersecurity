# explainer_helper.py

import lime
import lime.lime_tabular
import shap
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging
import re

# Suppress warnings from LIME/SHAP for cleaner output
warnings.filterwarnings("ignore")

# --- Global Logging Settings (to turn off SHAP's general INFO logs) ---
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('shap').setLevel(logging.WARNING)
# --- End Global Logging Settings ---


def get_lime_explanation(
    model,
    instance: np.ndarray,
    feature_names: list,
    training_data: np.ndarray,
    class_names: list = ['Class 0', 'Class 1'],
    num_features: int = None,
    categorical_feature_indices: list = None
) -> list:
    """
    Generates a LIME explanation for a given instance.
    """
    if num_features is None:
        num_features = len(feature_names)

    training_data_cleaned = np.ascontiguousarray(training_data, dtype=np.float64)
    instance_cleaned = np.ascontiguousarray(instance.reshape(1, -1), dtype=np.float64)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=training_data_cleaned,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        categorical_features=categorical_feature_indices,
        feature_selection='none',
        discretize_continuous=False
    )

    explanation = explainer.explain_instance(
        data_row=instance_cleaned[0],
        predict_fn=model.predict_proba,
        num_features=num_features
    )
    return explanation.as_list()

def get_shap_explanation(
    model,
    instance: np.ndarray,
    feature_names: list,
    background_data: np.ndarray,
    class_names: list = ['Class 0', 'Class 1'],
    explainer_type: str = 'kernel'
) -> np.ndarray:
    """
    Generates SHAP explanations for a given instance using various SHAP explainers.
    """
    background_data_cleaned = np.ascontiguousarray(background_data, dtype=np.float32)
    predict_fn = lambda x: model.predict_proba(x)

    if explainer_type == 'kernel':
        explainer = shap.KernelExplainer(predict_fn, background_data_cleaned)
        shap_values = explainer.shap_values(instance, silent=True)
    elif explainer_type == 'linear':
        explainer = shap.LinearExplainer(model, background_data_cleaned)
        shap_values = explainer.shap_values(instance)
    elif explainer_type == 'tree':
        explainer = shap.TreeExplainer(model, background_data_cleaned)
        shap_values = explainer.shap_values(instance)
    elif explainer_type == 'deep':
        explainer = shap.DeepExplainer(model, background_data_cleaned)
        shap_values = explainer.shap_values(instance)
    else:
        raise ValueError(f"Unsupported SHAP explainer type: {explainer_type}")

    return shap_values

def _process_explanation_data_for_plot(explanation_data, feature_names: list, predicted_class: int = None) -> dict:
    """
    Helper to convert LIME/SHAP output to a {feature: importance} dictionary.
    CLEANS LIME FEATURE NAMES to remove value ranges and ensures mapping to original feature_names.
    """
    feature_importance = {}

    if isinstance(explanation_data, list) and all(isinstance(item, tuple) and len(item) == 2 for item in explanation_data):
        # This is LIME output
        for feature_str, weight in explanation_data:
            # FIX: More robust cleaning for LIME feature names.
            # Try to map the LIME feature string back to one of the original feature_names.
            # This handles cases like "Feature_X <= 0.5" or "Feature_X > 0.5" mapping to "Feature_X"
            # and also ensures exact matches are preserved.
            matched_name = None
            for original_name in feature_names:
                # Check if the LIME string starts with the original feature name
                # and then has a space or comparison operator.
                # Or if it's an exact match.
                if feature_str == original_name: # Exact match
                    matched_name = original_name
                    break
                # Check for "FeatureName operator value" pattern
                if re.match(rf"^{re.escape(original_name)}\s*[<>=!]+", feature_str):
                    matched_name = original_name
                    break
                # Check for "FeatureName=value" pattern (common for categorical)
                if re.match(rf"^{re.escape(original_name)}=+", feature_str):
                    matched_name = original_name
                    break

            if matched_name:
                feature_importance[matched_name] = weight
            else:
                # Fallback: if no match, use the previous regex cleaning
                cleaned_feature_name = re.split(r'[<>=!]+', feature_str)[0].strip()
                feature_importance[cleaned_feature_name] = weight
    elif isinstance(explanation_data, np.ndarray) or \
         (isinstance(explanation_data, list) and all(isinstance(item, np.ndarray) for item in explanation_data)):
        # This is SHAP output
        shap_values_raw = None
        if isinstance(explanation_data, list):
            if predicted_class is not None:
                predicted_class_int = int(predicted_class)
            else:
                predicted_class_int = 0

            if predicted_class_int < len(explanation_data):
                shap_values_raw = explanation_data[predicted_class_int]
            else:
                shap_values_raw = explanation_data[0]
        elif isinstance(explanation_data, np.ndarray):
            shap_values_raw = explanation_data

        if shap_values_raw is not None:
            if shap_values_raw.ndim == 2 and shap_values_raw.shape[0] == 1:
                shap_values_raw = shap_values_raw[0]
            elif shap_values_raw.ndim > 1 and shap_values_raw.shape[0] > 1:
                shap_values_raw = shap_values_raw[0]

            if shap_values_raw is not None:
                for i, feature in enumerate(feature_names):
                    if i < len(shap_values_raw):
                        feature_importance[feature] = shap_values_raw[i]
    return feature_importance

def print_explanation(explanation_data, feature_names: list, title: str, predicted_class: int = None):
    """
    Prints LIME or SHAP explanation in a readable format.
    """
    print(f"\n--- {title} ---\n")
    feature_importance_dict = _process_explanation_data_for_plot(explanation_data, feature_names, predicted_class)
    if not feature_importance_dict:
        print("No explanation data to display.")
        return
    sorted_features = sorted(feature_importance_dict.items(), key=lambda item: abs(item[1]), reverse=True)
    for feature, weight in sorted_features:
        print(f"  {feature}: {weight:.4f}")

def plot_explanation_comparison(
    exp_original,
    exp_adversarial,
    feature_names: list,
    sensitive_feature_name: str,
    plot_title: str,
    predicted_class: int = None,
    top_n_features: int = 5
):
    """
    Generates a bar plot comparing feature importances from original and adversarial explanations for a single instance.
    """
    # print(f"\n--- Plotting: {plot_title} ---")
    original_imp = _process_explanation_data_for_plot(exp_original, feature_names, predicted_class)
    adversarial_imp = _process_explanation_data_for_plot(exp_adversarial, feature_names, predicted_class)
    if not original_imp or not adversarial_imp:
        print("Cannot plot: Missing explanation data.")
        return
    sorted_original_features = sorted(original_imp.items(), key=lambda item: abs(item[1]), reverse=True)

    features_to_plot = []
    for feature, _ in sorted_original_features:
        if feature != sensitive_feature_name:
            features_to_plot.append(feature)
        if len(features_to_plot) >= top_n_features:
            break
    if sensitive_feature_name not in features_to_plot and sensitive_feature_name in feature_names:
        features_to_plot.append(sensitive_feature_name)
    features_to_plot.sort(key=lambda f: original_imp.get(f, 0), reverse=True)

    plot_data = []
    for feature in features_to_plot:
        plot_data.append({'Feature': feature, 'Importance': original_imp.get(feature, 0), 'Model': 'Original Biased'})
        plot_data.append({'Feature': feature, 'Importance': adversarial_imp.get(feature, 0), 'Model': 'Adversarial'})
    plot_df = pd.DataFrame(plot_data)

    plt.figure(figsize=(6, 3))
    ax = sns.barplot(x='Importance', y='Feature', hue='Model', data=plot_df, palette='viridis', ci=None)
    plt.title(plot_title, fontsize=16)
    plt.xlabel("Feature Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.axvline(x=0, color='grey', linestyle='--', linewidth=0.8)
    
    # Adjust legend to be single line at bottom
    plt.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.2), 
        ncol=2, 
        frameon=False,
        fontsize=12
    )
    
    plt.tight_layout()
    plt.show()


def _get_ranked_features_from_explanation(explanation_data, feature_names: list, predicted_class: int = None, top_n: int = 5) -> list:
    """
    Extracts the names of the top N features from a single LIME or SHAP explanation, ranked by importance.
    """
    feature_importance_dict = _process_explanation_data_for_plot(explanation_data, feature_names, predicted_class)
    if not feature_importance_dict:
        return []
    sorted_features = sorted(feature_importance_dict.items(), key=lambda item: abs(item[1]), reverse=True)
    top_features = [feature for feature, _ in sorted_features[:top_n]]
    return top_features


def get_feature_rank_frequencies(
    model,
    explainer_type: str,
    data_instances: np.ndarray,
    feature_names: list,
    predicted_classes: np.ndarray,
    training_data_for_explainer: np.ndarray,
    background_data_for_shap: np.ndarray,
    top_n: int = 5,
    class_names: list = ['Class 0', 'Class 1'],
    categorical_feature_indices: list = None
) -> dict:
    """
    Calculates the percentage frequency of each feature appearing at each specific rank
    (1st, 2nd, ..., Nth) across a set of data instances.
    """
    # print(f"--- Calculating {explainer_type.upper()} Feature Rank Frequencies (Top {top_n}) ---")

    feature_rank_counts = defaultdict(lambda: defaultdict(int))
    total_instances = data_instances.shape[0]

    for i in range(total_instances):
        instance = data_instances[i]
        predicted_class = predicted_classes[i]

        if explainer_type == 'lime':
            explanation = get_lime_explanation(
                model, instance, feature_names, training_data_for_explainer, class_names,
                categorical_feature_indices=categorical_feature_indices
            )
        elif explainer_type == 'shap':
            explanation = get_shap_explanation(
                model, instance.reshape(1, -1), feature_names, background_data_for_shap, class_names
            )
        else:
            raise ValueError("explainer_type must be 'lime' or 'shap'")

        ranked_features_for_instance = _get_ranked_features_from_explanation(
            explanation, feature_names, predicted_class, top_n
        )

        for rank_idx, feature in enumerate(ranked_features_for_instance):
            rank = rank_idx + 1
            if rank <= top_n:
                feature_rank_counts[feature][rank] += 1

    feature_rank_percentages = {}
    for feature, ranks_and_counts in feature_rank_counts.items():
        feature_rank_percentages[feature] = {
            rank: (count / total_instances) * 100
            for rank, count in ranks_and_counts.items()
        }

    return feature_rank_percentages

def _get_top_non_sensitive_features(frequency_data_dict: dict, sensitive_feature_name: str, num_top_features: int = 3) -> list:
    """
    Helper to identify the top non-sensitive features based on their overall frequency
    across all models and ranks.
    """
    total_feature_frequency = defaultdict(float)
    for model_name, model_data in frequency_data_dict.items():
        for feature, ranks_data in model_data.items():
            for rank, percentage in ranks_data.items():
                total_feature_frequency[feature] += percentage

    sorted_total_frequency = sorted(total_feature_frequency.items(), key=lambda item: item[1], reverse=True)

    top_non_sensitive = []
    for feature, _ in sorted_total_frequency:
        if feature != sensitive_feature_name:
            top_non_sensitive.append(feature)
        if len(top_non_sensitive) >= num_top_features:
            break
    return top_non_sensitive


def plot_feature_rank_distribution(
    frequency_data_dict: dict,
    explainer_name: str,
    sensitive_feature_name: str,
    plot_title: str,
    top_n: int = 5
):
    """
    Generates a stacked bar plot showing the percentage frequency of each feature
    at specific ranks (1st, 2nd, ..., Nth) for different models.
    """
    # print(f"\n--- Plotting: {plot_title} ---")

    # Define the custom color palette
    SPECIFIC_FEATURE_COLORS = {
        'color_1': '#b54228', # For sensitive feature
        'color_2': '#4ab0d7',
        'color_3': '#1771a4',
        'color_4': '#607780', # Fourth feature color
        'color_other': '#bbbbbb' # Very light grey for 'Other Features'
    }

    # Identify the top 3 non-sensitive features based on overall frequency
    top_non_sensitive_features = _get_top_non_sensitive_features(frequency_data_dict, sensitive_feature_name, num_top_features=3)

    # Create the ordered list of features that will get specific colors
    features_to_explicitly_color = [sensitive_feature_name] + top_non_sensitive_features
    features_to_explicitly_color = list(dict.fromkeys(features_to_explicitly_color))

    # Create a mapping from actual feature names to the specific hex colors
    assigned_colors_map = {}
    specific_color_values_list = [
        SPECIFIC_FEATURE_COLORS['color_1'],
        SPECIFIC_FEATURE_COLORS['color_2'],
        SPECIFIC_FEATURE_COLORS['color_3'],
        SPECIFIC_FEATURE_COLORS['color_4']
    ]

    for i, feature_name in enumerate(features_to_explicitly_color):
        if i < len(specific_color_values_list):
            assigned_colors_map[feature_name] = specific_color_values_list[i]
        else:
            assigned_colors_map[feature_name] = SPECIFIC_FEATURE_COLORS['color_other']

    assigned_colors_map['Other Features'] = SPECIFIC_FEATURE_COLORS['color_other']


    num_models = len(frequency_data_dict)
    fig, axes = plt.subplots(1, num_models, figsize=(6, 3), sharey=True) # Increased figsize
    if num_models == 1:
        axes = [axes]

    # Prepare rank labels for Y-axis
    rank_labels_full = [f"{r}st" if r == 1 else f"{r}nd" if r == 2 else f"{r}rd" if r == 3 else f"{r}th" for r in range(1, top_n + 1)]
    # print(f"DEBUG: rank_labels_full: {rank_labels_full}")


    for ax_idx, (model_name, model_data) in enumerate(frequency_data_dict.items()):
        ax = axes[ax_idx]
        # print(f"DEBUG: Processing model: {model_name}, ax_idx: {ax_idx}")

        plot_df_rows = []

        for feature, ranks_data in model_data.items():
            display_feature_name = feature
            if feature not in features_to_explicitly_color:
                display_feature_name = 'Other Features'

            for rank, percentage in ranks_data.items():
                if rank <= top_n:
                    plot_df_rows.append({
                        'Feature': display_feature_name,
                        'Rank': rank,
                        'Percentage': percentage
                    })

        if not plot_df_rows:
            ax.set_title(f"{model_name}\n(No data)", fontsize=14)
            ax.set_xlabel("")
            ax.set_ylabel("")
            # print(f"DEBUG: No plot data for {model_name}. Skipping plot.")
            continue

        plot_df = pd.DataFrame(plot_df_rows)
        # print(f"DEBUG: Plot DF for {model_name} (head):\n{plot_df.head()}")

        aggregated_df = plot_df.groupby(['Rank', 'Feature'])['Percentage'].sum().unstack(fill_value=0)

        aggregated_df = aggregated_df.reindex(range(1, top_n + 1), fill_value=0)
        # print(f"DEBUG: Aggregated DF for {model_name} (head):\n{aggregated_df.head()}")

        column_order_for_stacking = []
        for feat in features_to_explicitly_color:
            if feat in aggregated_df.columns:
                column_order_for_stacking.append(feat)
        if 'Other Features' in aggregated_df.columns and 'Other Features' not in column_order_for_stacking:
            column_order_for_stacking.append('Other Features')

        # Filter out columns that don't exist in aggregated_df (e.g., if 'Other Features' is not present)
        final_column_order = [col for col in column_order_for_stacking if col in aggregated_df.columns]
        aggregated_df = aggregated_df[final_column_order]

        colors_for_plot = [assigned_colors_map.get(col, SPECIFIC_FEATURE_COLORS['color_other']) for col in aggregated_df.columns]
        # print(f"DEBUG: Colors for plot for {model_name}: {colors_for_plot}")
        # print(f"DEBUG: Columns for plot for {model_name}: {aggregated_df.columns.tolist()}")

        aggregated_df.plot(kind='barh', stacked=True, ax=ax, color=colors_for_plot, width=0.8)

        ax.set_title(f"{model_name}", fontsize=14)

        ax.set_xlabel("") # This will be set globally later
        ax.set_ylabel("") # This will be set globally later

        ax.set_xlim(0, 100)
        ax.tick_params(axis='y', rotation=0)
        ax.invert_yaxis()

        # THIS IS THE KEY CHANGE: Always set yticklabels
        ax.set_yticklabels(rank_labels_full)
        # print(f"DEBUG: Setting yticklabels for subplot {model_name}: {rank_labels_full}") # Confirms it's being called

        # Optional: Debugging the labels that are actually set
        current_labels = [label.get_text() for label in ax.get_yticklabels()]
        # print(f"DEBUG: Actual yticklabels obtained for {model_name}: {current_labels}")


        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=True)

        ax.legend().set_visible(False)

    # Create a single common legend for features
    legend_elements = []
    for feature_name in features_to_explicitly_color:
        if feature_name in assigned_colors_map:
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', label=feature_name,
                                              markerfacecolor=assigned_colors_map[feature_name], markersize=10))

    other_features_present = False
    for model_data in frequency_data_dict.values():
        temp_df_rows = []
        for feature, ranks_data in model_data.items():
            display_feature_name = feature
            if feature not in features_to_explicitly_color:
                display_feature_name = 'Other Features'
            for rank, percentage in ranks_data.items():
                if rank <= top_n:
                    temp_df_rows.append({'Feature': display_feature_name, 'Rank': rank, 'Percentage': percentage})
        if temp_df_rows:
            temp_plot_df = pd.DataFrame(temp_df_rows)
            temp_aggregated_df = temp_plot_df.groupby(['Rank', 'Feature'])['Percentage'].sum().unstack(fill_value=0)
            if 'Other Features' in temp_aggregated_df.columns:
                other_features_present = True
                break

    if other_features_present and 'Other Features' in assigned_colors_map:
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', label='Other Features',
                                          markerfacecolor=assigned_colors_map['Other Features'], markersize=10))

    # Adjusted bbox_to_anchor for the legend to be slightly lower, and ncol to 3
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False, fontsize=10)

    # Adjusted X-axis label position to be lower and centered
    fig.text(0.5, 0.15, "Percentage (%)", ha='center', va='center', fontsize=12) # Moved from 0.15 to 0.1

    # Y-axis label position
    fig.text(0.01, 0.5, "Feature Importance Rank", ha='center', va='center', rotation='vertical', fontsize=12)


    fig.suptitle(plot_title, fontsize=18, y=1.02)
    # Adjusted rect to give more space at the bottom for a multi-line legend and X-axis label,
    # and also increased left margin slightly for y-labels on *all* subplots now.
    plt.tight_layout(rect=[0.05, 0.2, 1, 0.98]) # Increased left from 0.03 to 0.05, bottom from 0.25 to 0.2
    plt.show()