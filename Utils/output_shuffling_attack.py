import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import random
import plotly.graph_objects as go
import plotly.express as px

# --- Algorithm 1: f_prime_template (MODIFIED RETURN TYPE with quotes) ---
def f_prime_template(X_data: pd.DataFrame, protected_feature: str,
                     base_model: object, attack_function: callable,
                     superior_outcome_value: int = 1) -> 'tuple[np.ndarray, np.ndarray, np.ndarray]':
    """
    Implements the f' (attacked model) template by applying a specific attack function
    to permute the ordering of base model scores.

    Args:
        X_data (pd.DataFrame): The input features DataFrame, including the protected feature.
                               Its index is used as unique IDs.
        protected_feature (str): The name of the protected feature column in X_data.
        base_model (object): The trained base model (e.g., sklearn classifier). 
                             It must have a .predict_proba() method.
        attack_function (callable): The specific attack algorithm function (e.g., dominance_attack_algo,
                                    mixing_attack_algo, swapping_attack_algo).
                                    It should accept (id_sorted_by_y, p_sorted_by_y, y_sorted_by_y)
                                    and return a new permutation of IDs (np.ndarray).
        superior_outcome_value (int): The class label (0 or 1) that represents the "superior" outcome.
                                      Used for sorting scores. Defaults to 1.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - y_prime_final (np.ndarray): An array of perturbed scores (y') for the original X_data,
                                          aligned with its original index.
            - new_id_order_from_attack (np.ndarray): The IDs re-ordered by the attack function.
            - id_algo1_sorted (np.ndarray): The original IDs, sorted by the base model's score.
    """
    X_zeroed = X_data.copy()
    if protected_feature in X_zeroed.columns:
        X_zeroed[protected_feature] = 0
    y_base_scores: np.ndarray = base_model.predict_proba(X_zeroed)[:, superior_outcome_value]

    ids: np.ndarray = X_data.index.values
    p_values: np.ndarray = X_data[protected_feature].values

    df_for_algo1_sort = pd.DataFrame({
        'id': ids,
        'protected': p_values,
        'score': y_base_scores
    })

    df_algo1_sorted = df_for_algo1_sort.sort_values(by='score', ascending=(superior_outcome_value == 0))

    id_algo1_sorted: np.ndarray = df_algo1_sorted['id'].values
    p_algo1_sorted: np.ndarray = df_algo1_sorted['protected'].values
    y_algo1_sorted: np.ndarray = df_algo1_sorted['score'].values

    new_id_order_from_attack: np.ndarray = attack_function(
        id_algo1_sorted.copy(),
        p_algo1_sorted.copy(),
        y_algo1_sorted.copy()
    )

    df_shuffled = pd.DataFrame({
        'original_index': new_id_order_from_attack,
        'shuffled_score': y_algo1_sorted
    })

    y_prime_final: np.ndarray = df_shuffled.sort_values(by='original_index')['shuffled_score'].values

    return y_prime_final, new_id_order_from_attack, id_algo1_sorted

# --- Attack Algorithm 2: Dominance Attack ---
def dominance_attack_algo(id_sorted_by_y: np.ndarray, p_sorted_by_y: np.ndarray, y_sorted_by_y: np.ndarray,
                          protected_feature_value_for_dominance: int = 0) -> np.ndarray:
    temp_df_for_p_sort = pd.DataFrame({'id': id_sorted_by_y, 'protected': p_sorted_by_y})
    ascending_order: bool = (protected_feature_value_for_dominance == 0) 
    temp_df_for_p_sort_sorted = temp_df_for_p_sort.sort_values(
        by='protected', 
        ascending=ascending_order
    )
    new_id_order_from_p_sort: np.ndarray = temp_df_for_p_sort_sorted['id'].values
    return new_id_order_from_p_sort

# --- Attack Algorithm 3: Mixing Attack ---
def mixing_attack_algo(id_sorted_by_y: np.ndarray, p_sorted_by_y: np.ndarray, y_sorted_by_y: np.ndarray,
                       protected_feature_value_for_protected_group: int = 1, bias_strength: float = 0.7) -> np.ndarray:
    id_protected_group_list: list = []
    scores_protected_group_list: list = []
    id_unprotected_group_list: list = []
    scores_unprotected_group_list: list = []

    for i in range(len(id_sorted_by_y)):
        if p_sorted_by_y[i] == protected_feature_value_for_protected_group:
            id_protected_group_list.append(id_sorted_by_y[i])
            scores_protected_group_list.append(y_sorted_by_y[i])
        else:
            id_unprotected_group_list.append(id_sorted_by_y[i])
            scores_unprotected_group_list.append(y_sorted_by_y[i])

    new_ids: list = []
    
    while id_protected_group_list or id_unprotected_group_list:
        if not id_protected_group_list:
            new_ids.extend(id_unprotected_group_list)
            break
        if not id_unprotected_group_list:
            new_ids.extend(id_protected_group_list)
            break
        
        if random.random() < bias_strength:
            new_ids.append(id_protected_group_list.pop(0))
            scores_protected_group_list.pop(0)
        elif scores_unprotected_group_list[0] <= scores_protected_group_list[0]:
            new_ids.append(id_unprotected_group_list.pop(0))
            scores_unprotected_group_list.pop(0)
        else:
            new_ids.append(id_unprotected_group_list.pop(0))
            scores_unprotected_group_list.pop(0)
            
    return np.array(new_ids)

# --- Attack Algorithm 4: Swapping Attack ---
def swapping_attack_algo(id_sorted_by_y: np.ndarray, p_sorted_by_y: np.ndarray, y_sorted_by_y: np.ndarray,
                          protected_group_value: int = 1) -> np.ndarray:
    N: int = len(id_sorted_by_y)
    ids_to_modify: np.ndarray = id_sorted_by_y.copy()
    p_to_modify: np.ndarray = p_sorted_by_y.copy()

    for i in range(N - 1):
        if p_to_modify[i] == protected_group_value and \
           p_to_modify[i+1] != protected_group_value:
            p_to_modify[i], p_to_modify[i+1] = p_to_modify[i+1], p_to_modify[i]
            ids_to_modify[i], ids_to_modify[i+1] = ids_to_modify[i+1], ids_to_modify[i]
    return ids_to_modify

# --- Plotting Function ---
def plot_top_bottom_shap(shap_values: shap.Explanation, feature_names: 'list[str]',
                         protected_feature: str = None, top_n: int = 3, bottom_n: int = 2,
                         figsize: 'tuple[int, int]' = (5, 2)) -> plt.Figure:
    if hasattr(shap_values, 'values'):
        shap_array: np.ndarray = shap_values.values
    else:
        shap_array: np.ndarray = shap_values

    mean_abs_shap = pd.DataFrame({
        'feature': [str(f)[:20] for f in feature_names],
        'importance': np.mean(np.abs(shap_array), axis=0),
        'original_index': range(len(feature_names))
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    plot_features = pd.DataFrame()
    top_features = mean_abs_shap.head(top_n)
    bottom_features = mean_abs_shap.tail(bottom_n)

    plot_features = pd.concat([top_features])
    
    if protected_feature and protected_feature in feature_names:
        protected_idx = feature_names.index(protected_feature)
        protected_row = mean_abs_shap[mean_abs_shap['original_index'] == protected_idx]
        
        is_protected_in_top = protected_idx in top_features['original_index'].values
        is_protected_in_bottom = protected_idx in bottom_features['original_index'].values

        if not is_protected_in_top and not is_protected_in_bottom:
            plot_features = pd.concat([plot_features, protected_row])
    
    plot_features = pd.concat([plot_features, bottom_features]).drop_duplicates(subset=['original_index']).sort_values('importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=figsize)

    print(plot_features)

    colors = []
    for idx in plot_features['original_index']:
        if protected_feature and feature_names[idx] == protected_feature:
            colors.append('#000000') # Black for protected feature
        else:
            colors.append('#ff125c') # Red for others

    bars = ax.barh(plot_features['feature'], plot_features['importance'], color=colors)
    
    ax.set_xlabel('Mean Absolute SHAP Value', fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.1)
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)

    for i, v in enumerate(plot_features['importance']):
        ax.text(v + (ax.get_xlim()[1] * 0.02), i, f"{v:.2f}", va='center', fontsize=6)

    if protected_feature and protected_feature in feature_names:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ff125c', label='Other Features'),
            Patch(facecolor='#000000', label=f'Protected: {protected_feature[:10]}')
        ]
        ax.legend(handles=legend_elements, fontsize=6, loc='lower right')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.tight_layout()
    return fig

# --- SHAP Explanation for Base Model ---
def Base_Model_Explanation(x_test: pd.DataFrame, base_model: object, protected_feature: str,
                           feature_names: 'list[str]', superior_outcome_value: int,
                           shap_sample_size: int = 1000, random_state: int = 42) -> None:
    print("\n" + "="*50)
    print("SHAP Explanations for the BASE Model (f)")
    print("="*50)

    BACKGROUND_SIZE: int = 100
    if len(x_test) > BACKGROUND_SIZE:
        background_data: np.ndarray = x_test.sample(BACKGROUND_SIZE, random_state=random_state).values
    else:
        background_data: np.ndarray = x_test.values

    if len(x_test) > shap_sample_size:
        random.seed(random_state)
        random_indices = random.sample(range(len(x_test)), shap_sample_size)
        x_test_for_shap: pd.DataFrame = x_test.iloc[random_indices].copy()
    else:
        x_test_for_shap: pd.DataFrame = x_test.copy()

    print(f"Using {len(x_test_for_shap)} samples from x_test for SHAP explanation calculation.")

    explainer_base = shap.Explainer(
        lambda X_val: base_model.predict_proba(pd.DataFrame(X_val, columns=feature_names))[:, superior_outcome_value],
        background_data,
        algorithm='permutation'
    )

    shap_values_base = explainer_base(x_test_for_shap.values)
    shap_values_base.feature_names = feature_names

    print(f"SHAP Summary (Base Model - using {len(x_test_for_shap)} test samples)")
    shap.plots.bar(shap_values_base, show=True)
    shap.plots.beeswarm(shap_values_base, show=True)

    if protected_feature in feature_names:
        protected_idx = feature_names.index(protected_feature)
        print(f"Mean |SHAP| for {protected_feature} (Base Model): {np.mean(np.abs(shap_values_base.values[:, protected_idx])):.4f}")
    else:
        print(f"Protected feature '{protected_feature}' not found in feature_names for SHAP analysis.")

    plot = plot_top_bottom_shap(
        shap_values=shap_values_base,
        feature_names=feature_names,
        protected_feature=protected_feature,
        top_n=3,
        bottom_n=2,
        figsize=(4, 2)
        )
    plot.show()

# --- run_attack_analysis ---
def run_attack_analysis(x_test: pd.DataFrame, base_model: object, feature_names: 'list[str]',
                        superior_outcome_value: int, protected_feature: str,
                        attack_type: str, attack_params: 'dict' = None,
                        shap_sample_size: int = 1000, random_state: int = 42) -> None:
    BACKGROUND_SIZE: int = 100
    if len(x_test) > BACKGROUND_SIZE:
        background_data: np.ndarray = x_test.sample(BACKGROUND_SIZE, random_state=random_state).values
    else:
        background_data: np.ndarray = x_test.values

    if len(x_test) > shap_sample_size:
        random.seed(random_state)
        random_indices = random.sample(range(len(x_test)), shap_sample_size)
        x_test_for_shap: pd.DataFrame = x_test.iloc[random_indices].copy()
    else:
        x_test_for_shap: pd.DataFrame = x_test.copy()

    print(f"Using {len(x_test_for_shap)} samples from x_test for SHAP explanation calculation.")

    if attack_type != 'none':
        print("\n" + "="*50)
        print(f"SHAP Explanations for the ATTACK Model (f') - {attack_type.capitalize()} Attack")
        print("="*50)

        attack_func: callable = None
        current_attack_params: 'dict' = attack_params if attack_params is not None else {}

        if attack_type == 'dominance':
            attack_func = lambda id_y, p_y, y_y: dominance_attack_algo(
                id_y, p_y, y_y,
                protected_feature_value_for_dominance=current_attack_params.get('protected_value_for_dominance', 0)
            )
        elif attack_type == 'mixing':
            attack_func = lambda id_y, p_y, y_y: mixing_attack_algo(
                id_y, p_y, y_y,
                protected_feature_value_for_protected_group=current_attack_params.get('protected_value_for_protected_group', 1),
                bias_strength=current_attack_params.get('bias_strength', 0.7)
            )
        elif attack_type == 'swapping':
            attack_func = lambda id_y, p_y, y_y: swapping_attack_algo(
                id_y, p_y, y_y,
                protected_group_value=current_attack_params.get('protected_value', 1)
            )
        else:
            print(f"Warning: Unknown attack type '{attack_type}'. No attack applied.")
            return

        def attack_model_wrapper_for_shap(X_array: np.ndarray) -> np.ndarray:
            X_df = pd.DataFrame(X_array, columns=feature_names, index=range(len(X_array)))
            y_prime_scores: np.ndarray = f_prime_template(
                X_data=X_df,
                protected_feature=protected_feature,
                base_model=base_model,
                attack_function=attack_func,
                superior_outcome_value=superior_outcome_value
            )[0]
            return y_prime_scores

        explainer_attack = shap.Explainer(
            attack_model_wrapper_for_shap,
            background_data,
            algorithm='permutation'
        )

        shap_values_attack = explainer_attack(x_test_for_shap.values)
        shap_values_attack.feature_names = feature_names

        print(f"SHAP Summary (Attack Model - using {len(x_test_for_shap)} test samples)")
        shap.plots.bar(shap_values_attack, show=True)
        shap.plots.beeswarm(shap_values_attack, show=True)

        if protected_feature in feature_names:
            protected_idx = feature_names.index(protected_feature)
            print(f"Mean |SHAP| for {protected_feature} (Attack Model - {attack_type.capitalize()}): {np.mean(np.abs(shap_values_attack.values[:, protected_idx])):.4f}")
        else:
            print(f"Protected feature '{protected_feature}' not found in feature_names for SHAP analysis.")

        plot = plot_top_bottom_shap(
            shap_values=shap_values_attack,
            feature_names=feature_names,
            protected_feature=protected_feature,
            top_n=3,
            bottom_n=2,
            figsize=(5, 2)
            )
        plot.show()

    plt.tight_layout()
    plt.show()

# --- Helper Functions to get attack info (scores, new ID order, original sorted IDs) ---
def get_dominance_attack_info(x_data_subset: pd.DataFrame, protected_feature: str,
                                base_model: object, superior_outcome_value: int,
                                protected_feature_value_for_dominance: int = 0) -> 'tuple[np.ndarray, np.ndarray, np.ndarray]':
    """Calculates perturbed scores, new ID order, and original sorted IDs using the dominance attack."""
    y_prime_final, new_id_order, original_sorted_ids = f_prime_template(
        X_data=x_data_subset,
        protected_feature=protected_feature,
        base_model=base_model,
        attack_function=lambda id_y, p_y, y_y: dominance_attack_algo(
            id_y, p_y, y_y,
            protected_feature_value_for_dominance=protected_feature_value_for_dominance
        ),
        superior_outcome_value=superior_outcome_value
    )
    return y_prime_final, new_id_order, original_sorted_ids

def get_mixing_attack_info(x_data_subset: pd.DataFrame, protected_feature: str,
                             base_model: object, superior_outcome_value: int,
                             protected_feature_value_for_protected_group: int = 1,
                             bias_strength: float = 0.7) -> 'tuple[np.ndarray, np.ndarray, np.ndarray]':
    """Calculates perturbed scores, new ID order, and original sorted IDs using the mixing attack."""
    y_prime_final, new_id_order, original_sorted_ids = f_prime_template(
        X_data=x_data_subset,
        protected_feature=protected_feature,
        base_model=base_model,
        attack_function=lambda id_y, p_y, y_y: mixing_attack_algo(
            id_y, p_y, y_y,
            protected_feature_value_for_protected_group=protected_feature_value_for_protected_group,
            bias_strength=bias_strength
        ),
        superior_outcome_value=superior_outcome_value
    )
    return y_prime_final, new_id_order, original_sorted_ids

def get_swapping_attack_info(x_data_subset: pd.DataFrame, protected_feature: str,
                               base_model: object, superior_outcome_value: int,
                               protected_group_value: int = 1) -> 'tuple[np.ndarray, np.ndarray, np.ndarray]':
    """Calculates perturbed scores, new ID order, and original sorted IDs using the swapping attack."""
    y_prime_final, new_id_order, original_sorted_ids = f_prime_template(
        X_data=x_data_subset,
        protected_feature=protected_feature,
        base_model=base_model,
        attack_function=lambda id_y, p_y, y_y: swapping_attack_algo(
            id_y, p_y, y_y,
            protected_group_value=protected_group_value
        ),
        superior_outcome_value=superior_outcome_value
    )
    return y_prime_final, new_id_order, original_sorted_ids


# --- Main Visualization Function for ALL Scores (Parallel Coordinates - 4 Axes, No Protected Status Axis) ---
def visualize_all_attack_scores_parallel_coords(x_test: pd.DataFrame, base_model: object, protected_feature: str,
                      superior_outcome_value: int = 1, sample_size: int = 500) -> go.Figure:
    """
    Visualizes the comparison of predicted scores from the base model and
    various attack scenarios using a single parallel coordinates plot with 4 axes.
    Lines are colored by protected status.

    Args:
        x_test (pd.DataFrame): The test features DataFrame.
        base_model (object): The trained base model (must have .predict_proba method).
        protected_feature (str): The name of the protected feature column in x_test.
        superior_outcome_value (int): The class label (0 or 1) that represents the "superior" outcome.
                                      Scores will be based on predict_proba for this class. Defaults to 1.
        sample_size (int): Number of samples to use for visualization to keep the plot manageable.
                           Defaults to 500.

    Returns:
        go.Figure: A Plotly Graph Objects figure for the parallel coordinates plot.
    """
    if len(x_test) > sample_size:
        x_vis = x_test.sample(sample_size, random_state=42).copy()
    else:
        x_vis = x_test.copy()

    base_scores = base_model.predict_proba(x_vis)[:, superior_outcome_value]
    
    # Use the 'get_X_attack_info' functions, but only take the first element (y_prime_final)
    attack_scores = {
        'dominance': get_dominance_attack_info(x_vis, protected_feature, base_model, superior_outcome_value)[0],
        'mixing': get_mixing_attack_info(x_vis, protected_feature, base_model, superior_outcome_value)[0],
        'swapping': get_swapping_attack_info(x_vis, protected_feature, base_model, superior_outcome_value)[0]
    }
    
    df_scores = pd.DataFrame({
        'id': x_vis.index,
        'protected_status': x_vis[protected_feature],
        'base_score': base_scores
    })
    df_scores = df_scores.set_index('id')

    for attack_name, scores in attack_scores.items():
        df_scores[f'{attack_name}_attack_score'] = pd.Series(scores, index=x_vis.index)

    columns_to_normalize = ['base_score'] + [col for col in df_scores.columns if '_attack_score' in col]
    
    for col in columns_to_normalize:
        min_val = df_scores[col].min()
        max_val = df_scores[col].max()
        if max_val > min_val:
            df_scores[col] = (df_scores[col] - min_val) / (max_val - min_val)
        else:
            df_scores[col] = 0.5 # If all values are the same, normalize to 0.5

    # Dimensions for the parallel coordinates plot - NO PROTECTED STATUS AXIS
    dimensions = [
        dict(range=[0, 1], label='Base Score', values=df_scores['base_score']),
        dict(range=[0, 1], label='Dominance Attack Score', values=df_scores['dominance_attack_score']),
        dict(range=[0, 1], label='Mixing Attack Score', values=df_scores['mixing_attack_score']),
        dict(range=[0, 1], label='Swapping Attack Score', values=df_scores['swapping_attack_score'])
    ]
    
    unique_protected_vals = sorted(df_scores['protected_status'].unique())
    
    # --- MODIFIED COLORSCALE HERE ---
    if len(unique_protected_vals) == 2:
        # Assuming 0 maps to light green and 1 maps to light yellow
        # You can swap these if your protected feature's values are different or if you prefer
        colorscale = [[0, '#8db581'], [1, '#f4b75e']] # Light Green and Light Yellow
    else:
        # Fallback for more than 2 protected values - will still use Plotly's qualitative,
        # as you only requested specific colors for 2 unique values.
        plotly_colors = px.colors.qualitative.Plotly
        colorscale = [[i / (len(unique_protected_vals) - 1), plotly_colors[i % len(plotly_colors)]] 
                      for i in range(len(unique_protected_vals))]
        
    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=df_scores['protected_status'],
            colorscale=colorscale,
            showscale=True,
            cmin=min(unique_protected_vals),
            cmax=max(unique_protected_vals),
            colorbar=dict(title=f'{protected_feature}<br>(Status)')
        ),
        dimensions=dimensions
    ))
    
    fig.update_layout(
        # title='Comparison of Attack Methods on Predicted Scores (Parallel Coordinates)',
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        width=1000,
        font=dict(family="Arial, sans-serif", size=10),
        hovermode='closest'
    )
    
    return fig


# --- New Visualization Function for ID Rank Changes (Parallel Coordinates - 4 Axes, No Protected Status Axis) ---
def visualize_all_attack_ranks_parallel_coords(x_test: pd.DataFrame, base_model: object, protected_feature: str,
                              superior_outcome_value: int = 1, sample_size: int = 500) -> go.Figure:
    """
    Visualizes the changes in ID ranks across different attack scenarios using a single
    parallel coordinates plot with 4 axes.
    Lines are colored by protected status.

    Args:
        x_test (pd.DataFrame): The test features DataFrame.
        base_model (object): The trained base model (must have .predict_proba method).
        protected_feature (str): The name of the protected feature column in x_test.
        superior_outcome_value (int): The class label (0 or 1) that represents the "superior" outcome.
                                      Used for defining the original ranking. Defaults to 1.
        sample_size (int): Number of samples to use for visualization. Defaults to 500.

    Returns:
        go.Figure: A Plotly Graph Objects figure for the parallel coordinates plot.
    """
    if len(x_test) > sample_size:
        x_vis = x_test.sample(sample_size, random_state=42).copy()
    else:
        x_vis = x_test.copy()

    df_ranks = pd.DataFrame({
        'id': x_vis.index.values,
        'protected_status': x_vis[protected_feature].values
    }).set_index('id')

    # Get original ranks from the base model by running f_prime_template with a "no-op" attack
    _, _, original_sorted_ids_full = f_prime_template(
        X_data=x_vis,
        protected_feature=protected_feature,
        base_model=base_model,
        attack_function=lambda id_y, p_y, y_y: id_y, # Attack function that returns original order
        superior_outcome_value=superior_outcome_value
    )
    
    original_rank_map = {id_val: i + 1 for i, id_val in enumerate(original_sorted_ids_full)}
    df_ranks['original_rank'] = df_ranks.index.map(original_rank_map)

    # Get ranks for each attack scenario
    attack_info = {
        'dominance': get_dominance_attack_info(x_vis, protected_feature, base_model, superior_outcome_value),
        'mixing': get_mixing_attack_info(x_vis, protected_feature, base_model, superior_outcome_value),
        'swapping': get_swapping_attack_info(x_vis, protected_feature, base_model, superior_outcome_value)
    }

    for attack_name, (scores, new_id_order, _) in attack_info.items(): # Unpack the tuple
        attack_rank_map = {id_val: i + 1 for i, id_val in enumerate(new_id_order)}
        df_ranks[f'{attack_name}_rank'] = df_ranks.index.map(attack_rank_map)

    # Dimensions for the parallel coordinates plot - NO PROTECTED STATUS AXIS
    dimensions = [
        dict(
            range=[1, sample_size], # Ranks typically start from 1
            label='Original Rank',
            values=df_ranks['original_rank']
        ),
        dict(
            range=[1, sample_size],
            label='Dominance Attack Rank',
            values=df_ranks['dominance_rank']
        ),
        dict(
            range=[1, sample_size],
            label='Mixing Attack Rank',
            values=df_ranks['mixing_rank']
        ),
        dict(
            range=[1, sample_size],
            label='Swapping Attack Rank',
            values=df_ranks['swapping_rank']
        )
    ]
    
    unique_protected_vals = sorted(df_ranks['protected_status'].unique())
    
    # --- MODIFIED COLORSCALE HERE ---
    if len(unique_protected_vals) == 2:
        # Assuming 0 maps to light green and 1 maps to light yellow
        colorscale = [[0, '#8db581'], [1, '#f4b75e']] # Light Green and Light Yellow
    else:
        # Fallback for more than 2 protected values
        plotly_colors = px.colors.qualitative.Plotly
        colorscale = [[i / (len(unique_protected_vals) - 1), plotly_colors[i % len(plotly_colors)]] 
                      for i in range(len(unique_protected_vals))]
        
    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=df_ranks['protected_status'],
            colorscale=colorscale,
            showscale=True,
            cmin=min(unique_protected_vals),
            cmax=max(unique_protected_vals),
            colorbar=dict(title=f'{protected_feature}<br>(Status)')
        ),
        dimensions=dimensions
    ))
    
    fig.update_layout(
        # title='ID Rank Changes Across Attack Methods (Parallel Coordinates)',
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        width=1000,
        font=dict(family="Arial, sans-serif", size=10),
        hovermode='closest'
    )
    
    return fig
