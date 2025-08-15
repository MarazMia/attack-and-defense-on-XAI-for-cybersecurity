import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import random
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.patches import Patch

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





def plot_global_lime_mean_abs_from_scratch(
    x_data: pd.DataFrame, 
    model: object,
    feature_names: 'list[str]',
    categorical_features: 'list[str]',
    superior_outcome_value: int,
    protected_feature: str = None, 
    top_n: int = 5,
    figsize: 'tuple[int, int]' = (6, 3),
    random_state: int = 42
) -> plt.Figure:
    """
    Generates a global mean absolute LIME explanation plot by explaining every instance
    in the provided data and then aggregating the results. Ensures the protected feature
    is always shown.
    
    Args:
        x_data (pd.DataFrame): The input features DataFrame.
        model (object or callable): The trained model object with a .predict_proba method
                                   or a callable prediction function.
        feature_names (list[str]): The names of the features.
        categorical_features (list[str]): The names of the categorical features.
        superior_outcome_value (int): The class label (0 or 1) to explain.
        protected_feature (str, optional): Name of the protected feature. Defaults to None.
        top_n (int): Number of top features to display. Defaults to 5.
        figsize (tuple[int, int]): Figure size. Defaults to (6, 3).
        random_state (int): Seed for reproducibility.
        
    Returns:
        plt.Figure: A Matplotlib Figure object.
    """
    if hasattr(model, 'predict_proba'):
        predict_fn = model.predict_proba
    elif callable(model):
        predict_fn = model
    else:
        raise ValueError("The 'model' argument must be an object with a .predict_proba method or a callable function.")

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=x_data.values,
        feature_names=feature_names,
        class_names=['other', 'superior'],
        categorical_features=[feature_names.index(feat) for feat in categorical_features],
        mode='classification',
        random_state=random_state
    )

    all_explanations = []
    
    print(f"Generating LIME explanations for {len(x_data)} samples to create a global plot...")
    
    for i in range(len(x_data)):
        exp = lime_explainer.explain_instance(
            data_row=x_data.iloc[i].values,
            predict_fn=predict_fn,
            num_features=len(feature_names),
            labels=[superior_outcome_value]
        )
        
        explanation_dict = dict(exp.as_list(label=superior_outcome_value))
        all_explanations.append(explanation_dict)

    df_explanations = pd.DataFrame(all_explanations).fillna(0)
    
    aggregated_importance = pd.Series(index=feature_names, data=0.0)

    for feature_name in feature_names:
        relevant_cols = [col for col in df_explanations.columns if col.startswith(feature_name)]
        if relevant_cols:
            aggregated_importance[feature_name] = df_explanations[relevant_cols].abs().mean().sum()
    
    sorted_importance = aggregated_importance.sort_values(ascending=False)
    
    features_to_plot = sorted_importance.head(top_n).index.tolist()
    
    if protected_feature and protected_feature not in features_to_plot:
        features_to_plot.append(protected_feature)
        features_to_plot = sorted_importance[features_to_plot].sort_values(ascending=False).index.tolist()

    plot_data = sorted_importance[features_to_plot].to_frame(name='importance').reset_index()
    plot_data = plot_data.rename(columns={'index': 'feature'})

    fig, ax = plt.subplots(figsize=figsize)
    
    colors = []
    for feature in plot_data['feature']:
        if protected_feature and feature == protected_feature:
            colors.append('black')
        else:
            colors.append('red')

    ax.barh(plot_data['feature'], plot_data['importance'], color=colors)
    
    ax.set_xlabel('Mean Absolute LIME Weight', fontsize=8)
    ax.set_title('Top Features by Mean Absolute LIME Importance', fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=8)

    for i, v in enumerate(plot_data['importance']):
        ax.text(v + (ax.get_xlim()[1] * 0.02), i, f"{v:.2f}", va='center', fontsize=6)
        
    if protected_feature and protected_feature in plot_data['feature'].values:
        legend_elements = [
            Patch(facecolor='red', label='Other Features'),
            Patch(facecolor='black', label=f'Protected: {protected_feature}')
        ]
        ax.legend(handles=legend_elements, fontsize=6, loc='lower right')
        
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)
    
    fig.tight_layout()
    plt.show()
    return fig





# --- Main Usage ---

def Base_Model_Explanation_Global(x_test: pd.DataFrame, base_model: object, protected_feature: str,
                                feature_names: 'list[str]', categorical_features: 'list[str]',
                                superior_outcome_value: int) -> None:
    print("\n" + "="*50)
    print("LIME Global Explanations for the BASE Model (f)")
    print("="*50)
    
    plot_global_lime_mean_abs_from_scratch(
        x_data=x_test,
        model=base_model,  # Pass the model object directly
        protected_feature=protected_feature,
        feature_names=feature_names,
        categorical_features=categorical_features,
        superior_outcome_value=superior_outcome_value
    )





def run_attack_analysis_LIME_global(x_test: pd.DataFrame, base_model: object, feature_names: 'list[str]',
                                    categorical_features: 'list[str]', superior_outcome_value: int, 
                                    protected_feature: str, attack_type: str, attack_params: 'dict' = None,
                                    random_state: int = 42) -> None:
    """
    Runs the output shuffling attack on a LIME explainer and visualizes the aggregated
    results with a global mean absolute plot.
    """
    print("\n" + "="*50)
    print(f"LIME Global Explanations for the ATTACK Model (f') - {attack_type.capitalize()} Attack")
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

    # Wrapper function for the attacked model
    def attack_model_wrapper_for_lime(X_array: np.ndarray) -> np.ndarray:
        X_df = pd.DataFrame(X_array, columns=feature_names, index=range(len(X_array)))
        y_prime_scores_superior = f_prime_template(
            X_data=X_df,
            protected_feature=protected_feature,
            base_model=base_model,
            attack_function=attack_func,
            superior_outcome_value=superior_outcome_value
        )[0]
        y_prime_scores_other = 1 - y_prime_scores_superior
        return np.vstack([y_prime_scores_other, y_prime_scores_superior]).T

    # Generate and visualize explanation from the attack model
    # Pass the wrapper function to the 'model' argument
    plot_global_lime_mean_abs_from_scratch(
        x_data=x_test,
        model=attack_model_wrapper_for_lime,  # Corrected line
        feature_names=feature_names,
        categorical_features=categorical_features,
        superior_outcome_value=superior_outcome_value,
        protected_feature=protected_feature,
        top_n=5
    )

    plt.tight_layout()
    plt.show()





# --- New LIME Specific Functions for Attack and Explanation ---

def LIME_Explanation(x_test: pd.DataFrame, base_model: object, protected_feature: str,
                     feature_names: 'list[str]', categorical_features: 'list[str]',
                     superior_outcome_value: int, random_state: int = 42) -> None:
    """
    Generates and prints LIME explanations for a base model.
    """
    print("\n" + "="*50)
    print("LIME Explanations for the BASE Model (f)")
    print("="*50)

    # LIME explainer setup
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=x_test.values,
        feature_names=feature_names,
        class_names=['other', 'superior'],
        categorical_features=[feature_names.index(feat) for feat in categorical_features],
        mode='classification',
        random_state=random_state
    )

    # Select an instance to explain (e.g., the first instance in x_test)
    instance_to_explain_idx = 0
    instance_to_explain = x_test.iloc[instance_to_explain_idx]

    # Generate and visualize explanation
    print(f"\nExplaining instance with ID {instance_to_explain.name}")
    explanation = lime_explainer.explain_instance(
        data_row=instance_to_explain.values,
        predict_fn=base_model.predict_proba,
        num_features=5,
        labels=[superior_outcome_value]
    )
    explanation.show_in_notebook(show_table=True)


def run_attack_analysis_LIME(x_test: pd.DataFrame, base_model: object, feature_names: 'list[str]',
                             categorical_features: 'list[str]', superior_outcome_value: int, 
                             protected_feature: str, attack_type: str, attack_params: 'dict' = None,
                             random_state: int = 42) -> None:
    """
    Runs the output shuffling attack on a LIME explainer and visualizes the results.
    """
    if attack_type == 'none':
        print("No attack type specified. Running base model explanation only.")
        LIME_Explanation(x_test, base_model, protected_feature, feature_names, categorical_features, superior_outcome_value, random_state)
        return

    print("\n" + "="*50)
    print(f"LIME Explanations for the ATTACK Model (f') - {attack_type.capitalize()} Attack")
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

    # Wrapper function for the attacked model
    def attack_model_wrapper_for_lime(X_array: np.ndarray) -> np.ndarray:
        # LIME perturbs data and gives a numpy array
        X_df = pd.DataFrame(X_array, columns=feature_names, index=range(len(X_array)))
        
        # We need the full probability array for LIME's `predict_fn`
        y_prime_scores_superior = f_prime_template(
            X_data=X_df,
            protected_feature=protected_feature,
            base_model=base_model,
            attack_function=attack_func,
            superior_outcome_value=superior_outcome_value
        )[0]
        
        # Create a dummy score for the 'other' class
        y_prime_scores_other = 1 - y_prime_scores_superior
        
        return np.vstack([y_prime_scores_other, y_prime_scores_superior]).T

    # LIME explainer setup for the attack model
    lime_explainer_attack = lime.lime_tabular.LimeTabularExplainer(
        training_data=x_test.values,
        feature_names=feature_names,
        class_names=['other', 'superior'],
        categorical_features=[feature_names.index(feat) for feat in categorical_features],
        mode='classification',
        random_state=random_state
    )

    # Select an instance to explain
    instance_to_explain_idx = 0
    instance_to_explain = x_test.iloc[instance_to_explain_idx]
    
    # Generate and visualize explanation from the attack model
    print(f"\nExplaining instance with ID {instance_to_explain.name}")
    explanation_attack = lime_explainer_attack.explain_instance(
        data_row=instance_to_explain.values,
        predict_fn=attack_model_wrapper_for_lime,
        num_features=5,
        labels=[superior_outcome_value]
    )
    explanation_attack.show_in_notebook(show_table=True)


# --- Re-using original visualization functions, they remain compatible ---
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


def visualize_all_attack_scores_parallel_coords(x_test: pd.DataFrame, base_model: object, protected_feature: str,
                                             superior_outcome_value: int = 1, sample_size: int = 500) -> go.Figure:
    """
    Visualizes the comparison of predicted scores from the base model and
    various attack scenarios using a single parallel coordinates plot with 4 axes.
    Lines are colored by protected status.
    """
    if len(x_test) > sample_size:
        x_vis = x_test.sample(sample_size, random_state=42).copy()
    else:
        x_vis = x_test.copy()

    base_scores = base_model.predict_proba(x_vis)[:, superior_outcome_value]
    
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
            df_scores[col] = 0.5

    dimensions = [
        dict(range=[0, 1], label='Base Score', values=df_scores['base_score']),
        dict(range=[0, 1], label='Dominance Attack Score', values=df_scores['dominance_attack_score']),
        dict(range=[0, 1], label='Mixing Attack Score', values=df_scores['mixing_attack_score']),
        dict(range=[0, 1], label='Swapping Attack Score', values=df_scores['swapping_attack_score'])
    ]
    
    unique_protected_vals = sorted(df_scores['protected_status'].unique())
    
    if len(unique_protected_vals) == 2:
        colorscale = [[0, '#8db581'], [1, '#f4b75e']]
    else:
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
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        width=1000,
        font=dict(family="Arial, sans-serif", size=10),
        hovermode='closest'
    )
    
    return fig


def visualize_all_attack_ranks_parallel_coords(x_test: pd.DataFrame, base_model: object, protected_feature: str,
                                              superior_outcome_value: int = 1, sample_size: int = 500) -> go.Figure:
    """
    Visualizes the changes in ID ranks across different attack scenarios using a single
    parallel coordinates plot with 4 axes.
    Lines are colored by protected status.
    """
    if len(x_test) > sample_size:
        x_vis = x_test.sample(sample_size, random_state=42).copy()
    else:
        x_vis = x_test.copy()

    df_ranks = pd.DataFrame({
        'id': x_vis.index.values,
        'protected_status': x_vis[protected_feature].values
    }).set_index('id')

    _, _, original_sorted_ids_full = f_prime_template(
        X_data=x_vis,
        protected_feature=protected_feature,
        base_model=base_model,
        attack_function=lambda id_y, p_y, y_y: id_y,
        superior_outcome_value=superior_outcome_value
    )
    
    original_rank_map = {id_val: i + 1 for i, id_val in enumerate(original_sorted_ids_full)}
    df_ranks['original_rank'] = df_ranks.index.map(original_rank_map)

    attack_info = {
        'dominance': get_dominance_attack_info(x_vis, protected_feature, base_model, superior_outcome_value),
        'mixing': get_mixing_attack_info(x_vis, protected_feature, base_model, superior_outcome_value),
        'swapping': get_swapping_attack_info(x_vis, protected_feature, base_model, superior_outcome_value)
    }

    for attack_name, (scores, new_id_order, _) in attack_info.items():
        attack_rank_map = {id_val: i + 1 for i, id_val in enumerate(new_id_order)}
        df_ranks[f'{attack_name}_rank'] = df_ranks.index.map(attack_rank_map)

    dimensions = [
        dict(
            range=[1, sample_size],
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
    
    if len(unique_protected_vals) == 2:
        colorscale = [[0, '#8db581'], [1, '#f4b75e']]
    else:
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
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        width=1000,
        font=dict(family="Arial, sans-serif", size=10),
        hovermode='closest'
    )
    
    return fig