# ood_generator.py

import numpy as np
import pandas as pd
import shap
from config import RANDOM_SEED, LIME_PERTURBATION_STD, LIME_PERTURBATION_MULTIPLIER, SHAP_N_SAMPLES_OOD, SHAP_N_KMEANS_BACKGROUND

def generate_lime_style_ood(
    X_in_distribution: np.ndarray,
    feature_names: list,
    perturbation_std: float = None,
    # multiplier parameter is now effectively ignored, as we target X_in_distribution.shape[0] OOD samples
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates out-of-distribution (OOD) samples that mimic LIME's perturbation strategy.
    LIME typically adds Gaussian noise to numerical features.
    The number of OOD samples generated will be equal to X_in_distribution.shape[0].

    Args:
        X_in_distribution (np.ndarray): The original in-distribution data (e.g., X_train_scaled).
        feature_names (list): List of feature names (not directly used for generation, but kept for signature consistency).
        perturbation_std (float): Standard deviation of the Gaussian noise.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - X_ood_perturbed: The generated OOD (perturbed) samples.
            - y_ood_label: Labels for the OOD detector (0 for OOD, 1 for in-distribution).
    """
    if perturbation_std is None:
        perturbation_std = LIME_PERTURBATION_STD

    np.random.seed(RANDOM_SEED)
    print(f"--- OOD Generator: Generating LIME-style OOD data ---")

    # Generate exactly X_in_distribution.shape[0] OOD samples
    # by adding noise to each in-distribution sample once.
    perturbed_xtrain = np.random.normal(0, perturbation_std, size=X_in_distribution.shape)
    X_ood_perturbed = X_in_distribution + perturbed_xtrain

    # Label these as OOD (0) for the OOD detector
    y_ood_label = np.zeros(X_ood_perturbed.shape[0])

    print(f"Generated {X_ood_perturbed.shape[0]} LIME-style OOD samples (equal to in-distribution samples).")
    return X_ood_perturbed, y_ood_label

def generate_shap_style_ood(
    X_in_distribution: np.ndarray,
    feature_names: list,
    n_samples: int = None, # This will now be overridden to match X_in_distribution.shape[0]
    n_kmeans: int = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates out-of-distribution (OOD) samples that mimic SHAP's KernelExplainer
    perturbation strategy (feature substitution from a background distribution).
    The number of OOD samples generated will be equal to X_in_distribution.shape[0].

    Args:
        X_in_distribution (np.ndarray): The original in-distribution data.
        feature_names (list): List of feature names.
        n_samples (int): Number of OOD samples to generate.
        n_kmeans (int): Number of clusters for SHAP's background distribution.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - X_ood_substituted: The generated OOD (substituted) samples.
            - y_ood_label: Labels for the OOD detector (0 for OOD, 1 for in-distribution).
    """
    # Override n_samples to match the number of in-distribution samples
    n_samples_to_generate = X_in_distribution.shape[0]
    if n_kmeans is None:
        n_kmeans = SHAP_N_KMEANS_BACKGROUND

    np.random.seed(RANDOM_SEED)
    print(f"--- OOD Generator: Generating SHAP-style OOD data ---\n")

    # Ensure X_in_distribution is a contiguous NumPy array with float type
    X_in_distribution_cleaned = np.ascontiguousarray(X_in_distribution, dtype=np.float32)

    background_distribution = shap.kmeans(X_in_distribution_cleaned, n_kmeans).data

    new_instances = []
    # Generate exactly n_samples_to_generate number of substitutions
    for _ in range(int(n_samples_to_generate)):
        i = np.random.choice(X_in_distribution_cleaned.shape[0])
        point = np.copy(X_in_distribution_cleaned[i, :])

        for _ in range(X_in_distribution_cleaned.shape[1]):
            j = np.random.choice(X_in_distribution_cleaned.shape[1])
            point[j] = background_distribution[np.random.choice(background_distribution.shape[0]), j]

        new_instances.append(point)

    X_ood_substituted = np.vstack(new_instances)

    y_ood_label = np.zeros(X_ood_substituted.shape[0])

    print(f"Generated {X_ood_substituted.shape[0]} SHAP-style OOD samples (equal to in-distribution samples).")
    return X_ood_substituted, y_ood_label

def combine_ood_data_for_detector_training(
    X_in_distribution: np.ndarray,
    X_ood_samples: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Combines in-distribution data with generated OOD samples for training an OOD detector.
    Assumes X_ood_samples has already been generated to match X_in_distribution's size.

    Args:
        X_in_distribution (np.ndarray): The original in-distribution data.
        X_ood_samples (np.ndarray): The generated OOD samples (should be equal in count to X_in_distribution).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - X_combined: Combined features for OOD detector training.
            - y_combined: Labels (1 for in-distribution, 0 for OOD).
    """
    y_in_distribution_label = np.ones(X_in_distribution.shape[0])

    X_combined = np.vstack((X_in_distribution, X_ood_samples))
    y_combined = np.concatenate((y_in_distribution_label, np.zeros(X_ood_samples.shape[0]))).astype(int)

    return X_combined, y_combined