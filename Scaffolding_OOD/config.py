# config.py

import numpy as np

# Global Parameters for the Attack Demonstration

# Random Seed for reproducibility
RANDOM_SEED = 42

# Outcome definitions for binary classification
POSITIVE_OUTCOME = 1
NEGATIVE_OUTCOME = 0

# Values for the synthetic sensitive feature
# PROTECTED_CLASS_VALUE will be the group that the biased model favors or disfavors
PROTECTED_CLASS_VALUE = 1
UNPROTECTED_CLASS_VALUE = 0

# --- Data Generation Parameters ---
N_SAMPLES_DATA = 1000
BIAS_CORRELATION_STRENGTH = 0.9 # How strongly the sensitive feature correlates with the target
BIAS_FAVOR_GROUP = POSITIVE_OUTCOME # Which target outcome the protected group is favored for

# --- OOD Generation Parameters ---
LIME_PERTURBATION_STD = 0.3 # Standard deviation for Gaussian noise in LIME-style perturbations
LIME_PERTURBATION_MULTIPLIER = 30 # How many times to perturb each original sample for OOD training

SHAP_N_SAMPLES_OOD = 20000 # Number of OOD samples to generate for SHAP-style training
SHAP_N_KMEANS_BACKGROUND = 10 # Number of clusters for SHAP background distribution

# --- OOD Detector Parameters ---
OOD_RF_ESTIMATORS = 100 # Number of estimators for the RandomForestClassifier in OOD detector
OOD_PRED_THRESHOLD = 0.5 # Probability threshold for OOD detector to classify as OOD

# --- Model Training Parameters ---
TEST_SIZE = 0.2 # Proportion of data to use for testing