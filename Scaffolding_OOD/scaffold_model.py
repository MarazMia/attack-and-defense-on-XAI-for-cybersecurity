# scaffold_model.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from copy import deepcopy

from sklearn.metrics import f1_score 

from ood_detector import OODDetector
# Import the functions directly, no need for the config object here
from ood_generator import generate_lime_style_ood, generate_shap_style_ood, combine_ood_data_for_detector_training
# Import config variables directly
from config import RANDOM_SEED, OOD_PRED_THRESHOLD, LIME_PERTURBATION_STD, LIME_PERTURBATION_MULTIPLIER, SHAP_N_SAMPLES_OOD, SHAP_N_KMEANS_BACKGROUND

class Adversarial_Model(object):
    def __init__(self, f_obscure, psi_display):
        self.f_obscure = f_obscure
        self.psi_display = psi_display
        self.perturbation_identifier: OODDetector = None
        self.feature_names = None
        self.ood_pred_threshold = OOD_PRED_THRESHOLD
        # self.ood_stats = {'total_samples': 0, 'ood_count': 0} 
        
        self._reset_ood_stats()  # Initialize stats tracking
    
    def _reset_ood_stats(self):
        """Reset OOD statistics counters"""
        self.ood_stats = {
            'last_batch_size': 0,
            'last_ood_count': 0,
            'cumulative_samples': 0,
            'cumulative_ood': 0
        } # Track OOD statistics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.perturbation_identifier is None:
            raise RuntimeError("Adversarial model is not trained yet. Call .train() first.")

        predictions_to_obscure = self.f_obscure.predict_proba(X)
        predictions_to_explain_by = self.psi_display.predict_proba(X)
        
        ood_probs = self.perturbation_identifier.predict_proba(X)
        is_in_distribution = (ood_probs[:, 1] >= self.ood_pred_threshold)
        
        # Update OOD statistics
        # batch_ood_count = np.sum(~is_in_distribution)
        # self.ood_stats['total_samples'] += X.shape[0]
        # self.ood_stats['ood_count'] += batch_ood_count
        

        # Update stats for the current batch only
        self.ood_stats['last_batch_size'] = X.shape[0]
        self.ood_stats['last_ood_count'] = np.sum(~is_in_distribution)
        self.ood_stats['cumulative_samples'] += X.shape[0]
        self.ood_stats['cumulative_ood'] += self.ood_stats['last_ood_count']


        sol = np.where(is_in_distribution[:, np.newaxis], predictions_to_obscure, predictions_to_explain_by)
        return sol

    def print_ood_stats(self, mode='last'):
        """
        Print OOD statistics.
        
        Parameters:
        -----------
        mode : str ('last'|'cumulative')
            'last' - shows only the most recent prediction batch
            'cumulative' - shows all predictions since last reset
        """
        if mode == 'last':
            total = self.ood_stats['last_batch_size']
            ood = self.ood_stats['last_ood_count']
        else:
            total = self.ood_stats['cumulative_samples']
            ood = self.ood_stats['cumulative_ood']
            
        if total == 0:
            print("No samples processed yet")
            return
            
        pct = (ood / total) * 100
        print("\n=== OOD Detection Report ===")
        print(f"Mode: {mode}")
        print(f"Samples processed: {total}")
        print(f"OOD samples detected: {ood} ({pct:.2f}%)")
        print("===========================")


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Scikit-learn style prediction. Follows from predict_proba.

        Parameters
        ----------
        X : np.ndarray

        Returns
        ----------
        A numpy array containing the binary class predictions.
        """
        pred_probs = self.predict_proba(X)
        return np.argmax(pred_probs, axis=1)

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """ Scikit-learn style accuracy scoring. """
        return np.sum(self.predict(X_test) == y_test) / y_test.size

    def get_feature_names(self) -> list:
        """ Access feature names."""
        if self.feature_names is None:
            raise RuntimeError("Model has not been trained with feature names.")
        return self.feature_names

    def fidelity(self, X: np.ndarray) -> float:
        """
        Get the fidelity of the adversarial model to the original predictions.
        High fidelity means that we're predicting f_obscure along the in-distribution data.

        Parameters:
        ----------
        X : np.ndarray

        Returns:
        ----------
        The fidelity score of the adversarial model's predictions to the model you're trying to obscure's predictions.
        """
        # This fidelity measures how often the adversarial model's prediction
        # matches the f_obscure model's prediction on the given X.
        # This is important to ensure the adversarial model doesn't change the
        # actual behavior on real data too much.
        return np.sum(self.predict(X) == self.f_obscure.predict(X)) / X.shape[0]

class Adversarial_Lime_Model(Adversarial_Model):
    """
    Lime adversarial model. Generates an adversarial model for LIME style explainers
    using the Adversarial_Model base class.
    """
    def __init__(self, f_obscure, psi_display, perturbation_std: float = LIME_PERTURBATION_STD):
        super().__init__(f_obscure, psi_display)
        self.perturbation_std = perturbation_std

    def train(self, X_in_distribution: np.ndarray, feature_names: list,
              perturbation_multiplier: int = LIME_PERTURBATION_MULTIPLIER, # Keep for consistency, but not passed
              ood_detector_estimator=None):
        """
        Trains the adversarial LIME model by training its OOD detector.

        Parameters:
        ----------
        X_in_distribution : np.ndarray
            The original in-distribution data (e.g., X_train_scaled).
        feature_names : list
            List of feature names.
        perturbation_multiplier : int
            Multiplier for generating OOD samples for LIME. (This argument is now ignored
            by generate_lime_style_ood, which automatically matches in-distribution size).
        ood_detector_estimator : object, optional
            A scikit-learn compatible classifier for the OOD detector.
            If None, RandomForestClassifier is used.
        """
        # print(f"--- Adversarial LIME Model: Training ---")
        self.feature_names = feature_names

        # 1. Generate LIME-style OOD data
        # FIX: Removed 'multiplier' argument as generate_lime_style_ood no longer accepts it.
        X_ood_lime, _ = generate_lime_style_ood(
            X_in_distribution,
            feature_names,
            perturbation_std=self.perturbation_std # Pass perturbation_std
        )

        # 2. Combine in-distribution and OOD data for detector training
        X_combined_for_ood, y_combined_for_ood = combine_ood_data_for_detector_training(
            X_in_distribution, X_ood_lime
        )

        # 3. Train the OOD detector
        self.perturbation_identifier = OODDetector(estimator=ood_detector_estimator)
        self.perturbation_identifier.train(X_combined_for_ood, y_combined_for_ood)

        # print(f"Adversarial LIME Model training complete.")
        return self





class Adversarial_Kernel_SHAP_Model(Adversarial_Model):
    """
    SHAP adversarial model. Generates an adversarial model for SHAP style perturbations
    (KernelExplainer) using the Adversarial_Model base class.
    """
    def __init__(self, f_obscure, psi_display):
        super().__init__(f_obscure, psi_display)

    def train(self, X_in_distribution: np.ndarray, feature_names: list,
              background_distribution: np.ndarray = None,
              n_samples: int = SHAP_N_SAMPLES_OOD, # Keep for consistency, but not passed
              n_kmeans: int = SHAP_N_KMEANS_BACKGROUND,
              ood_detector_estimator=None):
        """
        Trains the adversarial SHAP model by training its OOD detector.

        Parameters:
        ----------
        X_in_distribution : np.ndarray
            The original in-distribution data (e.g., X_train_scaled).
        feature_names : list
            List of feature names.
        background_distribution : np.ndarray, optional
            Custom background distribution for SHAP-style OOD generation.
            If None, SHAP's kmeans is used on X_in_distribution.
        n_samples : int
            Number of OOD samples to generate for SHAP. (This argument is now ignored
            by generate_shap_style_ood, which automatically matches in-distribution size).
        n_kmeans : int
            Number of clusters for SHAP's kmeans background if no custom background is provided.
        ood_detector_estimator : object, optional
            A scikit-learn compatible classifier for the OOD detector.
            If None, RandomForestClassifier is used.
        """
        # print(f"--- Adversarial SHAP Model: Training ---")
        self.feature_names = feature_names

        # 1. Generate SHAP-style OOD data
        # FIX: Removed 'n_samples' argument as generate_shap_style_ood no longer accepts it.
        X_ood_shap, _ = generate_shap_style_ood(
            X_in_distribution,
            feature_names,
            n_kmeans=n_kmeans # Pass n_kmeans
        )

        # 2. Combine in-distribution and OOD data for detector training
        X_combined_for_ood, y_combined_for_ood = combine_ood_data_for_detector_training(
            X_in_distribution, X_ood_shap
        )

        # 3. Train the OOD detector
        self.perturbation_identifier = OODDetector(estimator=ood_detector_estimator)
        self.perturbation_identifier.train(X_combined_for_ood, y_combined_for_ood)

        # print(f"Adversarial SHAP Model training complete.")
        return self



class CombinedAdversarialModel:
    def __init__(self, f_obscure, psi_display):
        self.f_obscure = f_obscure
        self.psi_display = psi_display
        self.lime_model = Adversarial_Lime_Model(f_obscure, psi_display)
        self.shap_model = Adversarial_Kernel_SHAP_Model(f_obscure, psi_display)
        self.feature_names = None
        self.ood_pred_threshold = OOD_PRED_THRESHOLD

    def train(self, X_in_distribution, feature_names):
        self.feature_names = feature_names
        self.lime_model.train(X_in_distribution, feature_names)
        self.shap_model.train(X_in_distribution, feature_names)
        return self

    def predict_proba(self, X):
        # Get predictions from both models
        pred_obscure = self.f_obscure.predict_proba(X)
        pred_display = self.psi_display.predict_proba(X)
        
        # Get OOD probabilities from both detectors
        lime_probs = self.lime_model.perturbation_identifier.predict_proba(X)[:, 1]
        shap_probs = self.shap_model.perturbation_identifier.predict_proba(X)[:, 1]
        
        # Combine predictions (OOD if either detector says OOD)
        is_in_distribution = (lime_probs >= self.ood_pred_threshold) & (shap_probs >= self.ood_pred_threshold)
        
        return np.where(is_in_distribution[:, np.newaxis], pred_obscure, pred_display)

    # [Include all other existing methods from Adversarial_Model...]
    def predict(self, X):
        pred_probs = self.predict_proba(X)
        return np.argmax(pred_probs, axis=1)

    def score(self, X_test, y_test):
        return np.sum(self.predict(X_test) == y_test) / y_test.size
    
    def score_f1(self, X_test, y_test):
        y_test = np.asarray(y_test)
        predictions = self.predict(X_test)
        return f1_score(y_test, predictions, average='binary')

    def fidelity(self, X):
        return np.sum(self.predict(X) == self.f_obscure.predict(X)) / X.shape[0]

    def print_ood_stats(self, X):
        lime_probs = self.lime_model.perturbation_identifier.predict_proba(X)[:, 1]
        shap_probs = self.shap_model.perturbation_identifier.predict_proba(X)[:, 1]
        
        lime_ood = np.sum(lime_probs < self.ood_pred_threshold)
        shap_ood = np.sum(shap_probs < self.ood_pred_threshold)
        combined_ood = np.sum((lime_probs < self.ood_pred_threshold) | (shap_probs < self.ood_pred_threshold))
        
        print("\n=== Combined OOD Detection Report ===")
        print(f"Samples processed: {len(X)}")
        print(f"LIME OOD detected: {lime_ood} ({lime_ood/len(X):.2%})")
        print(f"SHAP OOD detected: {shap_ood} ({shap_ood/len(X):.2%})")
        print(f"Combined OOD detected: {combined_ood} ({combined_ood/len(X):.2%})")
        print("====================================")


