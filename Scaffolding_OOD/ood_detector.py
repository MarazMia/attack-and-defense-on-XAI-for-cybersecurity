# ood_detector.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Import the specific configuration variables needed from config.py
from config import RANDOM_SEED, OOD_RF_ESTIMATORS, TEST_SIZE


class OODDetector:
    """
    A classifier to detect Out-of-Distribution (OOD) samples.
    It's trained to distinguish between original in-distribution data and perturbed/synthetic OOD data.
    """
    def __init__(self, estimator=None, rf_estimators: int = OOD_RF_ESTIMATORS): # Use imported OOD_RF_ESTIMATORS
        """
        Initializes the OOD Detector.

        Args:
            estimator: A scikit-learn compatible classifier (e.g., RandomForestClassifier).
                       If None, RandomForestClassifier is used by default.
            rf_estimators (int): Number of trees if using RandomForestClassifier.
        """
        if estimator is None:
            # FIX: Added class_weight='balanced' to handle class imbalance
            self.detector = RandomForestClassifier(n_estimators=rf_estimators, random_state=RANDOM_SEED, class_weight='balanced') # Use imported RANDOM_SEED
        else:
            self.detector = estimator
        self.ood_training_task_report = None

    def train(self, X_combined_train: np.ndarray, y_combined_train: np.ndarray):
        """
        Trains the OOD detector.

        Args:
            X_combined_train (np.ndarray): Combined features (in-distribution + OOD samples).
            y_combined_train (np.ndarray): Labels (1 for in-distribution, 0 for OOD).
        """
        print(f"--- OOD Detector: Training detector ---")
        # Split for internal evaluation of the OOD detector's performance
        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
            X_combined_train, y_combined_train, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_combined_train # Use imported TEST_SIZE, RANDOM_SEED
        )

        self.detector.fit(X_train_split, y_train_split)

        # Evaluate and store performance
        y_pred_split = self.detector.predict(X_test_split)
        self.ood_training_task_report = classification_report(y_test_split, y_pred_split, output_dict=True)
        print(f"OOD Detector training complete. Accuracy: {self.ood_training_task_report['accuracy']:.4f}")
        print("OOD Detector Classification Report (1: In-Dist, 0: OOD):\n",
              classification_report(y_test_split, y_pred_split))


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the probability of samples being in-distribution (class 1) or OOD (class 0).

        Args:
            X (np.ndarray): Input samples to predict.

        Returns:
            np.ndarray: Probability array, where `[..., 0]` is P(OOD) and `[..., 1]` is P(In-Dist).
        """
        if self.detector is None:
            raise RuntimeError("OOD Detector has not been trained yet. Call .train() first.")
        return self.detector.predict_proba(X)

    def evaluate_training_ability(self):
        """
        Returns the classification report from the OOD detector's internal training evaluation.
        """
        return self.ood_training_task_report

# Example usage for testing (if run directly) - This part remains for local testing if needed
if __name__ == '__main__':
    # Dummy config for standalone testing
    # Note: When running main_attack_demo.py, this block is skipped.
    # It's only for testing ood_detector.py in isolation.
    class LocalConfig:
        RANDOM_SEED = 42
        OOD_RF_ESTIMATORS = 100
        TEST_SIZE = 0.2
        LIME_PERTURBATION_STD = 0.3
        LIME_PERTURBATION_MULTIPLIER = 3
        SHAP_N_SAMPLES_OOD = 100
        SHAP_N_KMEANS_BACKGROUND = 5
    # Temporarily override imported config values for local test
    RANDOM_SEED = LocalConfig.RANDOM_SEED
    OOD_RF_ESTIMATORS = LocalConfig.OOD_RF_ESTIMATORS
    TEST_SIZE = LocalConfig.TEST_SIZE
    LIME_PERTURBATION_STD = LocalConfig.LIME_PERTURBATION_STD
    LIME_PERTURBATION_MULTIPLIER = LocalConfig.LIME_PERTURBATION_MULTIPLIER
    SHAP_N_SAMPLES_OOD = LocalConfig.SHAP_N_SAMPLES_OOD
    SHAP_N_KMEANS_BACKGROUND = LocalConfig.SHAP_N_KMEANS_BACKGROUND


    # Import necessary functions for local testing
    # These imports are relative if ood_generator is in the same directory
    # or need to be adjusted based on your project structure for standalone runs
    from ood_generator import generate_lime_style_ood, combine_ood_data_for_detector_training

    # Create some dummy in-distribution data
    X_test_data = np.random.rand(100, 5) # 100 samples, 5 features
    feature_names_test = [f'feat_{i}' for i in range(5)]

    # Generate LIME OOD data for detector training
    X_lime_ood_test, _ = generate_lime_style_ood(X_test_data, feature_names_test)
    X_combined_test, y_combined_test = combine_ood_data_for_detector_training(X_test_data, X_lime_ood_test)

    # Initialize and train OOD detector
    ood_detector = OODDetector()
    ood_detector.train(X_combined_test, y_combined_test)

    # Test prediction on a sample
    sample_in_dist = X_test_data[0:1]
    sample_ood = X_lime_ood_test[0:1]

    ood_pred_threshold = 0.5 # Using a dummy threshold for this test
    print(f"\nProb of in-distribution for a real sample: {ood_detector.predict_proba(sample_in_dist)[0][1]:.4f}")
    print(f"Prob of in-distribution for an OOD sample: {ood_detector.predict_proba(sample_ood)[0][1]:.4f}")

    is_in_dist_real = ood_detector.predict_proba(sample_in_dist)[0][1] >= ood_pred_threshold
    is_in_dist_ood = ood_detector.predict_proba(sample_ood)[0][1] >= ood_pred_threshold
    print(f"Real sample classified as in-distribution: {is_in_dist_real}")
    print(f"OOD sample classified as in-distribution: {is_in_dist_ood}")