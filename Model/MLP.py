import pandas as pd
from sklearn.neural_network import MLPClassifier

def base_model(x_train: pd.DataFrame, y_train: pd.Series, protected_feature: str, 
              keep_protected_feature: bool = False) -> MLPClassifier:
    
    x_train_mod = x_train.copy()
    
    # Zeroing out the protected feature to simulate fair model training
    if not keep_protected_feature:
        x_train_mod[protected_feature] = 0 
        
    model = MLPClassifier(
        hidden_layer_sizes=(150, 75, 50), # Example: two hidden layers with 100 and 50 neurons
        activation='relu',            # Rectified Linear Unit activation
        solver='adam',                # Adam optimizer
        max_iter=200,                 # Max iterations for the solver to converge
        random_state=42,              # For reproducibility
        early_stopping=True,          # Stop if validation score doesn't improve
        n_iter_no_change=20,          # Number of iterations with no improvement to wait
        verbose=False                 # Set to True for training progress output
    )
    
    model.fit(x_train_mod, y_train)
    
    return model

