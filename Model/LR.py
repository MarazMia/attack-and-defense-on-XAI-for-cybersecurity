import pandas as pd
from sklearn.linear_model import LogisticRegression

def base_model(X_train: pd.DataFrame, y_train: pd.Series, protected_feature: str, keep_protected_feature: bool = False):
    X_train_mod = X_train.copy()
    # Zeroing out the protected feature to simulate fair model training
    if not keep_protected_feature:
        X_train_mod[protected_feature] = 0 
    
    model = LogisticRegression(max_iter=10000, random_state=42, solver='saga', tol=1e-4)
    model.fit(X_train_mod, y_train)
    return model



def base_model_():
    model = LogisticRegression(max_iter=10000, random_state=42, solver='saga', tol=1e-4)
    return model