import pandas as pd
from xgboost import XGBClassifier 

# --- Model Training Function ---
def base_model(X_train: pd.DataFrame, y_train: pd.Series, protected_feature: str, keep_protected_feature: bool = False):
    X_train_mod = X_train.copy()
    # Zeroing out the protected feature to simulate fair model training
    if not keep_protected_feature:
        X_train_mod[protected_feature] = 0 
    
    model = XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=0.1,
        objective='binary:logistic',
        random_state=42
    )
    model.fit(X_train_mod, y_train)
    return model