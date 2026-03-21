from dataclasses import dataclass

@dataclass
class XGBoostParams:
    '''Configured with conservative hyperparameters to prevent overfitting
    on small datasets (~70 samples)'''

    n_estimators: int = 500          # Max trees — early stopping finds the optimal number
    max_depth: int = 3               # Shallow trees to prevent overfitting
    learning_rate: float = 0.05      # Moderate learning rate
    min_child_weight: int = 3        # Higher value = more conservative
    subsample: float = 0.8           # Observation percentage used for each tree
    colsample_bytree: float = 0.8    # Feature percentage used for each tree
    reg_alpha : float = 1            # L1 regularization
    reg_lambda: float = 8            # L2 regularization
    random_state: int = 42
    gamma: float = 2
    early_stopping_rounds: int = 0  # Stop if no improvement for 20 rounds


# Search space for hyperparameter tuning
# n_estimators excluded: with early stopping it's just a ceiling, not a meaningful tunable param
XGBOOST_PARAM_GRID = {
    "n_estimators": [200, 500],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.05],
    "min_child_weight": [1, 3],
}