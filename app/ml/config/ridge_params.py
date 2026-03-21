from dataclasses import dataclass

@dataclass
class RidgeRegressionParams:
    alpha: float = 1.0
    fit_intercept: bool = True
    solver: str = "auto"


# Search space for hyperparameter tuning
RIDGE_PARAM_GRID = {
    "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
}