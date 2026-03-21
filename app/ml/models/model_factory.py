"""Factory for creating prediction models."""

import logging

from app.ml.config.ridge_params import RIDGE_PARAM_GRID, RidgeRegressionParams
from app.ml.config.xgboost_params import XGBOOST_PARAM_GRID, XGBoostParams
from app.ml.models.base_model import BaseModel
from app.ml.models.ridge_model import RidgeModel
from app.ml.models.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)

# Default hyperparameters for each model
_DEFAULT_PARAMS = {
    "ridge": RidgeRegressionParams,
    "xgboost": XGBoostParams,
}


class ModelFactory:
    """ Creates prediction model instances by name """

    _models = {
        "ridge": RidgeModel,
        "xgboost": XGBoostModel,
    }

    # Hyperparameter search grids, co-located with model registration.
    # Adding a new model: register it in _models AND provide its grid here.
    _param_grids: dict[str, dict] = {
        "ridge": RIDGE_PARAM_GRID,
        "xgboost": XGBOOST_PARAM_GRID,
    }

    @classmethod
    def create(cls, model_name: str, hyperparameters=None) -> BaseModel:
        """Create a model by name. If hyperparameters are not provided, use the default ones."""
        name = model_name.lower()

        if name not in cls._models:
            raise ValueError(f"Unknown model: '{model_name}'")

        if hyperparameters is None:
            hyperparameters = _DEFAULT_PARAMS[name]()

        model = cls._models[name](hyperparameters=hyperparameters)
        logger.info("Created model: %s", model)
        return model

    @classmethod
    def available_models(cls) -> list[str]:
        """Return list of available model names."""
        return list(cls._models.keys())

    @classmethod
    def get_param_grid(cls, model_name: str) -> dict:
        """Return the hyperparameter search grid for the given model."""
        name = model_name.lower()
        if name not in cls._param_grids:
            raise ValueError(f"No param grid registered for model: '{model_name}'")
        return cls._param_grids[name]
