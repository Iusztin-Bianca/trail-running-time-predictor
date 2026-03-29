"""
Base model abstract class.
All prediction models must inherit this class to ensure interchangeability.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all prediction models.

    Implements the Strategy pattern - each concrete model provides
    its own implementation of training and prediction.
    """

    def __init__(self, name: str, hyperparameters):
        self.name = name
        self.hyperparameters = hyperparameters
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model on the given data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self.is_fitted})"

