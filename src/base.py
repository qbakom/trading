"""
Base class for all predictors.

All team members should inherit from BasePredictor to ensure
consistent interface for model comparison.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class BasePredictor(ABC):
    """
    Abstract base class for all prediction models.

    All models (LSTM, XGBoost, etc.) must implement this interface
    to ensure consistent evaluation and comparison.

    Attributes:
        name: Model name for identification
        model: Underlying model instance
        is_fitted: Whether model has been trained
    """

    def __init__(self, name: str):
        self.name = name
        self.model: Optional[Any] = None
        self.is_fitted: bool = False
        self.feature_names: list[str] = []

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_val: Optional[np.ndarray | pd.DataFrame] = None,
        y_val: Optional[np.ndarray | pd.Series] = None,
    ) -> Dict[str, float]:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            Dictionary with training metrics
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Input features

        Returns:
            Array of predictions
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to file."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from file."""
        pass

    def evaluate(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        prev_values: Optional[np.ndarray | pd.Series] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            X: Test features
            y: Test targets
            prev_values: Previous values for direction calculation

        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score,
            mean_absolute_error,
            mean_squared_error,
        )

        y_pred = self.predict(X)
        y_true = np.array(y)

        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
        }

        # Directional accuracy (if prev_values provided)
        if prev_values is not None:
            prev = np.array(prev_values)
            actual_dir = np.sign(y_true - prev)
            pred_dir = np.sign(y_pred - prev)
            metrics["directional_accuracy"] = accuracy_score(actual_dir, pred_dir)

        return metrics

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.name}', {status})"
