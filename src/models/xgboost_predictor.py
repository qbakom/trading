"""
XGBoost Predictor implementation.

Inherits from BasePredictor for consistent interface across team models.
"""

from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from ..base import BasePredictor


class XGBoostPredictor(BasePredictor):
    """
    XGBoost Regressor for time series prediction.

    Implements BasePredictor interface for team compatibility.

    Usage:
        model = XGBoostPredictor()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = model.evaluate(X_test, y_test, prev_close)
    """

    def __init__(self, name: str = "XGBoost"):
        super().__init__(name)
        self.best_params: Dict[str, Any] = {}
        self.cv_results: Optional[pd.DataFrame] = None

    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_val: Optional[np.ndarray | pd.DataFrame] = None,
        y_val: Optional[np.ndarray | pd.Series] = None,
        n_splits: int = 5,
        param_grid: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Train XGBoost with GridSearchCV using TimeSeriesSplit.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (unused, for interface compatibility)
            y_val: Validation targets (unused, for interface compatibility)
            n_splits: Number of CV folds
            param_grid: Hyperparameter grid

        Returns:
            Dictionary with training metrics
        """
        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)

        # Default parameter grid
        if param_grid is None:
            param_grid = {
                "n_estimators": [200, 400],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.7, 0.9],
                "colsample_bytree": [0.7, 0.9],
                "min_child_weight": [1, 5],
                "reg_lambda": [1.0, 5.0],
            }

        # TimeSeriesSplit preserves temporal order
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Base model
        xgb = XGBRegressor(
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
        )

        # Grid search
        print(f"Training {self.name} with GridSearchCV...")
        grid_search = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.cv_results = pd.DataFrame(grid_search.cv_results_)
        self.is_fitted = True

        # Training metrics
        y_pred_train = self.model.predict(X_train)
        train_rmse = np.sqrt(np.mean((y_train - y_pred_train) ** 2))
        cv_rmse = -grid_search.best_score_

        print(f"\nBest parameters: {self.best_params}")
        print(f"CV RMSE: {cv_rmse:.4f}")
        print(f"Train RMSE: {train_rmse:.4f}")

        return {
            "cv_rmse": cv_rmse,
            "train_rmse": train_rmse,
            "best_params": self.best_params,
        }

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def save(self, path: str) -> None:
        """Save model to file."""
        if not self.is_fitted:
            raise RuntimeError("No model to save.")

        save_data = {
            "model": self.model,
            "best_params": self.best_params,
            "feature_names": self.feature_names,
            "name": self.name,
        }
        joblib.dump(save_data, path)
        print(f"Model saved to: {path}")

    def load(self, path: str) -> None:
        """Load model from file."""
        save_data = joblib.load(path)
        self.model = save_data["model"]
        self.best_params = save_data["best_params"]
        self.feature_names = save_data["feature_names"]
        self.name = save_data.get("name", "XGBoost")
        self.is_fitted = True
        print(f"Model loaded from: {path}")

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance from trained model."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_names,
        ).sort_values(ascending=False)
