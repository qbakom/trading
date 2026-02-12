"""
Legacy model definitions for Time Series Prediction.

Prefer using implementations from the `models/` package to keep the team
consistent:
- src.models.XGBoostPredictor
- src.models.MetaFilter
"""

from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from .data_loader import BASE_FEATURES


class XGBoostPredictor:
    """
    XGBoost Regressor for price prediction.

    Input: VXN, DXY, BB_PB, EMA50, Close
    Output: Predicted close price at t+1

    Attributes:
        model: Trained XGBRegressor instance
        best_params: Best hyperparameters from GridSearchCV
        feature_names: List of input feature names
    """

    def __init__(self):
        self.model: Optional[XGBRegressor] = None
        self.best_params: Dict[str, Any] = {}
        self.feature_names: List[str] = BASE_FEATURES

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_splits: int = 5,
        param_grid: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Train XGBoost with GridSearchCV using TimeSeriesSplit.

        Args:
            X_train: Training features
            y_train: Training targets (next day close)
            n_splits: Number of CV folds
            param_grid: Hyperparameter grid (optional)

        Returns:
            Dictionary with training metrics
        """
        # Store feature names from input
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

        # Grid search with RMSE scoring
        grid_search = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=1,
        )

        print("Training XGBoost with GridSearchCV...")
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        # Training metrics
        y_pred_train = self.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        cv_rmse = -grid_search.best_score_

        print(f"\nBest parameters: {self.best_params}")
        print(f"CV RMSE: {cv_rmse:.4f}")
        print(f"Train RMSE: {train_rmse:.4f}")

        return {
            "cv_rmse": cv_rmse,
            "train_rmse": train_rmse,
            "best_params": self.best_params,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate price predictions.

        Args:
            X: Features (OHLCV)

        Returns:
            Array of predicted prices
        """
        assert self.model is not None, "Model not trained. Call train() first."
        return self.model.predict(X)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        prev_close: pd.Series,
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            X: Test features
            y: Test targets (actual prices)
            prev_close: Previous day close (for direction calculation)

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)

        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        # Directional accuracy
        actual_dir = np.sign(y.values - prev_close.values)
        pred_dir = np.sign(y_pred - prev_close.values)
        dir_accuracy = accuracy_score(actual_dir, pred_dir)

        return {
            "rmse": rmse,
            "mae": mae,
            "directional_accuracy": dir_accuracy,
        }

    def save_model(self, path: str) -> None:
        """Save model to file."""
        assert self.model is not None, "No model to save"

        save_data = {
            "model": self.model,
            "best_params": self.best_params,
            "feature_names": self.feature_names,
        }
        joblib.dump(save_data, path)
        print(f"Model saved to: {path}")

    def load_model(self, path: str) -> None:
        """Load model from file."""
        save_data = joblib.load(path)
        self.model = save_data["model"]
        self.best_params = save_data["best_params"]
        self.feature_names = save_data["feature_names"]
        print(f"Model loaded from: {path}")


class MetaFilter:
    """
    Meta-Model (Random Forest Classifier) that learns when to trust base model.

    Target: 1 if base model predicted direction correctly, 0 otherwise

    Features:
    - Base features (VXN, DXY, BB_PB, EMA50, Close)
    - Base model prediction + implied signal + confidence
    """

    META_FEATURES = [
        # Base features
        "VXN", "DXY", "BB_PB", "EMA50", "Close",
        # Prediction features
        "Predicted_Price", "Implied_Signal", "Model_Confidence",
    ]

    def __init__(self):
        self.model: Optional[RandomForestClassifier] = None
        self.best_params: Dict[str, Any] = {}
        self.feature_names: List[str] = []

    def prepare_meta_features(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
    ) -> pd.DataFrame:
        """
        Prepare features for meta-model training/prediction.

        Args:
            df: Full dataframe with OHLCV and volatility features
            predictions: Base model predictions

        Returns:
            DataFrame with meta-model features
        """
        meta_df = df.copy()

        # Base model prediction
        meta_df["Predicted_Price"] = predictions

        # Implied signal: 1 if predicted UP, 0 if predicted DOWN
        meta_df["Implied_Signal"] = (
            meta_df["Predicted_Price"] > meta_df["Prev_Close"]
        ).astype(int)

        # Model confidence: how far prediction is from current price (%)
        meta_df["Model_Confidence"] = (
            (meta_df["Predicted_Price"] - meta_df["Prev_Close"]).abs()
            / meta_df["Prev_Close"]
        )

        # Select available features
        available_features = [
            f for f in self.META_FEATURES if f in meta_df.columns
        ]
        self.feature_names = available_features

        return meta_df[available_features]

    def create_meta_target(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
    ) -> pd.Series:
        """
        Create meta-model target.

        Target = 1 if base model predicted direction correctly, 0 otherwise.

        Args:
            df: Full dataframe with actual prices
            predictions: Base model predictions

        Returns:
            Series with binary meta-labels
        """
        # Actual direction
        actual_direction = (df["Target"] > df["Prev_Close"]).astype(int)

        # Predicted direction
        predicted_direction = (predictions > df["Prev_Close"].values).astype(int)

        # Meta target: 1 if prediction was correct
        meta_target = (actual_direction.values == predicted_direction).astype(int)

        return pd.Series(meta_target, index=df.index, name="Meta_Target")

    def train(
        self,
        X_meta: pd.DataFrame,
        y_meta: pd.Series,
        param_grid: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Train Random Forest meta-model with GridSearchCV.

        Args:
            X_meta: Meta-model features
            y_meta: Meta-model targets
            param_grid: Hyperparameter grid

        Returns:
            Training metrics
        """
        if param_grid is None:
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "min_samples_leaf": [5, 10, 20],
                "class_weight": ["balanced", None],
            }

        rf = RandomForestClassifier(random_state=42, n_jobs=-1)

        # TimeSeriesSplit to avoid look-ahead bias
        tscv = TimeSeriesSplit(n_splits=3)
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=tscv,
            scoring="precision",
            n_jobs=-1,
            verbose=1,
        )

        print("\nTraining Meta-Filter with GridSearchCV...")
        grid_search.fit(X_meta, y_meta)

        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        # Training metrics
        y_pred_train = self.model.predict(X_meta)
        train_precision = precision_score(y_meta, y_pred_train, zero_division=0)
        train_accuracy = accuracy_score(y_meta, y_pred_train)

        print(f"\nBest parameters: {self.best_params}")
        print(f"Train Precision: {train_precision:.2%}")
        print(f"Train Accuracy: {train_accuracy:.2%}")

        return {
            "train_precision": train_precision,
            "train_accuracy": train_accuracy,
            "best_params": self.best_params,
        }

    def predict(self, X_meta: pd.DataFrame) -> np.ndarray:
        """
        Generate meta-predictions (1=trust signal, 0=don't trust).

        Args:
            X_meta: Meta-model features

        Returns:
            Array of binary predictions
        """
        assert self.model is not None, "Model not trained. Call train() first."
        return self.model.predict(X_meta)

    def predict_proba(self, X_meta: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X_meta: Meta-model features

        Returns:
            Array of probabilities for class 1 (trust)
        """
        assert self.model is not None, "Model not trained"
        return self.model.predict_proba(X_meta)[:, 1]

    def evaluate(
        self,
        X_meta: pd.DataFrame,
        y_meta: pd.Series,
    ) -> Dict[str, float]:
        """
        Evaluate meta-model.

        Args:
            X_meta: Test features
            y_meta: Test targets

        Returns:
            Evaluation metrics
        """
        y_pred = self.predict(X_meta)

        precision = precision_score(y_meta, y_pred, zero_division=0)
        accuracy = accuracy_score(y_meta, y_pred)

        # Base accuracy (without filter)
        base_accuracy = y_meta.mean()

        return {
            "precision": precision,
            "accuracy": accuracy,
            "base_accuracy": base_accuracy,
            "precision_improvement": precision - base_accuracy,
        }

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance from trained model."""
        assert self.model is not None, "Model not trained"

        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_names,
        ).sort_values(ascending=False)

    def save_model(self, path: str) -> None:
        """Save model to file."""
        assert self.model is not None, "No model to save"

        save_data = {
            "model": self.model,
            "best_params": self.best_params,
            "feature_names": self.feature_names,
        }
        joblib.dump(save_data, path)
        print(f"Meta-model saved to: {path}")

    def load_model(self, path: str) -> None:
        """Load model from file."""
        save_data = joblib.load(path)
        self.model = save_data["model"]
        self.best_params = save_data["best_params"]
        self.feature_names = save_data["feature_names"]
        print(f"Meta-model loaded from: {path}")
