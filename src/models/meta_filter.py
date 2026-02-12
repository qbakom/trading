"""
Meta-Filter implementation.

Decision filter that learns when to trust base model predictions.
"""

from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import GridSearchCV


class MetaFilter:
    """
    Meta-Model (Random Forest Classifier) that learns when to trust base model.

    Target: 1 if base model predicted direction correctly, 0 otherwise

    Usage:
        meta = MetaFilter()
        X_meta = meta.prepare_features(df, predictions)
        y_meta = meta.create_target(df, predictions)
        meta.fit(X_meta, y_meta)
        signals = meta.predict(X_meta_test)
    """

    # Features used by meta-model
    META_FEATURES = [
        # Base features
        "VXN", "DXY", "BB_PB", "EMA50", "Close",
        # Prediction features (added dynamically)
        "Predicted_Price", "Implied_Signal", "Model_Confidence",
    ]

    def __init__(self, name: str = "MetaFilter"):
        self.name = name
        self.model: Optional[RandomForestClassifier] = None
        self.best_params: Dict[str, Any] = {}
        self.feature_names: list[str] = []
        self.is_fitted: bool = False

    def prepare_features(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
    ) -> pd.DataFrame:
        """
        Prepare features for meta-model.

        Args:
            df: Full dataframe with base features and volatility
            predictions: Base model predictions

        Returns:
            DataFrame with meta-model features
        """
        meta_df = df.copy()

        # Add prediction features
        meta_df["Predicted_Price"] = predictions

        # Implied signal: 1 if predicted UP, 0 if DOWN
        meta_df["Implied_Signal"] = (
            meta_df["Predicted_Price"] > meta_df["Prev_Close"]
        ).astype(int)

        # Model confidence: how far prediction is from current price (%)
        meta_df["Model_Confidence"] = (
            (meta_df["Predicted_Price"] - meta_df["Prev_Close"]).abs()
            / meta_df["Prev_Close"]
        )

        # Select available features
        available_features = [f for f in self.META_FEATURES if f in meta_df.columns]
        self.feature_names = available_features

        return meta_df[available_features]

    def create_target(
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

    def fit(
        self,
        X_meta: pd.DataFrame,
        y_meta: pd.Series,
        param_grid: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Train Random Forest meta-model.

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

        print(f"\nTraining {self.name} with GridSearchCV...")
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=3,
            scoring="precision",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_meta, y_meta)

        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.is_fitted = True

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
        Generate binary predictions (1=trust, 0=don't trust).

        Args:
            X_meta: Meta-model features

        Returns:
            Array of binary predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(X_meta)

    def predict_proba(self, X_meta: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X_meta: Meta-model features

        Returns:
            Array of probabilities for class 1 (trust)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
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
        base_accuracy = y_meta.mean()

        return {
            "precision": precision,
            "accuracy": accuracy,
            "base_accuracy": base_accuracy,
            "precision_improvement": precision - base_accuracy,
        }

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance from trained model."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_names,
        ).sort_values(ascending=False)

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
        print(f"Meta-model saved to: {path}")

    def load(self, path: str) -> None:
        """Load model from file."""
        save_data = joblib.load(path)
        self.model = save_data["model"]
        self.best_params = save_data["best_params"]
        self.feature_names = save_data["feature_names"]
        self.name = save_data.get("name", "MetaFilter")
        self.is_fitted = True
        print(f"Meta-model loaded from: {path}")
