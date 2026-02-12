"""
LSTM Predictor Template.

Template for teammates implementing LSTM models.
Inherits from BasePredictor for consistent interface.

REQUIREMENTS:
    pip install tensorflow  # or torch

Usage:
    model = LSTMPredictor()
    model.fit(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
    predictions = model.predict(X_test_seq)
"""

from typing import Any, Dict, Optional

import numpy as np

from ..base import BasePredictor


class LSTMPredictor(BasePredictor):
    """
    LSTM Neural Network for time series prediction.

    TEMPLATE - Implement the methods below.

    Input shape: (samples, sequence_length, n_features)
    Output: predictions array

    Usage:
        from src.data_loader import DataLoader
        from src.config import TargetType

        loader = DataLoader(ticker="AAPL", target=TargetType.PRICE)
        data = loader.load()

        model = LSTMPredictor()
        model.fit(
            data.X_train_seq, data.y_train_seq,
            data.X_val_seq, data.y_val_seq
        )
        predictions = model.predict(data.X_test_seq)
    """

    def __init__(
        self,
        name: str = "LSTM",
        units: int = 50,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
    ):
        super().__init__(name)
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.history: Optional[Any] = None

    def _build_model(self, input_shape: tuple) -> Any:
        """
        Build LSTM model architecture.

        IMPLEMENT THIS METHOD.

        Example with Keras:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout

            model = Sequential([
                LSTM(self.units, return_sequences=True, input_shape=input_shape),
                Dropout(self.dropout),
                LSTM(self.units),
                Dropout(self.dropout),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            return model

        Example with PyTorch:
            import torch.nn as nn

            class LSTMModel(nn.Module):
                def __init__(self, input_size, hidden_size):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                    self.fc = nn.Linear(hidden_size, 1)
                ...
            return LSTMModel(input_shape[1], self.units)
        """
        raise NotImplementedError("Implement _build_model() for your LSTM architecture")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Train LSTM model.

        IMPLEMENT THIS METHOD.

        Args:
            X_train: shape (samples, sequence_length, n_features)
            y_train: shape (samples,)
            X_val: validation features
            y_val: validation targets

        Returns:
            Dictionary with training metrics

        Example:
            self.model = self._build_model(X_train.shape[1:])
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=1
            )
            self.is_fitted = True
            return {"train_loss": self.history.history['loss'][-1]}
        """
        raise NotImplementedError("Implement fit() method")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        IMPLEMENT THIS METHOD.

        Args:
            X: shape (samples, sequence_length, n_features)

        Returns:
            predictions array shape (samples,)

        Example:
            return self.model.predict(X).flatten()
        """
        raise NotImplementedError("Implement predict() method")

    def save(self, path: str) -> None:
        """
        Save model to file.

        IMPLEMENT THIS METHOD.

        Example (Keras):
            self.model.save(path)

        Example (PyTorch):
            torch.save(self.model.state_dict(), path)
        """
        raise NotImplementedError("Implement save() method")

    def load(self, path: str) -> None:
        """
        Load model from file.

        IMPLEMENT THIS METHOD.
        """
        raise NotImplementedError("Implement load() method")


# ============================================================================
# EXAMPLE IMPLEMENTATION (uncomment and modify as needed)
# ============================================================================
#
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
#
# class KerasLSTMPredictor(BasePredictor):
#     def __init__(self, name="LSTM-Keras", units=50, dropout=0.2, epochs=100):
#         super().__init__(name)
#         self.units = units
#         self.dropout = dropout
#         self.epochs = epochs
#
#     def fit(self, X_train, y_train, X_val=None, y_val=None):
#         model = Sequential([
#             LSTM(self.units, return_sequences=True, input_shape=X_train.shape[1:]),
#             Dropout(self.dropout),
#             LSTM(self.units),
#             Dropout(self.dropout),
#             Dense(1)
#         ])
#         model.compile(optimizer='adam', loss='mse')
#
#         callbacks = [EarlyStopping(patience=10, restore_best_weights=True)]
#         val_data = (X_val, y_val) if X_val is not None else None
#
#         history = model.fit(
#             X_train, y_train,
#             validation_data=val_data,
#             epochs=self.epochs,
#             batch_size=32,
#             callbacks=callbacks,
#             verbose=1
#         )
#
#         self.model = model
#         self.is_fitted = True
#         return {"train_loss": history.history['loss'][-1]}
#
#     def predict(self, X):
#         return self.model.predict(X).flatten()
#
#     def save(self, path):
#         self.model.save(path)
#
#     def load(self, path):
#         self.model = load_model(path)
#         self.is_fitted = True
