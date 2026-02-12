"""
Model implementations for Time Series Prediction.

Available models:
- XGBoostPredictor: XGBoost regressor for price/return prediction
- MetaFilter: Random Forest classifier for signal filtering
- LSTMPredictor: LSTM neural network (template for teammates)
"""

from .xgboost_predictor import XGBoostPredictor
from .meta_filter import MetaFilter

__all__ = ["XGBoostPredictor", "MetaFilter"]
