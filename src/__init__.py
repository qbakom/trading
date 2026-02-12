"""
Time Series Prediction System

Reusable framework for time series prediction with multiple models.
Supports XGBoost, LSTM, and other models with consistent interface.

Usage:
    from src.data_loader import DataLoader
    from src.config import TargetType
    from src.models import XGBoostPredictor, MetaFilter

    # Load data
    loader = DataLoader(ticker="AAPL", target=TargetType.PRICE)
    data = loader.load()

    # XGBoost (tabular)
    xgb = XGBoostPredictor()
    xgb.fit(data.X_train, data.y_train)
    predictions = xgb.predict(data.X_test)

    # LSTM (sequences)
    lstm = LSTMPredictor()  # implement from template
    lstm.fit(data.X_train_seq, data.y_train_seq)
"""

__version__ = "2.0.0"
