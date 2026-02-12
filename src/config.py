"""
Configuration module for Time Series Prediction System.

Defines available features, targets, and default settings.
All team members should use the same configuration for comparability.
"""

from enum import Enum
from typing import List


class TargetType(Enum):
    """Types of prediction targets."""
    PRICE = "price"           # Close[t+1] - absolute price
    RETURN = "return"         # (Close[t+1] - Close[t]) / Close[t]
    DIRECTION = "direction"   # 1 if Close[t+1] > Close[t], else 0


class FeatureSet(Enum):
    """Predefined feature sets for different use cases."""
    OHLCV = "ohlcv"
    TECHNICAL = "technical"
    MACRO = "macro"
    FULL = "full"


# Feature definitions
FEATURE_SETS = {
    FeatureSet.OHLCV: ["Open", "High", "Low", "Close", "Volume"],
    FeatureSet.TECHNICAL: ["BB_PB", "EMA50"],
    FeatureSet.MACRO: ["VXN", "DXY"],
    FeatureSet.FULL: ["VXN", "DXY", "BB_PB", "EMA50", "Close"],
}


# Default configuration (agreed by team)
DEFAULT_CONFIG = {
    "features": ["VXN", "DXY", "BB_PB", "EMA50", "Close"],
    "target": TargetType.PRICE,
    "train_end": "2018-12-31",
    "val_end": "2021-12-31",
    "sequence_length": 60,  # For LSTM: 60 days lookback
}


# Assets for evaluation
TICKERS = ["AAPL", "MSFT", "EURUSD=X", "BTC-USD", "GC=F"]
