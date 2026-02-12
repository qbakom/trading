"""
Data Loader for Time Series Prediction.

Supports loading price data, macro indicators, technical features,
and creating targets for different prediction tasks.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

from .config import DEFAULT_CONFIG, TargetType


# Feature columns (base model default)
FEATURES = ["VXN", "DXY", "BB_PB", "EMA50", "Close"]
# Backwards-compatible alias
BASE_FEATURES = FEATURES


@dataclass
class DataSplit:
    """Train/Val/Test splits."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    @property
    def X_train(self) -> pd.DataFrame:
        return self.train[FEATURES]

    @property
    def y_train(self) -> pd.Series:
        return self.train["Target"]

    @property
    def X_val(self) -> pd.DataFrame:
        return self.val[FEATURES]

    @property
    def y_val(self) -> pd.Series:
        return self.val["Target"]

    @property
    def X_test(self) -> pd.DataFrame:
        return self.test[FEATURES]

    @property
    def y_test(self) -> pd.Series:
        return self.test["Target"]


@dataclass
class LoadedData:
    """Container for dataset splits and feature configuration."""

    df_train: pd.DataFrame
    df_val: pd.DataFrame
    df_test: pd.DataFrame
    feature_columns: list[str]
    sequence_length: int = DEFAULT_CONFIG["sequence_length"]

    @property
    def X_train(self) -> pd.DataFrame:
        return self.df_train[self.feature_columns]

    @property
    def y_train(self) -> pd.Series:
        return self.df_train["Target"]

    @property
    def X_val(self) -> pd.DataFrame:
        return self.df_val[self.feature_columns]

    @property
    def y_val(self) -> pd.Series:
        return self.df_val["Target"]

    @property
    def X_test(self) -> pd.DataFrame:
        return self.df_test[self.feature_columns]

    @property
    def y_test(self) -> pd.Series:
        return self.df_test["Target"]

    def get_sequences(self, split: str = "train") -> tuple[np.ndarray, np.ndarray]:
        """Return LSTM-ready sequences for a split (train/val/test)."""
        df_map = {
            "train": self.df_train,
            "val": self.df_val,
            "test": self.df_test,
        }
        if split not in df_map:
            raise ValueError("split must be one of: train, val, test")
        return create_sequences(
            df_map[split],
            feature_columns=self.feature_columns,
            seq_length=self.sequence_length,
        )


class DataLoader:
    """
    High-level data loader used by the main pipeline.

    Provides train/val/test splits plus enriched features for meta-models.
    """

    def __init__(
        self,
        ticker: str = "AAPL",
        target: TargetType = TargetType.PRICE,
        feature_columns: Optional[Sequence[str]] = None,
        train_end: str = DEFAULT_CONFIG["train_end"],
        val_end: str = DEFAULT_CONFIG["val_end"],
        sequence_length: int = DEFAULT_CONFIG["sequence_length"],
        normalize_features: bool = False,
    ) -> None:
        self.ticker = ticker
        self.target = target
        self.feature_columns = list(feature_columns or DEFAULT_CONFIG["features"])
        self.train_end = train_end
        self.val_end = val_end
        self.sequence_length = sequence_length
        self.normalize_features = normalize_features

    def load(self) -> LoadedData:
        """Load and prepare data according to config."""
        print(f"Loading {self.ticker}...")

        df = _fetch_ohlcv(self.ticker)

        # Macro indicators
        vxn = _fetch_ohlcv("^VXN", "VXN")
        dxy = _fetch_ohlcv("DX-Y.NYB", "DXY")

        df["VXN"] = vxn["Close"]
        df["DXY"] = dxy["Close"]
        df["VXN"] = df["VXN"].ffill()
        df["DXY"] = df["DXY"].ffill()

        # Technical indicators
        df = _add_indicators(df)

        # Previous close (for directional accuracy & meta features)
        df["Prev_Close"] = df["Close"].shift(1)

        # Create target
        if self.target == TargetType.PRICE:
            df["Target"] = df["Close"].shift(-1)
        elif self.target == TargetType.RETURN:
            df["Target"] = (df["Close"].shift(-1) - df["Close"]) / df["Close"]
        elif self.target == TargetType.DIRECTION:
            df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        else:
            raise ValueError(f"Unsupported target: {self.target}")

        # Drop NaNs from indicators/target
        df = df.dropna()

        # Normalize features only (optional)
        if self.normalize_features:
            df[self.feature_columns] = df[self.feature_columns].pct_change().dropna()
            df = df.dropna()

        # Split by date
        df_train = df[df.index <= self.train_end].copy()
        df_val = df[(df.index > self.train_end) & (df.index <= self.val_end)].copy()
        df_test = df[df.index > self.val_end].copy()

        print(f"  Train: {len(df_train)} rows")
        print(f"  Val:   {len(df_val)} rows")
        print(f"  Test:  {len(df_test)} rows")

        return LoadedData(
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            feature_columns=self.feature_columns,
            sequence_length=self.sequence_length,
        )


def load_data(
    ticker: str = "AAPL",
    train_end: str = "2018-12-31",
    val_end: str = "2021-12-31",
    normalize: bool = True,
) -> DataSplit:
    """
    Load and prepare data.

    Args:
        ticker: Asset symbol
        train_end: End of training period
        val_end: End of validation period
        normalize: Whether to normalize data (percent change)

    Returns:
        DataSplit with train/val/test DataFrames
    """
    print(f"Loading {ticker}...")

    # 1. Fetch price data
    df = _fetch_ohlcv(ticker)

    # 2. Fetch macro indicators
    vdx = _fetch_ohlcv("^VXN", "VXN")
    dxy = _fetch_ohlcv("DX-Y.NYB", "DXY")

    df["VXN"] = vdx["Close"]
    df["DXY"] = dxy["Close"]
    df["VXN"] = df["VXN"].ffill()
    df["DXY"] = df["DXY"].ffill()

    # 3. Calculate technical indicators
    df = _add_indicators(df)

    # 4. Create target (next day close)
    df["Target"] = df["Close"].shift(-1)

    # 5. Drop NaN
    df = df.dropna()

    # 6. Select columns
    df = df[FEATURES + ["Target"]].copy()

    # 7. Normalize (percent change from previous day)
    if normalize:
        df = df.pct_change().dropna()

    # 8. Split by date
    train = df[df.index <= train_end].copy()
    val = df[(df.index > train_end) & (df.index <= val_end)].copy()
    test = df[df.index > val_end].copy()

    print(f"  Train: {len(train)} rows")
    print(f"  Val:   {len(val)} rows")
    print(f"  Test:  {len(test)} rows")
    print(f"  Columns: {list(df.columns)}")

    return DataSplit(train=train, val=val, test=test)


def _fetch_ohlcv(ticker: str, name: Optional[str] = None) -> pd.DataFrame:
    """Fetch OHLCV from Yahoo Finance."""
    df = yf.download(ticker, start="2000-01-01", progress=False, auto_adjust=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f"  Fetched {name or ticker}: {len(df)} rows")
    return df


def create_sequences(
    df: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    seq_length: int = 60,
):
    """
    Create sequences for LSTM.

    Args:
        df: DataFrame with features + Target
        seq_length: Number of timesteps (default 60 days)

    Returns:
        X: shape (samples, seq_length, n_features)
        y: shape (samples,)
    """
    feature_columns = list(feature_columns or FEATURES)
    features = df[feature_columns].values
    target = df["Target"].values

    X, y = [], []
    for i in range(seq_length, len(df)):
        X.append(features[i - seq_length:i])
        y.append(target[i])

    return np.array(X), np.array(y)


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators."""
    df = df.copy()

    # Bollinger Bands (20, 2)
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df["BB_PB"] = (df["Close"] - lower) / (upper - lower)

    # EMA50
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    return df


def export_results(
    dates: pd.Index,
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str,
    output_path: str,
    meta_signal: Optional[np.ndarray] = None,
) -> None:
    """Export predictions to CSV for later analysis."""
    out = pd.DataFrame(
        {
            "Date": dates,
            "Actual": actual,
            "Predicted": predicted,
            "Model": model_name,
        }
    )
    if meta_signal is not None:
        out["Meta_Signal"] = meta_signal

    out.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
