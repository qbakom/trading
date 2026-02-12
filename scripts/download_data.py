"""
Download and cache financial data from Yahoo Finance.
Run this script once to download data, then notebooks can load from cache.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
import time

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

TICKERS = {
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "EURUSD": "EURUSD=X",
    "BTC-USD": "BTC-USD",
    "GC": "GC=F",
    "VXN": "^VXN",
    "DXY": "DX-Y.NYB",
}

START_DATE = "1990-01-01"


def download_ticker(symbol: str, name: str) -> pd.DataFrame | None:
    """Download a single ticker with retry logic."""
    print(f"Downloading {name} ({symbol})...")

    for attempt in range(3):
        try:
            df = yf.download(
                symbol,
                start=START_DATE,
                auto_adjust=True,
                progress=False,
            )

            # Clean MultiIndex if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if len(df) > 0:
                print(f"  OK: {len(df)} rows")
                return df
            else:
                print(f"  Empty result, retrying...")

        except Exception as e:
            print(f"  Error: {e}, retrying...")

        time.sleep(2)  # Wait before retry

    print(f"  FAILED after 3 attempts")
    return None


def main():
    print("=" * 60)
    print("Downloading financial data from Yahoo Finance")
    print("=" * 60)

    all_data = {}

    for name, symbol in TICKERS.items():
        df = download_ticker(symbol, name)
        if df is not None:
            all_data[name] = df

            # Save individual ticker
            path = DATA_DIR / f"{name}.csv"
            df.to_csv(path)
            print(f"  Saved: {path}")

        time.sleep(1)  # Rate limiting

    # Build combined signals dataframe (like in notebook)
    if "AAPL" in all_data:
        print("\nBuilding combined signals dataframe...")

        signals = pd.DataFrame(all_data["AAPL"]["Close"])
        signals.columns = ["Close"]

        # Technical indicators
        signals["EMA50"] = signals["Close"].ewm(span=50, adjust=False).mean()

        # Bollinger Bands %B
        sma20 = signals["Close"].rolling(20).mean()
        std20 = signals["Close"].rolling(20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        signals["BB_PB"] = (signals["Close"] - lower) / (upper - lower)

        # Macro indicators
        if "VXN" in all_data:
            signals["VXN"] = all_data["VXN"]["Close"]
            signals["VXN"] = signals["VXN"].ffill()

        if "DXY" in all_data:
            signals["DXY"] = all_data["DXY"]["Close"]
            signals["DXY"] = signals["DXY"].ffill()

        # Save combined
        path = DATA_DIR / "signals_combined.csv"
        signals.to_csv(path)
        print(f"Saved combined signals: {path}")
        print(f"  Shape: {signals.shape}")
        print(f"  Columns: {list(signals.columns)}")
        print(f"  Date range: {signals.index.min()} to {signals.index.max()}")

    print("\n" + "=" * 60)
    print("DONE - Data saved to data/ directory")
    print("=" * 60)


if __name__ == "__main__":
    main()
