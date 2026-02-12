import pandas as pd
import pandas_ta as ta
import yfinance as yf

# 1. Pobierz dane
import os

# Define paths relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

CACHE_FILE = os.path.join(DATA_DIR, 'AAPL_cache.csv')

if os.path.exists(CACHE_FILE):
    print(f"Loading data from {CACHE_FILE}...")
    df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
else:
    print(f"Cache file not found at {CACHE_FILE}. Downloading...")
    df = yf.download('AAPL', start='2000-01-01', auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.to_csv(CACHE_FILE)

# Ensure columns are flat if not already (safeguard)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# 2. Policz wskaźniki
df['RSI'] = df.ta.rsi(length=14)
df['SMA_20'] = df.ta.sma(length=20)
df['ATR'] = df.ta.atr(length=14)

# 3. Lagi
for lag in [1, 2, 3, 5, 10]:
    df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
    df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)

# 4. Target
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

# Zapisz do CSV
df.to_csv(os.path.join(DATA_DIR, 'data_prepared.csv'))
print(f"Saved {len(df)} rows")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 3 rows:\n{df.head(3)}")
print(f"\nShape: {df.shape}")
