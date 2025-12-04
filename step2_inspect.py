import pandas as pd
import pandas_ta as ta
import yfinance as yf

# 1. Pobierz dane
df = yf.download('AAPL', start='2024-01-01', auto_adjust=False)
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
df.to_csv('data_prepared.csv')
print(f"Saved {len(df)} rows")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 3 rows:\n{df.head(3)}")
print(f"\nShape: {df.shape}")
