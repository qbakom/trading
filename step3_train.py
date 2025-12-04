import pandas as pd
import pandas_ta as ta
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Przygotuj dane (skopiuj z step1.py)
df = yf.download('AAPL', start='2024-01-01', auto_adjust=False)
df.columns = df.columns.get_level_values(0)
df['RSI'] = df.ta.rsi(length=14)
df['SMA_20'] = df.ta.sma(length=20)
df['ATR'] = df.ta.atr(length=14)
for lag in [1, 2, 3, 5, 10]:
    df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
    df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

# Wybierz tylko cechy (usuń target i oryginalne OHLCV)
feature_cols = [col for col in df.columns if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
X = df[feature_cols]
y = df['Target']

# Podziel na train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Trenuj model
model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

# Ewaluacja
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: ${mae:.2f}")
print(f"RMSE: ${rmse:.2f}")
print(f"\nFeatures used: {feature_cols}")
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
