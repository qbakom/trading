import pandas as pd
import pandas_ta as ta
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

# 1. Pobieramy WIĘCEJ danych (od 2018) + VIX (Indeks Strachu)
print("Pobieranie danych (AAPL + VIX)...")
df = yf.download('AAPL', start='2018-01-01', auto_adjust=False)
# Spłaszczamy MultiIndex
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Pobieramy VIX (Context)
vix = yf.download('^VIX', start='2018-01-01', auto_adjust=False)
if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = vix.columns.get_level_values(0)

# 2. Feature Engineering
df['RSI'] = df.ta.rsi(length=14)
df['SMA_50'] = df.ta.sma(length=50)
df['ATR'] = df.ta.atr(length=14)

# Dołączamy VIX do danych Apple (fill forward na wypadek dziur)
df['VIX'] = vix['Close']
df['VIX'] = df['VIX'].ffill()

# Feature: Czy VIX jest wysoki? (Powyżej 20 to strach, powyżej 30 to panika)
df['VIX_High'] = (df['VIX'] > 25).astype(int)

# Feature: Odległość od średniej (Mean Reversion)
df['Dist_SMA'] = (df['Close'] - df['SMA_50']) / df['SMA_50']

for lag in [1, 2, 3, 5]:
    df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)

df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

# 3. Podział Chronologiczny
feature_cols = ['RSI', 'ATR', 'VIX', 'Dist_SMA', 'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_5']
X = df[feature_cols]
y = df['Target']

# Bierzemy ostatni rok jako test
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# 4. Trening Modelu Bazowego
print("Trening modelu regresyjnego (XGBoost)...")
# Zwiększamy moc modelu, żeby miał chociaż 51% skuteczności
model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# 5. Generowanie wyników
results = X_test.copy()
results['Actual_Close'] = y_test
results['Predicted_Close'] = model.predict(X_test)
# Ważne: Musimy znać cenę z "wczoraj" dla tego wiersza (czyli Close_Lag_1)
results['Prev_Close'] = results['Close_Lag_1'] 

results.to_csv('colleagues_predictions.csv')
print(f"Zapisano nowe wyniki. VIX dodany. Zakres danych: {len(df)} dni.")