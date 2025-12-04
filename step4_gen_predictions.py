import pandas as pd
import pandas_ta as ta
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

# 1. Więcej danych! (2020 zamiast 2024)
print("Pobieranie danych...")
df = yf.download('AAPL', start='2020-01-01', auto_adjust=False)
df.columns = df.columns.get_level_values(0)

# 2. Features
df['RSI'] = df.ta.rsi(length=14)
df['SMA_20'] = df.ta.sma(length=20)
df['ATR'] = df.ta.atr(length=14)
# Dodajemy VIX (Indeks strachu) - WAŻNE dla Meta-Modelu
# Pobieramy osobno i łączymy, ale na razie uprośćmy: bazujemy na zmienności Apple
df['Rolling_Std'] = df['Close'].pct_change().rolling(20).std()

for lag in [1, 2, 3, 5]:
    df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)

df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

# 3. Podział
feature_cols = [col for col in df.columns if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
X = df[feature_cols]
y = df['Target']

# Bez shuffle, żeby zachować chronologię!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. Trening Modelu Bazowego (Udajemy kolegów)
print("Trening modelu regresyjnego (XGBoost)...")
model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05)
model.fit(X_train, y_train)

# 5. Generowanie predykcji i ZAPIS DO PLIKU
# Tworzymy DataFrame, który symuluje plik od kolegów
results = X_test.copy()
results['Actual_Close'] = y_test
results['Predicted_Close'] = model.predict(X_test)

# Dodajemy oryginalną cenę z dnia wczorajszego (Close_Lag_1), żeby wiedzieć czy model przewiduje wzrost czy spadek
# (Bo X_test ma znormalizowane dane lub lagi, musimy mieć punkt odniesienia)
results['Prev_Close'] = results['Close_Lag_1'] 

results.to_csv('colleagues_predictions.csv')
print(f"\nSukces! Zapisano wyniki do 'colleagues_predictions.csv'.")
print(f"MAE modelu kolegów: {mean_absolute_error(y_test, results['Predicted_Close']):.2f}")
print("Teraz odpal step5_meta_model.py")