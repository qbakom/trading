import pandas as pd
import pandas_ta as ta
import yfinance as yf

# 1. Pobierz dane
# 1. Pobierz dane
import os

# Define paths relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CACHE_FILE = os.path.join(DATA_DIR, 'AAPL_cache.csv')

if os.path.exists(CACHE_FILE):
    df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
else:
    df = yf.download('AAPL', start='2000-01-01', auto_adjust=False)
    # Ensure directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.to_csv(CACHE_FILE)

# Flatten MultiIndex columns (yfinance returns MultiIndex even for single ticker)
df.columns = df.columns.get_level_values(0)

# 2. Policz wskaźniki (To jest ta "inteligencja" dla XGBoosta)
df['RSI'] = df.ta.rsi(length=14)
df['SMA_20'] = df.ta.sma(length=20)
df['ATR'] = df.ta.atr(length=14) # Zmienność

# 3. Zrób LAGI (To jest kluczowe! XGBoost musi widzieć historię w jednym wierszu)
for lag in [1, 2, 3, 5, 10]:
    df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
    df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)

# 4. Target (Co przewidujemy? Cenę jutro)
df['Target'] = df['Close'].shift(-1)

# Usuń puste wiersze (bo lagi zrobiły dziury na początku)
df.dropna(inplace=True)

print("Dane dla XGBoost gotowe. X to wszystkie kolumny oprócz Target, Y to Target.")