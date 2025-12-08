import pandas as pd
import yfinance as yf

# 1. Wczytaj ich plik (załóżmy, że nazywa się 'raw_predictions.csv')
# Sprawdź czy używają przecinka czy średnika!
df_colleagues = pd.read_csv('raw_predictions.csv', parse_dates=True, index_col=0) 

# 2. Pobierz historię AAPL, żeby mieć pewność co do cen
df_market = yf.download('AAPL', start='2000-01-01', auto_adjust=False)
if isinstance(df_market.columns, pd.MultiIndex):
    df_market.columns = df_market.columns.get_level_values(0)

# 3. Złącz ich predykcje z Twoimi danymi (po dacie)
# To dodaje brakujące kolumny (Volume, High, Low) do ich pliku
df_merged = df_market.join(df_colleagues[['Predicted_Close']], how='inner')

# 4. Dodaj kolumnę Prev_Close (Kluczowe dla Twojego kodu!)
df_merged['Prev_Close'] = df_merged['Close'].shift(1)
df_merged['Actual_Close'] = df_merged['Close'] # Dla pewności nazewnictwa

df_merged.dropna(inplace=True)

# 5. Zapisz w formacie gotowym dla step5
df_merged.to_csv('colleagues_predictions.csv')
print("Plik przekonwertowany! Możesz odpalać step5.")