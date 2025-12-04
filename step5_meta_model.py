import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score
import matplotlib.pyplot as plt

# 1. Wczytaj "plik od kolegów"
df = pd.read_csv('colleagues_predictions.csv', index_col=0, parse_dates=True)

# 2. LABELING (Oceniamy pracę kolegów)

# Krok A: Jaki sygnał daje ich model?
# Jeśli Predicted > Prev_Close -> 1 (KUPUJ/WZROST)
# Jeśli Predicted < Prev_Close -> -1 (SPRZEDAJ/SPADEK)
df['Signal'] = np.where(df['Predicted_Close'] > df['Prev_Close'], 1, -1)

# Krok B: Czy mieli rację? (Tworzenie Targetu dla Meta-Modelu)
# Obliczamy faktyczny kierunek rynku
df['Actual_Direction'] = np.where(df['Actual_Close'] > df['Prev_Close'], 1, -1)

# Meta_Label: 1 jeśli Signal == Actual_Direction, 0 jeśli się pomylili
df['Meta_Target'] = (df['Signal'] == df['Actual_Direction']).astype(int)

print("Statystyki modelu kolegów (przed filtrem):")
print(f"Skuteczność kierunkowa (Accuracy): {df['Meta_Target'].mean():.2%}")

# 3. FEATURE ENGINEERING DLA META-MODELU
# Na czym Twój model ma się uczyć filtrowania?
# Używamy cech, które już są w pliku (RSI, ATR, Rolling_Std)
# Dodajemy: Pewność siebie modelu (jak dużą zmianę przewiduje?)
df['Model_Confidence'] = abs(df['Predicted_Close'] - df['Prev_Close']) / df['Prev_Close']

meta_features = ['RSI', 'ATR', 'Rolling_Std', 'Model_Confidence']
X_meta = df[meta_features]
y_meta = df['Meta_Target']

# 4. TRENING TWOJEGO FILTRA
# Dzielimy te wyniki na pół (uczymy się na pierwszej połowie testu kolegów, testujemy na drugiej)
split = int(len(df) * 0.5)
X_m_train, X_m_test = X_meta.iloc[:split], X_meta.iloc[split:]
y_m_train, y_m_test = y_meta.iloc[:split], y_meta.iloc[split:]

print("\nTrening Meta-Modelu (Random Forest)...")
# Random Forest jest świetny jako Meta-Model
meta_model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
meta_model.fit(X_m_train, y_m_train)

# 5. EWALUACJA
# Meta-Model mówi: 1 (Przepuść sygnał, jest bezpieczny), 0 (Zablokuj, model kłamie)
meta_preds = meta_model.predict(X_m_test)

print("\n--- WYNIKI TWOJEGO MODUŁU ---")
print(classification_report(y_m_test, meta_preds))

# Sprawdźmy precyzję dla klasy 1 (kiedy mówisz "TAK", czy masz rację?)
prec = precision_score(y_m_test, meta_preds)
print(f"Precyzja Twojego filtra: {prec:.2%}")
if prec > df['Meta_Target'].mean():
    print("SUKCES! Twój filtr poprawia wyniki modelu bazowego.")
else:
    print("UWAGA: Filtr wymaga dostrojenia (więcej danych/lepsze cechy).")

# 6. Symulacja Zysków (Equity Curve)
test_data = df.iloc[split:].copy()
test_data['Meta_Filter'] = meta_preds

# Obliczamy zwrot z tradingu (prosty: kupujemy na otwarciu, sprzedajemy na zamknięciu)
# Uproszczenie: zwrot to (Actual - Prev) / Prev * Sygnał
test_data['Return'] = (test_data['Actual_Close'] - test_data['Prev_Close']) / test_data['Prev_Close']
test_data['Strategy_Base'] = test_data['Return'] * test_data['Signal'] # Bierzemy wszystko
test_data['Strategy_Meta'] = test_data['Strategy_Base'] * test_data['Meta_Filter'] # Bierzemy tylko potwierdzone

# Skumulowany zysk
test_data['Equity_Base'] = (1 + test_data['Strategy_Base']).cumprod()
test_data['Equity_Meta'] = (1 + test_data['Strategy_Meta']).cumprod()

plt.figure(figsize=(10, 6))
plt.plot(test_data['Equity_Base'], label='RegLin (Bez Filtra)', color='red', alpha=0.6)
plt.plot(test_data['Equity_Meta'], label='Meta-Labeling (Z Meta-Filtrem)', color='green', linewidth=2)
plt.title('Porównanie Kapitału: Regresja vs Meta-Labeling')
plt.legend()
plt.grid(True)
plt.show() # Pokaże wykres, który wrzucisz do prezentacji