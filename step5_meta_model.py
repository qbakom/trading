import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score
import matplotlib.pyplot as plt
import joblib

# 1. Wczytaj "plik od kolegów"
df = pd.read_csv('stages/step4_predictions.csv', index_col=0, parse_dates=True)

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

# Updated Feature List with Market Regime indicators
meta_features = ['RSI', 'ATR', 'Rolling_Std', 'Model_Confidence', 'VIX_Slope', 'BB_Width', 'RSI_Dist']
# Check if all features exist
missing_features = [f for f in meta_features if f not in df.columns]
if missing_features:
    print(f"Warning: Missing features {missing_features}. Using available ones.")
    meta_features = [f for f in meta_features if f in df.columns]

X_meta = df[meta_features]
y_meta = df['Meta_Target']

# 4. TRENING TWOJEGO FILTRA
# Dzielimy te wyniki na pół (uczymy się na pierwszej połowie testu kolegów, testujemy na drugiej)
split = int(len(df) * 0.5)
X_m_train, X_m_test = X_meta.iloc[:split], X_meta.iloc[split:]
y_m_train, y_m_test = y_meta.iloc[:split], y_meta.iloc[split:]

print("\nTrening Meta-Modelu (GridSearchCV)...")

# Hyperparameter Grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],
    'min_samples_leaf': [3, 5, 10],
    'class_weight': ['balanced', None]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=3, n_jobs=-1, scoring='precision', verbose=1)

grid_search.fit(X_m_train, y_m_train)

meta_model = grid_search.best_estimator_
print(f"Najlepsze parametry: {grid_search.best_params_}")

# Feature Importance
importances = meta_model.feature_importances_
feat_importances = pd.Series(importances, index=X_m_test.columns)
print("\n--- CO JEST WAŻNE DLA FILTRA? ---")
print(feat_importances.sort_values(ascending=False))

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

# Advanced Metrics
def calculate_metrics(equity_curve):
    returns = equity_curve.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
    
    # Max Drawdown
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_dd = drawdown.min()
    
    return sharpe, max_dd

sharpe_base, dd_base = calculate_metrics(test_data['Equity_Base'])
sharpe_meta, dd_meta = calculate_metrics(test_data['Equity_Meta'])

print(f"\n--- METRYKI FINANSOWE ---")
print(f"Base Strategy: Sharpe={sharpe_base:.2f}, MaxDD={dd_base:.2%}")
print(f"Meta Strategy: Sharpe={sharpe_meta:.2f}, MaxDD={dd_meta:.2%}")

plt.figure(figsize=(10, 6))
plt.plot(test_data['Equity_Base'], label=f'Base (Sharpe={sharpe_base:.2f})', color='red', alpha=0.6)
plt.plot(test_data['Equity_Meta'], label=f'Meta-Filter (Sharpe={sharpe_meta:.2f})', color='green', linewidth=2)
plt.title('Porównanie Kapitału: Regresja vs Meta-Labeling')
plt.legend()
plt.grid(True)
plt.savefig('stages/step5_equity_curve.png', dpi=300, bbox_inches='tight')
print("\nWykres zapisany jako 'stages/step5_equity_curve.png'")

# 7. Zapisz model i dane dla interpretowalności (step6)
print("Zapisywanie modelu i danych dla step6...")
joblib.dump(meta_model, 'stages/step5_model_meta_rf.joblib')
data_step5 = {
    'X_m_train': X_m_train,
    'X_m_test': X_m_test,
    'y_m_train': y_m_train,
    'y_m_test': y_m_test,
    'meta_features': meta_features
}
joblib.dump(data_step5, 'stages/step5_data.pkl')
print("Zapisano 'stages/step5_model_meta_rf.joblib' i 'stages/step5_data.pkl'")
