import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score
import matplotlib.pyplot as plt

# 1. Wczytaj dane
df = pd.read_csv('colleagues_predictions.csv', index_col=0, parse_dates=True)

# 2. Labeling
df['Signal'] = np.where(df['Predicted_Close'] > df['Prev_Close'], 1, -1)
df['Actual_Direction'] = np.where(df['Actual_Close'] > df['Prev_Close'], 1, -1)

# Target: 1 jeśli mieli rację, 0 jeśli błąd
df['Meta_Target'] = (df['Signal'] == df['Actual_Direction']).astype(int)

# 3. Feature Engineering dla Meta-Modelu
# Używamy VIX, który dodaliśmy w kroku 4!
df['Model_Confidence'] = abs(df['Predicted_Close'] - df['Prev_Close']) / df['Prev_Close']

# Cechy, które wchodzą do Meta-Modelu
meta_features = ['RSI', 'ATR', 'VIX', 'Dist_SMA', 'Model_Confidence']
X_meta = df[meta_features]
y_meta = df['Meta_Target']

# 4. Trening
split = int(len(df) * 0.5)
X_m_train, X_m_test = X_meta.iloc[:split], X_meta.iloc[split:]
y_m_train, y_m_test = y_meta.iloc[:split], y_meta.iloc[split:]

print("\nTrening Meta-Modelu (Random Forest)...")
# Zmniejszamy min_samples_leaf, żeby model był bardziej agresywny
meta_model = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42)
meta_model.fit(X_m_train, y_m_train)

# 5. Feature Importance (Co jest ważne?)
importances = meta_model.feature_importances_
feat_importances = pd.Series(importances, index=X_m_test.columns)
print("\n--- CO JEST WAŻNE DLA FILTRA? ---")
print(feat_importances.sort_values(ascending=False))

# 6. Predykcja z niższym progiem (Soft Voting)
probs = meta_model.predict_proba(X_m_test)[:, 1]
# Domyślnie jest 0.5. Zniżamy do 0.52 (lekki filtr) lub podwyższamy. 
# Jeśli filtr nic nie puszczał, to znaczy że rzadko był pewien > 0.5.
threshold = 0.52 
meta_preds = (probs > threshold).astype(int)

print(f"\n--- WYNIKI (Próg pewności: {threshold}) ---")
print(f"Ile transakcji przepuścił filtr: {sum(meta_preds)} z {len(meta_preds)}")
print(classification_report(y_m_test, meta_preds))

# 7. Wykres
test_data = df.iloc[split:].copy()
test_data['Meta_Filter'] = meta_preds
test_data['Return'] = (test_data['Actual_Close'] - test_data['Prev_Close']) / test_data['Prev_Close']

# Symulacja
test_data['Strategy_Base'] = test_data['Return'] * test_data['Signal']
test_data['Strategy_Meta'] = test_data['Strategy_Base'] * test_data['Meta_Filter']

test_data['Equity_Base'] = (1 + test_data['Strategy_Base']).cumprod()
test_data['Equity_Meta'] = (1 + test_data['Strategy_Meta']).cumprod()

plt.figure(figsize=(10, 6))
plt.plot(test_data['Equity_Base'], label='VAM (Base)', color='red', alpha=0.5)
plt.plot(test_data['Equity_Meta'], label='Kuba (Meta-Filter)', color='green', linewidth=2)
plt.title('Porównanie Strategii (Z VIX i danymi od 2018)')
plt.legend()
plt.grid(True)
<<<<<<< Updated upstream
plt.show()
=======
plt.savefig('meta_model_comparison.png', dpi=300, bbox_inches='tight')
print("\nWykres zapisany jako 'meta_model_comparison.png'")
>>>>>>> Stashed changes
