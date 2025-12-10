import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
import xgboost
import os

# Ustawienie stylu wykresów
plt.style.use('seaborn-v0_8')

def analyze_xgboost():
    print("\n--- ANALIZA MODELU BAZOWEGO (XGBoost) ---")
    
    # 1. Wczytaj model i dane
    if not os.path.exists('stages/step4_model_xgboost.joblib') or not os.path.exists('stages/step4_data.pkl'):
        print("BŁĄD: Brak plików modelu lub danych. Uruchom najpierw step4_gen_predictions.py")
        return

    model = joblib.load('stages/step4_model_xgboost.joblib')
    data = joblib.load('stages/step4_data.pkl')
    X_test = data['X_test']
    feature_cols = data['feature_cols']
    
    print(f"Wczytano model XGBoost. Liczba cech: {len(feature_cols)}")

    # 2. Feature Importance (Wbudowane w XGBoost)
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance - XGBoost (Base Model)")
    plt.bar(range(X_test.shape[1]), importance[indices], align="center")
    plt.xticks(range(X_test.shape[1]), [feature_cols[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('stages/step6_feature_importance_xgboost.png', dpi=300)
    print("Zapisano 'stages/step6_feature_importance_xgboost.png'")
    plt.close()

    # 3. SHAP Analysis
    print("Obliczanie wartości SHAP dla XGBoost...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # SHAP Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary Plot - XGBoost")
    plt.tight_layout()
    plt.savefig('stages/step6_shap_summary_xgboost.png', dpi=300)
    print("Zapisano 'stages/step6_shap_summary_xgboost.png'")
    plt.close()

def analyze_meta_model():
    print("\n--- ANALIZA META-MODELU (Random Forest) ---")
    
    # 1. Wczytaj model i dane
    if not os.path.exists('stages/step5_model_meta_rf.joblib') or not os.path.exists('stages/step5_data.pkl'):
        print("BŁĄD: Brak plików modelu lub danych. Uruchom najpierw step5_meta_model.py")
        return

    model = joblib.load('stages/step5_model_meta_rf.joblib')
    data = joblib.load('stages/step5_data.pkl')
    X_m_test = data['X_m_test']
    meta_features = data['meta_features']
    
    print(f"Wczytano Meta-Model. Liczba cech: {len(meta_features)}")

    # 2. Feature Importance
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(8, 5))
    plt.title("Feature Importance - Meta Model (Random Forest)")
    plt.bar(range(X_m_test.shape[1]), importance[indices], align="center", color='green')
    plt.xticks(range(X_m_test.shape[1]), [meta_features[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig('stages/step6_feature_importance_meta.png', dpi=300)
    print("Zapisano 'stages/step6_feature_importance_meta.png'")
    plt.close()

    # 3. SHAP Analysis
    print("Obliczanie wartości SHAP dla Meta-Modelu...")
    # Dla Random Forest używamy TreeExplainer, ale uwaga na czas obliczeń
    explainer = shap.TreeExplainer(model)
    # Wybieramy klasę 1 (że sygnał jest poprawny)
    shap_values = explainer.shap_values(X_m_test)
    
    # Dla klasyfikacji binarnej shap_values to lista [values_class_0, values_class_1]
    # Interesuje nas klasa 1
    if isinstance(shap_values, list):
        shap_vals_target = shap_values[1]
    else:
        shap_vals_target = shap_values

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_vals_target, X_m_test, show=False)
    plt.title("SHAP Summary Plot - Meta Model (Class 1)")
    plt.tight_layout()
    plt.savefig('stages/step6_shap_summary_meta.png', dpi=300)
    print("Zapisano 'stages/step6_shap_summary_meta.png'")
    plt.close()

if __name__ == "__main__":
    analyze_xgboost()
    analyze_meta_model()
    print("\nAnaliza zakończona. Sprawdź pliki PNG.")
