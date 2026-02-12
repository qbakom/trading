"""
XGBoost Classifier - Direction Prediction (UP/DOWN).

Instead of predicting return value, we directly classify direction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from xgboost import XGBClassifier
import joblib
import os
from datetime import datetime

from src.data_loader import load_data, FEATURES

# =============================================================================
# Configuration
# =============================================================================
TICKER = "AAPL"
N_SPLITS = 5
RANDOM_STATE = 42
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =============================================================================
# 1. Load Data
# =============================================================================
print("=" * 70)
print("XGBOOST CLASSIFIER - DIRECTION PREDICTION")
print("=" * 70)
print(f"\nTicker: {TICKER}")
print(f"Features: {FEATURES}")

data = load_data(TICKER)

X_train, y_train_reg = data.X_train, data.y_train
X_val, y_val_reg = data.X_val, data.y_val
X_test, y_test_reg = data.X_test, data.y_test

# Convert to binary: 1 = UP (return > 0), 0 = DOWN (return <= 0)
y_train = (y_train_reg > 0).astype(int)
y_val = (y_val_reg > 0).astype(int)
y_test = (y_test_reg > 0).astype(int)

# Combine train + val for CV
X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])
y_train_full_reg = pd.concat([y_train_reg, y_val_reg])  # Keep for strategy calc

print(f"\nData shapes:")
print(f"  Train: {X_train.shape}")
print(f"  Val:   {X_val.shape}")
print(f"  Test:  {X_test.shape}")
print(f"  Train+Val: {X_train_full.shape}")

# =============================================================================
# 2. Class Distribution
# =============================================================================
print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

train_up = y_train_full.sum()
train_down = len(y_train_full) - train_up
test_up = y_test.sum()
test_down = len(y_test) - test_up

print(f"\nTrain+Val:")
print(f"  UP:   {train_up} ({train_up/len(y_train_full):.1%})")
print(f"  DOWN: {train_down} ({train_down/len(y_train_full):.1%})")
print(f"\nTest:")
print(f"  UP:   {test_up} ({test_up/len(y_test):.1%})")
print(f"  DOWN: {test_down} ({test_down/len(y_test):.1%})")

# =============================================================================
# 3. Grid Search
# =============================================================================
print("\n" + "=" * 70)
print("GRID SEARCH")
print("=" * 70)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "min_child_weight": [1, 3, 5],
    "reg_lambda": [0, 0.1, 1.0],
    "gamma": [0, 0.1],
    "scale_pos_weight": [1.0],  # Classes are roughly balanced
}

total = 1
for v in param_grid.values():
    total *= len(v)
print(f"Total combinations: {total:,}")

tscv = TimeSeriesSplit(n_splits=N_SPLITS)

xgb = XGBClassifier(
    random_state=RANDOM_STATE,
    n_jobs=-1,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    verbosity=0,
)

search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=tscv,
    scoring="accuracy",  # Optimize for accuracy
    n_jobs=-1,
    verbose=1,
)

print(f"Starting at {datetime.now().strftime('%H:%M:%S')}...")
search.fit(X_train_full, y_train_full)
print(f"Finished at {datetime.now().strftime('%H:%M:%S')}")

best_model = search.best_estimator_
best_params = search.best_params_
best_cv_acc = search.best_score_

print(f"\nBest parameters:")
for k, v in sorted(best_params.items()):
    print(f"  {k}: {v}")
print(f"\nBest CV Accuracy: {best_cv_acc:.4f}")

# =============================================================================
# 4. Evaluate
# =============================================================================
print("\n" + "=" * 70)
print("EVALUATION")
print("=" * 70)

y_pred_train = best_model.predict(X_train_full)
y_pred_test = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Metrics
train_acc = accuracy_score(y_train_full, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test)
test_recall = recall_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test)
test_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nClassification Metrics:")
print(f"  Train Accuracy: {train_acc:.4f}")
print(f"  Test Accuracy:  {test_acc:.4f}")
print(f"  Test Precision: {test_precision:.4f}")
print(f"  Test Recall:    {test_recall:.4f}")
print(f"  Test F1:        {test_f1:.4f}")
print(f"  Test AUC-ROC:   {test_auc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
print(f"\nConfusion Matrix:")
print(f"              Predicted DOWN | Predicted UP")
print(f"  Actual DOWN:  {cm[0,0]:4d}        | {cm[0,1]:4d}")
print(f"  Actual UP:    {cm[1,0]:4d}        | {cm[1,1]:4d}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_test, target_names=["DOWN", "UP"]))

# =============================================================================
# 5. Compare with Regressor
# =============================================================================
print("\n" + "=" * 70)
print("COMPARISON: CLASSIFIER vs REGRESSOR")
print("=" * 70)

# Load regressor results if available
regressor_path = os.path.join(MODEL_DIR, f"xgboost_{TICKER}.joblib")
if os.path.exists(regressor_path):
    reg_data = joblib.load(regressor_path)
    reg_dir_acc = reg_data["metrics"]["directional_accuracy"]
    print(f"\nRegressor Directional Accuracy: {reg_dir_acc:.4f}")
    print(f"Classifier Accuracy:            {test_acc:.4f}")
    print(f"Difference:                     {(test_acc - reg_dir_acc)*100:+.2f}pp")
else:
    print("\n(No regressor model found for comparison)")

# =============================================================================
# 6. Trading Strategy Simulation
# =============================================================================
print("\n" + "=" * 70)
print("TRADING STRATEGY")
print("=" * 70)

# Strategy: go long if predicted UP, stay out if predicted DOWN
strategy_returns = y_test_reg.values * y_pred_test
buyhold_returns = y_test_reg.values

cumret_strategy = (1 + strategy_returns).cumprod()
cumret_buyhold = (1 + buyhold_returns).cumprod()

sharpe_bh = np.sqrt(252) * buyhold_returns.mean() / buyhold_returns.std() if buyhold_returns.std() > 0 else 0
sharpe_xgb = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0

print(f"\nBuy & Hold:")
print(f"  Final Return: {(cumret_buyhold[-1] - 1)*100:.2f}%")
print(f"  Sharpe Ratio: {sharpe_bh:.2f}")
print(f"\nXGBoost Classifier Strategy:")
print(f"  Final Return: {(cumret_strategy[-1] - 1)*100:.2f}%")
print(f"  Sharpe Ratio: {sharpe_xgb:.2f}")

# =============================================================================
# 7. Feature Importance
# =============================================================================
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

importance = pd.Series(
    best_model.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

print(importance.to_string())

# =============================================================================
# 8. Save Model
# =============================================================================
model_path = os.path.join(MODEL_DIR, f"xgboost_classifier_{TICKER}.joblib")
save_data = {
    "model": best_model,
    "best_params": best_params,
    "feature_names": FEATURES,
    "metrics": {
        "cv_accuracy": best_cv_acc,
        "test_accuracy": test_acc,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_auc": test_auc,
    },
    "ticker": TICKER,
}
joblib.dump(save_data, model_path)
print(f"\nModel saved: {model_path}")

# =============================================================================
# 9. Plots
# =============================================================================
print("\n" + "=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Feature Importance
ax = axes[0, 0]
importance.plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Feature Importance (Gain)")
ax.set_xlabel("Importance")
ax.invert_yaxis()

# 2. Confusion Matrix Heatmap
ax = axes[0, 1]
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix\nAccuracy: {test_acc:.2%}")

# 3. Probability Distribution
ax = axes[0, 2]
ax.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.5, label="Actual DOWN", color="red", density=True)
ax.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.5, label="Actual UP", color="green", density=True)
ax.set_xlabel("Predicted Probability (UP)")
ax.set_ylabel("Density")
ax.set_title("Probability Distribution by Class")
ax.legend()
ax.axvline(x=0.5, color="black", linestyle="--", alpha=0.5)

# 4. ROC-like: Accuracy by Confidence Threshold
ax = axes[1, 0]
thresholds = np.linspace(0.3, 0.7, 21)
accuracies = []
coverages = []
for t in thresholds:
    high_conf = (y_pred_proba >= t) | (y_pred_proba <= (1-t))
    if high_conf.sum() > 0:
        acc = accuracy_score(y_test[high_conf], y_pred_test[high_conf])
        cov = high_conf.mean()
    else:
        acc = 0
        cov = 0
    accuracies.append(acc)
    coverages.append(cov)

ax.plot(thresholds, accuracies, 'b-', label="Accuracy")
ax.plot(thresholds, coverages, 'r--', label="Coverage")
ax.set_xlabel("Confidence Threshold")
ax.set_ylabel("Score")
ax.set_title("Accuracy vs Confidence Threshold")
ax.legend()
ax.grid(True, alpha=0.3)

# 5. Cumulative Returns
ax = axes[1, 1]
ax.plot(cumret_buyhold, label="Buy & Hold", color="blue", alpha=0.7)
ax.plot(cumret_strategy, label="XGBoost Classifier", color="green", alpha=0.7)
ax.set_xlabel("Trading Day")
ax.set_ylabel("Cumulative Return")
ax.set_title("Cumulative Returns (Test)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.98, f"Sharpe B&H: {sharpe_bh:.2f}\nSharpe XGB: {sharpe_xgb:.2f}",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

# 6. Rolling Accuracy
ax = axes[1, 2]
correct = (y_test.values == y_pred_test).astype(int)
rolling_acc = pd.Series(correct).rolling(50).mean()
ax.plot(rolling_acc, color="purple", alpha=0.7)
ax.axhline(y=0.5, color="red", linestyle="--", label="Random (50%)")
ax.axhline(y=test_acc, color="green", linestyle="--", label=f"Overall ({test_acc:.1%})")
ax.set_xlabel("Trading Day")
ax.set_ylabel("Rolling Accuracy (50-day)")
ax.set_title("Rolling Accuracy Over Time")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0.3, 0.7)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, f"xgboost_classifier_{TICKER}_results.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Saved: {plot_path}")

# =============================================================================
# 10. Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Ticker: {TICKER}
Features: {FEATURES}
Task: Binary Classification (UP/DOWN)

Best Parameters:
{pd.Series(best_params).to_string()}

Metrics:
  CV Accuracy:    {best_cv_acc:.4f}
  Test Accuracy:  {test_acc:.4f}
  Test Precision: {test_precision:.4f}
  Test Recall:    {test_recall:.4f}
  Test F1:        {test_f1:.4f}
  Test AUC-ROC:   {test_auc:.4f}

Strategy:
  Sharpe (Buy & Hold):  {sharpe_bh:.2f}
  Sharpe (Classifier):  {sharpe_xgb:.2f}

Files:
  Model: {model_path}
  Plot:  {plot_path}
""")

print("=" * 70)
print("DONE")
print("=" * 70)
plt.show()
