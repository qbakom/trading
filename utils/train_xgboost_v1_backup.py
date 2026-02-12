"""
XGBoost Training Script - Optimized for Best Performance.

Features: VXN, DXY, BB_PB, EMA50, Close
Target: Next day return (% change)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
import os
from datetime import datetime

from src.data_loader import load_data, FEATURES

# =============================================================================
# Configuration
# =============================================================================
TICKER = "AAPL"
N_SPLITS = 5  # TimeSeriesSplit folds
RANDOM_STATE = 42
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =============================================================================
# 1. Load Data
# =============================================================================
print("=" * 70)
print("XGBOOST TRAINING - OPTIMIZED")
print("=" * 70)
print(f"\nTicker: {TICKER}")
print(f"Features: {FEATURES}")

data = load_data(TICKER)

X_train, y_train = data.X_train, data.y_train
X_val, y_val = data.X_val, data.y_val
X_test, y_test = data.X_test, data.y_test

# Combine train + val for final training (use val for early stopping conceptually)
X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])

print(f"\nData shapes:")
print(f"  Train: {X_train.shape}")
print(f"  Val:   {X_val.shape}")
print(f"  Test:  {X_test.shape}")
print(f"  Train+Val (for CV): {X_train_full.shape}")

# =============================================================================
# 2. Define Hyperparameter Grid
# =============================================================================
# Extensive grid for thorough search
param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.02, 0.05, 0.1],
    "subsample": [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "reg_alpha": [0, 0.1, 0.5, 1.0],
    "reg_lambda": [1.0, 2.0, 5.0],
    "gamma": [0, 0.1, 0.2],
}

# Calculate total combinations
total_combinations = 1
for v in param_grid.values():
    total_combinations *= len(v)
print(f"\nTotal hyperparameter combinations: {total_combinations:,}")

# Use RandomizedSearchCV for efficiency (sample from grid)
N_ITER = 200  # Number of random samples to try
print(f"Using RandomizedSearchCV with {N_ITER} iterations")

# =============================================================================
# 3. Train with Cross-Validation
# =============================================================================
print("\n" + "=" * 70)
print("TRAINING WITH RANDOMIZED SEARCH CV")
print("=" * 70)

# TimeSeriesSplit preserves temporal order
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

# Base model
xgb = XGBRegressor(
    random_state=RANDOM_STATE,
    n_jobs=-1,
    objective="reg:squarederror",
    tree_method="hist",
    verbosity=0,
)

# Randomized search
search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=N_ITER,
    cv=tscv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=2,
    random_state=RANDOM_STATE,
    return_train_score=True,
)

print(f"\nStarting search at {datetime.now().strftime('%H:%M:%S')}...")
search.fit(X_train_full, y_train_full)
print(f"Finished at {datetime.now().strftime('%H:%M:%S')}")

# Best model
best_model = search.best_estimator_
best_params = search.best_params_
best_cv_score = -search.best_score_

print(f"\n{'='*70}")
print("BEST PARAMETERS")
print("=" * 70)
for param, value in sorted(best_params.items()):
    print(f"  {param}: {value}")
print(f"\nBest CV RMSE: {best_cv_score:.6f}")

# =============================================================================
# 4. Evaluate on Test Set
# =============================================================================
print("\n" + "=" * 70)
print("TEST SET EVALUATION")
print("=" * 70)

# Predictions
y_pred_train = best_model.predict(X_train_full)
y_pred_test = best_model.predict(X_test)

# Regression metrics
train_rmse = np.sqrt(mean_squared_error(y_train_full, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

print(f"\nRegression Metrics:")
print(f"  Train RMSE: {train_rmse:.6f}")
print(f"  Test RMSE:  {test_rmse:.6f}")
print(f"  Test MAE:   {test_mae:.6f}")
print(f"  Test R²:    {test_r2:.4f}")

# Directional accuracy (most important for trading)
# Direction: 1 if positive return, 0 if negative
actual_direction = (y_test > 0).astype(int)
pred_direction = (y_pred_test > 0).astype(int)
directional_accuracy = (actual_direction == pred_direction).mean()

print(f"\nDirectional Metrics:")
print(f"  Directional Accuracy: {directional_accuracy:.2%}")

# Confusion matrix for direction
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(actual_direction, pred_direction)
print(f"\nDirection Confusion Matrix:")
print(f"  TN (predicted DOWN, actual DOWN): {cm[0,0]}")
print(f"  FP (predicted UP, actual DOWN):   {cm[0,1]}")
print(f"  FN (predicted DOWN, actual UP):   {cm[1,0]}")
print(f"  TP (predicted UP, actual UP):     {cm[1,1]}")

# Precision/Recall for UP predictions
tp = cm[1, 1]
fp = cm[0, 1]
fn = cm[1, 0]
precision_up = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_up = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"\nUP Prediction Quality:")
print(f"  Precision (when we predict UP): {precision_up:.2%}")
print(f"  Recall (of actual UP days):     {recall_up:.2%}")

# =============================================================================
# 5. Feature Importance
# =============================================================================
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

feature_importance = pd.Series(
    best_model.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

print(feature_importance.to_string())

# =============================================================================
# 6. Save Model
# =============================================================================
model_path = os.path.join(MODEL_DIR, f"xgboost_{TICKER}.joblib")
save_data = {
    "model": best_model,
    "best_params": best_params,
    "feature_names": FEATURES,
    "metrics": {
        "cv_rmse": best_cv_score,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "directional_accuracy": directional_accuracy,
    },
    "ticker": TICKER,
}
joblib.dump(save_data, model_path)
print(f"\nModel saved to: {model_path}")

# =============================================================================
# 7. Visualizations
# =============================================================================
print("\n" + "=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Feature Importance
ax = axes[0, 0]
feature_importance.plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Feature Importance (Gain)")
ax.set_xlabel("Importance")
ax.invert_yaxis()

# 2. Actual vs Predicted (Test)
ax = axes[0, 1]
ax.scatter(y_test, y_pred_test, alpha=0.3, s=10)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
ax.set_xlabel("Actual Return")
ax.set_ylabel("Predicted Return")
ax.set_title(f"Actual vs Predicted (Test)\nR² = {test_r2:.4f}")
ax.grid(True, alpha=0.3)

# 3. Prediction Distribution
ax = axes[0, 2]
ax.hist(y_test, bins=50, alpha=0.5, label="Actual", color="blue")
ax.hist(y_pred_test, bins=50, alpha=0.5, label="Predicted", color="orange")
ax.set_xlabel("Return")
ax.set_ylabel("Frequency")
ax.set_title("Distribution: Actual vs Predicted")
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Residuals
ax = axes[1, 0]
residuals = y_test - y_pred_test
ax.scatter(y_pred_test, residuals, alpha=0.3, s=10)
ax.axhline(y=0, color="r", linestyle="--")
ax.set_xlabel("Predicted Return")
ax.set_ylabel("Residual")
ax.set_title("Residual Plot")
ax.grid(True, alpha=0.3)

# 5. Cumulative Returns (simulated strategy)
ax = axes[1, 1]
# Strategy: go long when predict UP, stay out when predict DOWN
strategy_returns = y_test.values * pred_direction
cumret_strategy = (1 + strategy_returns).cumprod()
cumret_buyhold = (1 + y_test.values).cumprod()

ax.plot(cumret_buyhold, label="Buy & Hold", color="blue", alpha=0.7)
ax.plot(cumret_strategy, label="XGBoost Strategy", color="green", alpha=0.7)
ax.set_xlabel("Trading Day")
ax.set_ylabel("Cumulative Return")
ax.set_title("Cumulative Returns (Test Period)")
ax.legend()
ax.grid(True, alpha=0.3)

# Calculate Sharpe (annualized, assuming 252 trading days)
sharpe_buyhold = np.sqrt(252) * y_test.mean() / y_test.std() if y_test.std() > 0 else 0
sharpe_strategy = np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0
ax.text(0.02, 0.98, f"Sharpe B&H: {sharpe_buyhold:.2f}\nSharpe XGB: {sharpe_strategy:.2f}",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

# 6. Confusion Matrix Heatmap
ax = axes[1, 2]
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"])
ax.set_xlabel("Predicted Direction")
ax.set_ylabel("Actual Direction")
ax.set_title(f"Direction Confusion Matrix\nAccuracy: {directional_accuracy:.2%}")

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, f"xgboost_{TICKER}_results.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Saved: {plot_path}")

# =============================================================================
# 8. CV Results Analysis
# =============================================================================
cv_results = pd.DataFrame(search.cv_results_)
cv_results_sorted = cv_results.sort_values("rank_test_score").head(20)

print("\n" + "=" * 70)
print("TOP 10 PARAMETER COMBINATIONS")
print("=" * 70)
for i, row in cv_results_sorted.head(10).iterrows():
    print(f"\nRank {row['rank_test_score']}: CV RMSE = {-row['mean_test_score']:.6f} (+/- {row['std_test_score']:.6f})")

# =============================================================================
# 9. Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Ticker: {TICKER}
Features: {FEATURES}

Best Hyperparameters:
{pd.Series(best_params).to_string()}

Performance:
  CV RMSE:              {best_cv_score:.6f}
  Test RMSE:            {test_rmse:.6f}
  Test MAE:             {test_mae:.6f}
  Test R²:              {test_r2:.4f}
  Directional Accuracy: {directional_accuracy:.2%}

Strategy Metrics (Test):
  Sharpe (Buy & Hold):  {sharpe_buyhold:.2f}
  Sharpe (XGBoost):     {sharpe_strategy:.2f}

Files:
  Model: {model_path}
  Plot:  {plot_path}
""")

print("=" * 70)
print("DONE")
print("=" * 70)
plt.show()
