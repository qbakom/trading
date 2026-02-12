"""
XGBoost Training Script v2 - Fixed underfitting issue.

Problem w v1: Model predykował stałą wartość (za silna regularyzacja).
Rozwiązanie: Mniejsza regularyzacja, lepszy grid, diagnostyka.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
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
print("XGBOOST TRAINING v2 - FIXED")
print("=" * 70)
print(f"\nTicker: {TICKER}")
print(f"Features: {FEATURES}")

data = load_data(TICKER)

X_train, y_train = data.X_train, data.y_train
X_val, y_val = data.X_val, data.y_val
X_test, y_test = data.X_test, data.y_test

# Combine train + val for CV
X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])

print(f"\nData shapes:")
print(f"  Train: {X_train.shape}")
print(f"  Val:   {X_val.shape}")
print(f"  Test:  {X_test.shape}")
print(f"  Train+Val: {X_train_full.shape}")

# =============================================================================
# 2. Data Diagnostics
# =============================================================================
print("\n" + "=" * 70)
print("DATA DIAGNOSTICS")
print("=" * 70)

print(f"\nTarget (y_train) statistics:")
print(f"  Mean:   {y_train_full.mean():.6f}")
print(f"  Std:    {y_train_full.std():.6f}")
print(f"  Min:    {y_train_full.min():.6f}")
print(f"  Max:    {y_train_full.max():.6f}")

print(f"\nFeature correlations with target:")
for col in FEATURES:
    corr = X_train_full[col].corr(y_train_full)
    print(f"  {col}: {corr:.4f}")

# =============================================================================
# 3. Define Parameter Grid (FIXED - less regularization)
# =============================================================================
# Poprzedni problem: reg_lambda=5.0 i gamma=0.2 były za duże
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3],
    "reg_alpha": [0, 0.01, 0.1],      # L1 - mała
    "reg_lambda": [0, 0.1, 1.0],      # L2 - mała (NIE 5.0!)
    "gamma": [0, 0.01],               # Mała (NIE 0.2!)
}

total = 1
for v in param_grid.values():
    total *= len(v)
print(f"\nTotal combinations: {total:,}")

# =============================================================================
# 4. Train - najpierw prosty model bez regularyzacji
# =============================================================================
print("\n" + "=" * 70)
print("TRAINING BASELINE (no regularization)")
print("=" * 70)

# Baseline - prosty model
baseline = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
baseline.fit(X_train_full, y_train_full)
baseline_pred = baseline.predict(X_test)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
print(f"Baseline RMSE: {baseline_rmse:.6f}")
print(f"Baseline predictions range: [{baseline_pred.min():.6f}, {baseline_pred.max():.6f}]")
print(f"Actual values range: [{y_test.min():.6f}, {y_test.max():.6f}]")

# Check if baseline is predicting constant
if baseline_pred.std() < 0.0001:
    print("\n*** WARNING: Baseline predicts near-constant values! ***")
    print("This suggests a data issue or extreme regularization.")

# =============================================================================
# 5. Grid Search with reduced regularization
# =============================================================================
print("\n" + "=" * 70)
print("GRID SEARCH (reduced regularization)")
print("=" * 70)

tscv = TimeSeriesSplit(n_splits=N_SPLITS)

xgb = XGBRegressor(
    random_state=RANDOM_STATE,
    n_jobs=-1,
    objective="reg:squarederror",
    tree_method="hist",
    verbosity=0,
)

# Use smaller grid for faster search
param_grid_small = {
    "n_estimators": [50, 100, 200],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.05, 0.1, 0.2],
    "min_child_weight": [1, 3],
    "reg_lambda": [0, 0.1],
    "gamma": [0],
}

search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid_small,
    cv=tscv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1,
)

print(f"Starting at {datetime.now().strftime('%H:%M:%S')}...")
search.fit(X_train_full, y_train_full)
print(f"Finished at {datetime.now().strftime('%H:%M:%S')}")

best_model = search.best_estimator_
best_params = search.best_params_
best_cv_rmse = -search.best_score_

print(f"\nBest parameters:")
for k, v in sorted(best_params.items()):
    print(f"  {k}: {v}")
print(f"\nBest CV RMSE: {best_cv_rmse:.6f}")

# =============================================================================
# 6. Evaluate
# =============================================================================
print("\n" + "=" * 70)
print("EVALUATION")
print("=" * 70)

y_pred_train = best_model.predict(X_train_full)
y_pred_test = best_model.predict(X_test)

# Check prediction variance
print(f"\nPrediction diagnostics:")
print(f"  Train pred std: {y_pred_train.std():.6f}")
print(f"  Test pred std:  {y_pred_test.std():.6f}")
print(f"  Test actual std: {y_test.std():.6f}")

if y_pred_test.std() < y_test.std() * 0.1:
    print("  *** WARNING: Predictions have very low variance! ***")

# Metrics
train_rmse = np.sqrt(mean_squared_error(y_train_full, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

print(f"\nRegression Metrics:")
print(f"  Train RMSE: {train_rmse:.6f}")
print(f"  Test RMSE:  {test_rmse:.6f}")
print(f"  Test MAE:   {test_mae:.6f}")
print(f"  Test R²:    {test_r2:.4f}")

# Direction
actual_dir = (y_test > 0).astype(int)
pred_dir = (y_pred_test > 0).astype(int)
dir_accuracy = (actual_dir == pred_dir).mean()

print(f"\nDirectional Accuracy: {dir_accuracy:.2%}")

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(actual_dir, pred_dir)
print(f"\nConfusion Matrix:")
print(f"  Predicted DOWN | Predicted UP")
print(f"  Actual DOWN: {cm[0,0]:4d} | {cm[0,1]:4d}")
print(f"  Actual UP:   {cm[1,0]:4d} | {cm[1,1]:4d}")

# Check if model always predicts same direction
if cm[0,0] + cm[1,0] == 0:
    print("\n  *** Model ALWAYS predicts UP! ***")
elif cm[0,1] + cm[1,1] == 0:
    print("\n  *** Model ALWAYS predicts DOWN! ***")

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

if importance.max() < 0.01:
    print("\n*** WARNING: All feature importances near zero! ***")

# =============================================================================
# 8. Save Model
# =============================================================================
model_path = os.path.join(MODEL_DIR, f"xgboost_{TICKER}.joblib")
save_data = {
    "model": best_model,
    "best_params": best_params,
    "feature_names": FEATURES,
    "metrics": {
        "cv_rmse": best_cv_rmse,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "directional_accuracy": dir_accuracy,
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
colors = ['green' if x > 0.1 else 'orange' if x > 0.01 else 'red' for x in importance]
importance.plot(kind="barh", ax=ax, color=colors)
ax.set_title("Feature Importance (Gain)")
ax.set_xlabel("Importance")
ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5)
ax.invert_yaxis()

# 2. Actual vs Predicted
ax = axes[0, 1]
ax.scatter(y_test, y_pred_test, alpha=0.3, s=10)
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
ax.set_xlabel("Actual Return")
ax.set_ylabel("Predicted Return")
ax.set_title(f"Actual vs Predicted\nR² = {test_r2:.4f}")
ax.grid(True, alpha=0.3)

# 3. Prediction Distribution
ax = axes[0, 2]
ax.hist(y_test, bins=50, alpha=0.5, label="Actual", color="blue", density=True)
ax.hist(y_pred_test, bins=50, alpha=0.5, label="Predicted", color="orange", density=True)
ax.set_xlabel("Return")
ax.set_ylabel("Density")
ax.set_title("Distribution Comparison")
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

# 5. Cumulative Returns
ax = axes[1, 1]
strategy_returns = y_test.values * pred_dir
cumret_strategy = (1 + strategy_returns).cumprod()
cumret_buyhold = (1 + y_test.values).cumprod()

ax.plot(cumret_buyhold, label="Buy & Hold", color="blue", alpha=0.7)
ax.plot(cumret_strategy, label="XGBoost Strategy", color="green", alpha=0.7)
ax.set_xlabel("Trading Day")
ax.set_ylabel("Cumulative Return")
ax.set_title("Cumulative Returns (Test)")
ax.legend()
ax.grid(True, alpha=0.3)

sharpe_bh = np.sqrt(252) * y_test.mean() / y_test.std() if y_test.std() > 0 else 0
sharpe_xgb = np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0
ax.text(0.02, 0.98, f"Sharpe B&H: {sharpe_bh:.2f}\nSharpe XGB: {sharpe_xgb:.2f}",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

# 6. Confusion Matrix
ax = axes[1, 2]
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"])
ax.set_xlabel("Predicted Direction")
ax.set_ylabel("Actual Direction")
ax.set_title(f"Direction Confusion Matrix\nAccuracy: {dir_accuracy:.2%}")

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, f"xgboost_{TICKER}_results.png")
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

Best Parameters:
{pd.Series(best_params).to_string()}

Metrics:
  CV RMSE:              {best_cv_rmse:.6f}
  Test RMSE:            {test_rmse:.6f}
  Test MAE:             {test_mae:.6f}
  Test R²:              {test_r2:.4f}
  Directional Accuracy: {dir_accuracy:.2%}

Strategy:
  Sharpe (Buy & Hold):  {sharpe_bh:.2f}
  Sharpe (XGBoost):     {sharpe_xgb:.2f}

Files:
  Model: {model_path}
  Plot:  {plot_path}
""")

print("=" * 70)
print("DONE")
print("=" * 70)
plt.show()
