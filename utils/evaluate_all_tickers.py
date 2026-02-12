"""
Evaluate XGBoost on all 5 tickers for the project.

Tickers: AAPL, MSFT, EURUSD=X, BTC-USD, GC=F
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
)
from xgboost import XGBRegressor
import joblib
import os
from datetime import datetime

from src.data_loader import load_data, FEATURES
from src.config import TICKERS

# =============================================================================
# Configuration
# =============================================================================
N_SPLITS = 5
RANDOM_STATE = 42
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Smaller grid for faster evaluation across multiple tickers
PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.05, 0.1, 0.2],
    "min_child_weight": [1, 3],
    "reg_lambda": [0, 0.1],
}


def evaluate_ticker(ticker: str) -> dict:
    """Train and evaluate XGBoost for a single ticker."""
    print(f"\n{'='*70}")
    print(f"TICKER: {ticker}")
    print("=" * 70)

    try:
        data = load_data(ticker)
    except Exception as e:
        print(f"  ERROR loading data: {e}")
        return {"ticker": ticker, "error": str(e)}

    X_train, y_train = data.X_train, data.y_train
    X_val, y_val = data.X_val, data.y_val
    X_test, y_test = data.X_test, data.y_test

    # Combine train + val
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    print(f"  Train+Val: {len(X_train_full)} rows, Test: {len(X_test)} rows")

    # Grid Search
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    xgb = XGBRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        objective="reg:squarederror",
        tree_method="hist",
        verbosity=0,
    )

    search = GridSearchCV(
        estimator=xgb,
        param_grid=PARAM_GRID,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )

    print(f"  Training...")
    start = datetime.now()
    search.fit(X_train_full, y_train_full)
    elapsed = (datetime.now() - start).total_seconds()
    print(f"  Finished in {elapsed:.1f}s")

    best_model = search.best_estimator_
    best_params = search.best_params_

    # Predictions
    y_pred_test = best_model.predict(X_test)

    # Regression metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    # Directional accuracy
    actual_dir = (y_test > 0).astype(int)
    pred_dir = (y_pred_test > 0).astype(int)
    dir_accuracy = (actual_dir == pred_dir).mean()

    # Confusion matrix
    cm = confusion_matrix(actual_dir, pred_dir)

    # Trading strategy
    strategy_returns = y_test.values * pred_dir
    buyhold_returns = y_test.values

    cumret_strategy = (1 + strategy_returns).cumprod()[-1] - 1
    cumret_buyhold = (1 + buyhold_returns).cumprod()[-1] - 1

    sharpe_bh = np.sqrt(252) * buyhold_returns.mean() / buyhold_returns.std() if buyhold_returns.std() > 0 else 0
    sharpe_xgb = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0

    # Feature importance
    importance = dict(zip(FEATURES, best_model.feature_importances_))

    # Save model
    model_path = os.path.join(MODEL_DIR, f"xgboost_{ticker.replace('=', '_')}.joblib")
    save_data = {
        "model": best_model,
        "best_params": best_params,
        "feature_names": FEATURES,
        "metrics": {
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "test_r2": test_r2,
            "directional_accuracy": dir_accuracy,
        },
        "ticker": ticker,
    }
    joblib.dump(save_data, model_path)

    results = {
        "ticker": ticker,
        "train_size": len(X_train_full),
        "test_size": len(X_test),
        "rmse": test_rmse,
        "mae": test_mae,
        "r2": test_r2,
        "dir_accuracy": dir_accuracy,
        "sharpe_bh": sharpe_bh,
        "sharpe_xgb": sharpe_xgb,
        "return_bh": cumret_buyhold,
        "return_xgb": cumret_strategy,
        "cm": cm,
        "best_params": best_params,
        "importance": importance,
        "y_test": y_test.values,
        "y_pred": y_pred_test,
    }

    print(f"  RMSE: {test_rmse:.4f}, Dir Acc: {dir_accuracy:.2%}, Sharpe: {sharpe_xgb:.2f}")

    return results


def main():
    print("=" * 70)
    print("XGBOOST EVALUATION - ALL TICKERS")
    print(f"Tickers: {TICKERS}")
    print(f"Features: {FEATURES}")
    print("=" * 70)

    all_results = []

    for ticker in TICKERS:
        result = evaluate_ticker(ticker)
        all_results.append(result)

    # Filter successful results
    valid_results = [r for r in all_results if "error" not in r]

    if not valid_results:
        print("\nNo valid results!")
        return

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary_df = pd.DataFrame([{
        "Ticker": r["ticker"],
        "Test Size": r["test_size"],
        "RMSE": r["rmse"],
        "MAE": r["mae"],
        "R²": r["r2"],
        "Dir Acc": r["dir_accuracy"],
        "Sharpe (B&H)": r["sharpe_bh"],
        "Sharpe (XGB)": r["sharpe_xgb"],
        "Return (B&H)": r["return_bh"],
        "Return (XGB)": r["return_xgb"],
    } for r in valid_results])

    print("\n" + summary_df.to_string(index=False))

    # Average metrics
    print("\n" + "-" * 40)
    print("AVERAGES:")
    print(f"  Dir Accuracy: {summary_df['Dir Acc'].mean():.2%}")
    print(f"  Sharpe XGB:   {summary_df['Sharpe (XGB)'].mean():.2f}")
    print(f"  R²:           {summary_df['R²'].mean():.4f}")

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "xgboost_all_tickers_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")

    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Directional Accuracy by Ticker
    ax = axes[0, 0]
    colors = ['green' if x > 0.52 else 'orange' if x > 0.50 else 'red'
              for x in summary_df["Dir Acc"]]
    ax.barh(summary_df["Ticker"], summary_df["Dir Acc"], color=colors)
    ax.axvline(x=0.5, color="red", linestyle="--", label="Random (50%)")
    ax.set_xlabel("Directional Accuracy")
    ax.set_title("Directional Accuracy by Ticker")
    ax.set_xlim(0.45, 0.60)

    # 2. Sharpe Ratios
    ax = axes[0, 1]
    x = np.arange(len(summary_df))
    width = 0.35
    ax.bar(x - width/2, summary_df["Sharpe (B&H)"], width, label="Buy & Hold", color="blue", alpha=0.7)
    ax.bar(x + width/2, summary_df["Sharpe (XGB)"], width, label="XGBoost", color="green", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["Ticker"], rotation=45)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratios Comparison")
    ax.legend()
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    # 3. R² Scores
    ax = axes[0, 2]
    colors = ['green' if x > 0 else 'red' for x in summary_df["R²"]]
    ax.barh(summary_df["Ticker"], summary_df["R²"], color=colors)
    ax.axvline(x=0, color="red", linestyle="--")
    ax.set_xlabel("R² Score")
    ax.set_title("R² by Ticker (>0 is better than mean)")

    # 4. Cumulative Returns (B&H)
    ax = axes[1, 0]
    ax.barh(summary_df["Ticker"], summary_df["Return (B&H)"] * 100, color="blue", alpha=0.7)
    ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)
    ax.set_xlabel("Return (%)")
    ax.set_title("Buy & Hold Returns (Test Period)")

    # 5. Cumulative Returns (XGB)
    ax = axes[1, 1]
    ax.barh(summary_df["Ticker"], summary_df["Return (XGB)"] * 100, color="green", alpha=0.7)
    ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)
    ax.set_xlabel("Return (%)")
    ax.set_title("XGBoost Strategy Returns (Test Period)")

    # 6. Feature Importance (averaged)
    ax = axes[1, 2]
    avg_importance = {}
    for feat in FEATURES:
        avg_importance[feat] = np.mean([r["importance"][feat] for r in valid_results])
    imp_series = pd.Series(avg_importance).sort_values()
    imp_series.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Average Importance")
    ax.set_title("Feature Importance (Averaged)")

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "xgboost_all_tickers_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {plot_path}")

    # Print best params for each ticker
    print("\n" + "=" * 70)
    print("BEST PARAMETERS BY TICKER")
    print("=" * 70)
    for r in valid_results:
        print(f"\n{r['ticker']}:")
        for k, v in sorted(r["best_params"].items()):
            print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
