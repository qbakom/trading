#!/usr/bin/env python3
"""
Main pipeline: XGBoost + Meta-Filter for Time Series Prediction.

Usage:
    python -m src.main --ticker AAPL
    python -m src.main --ticker AAPL --target return
    python -m src.main --all --export
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .config import TICKERS, TargetType
from .data_loader import DataLoader, export_results
from .models import MetaFilter, XGBoostPredictor


def sanitize_ticker(ticker: str) -> str:
    """Convert ticker to safe filename."""
    return ticker.replace("=", "").replace("-", "")


def calculate_trading_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    prev_close: np.ndarray,
    meta_signal: np.ndarray,
) -> Dict[str, float]:
    """Calculate trading performance metrics."""
    # Daily returns
    returns = (actual - prev_close) / prev_close

    # Signal: 1 if predicted UP, -1 if predicted DOWN
    signal = np.where(predicted > prev_close, 1, -1)

    # Base strategy: follow all signals
    strategy_base = returns * signal

    # Meta strategy: only follow approved signals
    strategy_meta = strategy_base * meta_signal

    # Sharpe ratio (annualized)
    def sharpe(r):
        if r.std() == 0:
            return 0.0
        return r.mean() / r.std() * np.sqrt(252)

    # Max drawdown
    def max_drawdown(r):
        equity = (1 + r).cumprod()
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        return dd.min()

    # Total return
    def total_return(r):
        return (1 + r).prod() - 1

    return {
        "sharpe_base": sharpe(strategy_base),
        "sharpe_meta": sharpe(strategy_meta),
        "max_dd_base": max_drawdown(strategy_base),
        "max_dd_meta": max_drawdown(strategy_meta),
        "return_base": total_return(strategy_base),
        "return_meta": total_return(strategy_meta),
        "n_trades_base": len(signal),
        "n_trades_meta": int(meta_signal.sum()),
    }


def run_pipeline(
    ticker: str = "AAPL",
    target: TargetType = TargetType.PRICE,
    export: bool = False,
    output_dir: str = "outputs",
    models_dir: str = "models",
) -> Dict:
    """
    Run the full training and evaluation pipeline.

    Args:
        ticker: Asset ticker symbol
        target: Target type (PRICE, RETURN, DIRECTION)
        export: Whether to export results
        output_dir: Output directory
        models_dir: Models directory

    Returns:
        Dictionary with all results
    """
    print("=" * 70)
    print(f"Time Series Prediction Pipeline: {ticker}")
    print(f"Target: {target.value}")
    print(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    Path(output_dir).mkdir(exist_ok=True)
    Path(models_dir).mkdir(exist_ok=True)

    # =========================================================================
    # STEP 1: Load data
    # =========================================================================
    print("\n[1/5] Loading data...")
    loader = DataLoader(ticker=ticker, target=target)
    data = loader.load()

    # =========================================================================
    # STEP 2: Train XGBoost (2000-2018)
    # =========================================================================
    print("\n[2/5] Training XGBoost Base Model...")

    base_model = XGBoostPredictor(name=f"XGBoost-{ticker}")
    train_metrics = base_model.fit(data.X_train, data.y_train)

    # =========================================================================
    # STEP 3: Generate OOS predictions (2019-2021) for Meta training
    # =========================================================================
    print("\n[3/5] Generating OOS predictions for Meta-Model...")

    val_predictions = base_model.predict(data.X_val)
    val_metrics = base_model.evaluate(data.X_val, data.y_val, data.df_val["Prev_Close"])

    print(f"Validation RMSE: {val_metrics['rmse']:.4f}")
    print(f"Validation Dir. Accuracy: {val_metrics['directional_accuracy']:.2%}")

    # =========================================================================
    # STEP 4: Train Meta-Filter (2019-2021)
    # =========================================================================
    print("\n[4/5] Training Meta-Filter...")

    meta_model = MetaFilter(name=f"Meta-{ticker}")
    X_meta_val = meta_model.prepare_features(data.df_val, val_predictions)
    y_meta_val = meta_model.create_target(data.df_val, val_predictions)

    print(f"Base model correct {y_meta_val.mean():.2%} of the time")

    meta_train_metrics = meta_model.fit(X_meta_val, y_meta_val)

    # =========================================================================
    # STEP 5: Final evaluation (2022-today)
    # =========================================================================
    print("\n[5/5] Final Evaluation on Test Set...")

    # Base model predictions
    test_predictions = base_model.predict(data.X_test)
    test_metrics = base_model.evaluate(
        data.X_test, data.y_test, data.df_test["Prev_Close"]
    )

    print(f"\nBase Model Test Metrics:")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  Dir. Accuracy: {test_metrics['directional_accuracy']:.2%}")

    # Meta-model predictions
    X_meta_test = meta_model.prepare_features(data.df_test, test_predictions)
    y_meta_test = meta_model.create_target(data.df_test, test_predictions)
    meta_test_predictions = meta_model.predict(X_meta_test)

    meta_test_metrics = meta_model.evaluate(X_meta_test, y_meta_test)

    print(f"\nMeta-Filter Test Metrics:")
    print(f"  Precision: {meta_test_metrics['precision']:.2%}")
    print(f"  Base Accuracy: {meta_test_metrics['base_accuracy']:.2%}")
    print(f"  Improvement: {meta_test_metrics['precision_improvement']:+.2%}")

    # Trading metrics
    trading_metrics = calculate_trading_metrics(
        actual=data.y_test.values,
        predicted=test_predictions,
        prev_close=data.df_test["Prev_Close"].values,
        meta_signal=meta_test_predictions,
    )

    print(f"\nTrading Performance:")
    print(f"  Base: Sharpe={trading_metrics['sharpe_base']:.2f}, Return={trading_metrics['return_base']:.2%}")
    print(f"  Meta: Sharpe={trading_metrics['sharpe_meta']:.2f}, Return={trading_metrics['return_meta']:.2%}")

    # =========================================================================
    # Save models and results
    # =========================================================================
    safe_ticker = sanitize_ticker(ticker)
    base_model.save(f"{models_dir}/xgboost_{safe_ticker}.joblib")
    meta_model.save(f"{models_dir}/meta_rf_{safe_ticker}.joblib")

    if export:
        export_results(
            dates=data.df_test.index,
            actual=data.y_test.values,
            predicted=test_predictions,
            model_name=base_model.name,
            output_path=f"{output_dir}/results_{safe_ticker}.csv",
            meta_signal=meta_test_predictions,
        )

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Ticker: {ticker} | Target: {target.value}")
    print(f"Base Model: RMSE={test_metrics['rmse']:.2f}, Dir.Acc={test_metrics['directional_accuracy']:.2%}")
    print(f"Meta-Filter: Precision={meta_test_metrics['precision']:.2%}, Improvement={meta_test_metrics['precision_improvement']:+.2%}")
    print("=" * 70)

    return {
        "ticker": ticker,
        "target": target,
        "base_model": base_model,
        "meta_model": meta_model,
        "data": data,
        "test_metrics": test_metrics,
        "meta_test_metrics": meta_test_metrics,
        "trading_metrics": trading_metrics,
        "test_predictions": test_predictions,
        "meta_predictions": meta_test_predictions,
    }


def run_all_tickers(
    target: TargetType = TargetType.PRICE,
    export: bool = True,
    output_dir: str = "outputs",
    models_dir: str = "models",
) -> pd.DataFrame:
    """Run pipeline for all tickers."""
    all_results = []

    for ticker in TICKERS:
        try:
            result = run_pipeline(
                ticker=ticker,
                target=target,
                export=export,
                output_dir=output_dir,
                models_dir=models_dir,
            )

            all_results.append({
                "Ticker": ticker,
                "RMSE": result["test_metrics"]["rmse"],
                "MAE": result["test_metrics"]["mae"],
                "Dir_Accuracy": result["test_metrics"]["directional_accuracy"],
                "Meta_Precision": result["meta_test_metrics"]["precision"],
                "Meta_Improvement": result["meta_test_metrics"]["precision_improvement"],
                "Sharpe_Base": result["trading_metrics"]["sharpe_base"],
                "Sharpe_Meta": result["trading_metrics"]["sharpe_meta"],
            })
        except Exception as e:
            print(f"\nERROR processing {ticker}: {e}")
            continue

    summary_df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY - ALL TICKERS")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    summary_df.to_csv(f"{output_dir}/summary_all.csv", index=False)
    print(f"\nSummary saved to: {output_dir}/summary_all.csv")

    return summary_df


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Time Series Prediction: XGBoost + Meta-Filter"
    )
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker symbol")
    parser.add_argument("--all", action="store_true", help="Run for all tickers")
    parser.add_argument(
        "--target",
        type=str,
        choices=["price", "return", "direction"],
        default="price",
        help="Target type",
    )
    parser.add_argument("--export", action="store_true", help="Export results")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--models-dir", type=str, default="models")

    args = parser.parse_args()

    target_map = {
        "price": TargetType.PRICE,
        "return": TargetType.RETURN,
        "direction": TargetType.DIRECTION,
    }
    target = target_map[args.target]

    if args.all:
        run_all_tickers(
            target=target,
            export=args.export,
            output_dir=args.output_dir,
            models_dir=args.models_dir,
        )
    else:
        run_pipeline(
            ticker=args.ticker,
            target=target,
            export=args.export,
            output_dir=args.output_dir,
            models_dir=args.models_dir,
        )


if __name__ == "__main__":
    main()
