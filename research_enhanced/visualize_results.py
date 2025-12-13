import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb
from utils import load_data, add_features, prepare_data

import argparse

def visualize(ticker='AAPL'):
    # Load Data
    df = load_data(ticker=ticker)
    # Save original Close prices for plotting before features modify df
    prices = df['Close'].copy()
    
    df = add_features(df)
    X, y = prepare_data(df)
    
    # Align prices with X (prepare_data drops initial rows and shifts target)
    # The 'y' target corresponds to the return from t to t+1. 
    # X index is 't'. 
    # So we need prices at 't' to calculate equity curve from returns.
    aligned_prices = prices.loc[X.index]
    
    # Load Model
    model_path = f'research_enhanced/models/best_xgboost_optuna_{ticker}.joblib'
    try:
        model = joblib.load(model_path)
    except:
        print(f"Model for {ticker} not found at {model_path}. Run training first.")
        return

    # Predict (Use all data for visualization to see full history, or split if strictly OOS)
    # For OOS visualization, let's look at the last 20% like in training
    split_idx = int(len(X) * 0.8)
    
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    prices_test = aligned_prices.iloc[split_idx:]
    
    preds = model.predict(X_test)
    
    # 1. Equity Curve Comparison
    # Strategy: 
    # If pred > 0, Long. 
    # If pred < 0, Short (or Cash? Assuming Long/Short for full symmetry)
    # Return = y_test * sign(pred)
    
    market_returns = y_test
    strategy_returns = y_test * np.sign(preds)
    
    # Cumulative Sum of Log Returns = Cumulative Log Return
    # exp(CumSum) = Normalized Price Path
    
    market_equity = np.exp(market_returns.cumsum())
    strategy_equity = np.exp(strategy_returns.cumsum())
    
    plt.figure(figsize=(12, 6))
    plt.plot(market_equity.index, market_equity, label='Buy & Hold (Market)', color='gray', alpha=0.6)
    plt.plot(strategy_equity.index, strategy_equity, label='XGBoost Strategy', color='blue')
    plt.title(f'Equity Curve: Strategy vs Market ({ticker} Out-of-Sample)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'research_enhanced/equity_curve_{ticker}.png')
    print(f"Saved equity_curve_{ticker}.png")
    
    # 2. Predicted vs Actual Scatter
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, preds, alpha=0.3, s=10)
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    plt.xlabel('Actual Log Return')
    plt.ylabel('Predicted Log Return')
    plt.title(f'Predicted vs Actual Returns ({ticker})')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'research_enhanced/pred_vs_actual_{ticker}.png')
    print(f"Saved pred_vs_actual_{ticker}.png")

    # 3. Recent Prediction Analysis
    results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': preds,
        'Price': prices_test
    })
    
    print(f"\nRecent Performance Metrics ({ticker} Test Set):")
    direction_acc = np.mean(np.sign(results['Actual']) == np.sign(results['Predicted']))
    print(f"Directional Accuracy: {direction_acc:.2%}")
    
    # Annualized Sharpe
    sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9) * np.sqrt(252)
    print(f"Sharpe Ratio: {sharpe:.4f}")

    print("\nLast 10 Predictions:")
    print(results.tail(10))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="AAPL")
    args = parser.parse_args()
    
    visualize(ticker=args.ticker)
