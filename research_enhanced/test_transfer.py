import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb
import os
from utils import load_data, add_features, prepare_data

def test_transfer(model_ticker='AAPL', target_ticker='MSFT'):
    print(f"Loading data for Target: {target_ticker}...")
    df = load_data(ticker=target_ticker)
    
    prices = df['Close'].copy()
    
    df = add_features(df)
    X, y = prepare_data(df)
    
    aligned_prices = prices.loc[X.index]
    
    model_path = f'research_enhanced/models/best_xgboost_optuna_{model_ticker}.joblib'
    if not os.path.exists(model_path):
         # Try generic name if specific not found
         model_path = 'research_enhanced/models/best_xgboost_optuna.joblib'
         
    print(f"Loading model trained on: {model_ticker} from {model_path}...")
    try:
        model = joblib.load(model_path)
    except:
        print(f"Model for {model_ticker} not found. Run training first.")
        return

    print("Generating predictions on target data...")
    preds = model.predict(X)
    
    market_returns = y
    strategy_returns = y * np.sign(preds)
    
    market_equity = np.exp(market_returns.cumsum())
    strategy_equity = np.exp(strategy_returns.cumsum())
    
    plt.figure(figsize=(12, 6))
    plt.plot(market_equity.index, market_equity, label=f'Buy & Hold ({target_ticker})', color='gray', alpha=0.6)
    plt.plot(strategy_equity.index, strategy_equity, label=f'Strategy (Model: {model_ticker})', color='purple')
    plt.title(f'Transfer learning: Model({model_ticker}) on Data({target_ticker})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    filename = f'research_enhanced/equity_curve_transfer_{model_ticker}_to_{target_ticker}.png'
    plt.savefig(filename)
    print(f"Saved {filename}")
    
    direction_match = np.sign(y) == np.sign(preds)
    direction_acc = np.mean(direction_match)
    
    sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9) * np.sqrt(252)
    
    print(f"\nTransfer Performance ({model_ticker} -> {target_ticker}):")
    print(f"Directional Accuracy: {direction_acc:.2%}")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    
    results = pd.DataFrame({
        'Actual': y,
        'Predicted': preds,
        'Price': aligned_prices
    })
    print("\nRecent Predictions:")
    print(results.tail(10))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ticker", type=str, default="AAPL")
    parser.add_argument("--target_ticker", type=str, default="MSFT")
    args = parser.parse_args()
    
    test_transfer(model_ticker=args.model_ticker, target_ticker=args.target_ticker)
