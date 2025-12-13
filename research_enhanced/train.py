import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os
import argparse
from utils import load_data, add_features, prepare_data

os.makedirs('research_enhanced/models', exist_ok=True)

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    direction_match = np.sign(y_true) == np.sign(y_pred)
    directional_accuracy = np.mean(direction_match)
    
    # Bug fix: Strategy return depends on REAL return (y_true), not predicted return
    strategy_returns = y_true * np.sign(y_pred)
    sharpe_ratio = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9) * np.sqrt(252)
    
    return rmse, directional_accuracy, sharpe_ratio

def objective(trial, X, y):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 10, 100),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'n_jobs': -1,
        'random_state': 42
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    sharpe_scores = []
    
    for train_index, val_index in tscv.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        model = xgb.XGBRegressor(**param)
        model.fit(X_train, y_train, verbose=False)
        
        preds = model.predict(X_val)
        _, _, sharpe = calculate_metrics(y_val, preds)
        sharpe_scores.append(sharpe)
        
    return np.mean(sharpe_scores)

def train_with_optuna(ticker='AAPL', timeout=3600):
    df = load_data(ticker=ticker)
    df = add_features(df)
    X, y = prepare_data(df)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), timeout=timeout)
    
    print(study.best_value)
    print(study.best_params)
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    best_params = study.best_params
    best_params['n_jobs'] = -1
    best_params['random_state'] = 42
    
    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_train, y_train)
    
    preds = final_model.predict(X_test)
    rmse, da, sharpe = calculate_metrics(y_test, preds)
    
    print(rmse)
    print(da)
    print(sharpe)
    
    joblib.dump(final_model, f'research_enhanced/models/best_xgboost_optuna_{ticker}.joblib')
    joblib.dump(study, f'research_enhanced/models/optuna_study_{ticker}.pkl')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--ticker", type=str, default="AAPL")
    args = parser.parse_args()
    
    train_with_optuna(ticker=args.ticker, timeout=args.timeout)
