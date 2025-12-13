import pandas as pd
import pandas_ta as ta
import yfinance as yf
import numpy as np
import os
import time
import random

def load_data(ticker='AAPL', start_date='2000-01-01', auto_adjust=False):
    cache_file = f"{ticker}_cache.csv"
    fallback_file = "data_prepared.csv"
    
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if len(df) > 0:
                return df
        except Exception:
            pass
            
    if ticker == 'AAPL' and os.path.exists(fallback_file):
        try:
            df = pd.read_csv(fallback_file, parse_dates=['Date'])
            if 'Date' in df.columns:
                df.set_index('Date', inplace=True)
            elif 'index' in df.columns:
                 df.set_index('index', inplace=True)
            
            req_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
            if all(c in df.columns for c in req_cols):
                return df
        except Exception:
            pass

    # Alpha Vantage
    api_key = "QD4BE7OCQT8DVK2Q"
    try:
        import urllib.request
        import io
        import sys
        
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}&datatype=csv&outputsize=full"
        print(f"Attempting Alpha Vantage for {ticker} via urllib...", flush=True)
        
        with urllib.request.urlopen(url) as response:
            data = response.read().decode('utf-8')
            
        print(f"AV Response prefix: {data[:200]}", flush=True)
        
        if "Error Message" in data or "Information" in data:
            print("Alpha Vantage returned error or limit message.", flush=True)
        else:
            df = pd.read_csv(io.StringIO(data))
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
                
                df.columns = [c.capitalize() for c in df.columns]
                
                if 'Close' in df.columns and len(df) > 0:
                    df = df[df.index >= pd.to_datetime(start_date)]
                    print("Alpha Vantage download success.", flush=True)
                    df.to_csv(cache_file)
                    return df
            else:
                 print(f"Alpha Vantage Columns unexpected: {df.columns}", flush=True)
                 
    except Exception as e:
        print(f"Alpha Vantage failed: {e}", flush=True)
            
    max_retries = 3
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, start=start_date, auto_adjust=False)
            df = df.loc[:, ~df.columns.duplicated()]
            if isinstance(df.columns, pd.MultiIndex):
                if 'Close' in df.columns.get_level_values(0):
                     df.columns = df.columns.get_level_values(0)
                else:
                     df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated()]
            if len(df) > 0:
                df.to_csv(cache_file)
                return df
                
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1) + random.random() * 5)
            
    raise RuntimeError("Failed to download data.")

def add_features(df):
    df = df.copy()
    
    eps = 1e-9

    def get_column(df, col):
        if col not in df.columns:
             return pd.Series(dtype=float)
        s = df[col]
        if isinstance(s, pd.DataFrame):
            return s.iloc[:, 0]
        return s
        
    def ensure_series(s):
        if isinstance(s, pd.DataFrame):
            return s.iloc[:, 0]
        return s

    close = get_column(df, 'Close')
    volume = get_column(df, 'Volume')
    
    df['Log_Return'] = np.log(close / close.shift(1).replace(0, np.nan))
    
    # 1. Trend
    res = df.ta.sma(length=20)
    sma_20 = ensure_series(res) if res is not None else close.rolling(20).mean()
    df['Dist_SMA_20'] = close / sma_20 - 1
    
    res = df.ta.sma(length=50)
    sma_50 = ensure_series(res) if res is not None else close.rolling(50).mean()
    df['Dist_SMA_50'] = close / sma_50 - 1
    
    res = df.ta.sma(length=200)
    sma_200 = ensure_series(res) if res is not None else close.rolling(200).mean()
    df['Dist_SMA_200'] = close / sma_200 - 1

    # 2. Momentum
    res = df.ta.rsi(length=14)
    df['RSI'] = ensure_series(res) if res is not None else pd.Series(np.nan, index=df.index)
    
    stoch = df.ta.stoch(k=14, d=3, smooth_k=3)
    if stoch is not None:
        if isinstance(stoch, pd.DataFrame):
            # Try to populate STOCH_k
            if 'STOCHk_14_3_3' in stoch.columns:
                df['STOCH_k'] = stoch['STOCHk_14_3_3']
            else:
                 df['STOCH_k'] = stoch.iloc[:, 0]
        else:
            df['STOCH_k'] = stoch

    res = df.ta.willr(length=14)
    df['WILLR'] = ensure_series(res) if res is not None else pd.Series(np.nan, index=df.index)
    
    # 3. Volatility
    res = df.ta.atr(length=14)
    atr = ensure_series(res) if res is not None else pd.Series(np.nan, index=df.index)
    df['ATR_Pct'] = atr / close
    
    bb = df.ta.bbands(length=20, std=2)
    if bb is not None and isinstance(bb, pd.DataFrame):
        # BBP is typically column 4, BBW column 3. Safe access by name is better but names vary.
        # usually columns: BBL, BBM, BBU, BBB, BBP
        if 'BBP_20_2.0' in bb.columns:
             df['BBP'] = bb['BBP_20_2.0']
        elif len(bb.columns) > 4:
             df['BBP'] = bb.iloc[:, 4]
             
        if 'BBB_20_2.0' in bb.columns:
             df['BBW'] = bb['BBB_20_2.0']
        elif len(bb.columns) > 3:
             df['BBW'] = bb.iloc[:, 3]

    # 4. Volume
    df['Vol_Change'] = np.log(volume / volume.shift(1).replace(0, np.nan))
    
    sma_vol_20 = volume.rolling(20).mean()
    df['Rel_Vol'] = volume / sma_vol_20

    lags = [1, 2, 3, 5, 10]
    for lag in lags:
        df[f'Log_Return_Lag_{lag}'] = df['Log_Return'].shift(lag)
        df[f'Vol_Change_Lag_{lag}'] = df['Vol_Change'].shift(lag)
        if 'RSI' in df.columns:
            df[f'RSI_Lag_{lag}'] = df['RSI'].shift(lag)
        if 'ATR_Pct' in df.columns:
            df[f'ATR_Pct_Lag_{lag}'] = df['ATR_Pct'].shift(lag)

    return df

def prepare_data(df, target_col='Log_Return', horizon=1, dropna=True):
    df = df.copy()
    
    df = df.loc[:, ~df.columns.duplicated()]
    
    target_series = df[target_col]
    if isinstance(target_series, pd.DataFrame):
        target_series = target_series.iloc[:, 0]
        
    df['Target'] = target_series.shift(-horizon)
    
    if dropna:
        df.dropna(inplace=True)
        
    cols_to_exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Target']
    feature_cols = [c for c in df.columns if c not in cols_to_exclude]
    
    X = df[feature_cols]
    y = df['Target']
    
    return X, y

if __name__ == "__main__":
    df = load_data()
    df = add_features(df)
    X, y = prepare_data(df)
    print(X.shape, y.shape)
