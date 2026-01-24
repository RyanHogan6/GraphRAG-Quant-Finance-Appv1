"""
CME Futures Feature Engineering
Calculate technical indicators and price momentum features
"""
import pandas as pd
import numpy as np

def calculate_returns(df):
    """Calculate daily and weekly returns"""
    df = df.sort_values('date')

    # Daily returns
    df['daily_return'] = df.groupby('commodity')['close'].pct_change()

    # Weekly returns (5-day)
    df['weekly_return'] = df.groupby('commodity')['close'].pct_change(periods=5)

    # Monthly returns (20-day approximation)
    df['monthly_return'] = df.groupby('commodity')['close'].pct_change(periods=20)

    return df


def calculate_moving_averages(df):
    """Calculate moving averages for trend analysis"""
    df = df.sort_values('date')

    for window in [5, 10, 20, 50, 200]:
        df[f'sma_{window}'] = df.groupby('commodity')['close'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )

    # Price relative to moving averages
    df['above_sma20'] = (df['close'] > df['sma_20']).astype(int)
    df['above_sma50'] = (df['close'] > df['sma_50']).astype(int)
    df['above_sma200'] = (df['close'] > df['sma_200']).astype(int)

    # Distance from moving averages (percentage)
    df['dist_from_sma20'] = ((df['close'] - df['sma_20']) / df['sma_20']) * 100
    df['dist_from_sma50'] = ((df['close'] - df['sma_50']) / df['sma_50']) * 100

    return df


def calculate_volatility(df):
    """Calculate volatility metrics"""
    df = df.sort_values('date')

    # Historical volatility (20-day)
    df['volatility_20d'] = df.groupby('commodity')['daily_return'].transform(
        lambda x: x.rolling(window=20, min_periods=10).std() * np.sqrt(252)  # Annualized
    )

    # Average True Range (ATR)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df.groupby('commodity')['close'].shift(1)),
            abs(df['low'] - df.groupby('commodity')['close'].shift(1))
        )
    )
    df['atr_14'] = df.groupby('commodity')['tr'].transform(
        lambda x: x.rolling(window=14, min_periods=1).mean()
    )

    # High-Low range as percentage of close
    df['daily_range_pct'] = ((df['high'] - df['low']) / df['close']) * 100

    return df


def calculate_momentum(df):
    """Calculate momentum indicators"""
    df = df.sort_values('date')

    # RSI (14-day)
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df['rsi_14'] = df.groupby('commodity')['close'].transform(calculate_rsi)

    # MACD
    df['ema_12'] = df.groupby('commodity')['close'].transform(
        lambda x: x.ewm(span=12, adjust=False).mean()
    )
    df['ema_26'] = df.groupby('commodity')['close'].transform(
        lambda x: x.ewm(span=26, adjust=False).mean()
    )
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df.groupby('commodity')['macd'].transform(
        lambda x: x.ewm(span=9, adjust=False).mean()
    )
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    return df


def calculate_price_levels(df):
    """Calculate support/resistance levels"""
    df = df.sort_values('date')

    # 52-week high/low
    df['high_52w'] = df.groupby('commodity')['high'].transform(
        lambda x: x.rolling(window=252, min_periods=1).max()
    )
    df['low_52w'] = df.groupby('commodity')['low'].transform(
        lambda x: x.rolling(window=252, min_periods=1).min()
    )

    # Distance from 52-week high/low
    df['pct_off_high'] = ((df['close'] - df['high_52w']) / df['high_52w']) * 100
    df['pct_off_low'] = ((df['close'] - df['low_52w']) / df['low_52w']) * 100

    # At 52-week high/low (within 2%)
    df['at_52w_high'] = (df['pct_off_high'] > -2).astype(int)
    df['at_52w_low'] = (df['pct_off_low'] < 2).astype(int)

    return df


def engineer_futures_features(df):
    """
    Main feature engineering pipeline for futures data

    Args:
        df: Raw futures DataFrame from downloader

    Returns:
        DataFrame with engineered features
    """
    if df.empty:
        return df

    print(f"  Engineering futures features: {len(df)} raw records")

    # Ensure date is datetime for calculations
    df['date'] = pd.to_datetime(df['date'])

    # Calculate all features
    df = calculate_returns(df)
    df = calculate_moving_averages(df)
    df = calculate_volatility(df)
    df = calculate_momentum(df)
    df = calculate_price_levels(df)

    # Convert date back to string for ArangoDB
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    # Add ingestion timestamp
    from datetime import datetime
    df['ingested_at'] = datetime.utcnow().isoformat()

    # Drop intermediate calculation columns
    df = df.drop(columns=['tr', 'ema_12', 'ema_26'], errors='ignore')

    # Replace NaN/inf with None for ArangoDB
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.where(pd.notnull(df), None)

    print(f"  Final futures data: {len(df)} records with {len(df.columns)} features")

    return df
