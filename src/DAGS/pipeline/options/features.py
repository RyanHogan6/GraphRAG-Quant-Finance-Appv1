"""
Options Flow Feature Engineering
Calculates unusual activity, sentiment signals, and historical comparisons
"""
import pandas as pd
import numpy as np

def calculate_historical_metrics(df):
    """
    Calculate rolling averages and historical context

    Args:
        df: DataFrame with options data (must have multiple days per ticker)

    Returns:
        DataFrame with historical metrics added
    """
    df = df.sort_values(['ticker', 'date'])

    # 20-day average volume
    df['call_volume_20d_avg'] = df.groupby('ticker')['call_volume'].transform(
        lambda x: x.rolling(window=20, min_periods=5).mean()
    )
    df['put_volume_20d_avg'] = df.groupby('ticker')['put_volume'].transform(
        lambda x: x.rolling(window=20, min_periods=5).mean()
    )
    df['total_volume_20d_avg'] = df.groupby('ticker')['total_volume'].transform(
        lambda x: x.rolling(window=20, min_periods=5).mean()
    )

    # Unusual volume detection (current vs 20-day average)
    df['call_volume_unusual'] = (
        df['call_volume'] / df['call_volume_20d_avg']
    ).fillna(1.0)

    df['put_volume_unusual'] = (
        df['put_volume'] / df['put_volume_20d_avg']
    ).fillna(1.0)

    df['total_volume_unusual'] = (
        df['total_volume'] / df['total_volume_20d_avg']
    ).fillna(1.0)

    # Flag highly unusual activity (>2x average)
    df['unusual_call_activity'] = (df['call_volume_unusual'] > 2.0).astype(int)
    df['unusual_put_activity'] = (df['put_volume_unusual'] > 2.0).astype(int)
    df['unusual_total_activity'] = (df['total_volume_unusual'] > 2.0).astype(int)

    return df


def calculate_iv_rank(df):
    """
    Calculate implied volatility rank (IV percentile over 52 weeks)

    Args:
        df: DataFrame with call_iv_avg and put_iv_avg

    Returns:
        DataFrame with IV rank added
    """
    df = df.sort_values(['ticker', 'date'])

    # Use average of call and put IV
    df['iv_avg'] = df[['call_iv_avg', 'put_iv_avg']].mean(axis=1)

    # 52-week IV rank (percentile)
    def calculate_percentile(series):
        """Calculate percentile rank (0-100)"""
        if len(series) < 2:
            return 50.0  # Default to middle if not enough data

        current = series.iloc[-1]
        if pd.isna(current):
            return None

        # Count how many historical values are below current
        below_count = (series < current).sum()
        percentile = (below_count / len(series)) * 100

        return percentile

    df['iv_rank'] = df.groupby('ticker')['iv_avg'].transform(
        lambda x: x.rolling(window=252, min_periods=20).apply(calculate_percentile, raw=False)
    )

    # Classify IV rank
    df['iv_rank_class'] = pd.cut(
        df['iv_rank'],
        bins=[0, 25, 50, 75, 100],
        labels=['Low', 'Medium', 'High', 'Very High'],
        include_lowest=True
    )

    return df


def calculate_sentiment_signals(df):
    """
    Generate sentiment signals based on put/call ratios and unusual activity

    Args:
        df: DataFrame with options metrics

    Returns:
        DataFrame with sentiment signals
    """
    # Put/call ratio interpretation
    # < 0.7 = Bullish (more calls than puts)
    # 0.7-1.0 = Neutral-Bullish
    # 1.0-1.5 = Neutral-Bearish
    # > 1.5 = Bearish (more puts than calls)

    df['sentiment_pc_ratio'] = pd.cut(
        df['put_call_volume_ratio'],
        bins=[0, 0.7, 1.0, 1.5, float('inf')],
        labels=['Bullish', 'Neutral-Bullish', 'Neutral-Bearish', 'Bearish'],
        include_lowest=True
    )

    # Bullish signals
    df['bullish_signal'] = (
        (df['call_volume_unusual'] > 1.5) &  # Elevated call buying
        (df['put_call_volume_ratio'] < 0.7)   # Low put/call ratio
    ).astype(int)

    # Bearish signals
    df['bearish_signal'] = (
        (df['put_volume_unusual'] > 1.5) &    # Elevated put buying
        (df['put_call_volume_ratio'] > 1.5)   # High put/call ratio
    ).astype(int)

    # Call sweep detection (very high call volume with low P/C ratio)
    df['potential_call_sweep'] = (
        (df['call_volume_unusual'] > 3.0) &
        (df['put_call_volume_ratio'] < 0.5)
    ).astype(int)

    # Put sweep detection (very high put volume with high P/C ratio)
    df['potential_put_sweep'] = (
        (df['put_volume_unusual'] > 3.0) &
        (df['put_call_volume_ratio'] > 2.0)
    ).astype(int)

    # High IV with unusual call buying (bullish speculation)
    df['high_iv_call_buying'] = (
        (df['iv_rank'] > 75) &
        (df['call_volume_unusual'] > 2.0)
    ).astype(int)

    return df


def calculate_momentum(df):
    """
    Calculate day-over-day changes in options metrics

    Args:
        df: DataFrame sorted by ticker and date

    Returns:
        DataFrame with momentum features
    """
    df = df.sort_values(['ticker', 'date'])

    # Put/call ratio change
    df['put_call_ratio_change'] = df.groupby('ticker')['put_call_volume_ratio'].diff()

    # Volume changes (absolute and percentage)
    df['call_volume_change'] = df.groupby('ticker')['call_volume'].diff()
    df['put_volume_change'] = df.groupby('ticker')['put_volume'].diff()

    df['call_volume_pct_change'] = df.groupby('ticker')['call_volume'].pct_change()
    df['put_volume_pct_change'] = df.groupby('ticker')['put_volume'].pct_change()

    # IV changes
    df['iv_change'] = df.groupby('ticker')['iv_avg'].diff()

    return df


def engineer_options_features(df):
    """
    Main feature engineering pipeline for options data

    Args:
        df: Raw options DataFrame from downloader

    Returns:
        DataFrame with engineered features
    """
    if df.empty:
        return df

    print(f"  Engineering options features: {len(df)} raw records")

    # Ensure date is datetime for calculations
    df['date'] = pd.to_datetime(df['date'])

    # Calculate all features
    df = calculate_historical_metrics(df)
    df = calculate_iv_rank(df)
    df = calculate_sentiment_signals(df)
    df = calculate_momentum(df)

    # Convert date back to string for ArangoDB
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    # Add ingestion timestamp
    from datetime import datetime
    df['ingested_at'] = datetime.utcnow().isoformat()
    df['data_source'] = 'Yahoo Finance Options'

    # Replace NaN/inf with None for ArangoDB
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.where(pd.notnull(df), None)

    print(f"  Final options data: {len(df)} records with {len(df.columns)} features")

    return df
