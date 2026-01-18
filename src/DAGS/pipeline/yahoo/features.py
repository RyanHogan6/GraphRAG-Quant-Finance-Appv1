"""Yahoo MarketData feature engineering - technical + fundamental indicators"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

def calculate_sma(df, periods=[5, 10, 20, 50, 200]):
    """Calculate Simple Moving Averages"""
    for period in periods:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
    return df

def calculate_ema(df, periods=[12, 26]):
    """Calculate Exponential Moving Averages"""
    for period in periods:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    return df

def calculate_macd(df):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    return df

def calculate_rsi(df, period=14):
    """Calculate Relative Strength Index"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    return df

def calculate_bollinger_bands(df, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    df['bb_middle'] = df['close'].rolling(window=period).mean()
    bb_std = df['close'].rolling(window=period).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * std_dev)
    df['bb_lower'] = df['bb_middle'] - (bb_std * std_dev)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    return df

def calculate_volatility(df):
    """Calculate volatility metrics"""
    df['daily_return'] = df['close'].pct_change()
    df['volatility_10d'] = df['daily_return'].rolling(window=10).std()
    df['volatility_30d'] = df['daily_return'].rolling(window=30).std()

    # True Range
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr_14'] = df['tr'].rolling(window=14).mean()
    return df

def calculate_volume_indicators(df):
    """Calculate volume-based indicators"""
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']

    # On-Balance Volume
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    return df

def calculate_momentum(df):
    """Calculate momentum indicators"""
    df['roc_10'] = df['close'].pct_change(periods=10) * 100
    df['roc_20'] = df['close'].pct_change(periods=20) * 100
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    return df

def calculate_distance_metrics(df):
    """Calculate distance from moving averages"""
    df['dist_from_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
    df['dist_from_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
    df['dist_from_sma200'] = (df['close'] - df['sma_200']) / df['sma_200']
    return df

def calculate_52week_metrics(df):
    """Calculate 52-week high/low metrics"""
    df['rolling_52w_high'] = df['high'].rolling(window=252).max()
    df['rolling_52w_low'] = df['low'].rolling(window=252).min()
    df['pct_from_52w_high'] = (df['close'] - df['rolling_52w_high']) / df['rolling_52w_high']
    df['pct_from_52w_low'] = (df['close'] - df['rolling_52w_low']) / df['rolling_52w_low']
    df.drop(['rolling_52w_high', 'rolling_52w_low'], axis=1, inplace=True)
    return df

def calculate_signals(df):
    """Calculate trading signals"""
    df['golden_cross'] = ((df['sma_50'] > df['sma_200']) &
                          (df['sma_50'].shift(1) <= df['sma_200'].shift(1))).astype(int)
    df['death_cross'] = ((df['sma_50'] < df['sma_200']) &
                         (df['sma_50'].shift(1) >= df['sma_200'].shift(1))).astype(int)
    df['above_sma20'] = (df['close'] > df['sma_20']).astype(int)
    df['above_sma50'] = (df['close'] > df['sma_50']).astype(int)
    df['above_sma200'] = (df['close'] > df['sma_200']).astype(int)
    return df

def calculate_time_features(df):
    """Calculate time-based features"""
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    return df

def fetch_fundamental_data(ticker):
    """Fetch fundamental data from yfinance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        fundamentals = {
            # Price targets
            'targetMeanPrice': info.get('targetMeanPrice'),
            'targetHighPrice': info.get('targetHighPrice'),
            'targetLowPrice': info.get('targetLowPrice'),
            'targetMedianPrice': info.get('targetMedianPrice'),

            # Analyst data
            'recommendationKey': info.get('recommendationKey'),
            'numberOfAnalystOpinions': info.get('numberOfAnalystOpinions'),

            # Earnings
            'forwardEps': info.get('forwardEps'),
            'trailingEps': info.get('trailingEps'),
            'earningsGrowth': info.get('earningsGrowth'),
            'earningsQuarterlyGrowth': info.get('earningsQuarterlyGrowth'),

            # Revenue
            'revenueGrowth': info.get('revenueGrowth'),
            'revenuePerShare': info.get('revenuePerShare'),

            # Profitability
            'returnOnEquity': info.get('returnOnEquity'),
            'returnOnAssets': info.get('returnOnAssets'),
            'grossMargins': info.get('grossMargins'),
            'ebitdaMargins': info.get('ebitdaMargins'),
            'operatingMargins': info.get('operatingMargins'),
            'profitMargins': info.get('profitMargins'),

            # Valuation
            'trailingPE': info.get('trailingPE'),
            'forwardPE': info.get('forwardPE'),
            'pegRatio': info.get('pegRatio'),
            'priceToBook': info.get('priceToBook'),
            'priceToSalesTrailing12Months': info.get('priceToSalesTrailing12Months'),
            'enterpriseToRevenue': info.get('enterpriseToRevenue'),
            'enterpriseToEbitda': info.get('enterpriseToEbitda'),

            # Debt & Cash
            'debtToEquity': info.get('debtToEquity'),
            'totalDebt': info.get('totalDebt'),
            'totalCash': info.get('totalCash'),
            'currentRatio': info.get('currentRatio'),
            'quickRatio': info.get('quickRatio'),

            # Cash Flow
            'freeCashflow': info.get('freeCashflow'),
            'operatingCashflow': info.get('operatingCashflow'),

            # Dividends
            'dividendRate': info.get('dividendRate'),
            'dividendYield': info.get('dividendYield'),
            'payoutRatio': info.get('payoutRatio'),
            'fiveYearAvgDividendYield': info.get('fiveYearAvgDividendYield'),

            # Risk & Price History
            'beta': info.get('beta'),
            'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh'),
            'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow'),
            'fiftyDayAverage': info.get('fiftyDayAverage'),
            'twoHundredDayAverage': info.get('twoHundredDayAverage'),

            # Company data for MarketData (already in Company collection)
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'country': info.get('country'),
            'city': info.get('city'),
            'website': info.get('website'),
            'fullTimeEmployees': info.get('fullTimeEmployees'),
            'marketCap': info.get('marketCap'),
            'sharesOutstanding': info.get('sharesOutstanding'),
        }

        return fundamentals
    except Exception as e:
        print(f"  Warning: Could not fetch fundamentals for {ticker}: {e}")
        return {}

def engineer_technical_features(df, include_fundamentals=False):
    """
    Engineer all technical and fundamental features for Yahoo MarketData

    Input: DataFrame with columns [date, ticker, open, high, low, close, volume]
    Output: DataFrame with 60+ engineered features

    Args:
        include_fundamentals: If True, fetch fundamental data (adds extra API calls)
                             If False, only calculate technical indicators (faster, avoids rate limits)
    """
    # Check for empty DataFrame or missing ticker column
    if df.empty or 'ticker' not in df.columns:
        print(f"  Warning: Empty DataFrame or missing ticker column")
        return df

    print(f"[FEATURES] Engineering features for {df['ticker'].nunique()} tickers...")
    if not include_fundamentals:
        print("  Skipping fundamentals (technical indicators only)")

    if len(df) < 200:
        print(f"  Warning: Insufficient data ({len(df)} rows)")
        return df

    # Group by ticker and process each separately
    result_dfs = []

    for ticker, group in df.groupby('ticker'):
        group = group.sort_values('date').copy()

        # Fetch fundamentals (once per ticker) - optional to avoid rate limiting
        if include_fundamentals:
            fundamentals = fetch_fundamental_data(ticker)
        else:
            fundamentals = {}

        # Technical indicators
        group = calculate_sma(group)
        group = calculate_ema(group)
        group = calculate_macd(group)
        group = calculate_rsi(group)
        group = calculate_bollinger_bands(group)
        group = calculate_volatility(group)
        group = calculate_volume_indicators(group)
        group = calculate_momentum(group)
        group = calculate_distance_metrics(group)
        group = calculate_52week_metrics(group)
        group = calculate_signals(group)
        group = calculate_time_features(group)

        # Add fundamentals (broadcast to all rows)
        for key, value in fundamentals.items():
            group[key] = value

        # Calculated fields
        if 'trailingEps' in group.columns and group['trailingEps'].notna().any():
            group['calculated_pe'] = group['close'] / group['trailingEps']

        if 'dividendRate' in group.columns and group['dividendRate'].notna().any():
            group['calculated_div_yield'] = group['dividendRate'] / group['close']

        if 'sharesOutstanding' in group.columns and group['sharesOutstanding'].notna().any():
            group['daily_market_cap'] = group['close'] * group['sharesOutstanding']

        result_dfs.append(group)

    result = pd.concat(result_dfs, ignore_index=True)
    print(f"  ✓ Engineered {len(result)} rows with {len(result.columns)} features")

    return result
