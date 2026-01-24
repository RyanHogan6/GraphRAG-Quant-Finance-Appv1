"""
CME/NYMEX Futures Prices Downloader
Fetches commodity futures prices to complement CFTC positioning data
"""
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Major commodity futures contracts
# Using continuous contracts (=F suffix)
FUTURES_CONTRACTS = {
    # Energy
    'CL=F': {'name': 'Crude Oil WTI', 'commodity': 'CRUDE_OIL', 'unit': 'USD/barrel', 'exchange': 'NYMEX'},
    'NG=F': {'name': 'Natural Gas', 'commodity': 'NATURAL_GAS', 'unit': 'USD/MMBtu', 'exchange': 'NYMEX'},
    'RB=F': {'name': 'Gasoline RBOB', 'commodity': 'GASOLINE', 'unit': 'USD/gallon', 'exchange': 'NYMEX'},
    'HO=F': {'name': 'Heating Oil', 'commodity': 'HEATING_OIL', 'unit': 'USD/gallon', 'exchange': 'NYMEX'},

    # Metals
    'GC=F': {'name': 'Gold', 'commodity': 'GOLD', 'unit': 'USD/oz', 'exchange': 'COMEX'},
    'SI=F': {'name': 'Silver', 'commodity': 'SILVER', 'unit': 'USD/oz', 'exchange': 'COMEX'},
    'HG=F': {'name': 'Copper', 'commodity': 'COPPER', 'unit': 'USD/lb', 'exchange': 'COMEX'},
    'PL=F': {'name': 'Platinum', 'commodity': 'PLATINUM', 'unit': 'USD/oz', 'exchange': 'NYMEX'},

    # Agriculture
    'ZC=F': {'name': 'Corn', 'commodity': 'CORN', 'unit': 'USD/bushel', 'exchange': 'CBOT'},
    'ZW=F': {'name': 'Wheat', 'commodity': 'WHEAT', 'unit': 'USD/bushel', 'exchange': 'CBOT'},
    'ZS=F': {'name': 'Soybeans', 'commodity': 'SOYBEANS', 'unit': 'USD/bushel', 'exchange': 'CBOT'},
    'ZL=F': {'name': 'Soybean Oil', 'commodity': 'SOYBEAN_OIL', 'unit': 'USD/lb', 'exchange': 'CBOT'},
    'CT=F': {'name': 'Cotton', 'commodity': 'COTTON', 'unit': 'USD/lb', 'exchange': 'ICE'},
    'KC=F': {'name': 'Coffee', 'commodity': 'COFFEE', 'unit': 'USD/lb', 'exchange': 'ICE'},
    'SB=F': {'name': 'Sugar', 'commodity': 'SUGAR', 'unit': 'USD/lb', 'exchange': 'ICE'},

    # Livestock
    'LE=F': {'name': 'Live Cattle', 'commodity': 'LIVE_CATTLE', 'unit': 'USD/lb', 'exchange': 'CME'},
    'HE=F': {'name': 'Lean Hogs', 'commodity': 'LEAN_HOGS', 'unit': 'USD/lb', 'exchange': 'CME'},
}

def fetch_futures_contract(symbol, metadata, days_back=90):
    """
    Fetch single futures contract data from Yahoo Finance

    Args:
        symbol: Futures symbol (e.g., 'CL=F')
        metadata: Dict with name, commodity, unit, exchange
        days_back: Number of days of historical data to fetch

    Returns:
        DataFrame with OHLCV data
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        print(f"  Fetching {metadata['name']} ({symbol})...")

        # Fetch data from yfinance
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval='1d')

        if df.empty:
            print(f"    ✗ No data for {symbol}")
            return None

        # Reset index to make Date a column
        df = df.reset_index()

        # Rename columns to match our schema
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        # Add metadata
        df['symbol'] = symbol
        df['commodity'] = metadata['commodity']
        df['commodity_name'] = metadata['name']
        df['unit'] = metadata['unit']
        df['exchange'] = metadata['exchange']
        df['data_source'] = 'Yahoo Finance'

        # Format date
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        # Select only needed columns
        df = df[['date', 'symbol', 'commodity', 'commodity_name', 'exchange', 'unit',
                 'open', 'high', 'low', 'close', 'volume', 'data_source']]

        print(f"    ✓ Fetched {len(df)} days")
        return df

    except Exception as e:
        print(f"    ✗ Error fetching {symbol}: {e}")
        return None


def fetch_all_futures_data(days_back=90, contracts=None):
    """
    Fetch all futures contracts data

    Args:
        days_back: Number of days of historical data (default 90 for incremental)
        contracts: List of symbols to fetch (default: all)

    Returns:
        DataFrame with all futures data
    """
    print(f"\nFetching CME/NYMEX futures data (last {days_back} days)...")

    if contracts is None:
        contracts = FUTURES_CONTRACTS.keys()

    all_data = []

    for symbol in contracts:
        if symbol not in FUTURES_CONTRACTS:
            print(f"  ⚠️  Unknown symbol: {symbol}")
            continue

        metadata = FUTURES_CONTRACTS[symbol]
        df = fetch_futures_contract(symbol, metadata, days_back)

        if df is not None and not df.empty:
            all_data.append(df)

    if not all_data:
        print("✗ No futures data fetched")
        return pd.DataFrame()

    # Combine all contracts
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n✓ Fetched {len(combined_df)} total records across {len(all_data)} contracts")

    return combined_df


def fetch_front_month_contracts(days_back=90):
    """
    Fetch only front-month (most liquid) contracts for key commodities
    Useful for initial testing with lower data volume
    """
    key_contracts = [
        'CL=F',  # Crude Oil
        'NG=F',  # Natural Gas
        'GC=F',  # Gold
        'SI=F',  # Silver
        'HG=F',  # Copper
        'ZC=F',  # Corn
        'ZW=F',  # Wheat
    ]

    return fetch_all_futures_data(days_back=days_back, contracts=key_contracts)
