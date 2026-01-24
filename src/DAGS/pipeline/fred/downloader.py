"""
FRED (Federal Reserve Economic Data) Downloader
Fetches key economic indicators from Federal Reserve
"""
import os
import pandas as pd
from fredapi import Fred
from datetime import datetime, timedelta

FRED_API_KEY = os.getenv('FRED_API_KEY')

# Core economic indicators
FRED_SERIES = {
    # Interest Rates (daily/weekly updates)
    'FEDFUNDS': 'Federal Funds Rate',
    'DGS10': '10-Year Treasury Yield',
    'DGS2': '2-Year Treasury Yield',
    'T10Y2Y': '10Y-2Y Treasury Spread',
    'MORTGAGE30US': '30-Year Mortgage Rate',

    # Inflation (monthly)
    'CPIAUCSL': 'Consumer Price Index (CPI)',
    'CPILFESL': 'Core CPI (ex food & energy)',
    'PCEPILFE': 'Core PCE (Fed\'s preferred)',

    # Employment (monthly)
    'UNRATE': 'Unemployment Rate',
    'PAYEMS': 'Nonfarm Payrolls',
    'ICSA': 'Initial Jobless Claims',

    # Economy (quarterly/monthly)
    'GDP': 'Gross Domestic Product',
    'GDPC1': 'Real GDP',
    'INDPRO': 'Industrial Production',
    'RSXFS': 'Retail Sales',
    'UMCSENT': 'Consumer Sentiment',

    # Markets
    'SP500': 'S&P 500 Index',
    'VIXCLS': 'VIX Volatility Index',

    # Money Supply
    'M2SL': 'M2 Money Supply',
    'WALCL': 'Fed Balance Sheet Size',

    # Commodity Prices (Energy)
    'DCOILWTICO': 'Crude Oil WTI',
    'DCOILBRENTEU': 'Crude Oil Brent',
    'GASREGW': 'Natural Gas Henry Hub',
    'DPROPANEMBTX': 'Propane Price',

    # Commodity Prices (Metals)
    'GOLDAMGBD228NLBM': 'Gold London Fixing',
    'SLVPRUSD': 'Silver Price',
    'PCOPPUSDM': 'Copper Price',

    # Commodity Prices (Agriculture)
    'PWHEAMTUSDM': 'Wheat Price',
    'PMAIZMTUSDM': 'Corn Price',
    'PSOYBUSDQ': 'Soybeans Price',
}

def fetch_fred_series(series_id, nice_name, years_back=None, days_back=None):
    """Fetch single FRED series"""
    if not FRED_API_KEY:
        print(f"  ⚠️  FRED_API_KEY not set")
        return None

    fred = Fred(api_key=FRED_API_KEY)

    # Calculate date range
    end_date = datetime.now()
    if years_back:
        start_date = end_date - timedelta(days=years_back * 365)
    elif days_back:
        start_date = end_date - timedelta(days=days_back)
    else:
        start_date = end_date - timedelta(days=90)  # Default 3 months

    try:
        series = fred.get_series(
            series_id,
            observation_start=start_date.strftime('%Y-%m-%d'),
            observation_end=end_date.strftime('%Y-%m-%d')
        )

        if series.empty:
            return None

        # Convert to DataFrame
        df = series.to_frame(name='value')
        df.index.name = 'date'
        df = df.reset_index()
        df['series_id'] = series_id
        df['series_name'] = nice_name
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        return df

    except Exception as e:
        print(f"  ✗ Error fetching {series_id}: {e}")
        return None

def fetch_all_fred_data(years_back=None, days_back=90):
    """
    Fetch all FRED economic data

    Args:
        years_back: Fetch N years of historical data (for initial backfill)
        days_back: Fetch N days of data (for incremental updates)
    """
    if years_back:
        print(f"\nFetching FRED data (last {years_back} years)...")
    else:
        print(f"\nFetching FRED data (last {days_back} days)...")

    all_series = []

    for series_id, nice_name in FRED_SERIES.items():
        df = fetch_fred_series(series_id, nice_name, years_back, days_back)
        if df is not None and not df.empty:
            all_series.append(df)
            print(f"  ✓ {series_id}: {len(df)} records")

    if not all_series:
        print("✗ No FRED data fetched")
        return pd.DataFrame()

    # Combine all series
    combined_df = pd.concat(all_series, ignore_index=True)
    print(f"✓ Fetched {len(combined_df)} total FRED records")

    return combined_df
