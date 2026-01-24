"""
EIA (Energy Information Administration) Data Downloader
Fetches energy commodity fundamentals: storage, production, exports
"""
import os
import requests
import pandas as pd
from datetime import datetime, timedelta

EIA_API_KEY = os.getenv('EIA_API_KEY')
EIA_BASE_URL = "https://api.eia.gov/v2"

def fetch_eia_series(series_id, frequency='weekly', years_back=None, weeks_back=4):
    """
    Generic EIA series fetcher

    Args:
        series_id: EIA series ID
        frequency: 'weekly' or 'monthly'
        years_back: How many years of data to fetch (overrides weeks_back)
        weeks_back: How many weeks of data to fetch (default 4)
    """
    if not EIA_API_KEY:
        print("⚠️  EIA_API_KEY not set")
        return pd.DataFrame()

    # Calculate date range
    end_date = datetime.now()
    if years_back:
        start_date = end_date - timedelta(days=years_back * 365)
    else:
        start_date = end_date - timedelta(weeks=weeks_back)

    params = {
        'api_key': EIA_API_KEY,
        'frequency': frequency,
        'data[0]': 'value',
        'start': start_date.strftime('%Y-%m-%d'),
        'sort[0][column]': 'period',
        'sort[0][direction]': 'desc',
        'length': 5000
    }

    try:
        response = requests.get(f"{EIA_BASE_URL}/{series_id}", params=params, timeout=30)

        if response.status_code != 200:
            print(f"  ⚠ EIA API error {response.status_code}: {series_id}")
            return pd.DataFrame()

        data = response.json()

        if 'response' in data and 'data' in data['response']:
            df = pd.DataFrame(data['response']['data'])
            return df

        return pd.DataFrame()

    except Exception as e:
        print(f"  ✗ Error fetching {series_id}: {e}")
        return pd.DataFrame()


def fetch_natgas_storage(years_back=None, weeks_back=4):
    """
    Fetch Natural Gas Underground Storage (Weekly)
    Most watched nat gas report - released Thursdays
    """
    print("  Fetching Natural Gas Storage (Weekly)...")

    # Working gas in storage
    df = fetch_eia_series(
        'natural-gas/stor/wkly/data',
        frequency='weekly',
        years_back=years_back,
        weeks_back=weeks_back
    )

    if df.empty:
        return pd.DataFrame()

    df['data_source'] = 'EIA_NATGAS_STORAGE'
    df['frequency'] = 'weekly'
    print(f"    ✓ {len(df)} records")
    return df


def fetch_crude_inventory(years_back=None, weeks_back=4):
    """
    Fetch Crude Oil & Petroleum Inventories (Weekly)
    Major market mover - released Wednesdays
    """
    print("  Fetching Crude Oil Inventories (Weekly)...")

    # U.S. Crude Oil Stocks
    df = fetch_eia_series(
        'petroleum/stoc/wstk/data',
        frequency='weekly',
        years_back=years_back,
        weeks_back=weeks_back
    )

    if df.empty:
        return pd.DataFrame()

    df['data_source'] = 'EIA_CRUDE_INVENTORY'
    df['frequency'] = 'weekly'
    print(f"    ✓ {len(df)} records")
    return df


def fetch_lng_exports(years_back=None, weeks_back=8):
    """
    Fetch LNG Exports (Monthly)
    Growing market importance
    """
    print("  Fetching LNG Exports (Monthly)...")

    df = fetch_eia_series(
        'natural-gas/move/expc/data',
        frequency='monthly',
        years_back=years_back,
        weeks_back=weeks_back
    )

    if df.empty:
        return pd.DataFrame()

    df['data_source'] = 'EIA_LNG_EXPORTS'
    df['frequency'] = 'monthly'
    print(f"    ✓ {len(df)} records")
    return df


def fetch_natgas_production(years_back=None, weeks_back=8):
    """
    Fetch Natural Gas Production (Monthly)
    Supply-side fundamentals
    """
    print("  Fetching Natural Gas Production (Monthly)...")

    df = fetch_eia_series(
        'natural-gas/prod/sum/data',
        frequency='monthly',
        years_back=years_back,
        weeks_back=weeks_back
    )

    if df.empty:
        return pd.DataFrame()

    df['data_source'] = 'EIA_NATGAS_PRODUCTION'
    df['frequency'] = 'monthly'
    print(f"    ✓ {len(df)} records")
    return df


def fetch_all_eia_data(years_back=None, weeks_back=4):
    """
    Fetch all EIA datasets

    Args:
        years_back: Fetch N years of historical data (for initial backfill)
        weeks_back: Fetch N weeks of data (for incremental updates)
    """
    if years_back:
        print(f"\nFetching EIA energy data (last {years_back} years)...")
    else:
        print(f"\nFetching EIA energy data (last {weeks_back} weeks)...")

    datasets = {
        'natgas_storage': fetch_natgas_storage(years_back, weeks_back),
        'crude_inventory': fetch_crude_inventory(years_back, weeks_back),
        'lng_exports': fetch_lng_exports(years_back, weeks_back),
        'natgas_production': fetch_natgas_production(years_back, weeks_back)
    }

    total_records = sum(len(df) for df in datasets.values() if not df.empty)
    print(f"✓ Fetched {total_records} total EIA records")

    return datasets
