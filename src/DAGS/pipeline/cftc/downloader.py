"""
CFTC Commitments of Traders (COT) Downloader - Incremental Updates
Fetches weekly commodity futures positioning data
"""
import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from io import BytesIO
from zipfile import ZipFile

CFTC_BASE_URL = "https://www.cftc.gov/files/dea/history"

def fetch_recent_cftc_data(weeks_back=4):
    """
    Fetch recent CFTC COT data
    Returns DataFrame with commodity positions
    """
    # CFTC publishes weekly data every Tuesday for previous Friday
    # File format: deaYYYYMM.zip contains weekly reports for that month

    current_date = datetime.now()
    end_date = current_date
    start_date = current_date - timedelta(weeks=weeks_back)

    # Determine which monthly files to download
    months_to_fetch = set()
    temp_date = start_date
    while temp_date <= end_date:
        months_to_fetch.add((temp_date.year, temp_date.month))
        temp_date += timedelta(days=32)
        temp_date = temp_date.replace(day=1)

    print(f"Fetching CFTC data for {len(months_to_fetch)} month(s)")

    all_frames = []

    for year, month in sorted(months_to_fetch):
        url = f"{CFTC_BASE_URL}/dea{year}{month:02d}.zip"

        try:
            print(f"  Downloading {year}-{month:02d}...")
            response = requests.get(url, timeout=30)

            if response.status_code != 200:
                print(f"  ⚠ File not available: {url}")
                continue

            # Extract and read annual.txt (contains disaggregated futures)
            with ZipFile(BytesIO(response.content)) as zf:
                # Look for annual.txt
                annual_files = [f for f in zf.namelist() if 'annual.txt' in f.lower()]

                if annual_files:
                    with zf.open(annual_files[0]) as f:
                        df = pd.read_csv(f, encoding='latin1', low_memory=False)
                        df['fetch_year'] = year
                        df['fetch_month'] = month
                        all_frames.append(df)
                        print(f"  ✓ Loaded {len(df)} records")

        except Exception as e:
            print(f"  ✗ Error downloading {year}-{month:02d}: {e}")
            continue

    if not all_frames:
        print("✗ No data fetched")
        return pd.DataFrame()

    combined_df = pd.concat(all_frames, ignore_index=True)

    # Filter to date range
    date_col = "As of Date in Form YYYY-MM-DD"
    if date_col in combined_df.columns:
        combined_df[date_col] = pd.to_datetime(combined_df[date_col], errors='coerce')
        combined_df = combined_df[combined_df[date_col] >= start_date]

    print(f"✓ Fetched {len(combined_df)} total records")
    return combined_df
