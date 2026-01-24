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

def fetch_recent_cftc_data(weeks_back=4):
    """
    Fetch recent CFTC COT data from current and previous year
    Returns DataFrame with commodity positions
    """
    current_year = datetime.now().year
    years_to_fetch = [current_year - 1, current_year]

    print(f"Fetching CFTC data for years: {years_to_fetch}")

    all_frames = []

    for year in years_to_fetch:
        url = f"https://www.cftc.gov/files/dea/history/deacot{year}.zip"

        try:
            print(f"  Downloading {year}...")
            response = requests.get(url, timeout=30)

            if response.status_code != 200:
                print(f"  ⚠ File not available for {year}")
                continue

            # Extract and read annual.txt (contains disaggregated futures)
            with ZipFile(BytesIO(response.content)) as zf:
                # Look for annual.txt
                annual_files = [f for f in zf.namelist() if 'annual.txt' in f.lower()]

                if annual_files:
                    with zf.open(annual_files[0]) as f:
                        df = pd.read_csv(f, encoding='latin1', low_memory=False)
                        all_frames.append(df)
                        print(f"  ✓ Loaded {len(df)} records from {year}")

        except Exception as e:
            print(f"  ✗ Error downloading {year}: {e}")
            continue

    if not all_frames:
        print("✗ No data fetched")
        return pd.DataFrame()

    combined_df = pd.concat(all_frames, ignore_index=True)

    # Filter to recent weeks
    date_col = "As of Date in Form YYYY-MM-DD"
    if date_col in combined_df.columns:
        combined_df[date_col] = pd.to_datetime(combined_df[date_col], errors='coerce')
        cutoff_date = datetime.now() - timedelta(weeks=weeks_back)
        combined_df = combined_df[combined_df[date_col] >= cutoff_date]

    print(f"✓ Fetched {len(combined_df)} records from last {weeks_back} weeks")
    return combined_df
