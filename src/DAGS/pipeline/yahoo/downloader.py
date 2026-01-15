import os, time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from .config import *

def fetch_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url)[0]
    return df['Symbol'].str.replace('.', '-', regex=False).tolist()

def download_stock_data(tickers, period='1mo', batch_size=50):
    """
    Download stock data for given tickers with batching and error handling

    Args:
        tickers: List of ticker symbols
        period: Time period ('1mo', '1y')
        batch_size: Number of tickers per batch (default 50 to avoid rate limits)
    """
    import time

    if period == '1mo':
        start_date = datetime.today().date() - timedelta(days=30)
    elif period == '1y':
        start_date = datetime.today().date() - timedelta(days=365)
    else:
        start_date = datetime.today().date() - timedelta(days=365)

    end_date = datetime.today().date()

    all_records = []
    failed_tickers = []

    print(f"Downloading {len(tickers)} tickers in batches of {batch_size}...")

    # Download in batches
    for batch_idx in range(0, len(tickers), batch_size):
        batch = tickers[batch_idx:batch_idx + batch_size]
        print(f"  Batch {batch_idx//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}: {len(batch)} tickers")

        try:
            # Download batch
            data = yf.download(
                batch,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                group_by='ticker',
                threads=True,
                progress=False
            )

            # Process each ticker in batch
            for ticker in batch:
                try:
                    # Extract ticker data
                    if len(batch) == 1:
                        df = data.copy()
                    elif isinstance(data.columns, pd.MultiIndex):
                        df = data[ticker].copy()
                    else:
                        df = data.copy()

                    # Reset index and standardize columns
                    df = df.reset_index()
                    df.columns = [col.lower() if isinstance(col, str) else col for col in df.columns]

                    # Skip if no data
                    if df.empty or len(df) == 0:
                        failed_tickers.append(ticker)
                        continue

                    # Add ticker column and format date
                    df['ticker'] = ticker
                    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

                    # Only keep rows with valid OHLCV data
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    if all(col in df.columns for col in required_cols):
                        df = df.dropna(subset=required_cols)
                        if len(df) > 0:
                            all_records.append(df)
                    else:
                        failed_tickers.append(ticker)

                except Exception as e:
                    print(f"    Error processing {ticker}: {e}")
                    failed_tickers.append(ticker)

            # Rate limit: wait between batches
            if batch_idx + batch_size < len(tickers):
                time.sleep(1)

        except Exception as e:
            print(f"  Batch download failed: {e}")
            failed_tickers.extend(batch)

    # Combine all successful downloads
    if all_records:
        result = pd.concat(all_records, ignore_index=True)
        print(f"  ✓ Successfully downloaded {len(all_records)} tickers, {len(result)} rows")
        if failed_tickers:
            print(f"  ⚠ Failed tickers: {len(failed_tickers)}/{len(tickers)}")
        return result

    print(f"  ✗ No data downloaded (all {len(tickers)} tickers failed)")
    return pd.DataFrame()

def download_yahoo_data(start_date=None, end_date=None, tickers=None):
    os.makedirs(DATA_RAW_PATH, exist_ok=True)
    if not tickers:
        tickers = fetch_sp500_tickers()
    if not start_date:
        start_date = datetime.today().date() - timedelta(days=365*YEARS)
    if not end_date:
        end_date = datetime.today().date()

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        data = yf.download(batch, start=start_date, end=end_date, interval=INTERVAL,
                           auto_adjust=True, group_by='ticker', threads=True, progress=False)
        for ticker in batch:
            try:
                df = data[ticker].copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
                df.reset_index(inplace=True)
                df.columns = [col.lower() for col in df.columns]
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])
                df_valid = df.dropna(subset=['open','high','low','close','volume'])
                if len(df_valid) < MIN_REQUIRED_ROWS:
                    print(f"Skipping {ticker}: insufficient rows")
                    continue
                info = yf.Ticker(ticker).info or {}
                for col in REVISED_STATIC_COLS:
                    df[col] = info.get(col)
                if 'sharesOutstanding' in df.columns:
                    df['market_cap'] = df['close'] * df['sharesOutstanding']
                output_path = os.path.join(DATA_RAW_PATH, f"{ticker}.csv")
                df.to_csv(output_path, index=False)
                print(f"Saved {ticker}: {len(df)} rows")
                time.sleep(0.1)
            except Exception as e:
                print(f"Error: {ticker}: {e}")
