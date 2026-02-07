import os, time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from .config import *

def fetch_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url)[0]
    return df['Symbol'].str.replace('.', '-', regex=False).tolist()

def download_stock_data(tickers, period='1mo', batch_size=1, sleep_between_tickers=0.5):
    """
    Download stock data ONE TICKER AT A TIME with delays to avoid rate limiting

    Args:
        tickers: List of ticker symbols
        period: Time period ('1mo', '1y', '6mo')
        batch_size: DEPRECATED - always downloads one at a time now
        sleep_between_tickers: Seconds to wait between each ticker (default 0.5)
    """
    import time
    import random

    # NOTE: yfinance now requires curl_cffi sessions, not requests sessions
    # Let yfinance handle session management internally
    # See: https://github.com/ranaroussi/yfinance/issues/1729

    if period == '1mo':
        start_date = datetime.today().date() - timedelta(days=30)
    elif period == '6mo':
        start_date = datetime.today().date() - timedelta(days=180)
    elif period == '1y':
        start_date = datetime.today().date() - timedelta(days=365)
    else:
        start_date = datetime.today().date() - timedelta(days=365)

    end_date = datetime.today().date()

    all_records = []
    failed_tickers = []

    print(f"Downloading {len(tickers)} tickers ONE AT A TIME...")
    print(f"Rate limiting: {sleep_between_tickers}s between tickers (with random jitter)")
    print("This will take approximately {:.1f} minutes".format(len(tickers) * sleep_between_tickers / 60))

    # Initial delay before first request
    time.sleep(2)

    # Download one ticker at a time to avoid rate limiting
    for idx, ticker in enumerate(tickers, 1):
        try:
            # Progress update every 50 tickers (show first 3 failures for debugging)
            if idx % 50 == 0 or idx == len(tickers):
                print(f"  Progress: {idx}/{len(tickers)} ({idx/len(tickers)*100:.1f}%) - Success: {len(all_records)}, Failed: {len(failed_tickers)}")

            # Show first 3 ticker attempts for debugging
            if idx <= 3:
                print(f"  [DEBUG] Attempting ticker: {ticker}")

            # Download single ticker (yfinance handles session internally)
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False
            )

            # Show first 3 results for debugging
            if idx <= 3:
                print(f"  [DEBUG] Result for {ticker}: {len(data)} rows, empty={data.empty}")
                if not data.empty:
                    print(f"  [DEBUG] Columns: {list(data.columns)}")
                    print(f"  [DEBUG] First row: {data.iloc[0].to_dict()}")
                else:
                    print(f"  [DEBUG] Data is empty - ticker may be delisted or invalid")

            # Check if data is empty
            if data.empty or len(data) == 0:
                if idx <= 5:
                    print(f"  [WARN] {ticker}: No data returned from Yahoo Finance")
                failed_tickers.append(ticker)
                # Add small delay even on failure
                time.sleep(0.1)
                continue

            # Process data
            df = data.copy()
            df = df.reset_index()

            # Debug: show columns before processing
            if idx <= 3:
                print(f"  [DEBUG] Columns before processing: {list(df.columns)}")

            # Flatten MultiIndex columns (yfinance returns tuples like ('Close', 'ABT'))
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                if idx <= 3:
                    print(f"  [DEBUG] Flattened MultiIndex columns")

            # Lowercase all column names
            df.columns = [col.lower() if isinstance(col, str) else str(col).lower() for col in df.columns]

            # Debug: show columns after processing
            if idx <= 3:
                print(f"  [DEBUG] Columns after processing: {list(df.columns)}")

            # Add ticker column
            df['ticker'] = ticker

            # Format date column (handle both 'date' and 'index' names)
            date_col = None
            if 'date' in df.columns:
                date_col = 'date'
            elif 'index' in df.columns:
                date_col = 'index'
                df = df.rename(columns={'index': 'date'})
            else:
                # Find any datetime column
                for col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        date_col = col
                        df = df.rename(columns={col: 'date'})
                        break

            if date_col is None:
                if idx <= 5:
                    print(f"  [ERROR] {ticker}: No date column found in {list(df.columns)}")
                failed_tickers.append(ticker)
                continue

            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

            # Only keep rows with valid OHLCV data
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if all(col in df.columns for col in required_cols):
                df = df.dropna(subset=required_cols)
                if len(df) > 0:
                    # Fetch fundamental data for this ticker
                    try:
                        info = yf.Ticker(ticker).info or {}
                        for col in REVISED_STATIC_COLS:
                            df[col] = info.get(col)
                        if idx <= 3:
                            print(f"  [DEBUG] Added {len(REVISED_STATIC_COLS)} fundamental fields")
                    except Exception as e:
                        if idx <= 3:
                            print(f"  [WARN] Could not fetch fundamentals for {ticker}: {e}")

                    all_records.append(df)
                else:
                    failed_tickers.append(ticker)
            else:
                failed_tickers.append(ticker)

            # Rate limiting with random jitter to appear more human-like
            if idx < len(tickers):
                # Add random 20-50% variation to sleep time
                jitter = random.uniform(1.2, 1.5)
                time.sleep(sleep_between_tickers * jitter)

        except Exception as e:
            # Show first 5 errors for debugging
            if idx <= 5:
                print(f"  [ERROR] {ticker}: {type(e).__name__}: {str(e)}")
            failed_tickers.append(ticker)
            # Longer delay on error to avoid cascading rate limits
            time.sleep(2.0)
            continue

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
