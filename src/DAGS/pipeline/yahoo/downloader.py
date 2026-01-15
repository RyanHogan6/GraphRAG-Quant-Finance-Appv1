import os, time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from .config import *

def fetch_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url)[0]
    return df['Symbol'].str.replace('.', '-', regex=False).tolist()

def download_stock_data(tickers, period='1mo'):
    """Download stock data for given tickers - simplified for scheduler"""
    if period == '1mo':
        start_date = datetime.today().date() - timedelta(days=30)
    elif period == '1y':
        start_date = datetime.today().date() - timedelta(days=365)
    else:
        start_date = datetime.today().date() - timedelta(days=365)

    end_date = datetime.today().date()

    data = yf.download(tickers, start=start_date, end=end_date,
                       auto_adjust=True, group_by='ticker', threads=True, progress=False)

    # Convert to long format DataFrame
    records = []
    for ticker in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df = data[ticker].copy()
            else:
                df = data.copy()

            df = df.reset_index()
            df.columns = [col.lower() if isinstance(col, str) else col for col in df.columns]
            df['ticker'] = ticker
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            records.append(df)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    if records:
        return pd.concat(records, ignore_index=True)
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
