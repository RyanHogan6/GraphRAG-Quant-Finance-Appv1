"""
Options Flow Downloader
Fetches options activity data to detect unusual volume and sentiment
"""
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time

def fetch_options_for_ticker(ticker, current_price=None):
    """
    Fetch options data for a single ticker

    Args:
        ticker: Stock ticker symbol
        current_price: Current stock price (fetched if not provided)

    Returns:
        Dict with aggregated options metrics
    """
    try:
        stock = yf.Ticker(ticker)

        # Get current price if not provided
        if current_price is None:
            hist = stock.history(period='1d')
            if hist.empty:
                return None
            current_price = hist['Close'].iloc[-1]

        # Get options expirations
        expirations = stock.options

        if not expirations or len(expirations) == 0:
            return None

        # Use next 2 expirations (near-term activity)
        near_term_expirations = expirations[:min(2, len(expirations))]

        all_calls = []
        all_puts = []

        for exp_date in near_term_expirations:
            try:
                chain = stock.option_chain(exp_date)

                # Filter to near-the-money (within 20% of current price)
                calls = chain.calls
                puts = chain.puts

                calls = calls[
                    (calls['strike'] >= current_price * 0.8) &
                    (calls['strike'] <= current_price * 1.2)
                ]
                puts = puts[
                    (puts['strike'] >= current_price * 0.8) &
                    (puts['strike'] <= current_price * 1.2)
                ]

                all_calls.append(calls)
                all_puts.append(puts)

            except Exception as e:
                print(f"    ⚠️  Could not fetch {exp_date}: {e}")
                continue

        if not all_calls and not all_puts:
            return None

        # Combine all expirations
        calls_df = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
        puts_df = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()

        # Calculate aggregate metrics
        metrics = {
            'ticker': ticker,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'stock_price': round(current_price, 2),

            # Volume metrics
            'call_volume': int(calls_df['volume'].sum()) if not calls_df.empty else 0,
            'put_volume': int(puts_df['volume'].sum()) if not puts_df.empty else 0,
            'total_volume': 0,  # Will be calculated

            # Open interest
            'call_open_interest': int(calls_df['openInterest'].sum()) if not calls_df.empty else 0,
            'put_open_interest': int(puts_df['openInterest'].sum()) if not puts_df.empty else 0,
            'total_open_interest': 0,  # Will be calculated

            # Put/call ratios
            'put_call_volume_ratio': 0.0,  # Will be calculated
            'put_call_oi_ratio': 0.0,  # Will be calculated

            # Implied volatility (average of near-the-money options)
            'call_iv_avg': float(calls_df['impliedVolatility'].mean()) if not calls_df.empty else None,
            'put_iv_avg': float(puts_df['impliedVolatility'].mean()) if not puts_df.empty else None,

            # Premium flow (volume * last price * 100 shares per contract)
            'call_premium': 0.0,  # Will be calculated
            'put_premium': 0.0,  # Will be calculated

            # Number of contracts
            'call_contracts': len(calls_df),
            'put_contracts': len(puts_df),
        }

        # Calculate derived metrics
        metrics['total_volume'] = metrics['call_volume'] + metrics['put_volume']
        metrics['total_open_interest'] = metrics['call_open_interest'] + metrics['put_open_interest']

        # Put/call ratios (handle division by zero)
        if metrics['call_volume'] > 0:
            metrics['put_call_volume_ratio'] = round(metrics['put_volume'] / metrics['call_volume'], 3)

        if metrics['call_open_interest'] > 0:
            metrics['put_call_oi_ratio'] = round(metrics['put_open_interest'] / metrics['call_open_interest'], 3)

        # Premium flow (approximate - volume * last price * 100)
        if not calls_df.empty and 'lastPrice' in calls_df.columns and 'volume' in calls_df.columns:
            metrics['call_premium'] = float((calls_df['lastPrice'] * calls_df['volume'] * 100).sum())

        if not puts_df.empty and 'lastPrice' in puts_df.columns and 'volume' in puts_df.columns:
            metrics['put_premium'] = float((puts_df['lastPrice'] * puts_df['volume'] * 100).sum())

        return metrics

    except Exception as e:
        print(f"    ✗ Error fetching options for {ticker}: {e}")
        return None


def fetch_options_for_tickers(tickers, delay=0.5):
    """
    Fetch options data for multiple tickers

    Args:
        tickers: List of ticker symbols
        delay: Delay between requests in seconds (to avoid rate limiting)

    Returns:
        DataFrame with options metrics
    """
    all_data = []

    print(f"Fetching options data for {len(tickers)} tickers...")

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{len(tickers)}] {ticker}...", end=' ')

        data = fetch_options_for_ticker(ticker)

        if data:
            all_data.append(data)
            print(f"✓ Vol: {data['total_volume']:,}, P/C: {data['put_call_volume_ratio']:.2f}")
        else:
            print("✗ No data")

        # Rate limiting
        if delay > 0 and i < len(tickers):
            time.sleep(delay)

    if not all_data:
        print("✗ No options data fetched")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    print(f"\n✓ Fetched options data for {len(df)} tickers")

    return df


def get_tickers_from_company_collection(db=None):
    """
    Fetch all tickers from the Company collection in ArangoDB

    Args:
        db: ArangoDB connection (if None, will create new connection)

    Returns:
        List of ticker symbols
    """
    import os
    from arango import ArangoClient

    # Connect to database if not provided
    if db is None:
        url = os.getenv('ARANGO_URL') or os.getenv('ARANGO_HOST')
        db_name = os.getenv('ARANGO_DATABASE') or os.getenv('ARANGO_DB')
        username = os.getenv('ARANGO_USERNAME', 'root')
        password = os.getenv('ARANGO_PASSWORD')

        if not all([url, db_name, password]):
            print("⚠️  No ArangoDB credentials found, using fallback ticker list")
            return get_fallback_tickers()

        try:
            client = ArangoClient(hosts=url)
            db = client.db(db_name, username=username, password=password)
        except Exception as e:
            print(f"⚠️  Could not connect to ArangoDB: {e}")
            return get_fallback_tickers()

    try:
        # Query all tickers from Company collection
        query = """
        FOR company IN Company
            FILTER company.ticker != null
            FILTER company.ticker != ""
            RETURN DISTINCT company.ticker
        """

        tickers = list(db.aql.execute(query))

        if not tickers:
            print("⚠️  No tickers found in Company collection, using fallback")
            return get_fallback_tickers()

        print(f"✓ Loaded {len(tickers)} tickers from Company collection")
        return sorted(tickers)

    except Exception as e:
        print(f"⚠️  Error fetching from Company collection: {e}")
        return get_fallback_tickers()


def get_fallback_tickers():
    """
    Fallback list of liquid tickers if Company collection is unavailable
    """
    # Top liquid names with active options
    mega_caps = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA']
    defense = ['LMT', 'RTX', 'NOC', 'BA', 'GD', 'LHX']
    energy = ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC']
    financials = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C']

    all_tickers = mega_caps + defense + energy + financials
    return list(set(all_tickers))


def get_sp500_tickers_with_options():
    """
    Get tickers from Company collection
    Maintained for backwards compatibility
    """
    return get_tickers_from_company_collection()


def fetch_options_sp500_subset(max_tickers=50, delay=1.0):
    """
    Fetch options for a subset of S&P 500 (most liquid names)

    Args:
        max_tickers: Maximum number of tickers to fetch
        delay: Delay between requests (seconds)

    Returns:
        DataFrame with options data
    """
    tickers = get_sp500_tickers_with_options()[:max_tickers]
    return fetch_options_for_tickers(tickers, delay=delay)
