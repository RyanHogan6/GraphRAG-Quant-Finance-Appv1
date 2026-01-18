"""Kalshi API downloader"""
import requests
import pandas as pd
from datetime import datetime
from .config import KALSHI_API_URL

def fetch_events():
    """Fetch events to get categories"""
    url = f"{KALSHI_API_URL}/events"
    params = {'limit': 200, 'status': 'open'}  # API uses 'open' for filter

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        events = data.get('events', [])
        return {e['event_ticker']: e for e in events}
    except Exception as e:
        print(f"Error fetching events: {e}")
        return {}

def fetch_all_markets(status_filter=None, limit=None):
    """
    Fetch all markets from Kalshi with proper category mapping

    Args:
        status_filter: None (all), 'open' (active only), 'closed', 'settled'
        limit: Max markets to fetch (None = all markets via pagination)
    """
    # First fetch events to get categories
    print("Fetching events for category mapping...")
    events_map = fetch_events()
    print(f"Found {len(events_map)} events")

    # Now fetch markets with pagination
    url = f"{KALSHI_API_URL}/markets"
    all_markets = []
    cursor = None

    while True:
        params = {'limit': 200}  # Max per page

        if status_filter:
            params['status'] = status_filter

        if cursor:
            params['cursor'] = cursor

        print(f"  Fetching batch (total so far: {len(all_markets)})...", end='\r')

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        markets_batch = data.get('markets', [])
        if not markets_batch:
            break

        all_markets.extend(markets_batch)

        # Check for next page
        cursor = data.get('cursor')
        if not cursor:
            break

        # Check limit
        if limit and len(all_markets) >= limit:
            all_markets = all_markets[:limit]
            break

        import time
        time.sleep(0.2)  # Rate limiting

    print(f"\n  Fetched {len(all_markets)} total markets")

    markets = []
    for market in all_markets:
        # Get event to extract category
        event_ticker = market.get('event_ticker')
        event = events_map.get(event_ticker, {})
        category = event.get('category', 'Other')

        # Kalshi uses yes_bid, yes_ask, and last_price
        # Use yes_bid (current best bid) as primary probability indicator
        yes_bid = market.get('yes_bid')
        yes_ask = market.get('yes_ask')
        last_price = market.get('last_price')

        # Priority: last_price > midpoint of bid/ask > yes_bid
        if last_price is not None and last_price > 0:
            yes_probability = last_price / 100
        elif yes_bid is not None and yes_ask is not None:
            yes_probability = ((yes_bid + yes_ask) / 2) / 100
        elif yes_bid is not None:
            yes_probability = yes_bid / 100
        else:
            yes_probability = None

        no_probability = (1 - yes_probability) if yes_probability is not None else None

        markets.append({
            'market_id': market.get('ticker'),
            'market_ticker': market.get('ticker'),
            'title': market.get('title'),
            'category': category,
            'event_ticker': event_ticker,
            'status': market.get('status'),
            'close_time': market.get('close_time'),
            'yes_probability': yes_probability,
            'no_probability': no_probability,
            'yes_bid': yes_bid,
            'yes_ask': yes_ask,
            'last_price': last_price,
            'volume': market.get('volume', 0),
            'volume_24h': market.get('volume_24h', 0),
            'open_interest': market.get('open_interest', 0),
            'fetched_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        })

    print(f"Processed {len(markets)} markets with categories")
    return pd.DataFrame(markets)
