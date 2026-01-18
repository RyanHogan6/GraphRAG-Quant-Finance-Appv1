"""Kalshi API downloader"""
import requests
import pandas as pd
from datetime import datetime
from .config import KALSHI_API_URL

def fetch_events():
    """Fetch events to get categories"""
    url = f"{KALSHI_API_URL}/events"
    params = {'limit': 200, 'status': 'active'}

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        events = data.get('events', [])
        return {e['event_ticker']: e for e in events}
    except Exception as e:
        print(f"Error fetching events: {e}")
        return {}

def fetch_all_markets():
    """Fetch all markets from Kalshi with proper category mapping"""
    # First fetch events to get categories
    print("Fetching events for category mapping...")
    events_map = fetch_events()
    print(f"Found {len(events_map)} events")

    # Now fetch markets
    url = f"{KALSHI_API_URL}/markets"
    params = {
        'limit': 1000,
        'status': 'active'  # Only fetch markets that are actively trading
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    markets = []
    for market in data.get('markets', []):
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

    print(f"Fetched {len(markets)} active markets")
    return pd.DataFrame(markets)
