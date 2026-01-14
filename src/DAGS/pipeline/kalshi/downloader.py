"""Kalshi API downloader"""
import requests
import pandas as pd
from datetime import datetime
from .config import KALSHI_API_URL

def fetch_all_markets():
    """Fetch active markets from Kalshi"""
    url = f"{KALSHI_API_URL}/markets"
    params = {
        'status': 'active',
        'limit': 1000
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    markets = []
    for market in data.get('markets', []):
        markets.append({
            'market_id': market.get('ticker'),
            'title': market.get('title'),
            'category': market.get('category'),
            'status': market.get('status'),
            'close_time': market.get('close_time'),
            'yes_price': market.get('yes_bid', 0) / 100,  # Convert cents to dollars
            'no_price': market.get('no_bid', 0) / 100,
            'volume': market.get('volume', 0),
            'volume_24h': market.get('volume_24h', 0),
            'open_interest': market.get('open_interest', 0),
            'fetched_at': datetime.now().isoformat()
        })

    return pd.DataFrame(markets)
