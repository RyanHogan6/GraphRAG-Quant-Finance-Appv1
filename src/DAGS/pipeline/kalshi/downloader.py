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
        # Kalshi uses last_price as the probability (0-100 range)
        # Convert to 0-1 range for consistency with Polymarket
        last_price = market.get('last_price', 0)
        yes_probability = last_price / 100 if last_price else None
        no_probability = (100 - last_price) / 100 if last_price else None

        markets.append({
            'market_id': market.get('ticker'),
            'title': market.get('title'),
            'category': market.get('category'),
            'status': market.get('status'),
            'close_time': market.get('close_time'),
            'yes_probability': yes_probability,
            'no_probability': no_probability,
            'volume': market.get('volume', 0),
            'volume_24h': market.get('volume_24h', 0),
            'open_interest': market.get('open_interest', 0),
            'fetched_at': datetime.now().isoformat()
        })

    return pd.DataFrame(markets)
