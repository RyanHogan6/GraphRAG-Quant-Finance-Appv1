"""
Polymarket Downloader Module
Fetches market data and current prices from Polymarket Gamma API
"""

import pandas as pd
import requests
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import (
    GAMMA_BASE_URL,
    DATA_RAW,
    BATCH_SIZE,
    RATE_LIMIT_DELAY,
    API_TIMEOUT,
    TOP_MARKETS_LIMIT
)

# ============================================================================
# MARKET FETCHING
# ============================================================================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_all_markets() -> pd.DataFrame:
    """
    Fetch all active markets from Polymarket Gamma API with pagination.

    Returns:
        DataFrame with market data including:
        - market_id, condition_id, question, description
        - volume, volume_24h, liquidity
        - closed, category, outcomes, outcome_prices
        - yes_probability, no_probability
    """

    print("\n[DOWNLOADER] Fetching all active markets from Gamma API...")
    print("-" * 80)

    markets = []
    offset = 0
    limit = BATCH_SIZE

    while True:
        try:
            url = f"{GAMMA_BASE_URL}/markets"
            params = {
                'limit': limit,
                'offset': offset,
                'active': 'true',
                'closed': 'false'
            }

            response = requests.get(url, params=params, timeout=API_TIMEOUT)
            response.raise_for_status()

            batch = response.json()

            if not batch:
                break

            markets.extend(batch)
            offset += limit

            print(f"  Fetched {len(markets)} markets...", end='\r')

            time.sleep(RATE_LIMIT_DELAY)

        except Exception as e:
            print(f"\n  [WARN] Error fetching markets at offset {offset}: {e}")
            break

    print(f"\n  [OK] Fetched {len(markets)} active markets")

    # Parse market data into structured format
    markets_data = []

    for market in markets:
        try:
            # Extract outcome prices - API returns as JSON string, need to parse
            outcome_prices_raw = market.get('outcomePrices', [])
            yes_prob = None
            no_prob = None

            # Parse outcomePrices if it's a JSON string
            outcome_prices = outcome_prices_raw
            if isinstance(outcome_prices_raw, str):
                try:
                    outcome_prices = json.loads(outcome_prices_raw)
                except (json.JSONDecodeError, TypeError):
                    outcome_prices = []

            if isinstance(outcome_prices, list) and len(outcome_prices) >= 2:
                try:
                    yes_prob = float(outcome_prices[0]) if outcome_prices[0] is not None else None
                    no_prob = float(outcome_prices[1]) if outcome_prices[1] is not None else None
                except (ValueError, TypeError):
                    pass

            markets_data.append({
                'market_id': market.get('id'),
                'condition_id': market.get('conditionId'),
                'question': market.get('question'),
                'description': market.get('description', ''),
                'end_date': market.get('endDate'),
                'game_start_time': market.get('gameStartTime'),
                'market_slug': market.get('marketSlug'),
                'min_incentive_size': market.get('minIncentiveSize', 0),
                'max_incentive_spread': market.get('maxIncentiveSpread', 0),
                'volume': float(market.get('volume', 0)),
                'volume_24h': float(market.get('volume24hr', 0)),
                'liquidity': float(market.get('liquidity', 0)),
                'closed': market.get('closed', False),
                'archived': market.get('archived', False),
                'new': market.get('new', False),
                'featured': market.get('featured', False),
                'submitted_by': market.get('submittedBy', ''),
                'category': market.get('category', 'Other'),
                'tags': json.dumps(market.get('tags', [])),
                'outcomes': json.dumps(market.get('outcomes', [])),
                'outcome_prices': json.dumps(market.get('outcomePrices', [])),
                'yes_probability': yes_prob,
                'no_probability': no_prob,
                'reward_min_size': market.get('rewardMinSize', 0),
                'reward_max_spread': market.get('rewardMaxSpread', 0),
                'accepting_orders': market.get('acceptingOrders', True),
                'enable_order_book': market.get('enableOrderBook', True),
                'neg_risk': market.get('negRisk', False),
                'fetched_at': datetime.now().isoformat()
            })
        except Exception as e:
            print(f"  [WARN] Error parsing market {market.get('id')}: {e}")
            continue

    markets_df = pd.DataFrame(markets_data)

    if len(markets_df) > 0:
        print(f"  [OK] Parsed {len(markets_df)} markets")
        print(f"  [OK] Categories: {markets_df['category'].nunique()}")
        print(f"  [OK] Total volume: ${markets_df['volume'].sum():,.0f}")
        print(f"  [OK] Total 24h volume: ${markets_df['volume_24h'].sum():,.0f}")

        # Save snapshot to CSV (optional)
        os.makedirs(DATA_RAW, exist_ok=True)
        snapshot_path = os.path.join(DATA_RAW, f'markets_snapshot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        markets_df.to_csv(snapshot_path, index=False)
        print(f"  [OK] Saved snapshot: {snapshot_path}")
    else:
        print("  [WARN] No markets fetched!")

    return markets_df


# ============================================================================
# PRICE FETCHING FOR TOP MARKETS
# ============================================================================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_current_market_price(market_slug: str, market_id: str) -> Optional[Dict]:
    """
    Fetch current price snapshot for a single market from Gamma API.

    Args:
        market_slug: Market slug identifier
        market_id: Market ID

    Returns:
        Dict with current price data or None if unavailable
    """
    try:
        url = f"{GAMMA_BASE_URL}/markets/{market_slug}"

        response = requests.get(url, timeout=API_TIMEOUT)

        if response.status_code != 200:
            return None

        data = response.json()

        if not data:
            return None

        # Extract price data
        outcome_prices = data.get('outcomePrices', [])

        if not outcome_prices or len(outcome_prices) < 2:
            return None

        # Polymarket markets are binary (Yes/No)
        yes_price = float(outcome_prices[0]) if len(outcome_prices) > 0 else 0.5
        no_price = float(outcome_prices[1]) if len(outcome_prices) > 1 else 0.5

        return {
            'market_id': market_id,
            'market_slug': market_slug,
            'condition_id': data.get('conditionId'),
            'timestamp': int(datetime.now().timestamp()),
            'datetime': datetime.now().isoformat(),
            'yes_price': yes_price,
            'no_price': no_price,
            'volume': float(data.get('volume', 0)),
            'volume_24h': float(data.get('volume24hr', 0)),
            'liquidity': float(data.get('liquidity', 0))
        }

    except Exception as e:
        return None


def fetch_current_prices(markets_df: pd.DataFrame, top_n: int = TOP_MARKETS_LIMIT) -> pd.DataFrame:
    """
    Fetch current prices for top N markets by 24h volume.

    Args:
        markets_df: DataFrame of all markets
        top_n: Number of top markets to fetch prices for

    Returns:
        DataFrame with current price snapshots
    """

    print(f"\n[DOWNLOADER] Fetching current prices for top {top_n} markets...")
    print("-" * 80)

    # Select top markets by volume
    top_markets = markets_df.nlargest(top_n, 'volume_24h')
    print(f"  Selected top {len(top_markets)} markets by 24h volume")

    all_prices = []
    success_count = 0
    fail_count = 0

    for idx, row in top_markets.iterrows():
        market_slug = row['market_slug']
        market_id = row['market_id']
        question = row['question'][:50] if pd.notna(row['question']) else "Unknown"

        print(f"  [{success_count + fail_count + 1}/{len(top_markets)}] {question}...", end='\r')

        price_data = fetch_current_market_price(market_slug, market_id)

        if price_data:
            price_data['question'] = row['question']
            price_data['category'] = row['category']
            all_prices.append(price_data)
            success_count += 1
        else:
            fail_count += 1

        time.sleep(RATE_LIMIT_DELAY)

    print("\n")
    print(f"  [OK] Successfully fetched prices for {success_count} markets")
    print(f"  [WARN] {fail_count} markets had no price data")

    if all_prices:
        prices_df = pd.DataFrame(all_prices)

        print(f"  [OK] Total price points: {len(prices_df):,}")
        print(f"  [OK] Average yes price: {prices_df['yes_price'].mean():.3f}")

        # Save prices snapshot
        os.makedirs(DATA_RAW, exist_ok=True)
        prices_path = os.path.join(DATA_RAW, f'market_prices_current_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        prices_df.to_csv(prices_path, index=False)
        print(f"  [OK] Saved prices: {prices_path}")

        return prices_df
    else:
        print("  [WARN] No price data fetched")
        return pd.DataFrame()


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def download_polymarket_data() -> tuple:
    """
    Convenience function to fetch all market data and current prices.

    Returns:
        Tuple of (markets_df, prices_df)
    """

    print("\n" + "="*80)
    print("POLYMARKET DATA DOWNLOAD")
    print("="*80)

    # Fetch all markets
    markets_df = fetch_all_markets()

    # Fetch prices for top markets
    prices_df = fetch_current_prices(markets_df) if len(markets_df) > 0 else pd.DataFrame()

    print("\n" + "="*80)
    print("[OK] DOWNLOAD COMPLETE")
    print("="*80)
    print(f"  Markets: {len(markets_df):,}")
    print(f"  Price snapshots: {len(prices_df):,}")
    print(f"  Total 24h volume: ${markets_df['volume_24h'].sum():,.0f}" if len(markets_df) > 0 else "")
    print("="*80 + "\n")

    return markets_df, prices_df


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    # Test the downloader
    markets, prices = download_polymarket_data()

    if len(markets) > 0:
        print(f"\nTop 5 markets by volume:")
        print(markets.nlargest(5, 'volume_24h')[['question', 'volume_24h', 'category']])
