"""
Polymarket Trader Tracker Module
Fetches trader position data from Polymarket subgraph (GraphQL API)
Uses reliable subgraph approach instead of per-market orderbook scraping
"""

import pandas as pd
import requests
from datetime import datetime
from typing import List, Dict, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import (
    LEADERBOARD_URL,
    POSITIONS_URL,
    MIN_TRADER_VOLUME,
    WHALE_THRESHOLD,
    API_TIMEOUT,
    RATE_LIMIT_DELAY
)

# ============================================================================
# LEADERBOARD API (NEW APPROACH - 2026)
# ============================================================================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_traders_from_subgraph(min_volume: float = MIN_TRADER_VOLUME) -> List[Dict]:
    """
    Fetch all active traders from Polymarket leaderboard API.

    NOTE: Function name kept for backwards compatibility, but now uses
    the new Data API leaderboard endpoint instead of deprecated subgraph.

    The new API provides:
    - Trader wallet addresses
    - Trading volume
    - PnL (profit/loss)
    - Ranking data

    Note: Position data is NOT available in leaderboard API.
    For position data, use get_user_positions(address) separately.

    Args:
        min_volume: Minimum total volume ($) for trader inclusion

    Returns:
        List of trader dicts compatible with parse_trader_positions()
    """

    print("\n[TRADER TRACKER] Fetching traders from Polymarket leaderboard API...")
    print("-" * 80)

    all_traders = []

    # Fetch multiple pages to get top traders
    # The API limits to 50 per page, max offset 1000
    categories = ['OVERALL']  # Can add 'POLITICS', 'CRYPTO', 'SPORTS' etc. for more data

    for category in categories:
        print(f"  Fetching category: {category}")
        offset = 0
        limit = 50  # Max per request

        while offset < 1000:  # API limit
            try:
                params = {
                    'category': category,
                    'timePeriod': 'MONTH',  # Get monthly top traders
                    'orderBy': 'VOL',  # Sort by volume
                    'limit': limit,
                    'offset': offset
                }

                response = requests.get(
                    LEADERBOARD_URL,
                    params=params,
                    timeout=API_TIMEOUT
                )

                response.raise_for_status()
                traders_page = response.json()

                if not traders_page or len(traders_page) == 0:
                    break  # No more traders

                # Filter by minimum volume
                filtered = [t for t in traders_page if float(t.get('vol', 0)) >= min_volume]

                if not filtered:
                    break  # Below minimum volume threshold

                all_traders.extend(filtered)
                offset += limit

                print(f"    Fetched {len(filtered)} traders (offset {offset})...", end='\r')

            except Exception as e:
                print(f"  [WARN] Failed to fetch page at offset {offset}: {e}")
                break

        print()  # Newline after progress

    # Remove duplicates (same wallet might appear in multiple categories)
    unique_traders = {}
    for trader in all_traders:
        wallet = trader.get('proxyWallet', '')
        if wallet and wallet not in unique_traders:
            unique_traders[wallet] = trader

    traders = list(unique_traders.values())

    if traders:
        total_volume = sum(float(t.get('vol', 0)) for t in traders)
        whales = sum(1 for t in traders if float(t.get('vol', 0)) > WHALE_THRESHOLD)
        print(f"  [OK] Fetched {len(traders)} unique traders from leaderboard")
        print(f"  [OK] Min volume filter: ${min_volume:,.0f}")
        print(f"  [OK] Total volume: ${total_volume:,.0f}")
        print(f"  [OK] Whales (>${WHALE_THRESHOLD:,.0f}): {whales}")
    else:
        print("  [WARN] No traders fetched from leaderboard")

    return traders


# ============================================================================
# POSITION FETCHING (PER TRADER)
# ============================================================================

import time

def fetch_positions_for_trader(wallet_address: str, limit: int = 20) -> List[Dict]:
    """
    Fetch current positions for a specific trader.

    Args:
        wallet_address: Trader's wallet address (0x-prefixed)
        limit: Maximum number of positions to fetch (default: 20)

    Returns:
        List of position dicts
    """
    try:
        params = {
            'user': wallet_address,
            'limit': limit,
            'sortBy': 'TOKENS',  # Sort by position size
            'sortDirection': 'DESC'
        }

        response = requests.get(
            POSITIONS_URL,
            params=params,
            timeout=API_TIMEOUT
        )

        response.raise_for_status()
        positions = response.json()

        return positions if positions else []

    except Exception as e:
        # Don't print errors for every trader to avoid spam
        return []


# ============================================================================
# TRADER POSITION PARSING
# ============================================================================

def parse_trader_positions(
    traders_raw: List[Dict],
    market_condition_map: Dict[str, str],
    fetch_positions: bool = True,
    max_traders_for_positions: int = 100
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse leaderboard trader data into normalized DataFrames.

    Optionally fetches position data for traders (makes additional API calls).

    Args:
        traders_raw: Raw trader data from leaderboard API
        market_condition_map: Mapping of condition_id → market _id from ArangoDB
        fetch_positions: Whether to fetch position data for traders (default: True)
        max_traders_for_positions: Max number of traders to fetch positions for (default: 100)

    Returns:
        Tuple of (traders_df, positions_df)
    """

    print("\n[TRADER TRACKER] Parsing trader data...")
    print("-" * 80)

    if not traders_raw:
        print("  [WARN] No traders to parse")
        return pd.DataFrame(), pd.DataFrame()

    traders_data = []
    positions_data = []

    # Limit position fetching to avoid too many API calls
    traders_to_fetch = traders_raw[:max_traders_for_positions] if fetch_positions else []

    for idx, trader in enumerate(traders_raw):
        try:
            # New API uses 'proxyWallet' instead of 'address'
            trader_address = trader.get('proxyWallet', '')

            if not trader_address:
                continue

            # Clean address for _key (remove 0x prefix, limit length)
            trader_key = trader_address.replace('0x', '').replace('-', '_')[:64]

            # New API uses 'vol' instead of 'totalVolume'
            total_volume = float(trader.get('vol', 0))
            is_whale = total_volume > WHALE_THRESHOLD

            # New API uses 'pnl' instead of 'totalProfit'
            total_pnl = float(trader.get('pnl', 0))

            # Create trader document
            trader_doc = {
                'trader_key': trader_key,
                'address': trader_address,
                'username': trader.get('userName', ''),
                'rank': int(trader.get('rank', 0)),
                'total_volume': total_volume,
                'total_trades': 0,  # Not available in leaderboard API
                'total_profit': total_pnl,
                'is_whale': is_whale,
                'verified_badge': trader.get('verifiedBadge', False),
                'x_username': trader.get('xUsername', ''),
                'profile_image': trader.get('profileImage', ''),
                'fetched_at': datetime.now().isoformat()
            }

            traders_data.append(trader_doc)

            # Fetch positions for top traders only
            if fetch_positions and idx < max_traders_for_positions:
                positions = fetch_positions_for_trader(trader_address, limit=20)

                for position in positions:
                    try:
                        condition_id = position.get('conditionId', '')

                        # Get market key from condition map
                        market_id = market_condition_map.get(condition_id)
                        if not market_id:
                            continue  # Skip positions for markets we don't have

                        market_key = market_id.split('/')[-1] if '/' in market_id else market_id

                        # Create position document
                        position_key = f"{trader_key}_{market_key}"[:64]

                        # Map API fields to database schema
                        position_doc = {
                            'position_key': position_key,
                            'position_id': position.get('positionId', ''),
                            'trader_address': trader_address,
                            'trader_key': trader_key,
                            'market_condition_id': condition_id,
                            'market_key': market_key,
                            'market_question': position.get('title', ''),
                            'outcome_index': int(position.get('outcomeIndex', 0)),
                            'size': float(position.get('size', 0)),
                            'average_price': float(position.get('avgPrice', 0)),
                            'realized_profit': float(position.get('cashPnl', 0)),
                            'unrealizedProfit': float(position.get('unrealizedPnl', 0)),
                            'current_value': float(position.get('currentValue', 0)),
                            'current_price': float(position.get('curPrice', 0)),
                            'redeemable': position.get('redeemable', False),
                            'fetched_at': datetime.now().isoformat()
                        }

                        positions_data.append(position_doc)

                    except Exception as e:
                        continue

                # Rate limiting - small delay between requests
                time.sleep(RATE_LIMIT_DELAY)

                # Progress indicator
                if (idx + 1) % 10 == 0:
                    print(f"    Fetched positions for {idx+1}/{max_traders_for_positions} traders...", end='\r')

        except Exception as e:
            print(f"  [WARN] Error parsing trader {trader.get('proxyWallet')}: {e}")
            continue

    # Convert to DataFrames
    traders_df = pd.DataFrame(traders_data)
    positions_df = pd.DataFrame(positions_data)

    print(f"\n  [OK] Parsed {len(traders_df)} traders")

    if fetch_positions:
        print(f"  [OK] Fetched positions for top {min(len(traders_raw), max_traders_for_positions)} traders")
        print(f"  [OK] Found {len(positions_df)} positions")
    else:
        print(f"  [NOTE] Position fetching disabled")

    if len(traders_df) > 0:
        whales = traders_df['is_whale'].sum()
        avg_volume = traders_df['total_volume'].mean()
        avg_pnl = traders_df['total_profit'].mean()
        print(f"  [OK] Whales: {whales}")
        print(f"  [OK] Average volume per trader: ${avg_volume:,.0f}")
        print(f"  [OK] Average PnL per trader: ${avg_pnl:,.0f}")

    if len(positions_df) > 0:
        avg_position_size = positions_df['size'].mean()
        avg_position_pnl = positions_df['realized_profit'].mean()
        print(f"  [OK] Average position size: {avg_position_size:,.0f} tokens")
        print(f"  [OK] Average position PnL: ${avg_position_pnl:,.0f}")

    return traders_df, positions_df


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def fetch_and_parse_traders(market_condition_map: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to fetch and parse trader data in one call.

    Args:
        market_condition_map: Mapping of condition_id → market _id

    Returns:
        Tuple of (traders_df, positions_df)
    """

    print("\n" + "="*80)
    print("POLYMARKET TRADER TRACKING")
    print("="*80)

    # Fetch from subgraph
    traders_raw = fetch_traders_from_subgraph(min_volume=MIN_TRADER_VOLUME)

    # Parse into DataFrames
    traders_df, positions_df = parse_trader_positions(traders_raw, market_condition_map)

    print("\n" + "="*80)
    print("[OK] TRADER TRACKING COMPLETE")
    print("="*80)
    print(f"  Traders: {len(traders_df):,}")
    print(f"  Positions: {len(positions_df):,}")
    if len(traders_df) > 0:
        print(f"  Whales: {traders_df['is_whale'].sum()}")
        print(f"  Total tracked volume: ${traders_df['total_volume'].sum():,.0f}")
    print("="*80 + "\n")

    return traders_df, positions_df


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    # Test the trader tracker
    print("Testing trader tracker (without market map)...")

    traders_raw = fetch_traders_from_subgraph(min_volume=1000)

    if traders_raw:
        print(f"\nSample trader data:")
        print(f"  Address: {traders_raw[0].get('address')}")
        print(f"  Total volume: ${float(traders_raw[0].get('totalVolume', 0)):,.0f}")
        print(f"  Positions: {len(traders_raw[0].get('positions', []))}")

        # Mock market map for testing
        mock_map = {}
        for trader in traders_raw[:5]:
            for pos in trader.get('positions', []):
                cond_id = pos.get('market', {}).get('conditionId')
                if cond_id:
                    mock_map[cond_id] = f"prediction_markets_polymarket/mock_{cond_id}"

        traders_df, positions_df = parse_trader_positions(traders_raw, mock_map)

        print(f"\nParsed DataFrames:")
        print(f"  Traders: {len(traders_df)} rows")
        print(f"  Positions: {len(positions_df)} rows")
    else:
        print("  [WARN] No traders fetched")
