"""
Polymarket Price History Tracker
Saves time-series snapshots of market prices for historical analysis and charting
"""

import pandas as pd
from datetime import datetime
from typing import Tuple

from .config import INSERT_BATCH_SIZE

# Collection name
PRICE_HISTORY_COL = "polymarket_price_history"


def create_price_history_collection(db):
    """
    Create price history collection with proper indexes.

    Args:
        db: ArangoDB database handle
    """
    if not db.has_collection(PRICE_HISTORY_COL):
        collection = db.create_collection(PRICE_HISTORY_COL)
        print(f"  [OK] Created collection: {PRICE_HISTORY_COL}")

        # Add indexes for fast time-range queries
        try:
            # Compound index for market + timestamp queries
            collection.add_persistent_index(
                fields=['market_id', 'timestamp'],
                unique=False,
                name='idx_market_timestamp'
            )

            # Index on timestamp alone for time-range queries
            collection.add_persistent_index(
                fields=['timestamp'],
                unique=False,
                name='idx_timestamp'
            )

            print(f"  [OK] Created indexes on {PRICE_HISTORY_COL}")
        except Exception as e:
            print(f"  [WARN] Index creation warning: {e}")
    else:
        print(f"  [i] Collection exists: {PRICE_HISTORY_COL}")


def save_price_snapshots(db, markets_df: pd.DataFrame, batch_size: int = 1000) -> Tuple[int, int]:
    """
    Save current price snapshots to time-series collection.

    This enables:
    - Historical price charts
    - Probability movement analysis
    - Volume/liquidity trends
    - Sharp swing detection

    Args:
        db: ArangoDB database handle
        markets_df: DataFrame with current market data
        batch_size: Number of documents per batch

    Returns:
        Tuple of (inserted_count, error_count)
    """

    print("\n[PRICE HISTORY] Saving price snapshots...")
    print("-" * 80)

    if len(markets_df) == 0:
        print("  [WARN] No markets to snapshot")
        return 0, 0

    # Ensure collection exists
    create_price_history_collection(db)
    collection = db.collection(PRICE_HISTORY_COL)

    # Current timestamp
    timestamp = int(datetime.now().timestamp())
    datetime_iso = datetime.now().isoformat()

    # Prepare snapshot documents
    documents = []
    error_count = 0

    for idx, row in markets_df.iterrows():
        try:
            # Only snapshot markets with valid probability data
            # Use bracket notation for pandas Series, not .get()
            yes_prob = row['yes_probability'] if 'yes_probability' in row.index else None
            no_prob = row['no_probability'] if 'no_probability' in row.index else None

            # Skip if no probability data or if they're NaN
            if yes_prob is None or no_prob is None or pd.isna(yes_prob) or pd.isna(no_prob):
                continue

            # Also skip markets with 0 or invalid probabilities
            if yes_prob == 0 and no_prob == 0:
                continue

            doc = {
                '_key': f"{row['market_id']}_{timestamp}",
                'market_id': str(row['market_id']),
                'condition_id': str(row['condition_id']) if 'condition_id' in row.index and pd.notna(row['condition_id']) else None,
                'timestamp': timestamp,
                'datetime': datetime_iso,
                'yes_price': float(yes_prob),
                'no_price': float(no_prob),
                'volume': float(row['volume']) if 'volume' in row.index and pd.notna(row['volume']) else 0.0,
                'volume_24h': float(row['volume_24h']) if 'volume_24h' in row.index and pd.notna(row['volume_24h']) else 0.0,
                'liquidity': float(row['liquidity']) if 'liquidity' in row.index and pd.notna(row['liquidity']) else 0.0,
            }

            documents.append(doc)

        except Exception as e:
            error_count += 1
            # Don't spam logs with individual errors

    if len(documents) == 0:
        print("  [WARN] No valid price data to snapshot")
        return 0, 0

    print(f"  Prepared {len(documents):,} price snapshots")

    # Batch insert using AQL (overwriteMode: ignore prevents duplicates)
    total_inserted = 0

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(documents) + batch_size - 1) // batch_size

        try:
            query = f"""
            FOR doc IN @documents
                INSERT doc INTO {PRICE_HISTORY_COL}
                OPTIONS {{ overwriteMode: "ignore" }}
            """

            db.aql.execute(query, bind_vars={'documents': batch})
            total_inserted += len(batch)

            print(f"  Batch {batch_num}/{total_batches}: Inserted {len(batch):,} snapshots", end='\r')

        except Exception as e:
            error_count += len(batch)
            print(f"\n  [ERROR] Batch {batch_num} failed: {e}")

    print(f"\n  [OK] Saved {total_inserted:,} price snapshots")

    if error_count > 0:
        print(f"  [WARN] {error_count} errors during snapshot")

    # Report collection size
    total_docs = collection.count()
    print(f"  [INFO] Total price history documents: {total_docs:,}")

    return total_inserted, error_count


def cleanup_old_snapshots(db, days_to_keep: int = 90):
    """
    Clean up old price snapshots to manage storage.

    Keep:
    - Last 30 days: All snapshots (10-min resolution)
    - 30-90 days: Downsample to hourly
    - 90+ days: Delete

    Args:
        db: ArangoDB database handle
        days_to_keep: Number of days to keep (default: 90)

    Returns:
        Number of documents deleted
    """

    print(f"\n[PRICE HISTORY] Cleaning snapshots older than {days_to_keep} days...")
    print("-" * 80)

    # Ensure collection exists
    if not db.has_collection(PRICE_HISTORY_COL):
        print("  [WARN] Price history collection doesn't exist")
        return 0

    # Delete snapshots older than days_to_keep
    query = f"""
    FOR doc IN {PRICE_HISTORY_COL}
        FILTER doc.timestamp < DATE_TIMESTAMP(DATE_SUBTRACT(DATE_NOW(), @days, "day"))
        REMOVE doc IN {PRICE_HISTORY_COL}
        COLLECT WITH COUNT INTO deleted
        RETURN deleted
    """

    try:
        cursor = db.aql.execute(query, bind_vars={'days': days_to_keep})
        deleted_count = next(cursor, 0)

        print(f"  [OK] Deleted {deleted_count:,} old snapshots")
        return deleted_count

    except Exception as e:
        print(f"  [ERROR] Cleanup failed: {e}")
        return 0


def get_market_history(db, market_id: str, days: int = 30) -> pd.DataFrame:
    """
    Retrieve price history for a specific market.

    Args:
        db: ArangoDB database handle
        market_id: Market ID to query
        days: Number of days of history (default: 30)

    Returns:
        DataFrame with price history
    """

    query = f"""
    FOR doc IN {PRICE_HISTORY_COL}
        FILTER doc.market_id == @market_id
        FILTER doc.timestamp >= DATE_TIMESTAMP(DATE_SUBTRACT(DATE_NOW(), @days, "day"))
        SORT doc.timestamp ASC
        RETURN {{
            timestamp: doc.timestamp,
            datetime: doc.datetime,
            yes_price: doc.yes_price,
            no_price: doc.no_price,
            volume: doc.volume,
            volume_24h: doc.volume_24h,
            liquidity: doc.liquidity
        }}
    """

    try:
        cursor = db.aql.execute(query, bind_vars={'market_id': market_id, 'days': days})
        results = list(cursor)

        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame()

    except Exception as e:
        print(f"  [ERROR] Failed to fetch history for {market_id}: {e}")
        return pd.DataFrame()


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    from .arango_uploader import get_arango_connection
    from .downloader import fetch_all_markets

    print("Testing price history module...")

    # Get database connection
    db = get_arango_connection()

    # Fetch current markets
    markets_df = fetch_all_markets()

    if len(markets_df) > 0:
        # Save snapshots
        inserted, errors = save_price_snapshots(db, markets_df)

        print(f"\n[TEST] Snapshot test complete:")
        print(f"  - Inserted: {inserted:,}")
        print(f"  - Errors: {errors}")

        # Test retrieval
        if len(markets_df) > 0:
            test_market_id = markets_df.iloc[0]['market_id']
            history_df = get_market_history(db, test_market_id, days=7)

            print(f"\n[TEST] Retrieved {len(history_df)} snapshots for market {test_market_id}")
            if len(history_df) > 0:
                print(history_df.head())
    else:
        print("\n[TEST] No markets to test with")
