"""Kalshi ArangoDB uploader"""
import pandas as pd
from arango import ArangoClient
from datetime import datetime
from .config import *

_db_connection = None

def get_arango_connection():
    global _db_connection
    if _db_connection:
        return _db_connection

    client = ArangoClient(hosts=ARANGO_HOST)
    db = client.db(DB_NAME, username=USERNAME, password=PASSWORD)
    _db_connection = db
    return db

def upsert_markets(db, markets_df: pd.DataFrame, truncate_first=False):
    """
    Upsert Kalshi markets with memory-efficient batching

    Args:
        truncate_first: If True, clears collection before inserting (default: False, updates existing)
    """
    print("\n[UPLOADER] Upserting Kalshi markets...")

    if len(markets_df) == 0:
        return 0, 0, 0

    if truncate_first:
        # Truncate collection to remove old data
        print("  Truncating old data...")
        collection = db.collection(MARKET_COL)
        collection.truncate()
        print("  ✓ Old data cleared")
    else:
        print("  Updating existing records (keeping historical data)...")

    # Memory-efficient: Process in smaller batches (100 instead of 250)
    # For 102k records: 1,020 batches instead of 408
    batch_size = 100
    total_rows = len(markets_df)
    total_batches = (total_rows - 1) // batch_size + 1

    print(f"  Processing {total_rows:,} records in {total_batches:,} batches of {batch_size}...")

    inserted_total, updated_total = 0, 0

    # Process dataframe in chunks to avoid building huge docs list in memory
    for batch_num in range(0, total_rows, batch_size):
        batch_df = markets_df.iloc[batch_num:batch_num+batch_size]

        # Build docs for this batch only
        docs = []
        for _, row in batch_df.iterrows():
            doc = {
                '_key': str(row['market_id']),
                'market_ticker': row['market_id'],
                'title': str(row['title']),
                'category': str(row['category']),
                'status': str(row['status']),
                'close_time': str(row['close_time']),
                'yes_probability': float(row['yes_probability']) if pd.notna(row['yes_probability']) else None,
                'no_probability': float(row['no_probability']) if pd.notna(row['no_probability']) else None,
                'volume': float(row['volume']),
                'volume_24h': float(row['volume_24h']),
                'open_interest': float(row['open_interest']),
                'fetched_at': row['fetched_at'],
                'updated_at': datetime.now().isoformat()
            }

            # Add embedding if present
            if 'title_embedding' in row.index:
                embedding = row['title_embedding']
                # Check if embedding is valid (list or array, not None/NaN)
                if embedding is not None:
                    if hasattr(embedding, 'tolist'):
                        doc['title_embedding'] = embedding.tolist()
                    elif isinstance(embedding, list) and len(embedding) > 0:
                        doc['title_embedding'] = embedding

            docs.append(doc)

        # Upload this batch
        try:
            query = f"""
            FOR doc IN @documents
                UPSERT {{ _key: doc._key }}
                INSERT doc
                UPDATE doc
                IN {MARKET_COL}
            """
            db.aql.execute(query, bind_vars={'documents': docs})

            # Track progress
            updated_total += len(docs)
            current_batch = (batch_num // batch_size) + 1

            # Progress logging every 100 batches (10k records)
            if current_batch % 100 == 0 or current_batch == total_batches:
                print(f"  Progress: {updated_total:,}/{total_rows:,} records ({updated_total/total_rows*100:.1f}%) - Batch {current_batch:,}/{total_batches:,}")

        except Exception as e:
            print(f"  ✗ Batch {current_batch} failed: {e}")
            # Continue with next batch instead of failing completely
            continue

        # Clear batch from memory
        docs = None
        batch_df = None

    # Estimate inserted vs updated (rough approximation)
    inserted = updated_total // 2
    updated = updated_total - inserted

    print(f"[OK] Processed: {updated_total:,} records (est. {inserted:,} inserted, {updated:,} updated)")
    return inserted, updated, 0
