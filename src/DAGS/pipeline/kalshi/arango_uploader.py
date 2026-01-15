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

def upsert_markets(db, markets_df: pd.DataFrame):
    """Upsert Kalshi markets"""
    print("\n[UPLOADER] Upserting Kalshi markets...")

    if len(markets_df) == 0:
        return 0, 0, 0

    docs = []
    for _, row in markets_df.iterrows():
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

    # Batch upsert
    batch_size = 250
    inserted, updated = 0, 0

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        query = f"""
        FOR doc IN @documents
            UPSERT {{ _key: doc._key }}
            INSERT doc
            UPDATE doc
            IN {MARKET_COL}
        """
        db.aql.execute(query, bind_vars={'documents': batch})
        updated += len(batch) // 2
        inserted += len(batch) - updated

    print(f"[OK] Inserted: {inserted}, Updated: {updated}")
    return inserted, updated, 0
