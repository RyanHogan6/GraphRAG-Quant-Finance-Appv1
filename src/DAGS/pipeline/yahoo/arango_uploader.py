"""
Yahoo Finance MarketData ArangoDB Uploader
Uploads stock price data with technical indicators to MarketData collection
"""
import os
from datetime import datetime
from arango import ArangoClient
import numpy as np
import pandas as pd

MARKETDATA_COLLECTION = "MarketData"
COMPANY_COLLECTION = "Company"

# Edge definitions
EDGE_DEFINITIONS = {
    'HAS_MARKETDATA': {
        'from': [COMPANY_COLLECTION],
        'to': [MARKETDATA_COLLECTION],
        'description': 'Links companies to their daily market data (OHLCV + indicators)'
    }
}


def get_arango_connection():
    """Connect to ArangoDB"""
    url = os.getenv('ARANGO_URL') or os.getenv('ARANGO_HOST')
    db_name = os.getenv('ARANGO_DATABASE') or os.getenv('ARANGO_DB')
    username = os.getenv('ARANGO_USERNAME', 'root')
    password = os.getenv('ARANGO_PASSWORD')

    client = ArangoClient(hosts=url)
    return client.db(db_name, username=username, password=password)


def setup_collections(db):
    """Ensure collections and edges exist"""
    # Create MarketData collection
    if not db.has_collection(MARKETDATA_COLLECTION):
        db.create_collection(MARKETDATA_COLLECTION)
        print(f"  Created {MARKETDATA_COLLECTION} collection")

    # Create edge collections
    for edge_name, config in EDGE_DEFINITIONS.items():
        if not db.has_collection(edge_name):
            db.create_collection(edge_name, edge=True)
            print(f"  Created {edge_name} edge collection")


def clean_value(val):
    """Clean values for ArangoDB (remove NaN/inf)"""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if np.isnan(val) or np.isinf(val):
            return None
        if isinstance(val, (np.integer, np.int64, np.int32)):
            return int(val)
        elif isinstance(val, (np.floating, np.float64, np.float32)):
            return float(val)
    return val


def upsert_market_data(db, df):
    """
    Upload market data to ArangoDB and create graph edges

    Args:
        db: ArangoDB connection
        df: DataFrame with market data (must have ticker + date columns)

    Returns:
        Tuple of (inserted, updated, edge_counts)
    """
    if df.empty:
        return 0, 0, {}

    setup_collections(db)

    market_col = db.collection(MARKETDATA_COLLECTION)

    inserted = 0
    updated = 0
    docs = []

    for _, row in df.iterrows():
        ticker = str(row.get('ticker', '')).strip()
        date = str(row.get('date', '')).strip()

        if not ticker or not date:
            continue

        # Ensure date is in YYYY-MM-DD format (strip time if present)
        date = date.split(' ')[0]  # Remove timestamp if present

        # Create unique key from ticker + date
        key = f"{ticker}_{date.replace('-', '_')}"

        # Build document
        doc = {
            "_key": key,
            "ticker": ticker,
            "date": date
        }

        # Add all fields with null/nan cleaning
        for col in df.columns:
            if col not in ['ticker', 'date']:
                val = clean_value(row.get(col))
                if val is not None:
                    doc[col] = val

        docs.append(doc)

    # Batch upsert documents
    if docs:
        print(f"  Total market data docs to upsert: {len(docs)}")
        for i in range(0, len(docs), 500):
            batch = docs[i:i+500]
            try:
                # Use ticker+date as match criteria (respects unique index)
                result = db.aql.execute(
                    f"""
                    FOR doc IN @docs
                        UPSERT {{ticker: doc.ticker, date: doc.date}}
                        INSERT doc
                        UPDATE doc
                        IN {MARKETDATA_COLLECTION}
                        RETURN {{new: NEW, old: OLD}}
                    """,
                    bind_vars={'docs': batch}
                )
                for r in result:
                    if r['old']:
                        updated += 1
                    else:
                        inserted += 1
            except Exception as e:
                print(f"  ✗ Batch upsert error: {e}")

    # Create graph edges
    edge_counts = create_marketdata_edges(db, docs)

    return inserted, updated, edge_counts


def create_marketdata_edges(db, market_docs):
    """
    Create edges linking companies to their market data

    Edge: Company → MarketData (HAS_MARKETDATA)
    """
    edge_counts = {}

    if not market_docs:
        return edge_counts

    print(f"  Creating market data edges...")

    # Edge: Company → MarketData (by ticker)
    try:
        edges = []
        # Group by ticker to avoid duplicate edges
        tickers_in_batch = set(doc['ticker'] for doc in market_docs)

        for ticker in tickers_in_batch:
            # Find company
            query = """
            FOR company IN Company
                FILTER company.ticker == @ticker
                LIMIT 1
                RETURN company._key
            """
            company_keys = list(db.aql.execute(query, bind_vars={'ticker': ticker}))

            if company_keys:
                company_key = company_keys[0]
                # Get all market data records for this ticker in this batch
                market_keys = [doc['_key'] for doc in market_docs if doc['ticker'] == ticker]

                for market_key in market_keys:
                    edge_key = f"{company_key}_{market_key}"
                    edges.append({
                        "_key": edge_key,
                        "_from": f"{COMPANY_COLLECTION}/{company_key}",
                        "_to": f"{MARKETDATA_COLLECTION}/{market_key}",
                        "relationship": "has_market_data"
                    })

        if edges:
            for i in range(0, len(edges), 500):
                batch = edges[i:i+500]
                db.aql.execute(
                    "FOR edge IN @edges INSERT edge INTO HAS_MARKETDATA OPTIONS {overwriteMode: 'ignore'}",
                    bind_vars={'edges': batch}
                )
            edge_counts['company_to_marketdata'] = len(edges)
            print(f"    ✓ Created {len(edges)} company→market edges")

    except Exception as e:
        print(f"    ✗ Error creating market edges: {e}")

    return edge_counts
