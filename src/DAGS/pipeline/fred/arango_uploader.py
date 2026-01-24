"""
FRED ArangoDB Uploader with Graph Edge Creation
Links economic data to commodities, stocks, and prediction markets
"""
import os
from datetime import datetime
from arango import ArangoClient
import numpy as np

ECONOMIC_DATA_COL = "EconomicData"

# Edge definitions for graph connections
EDGE_DEFINITIONS = {
    'HAS_ECONOMIC_CONTEXT': {
        'from': ['commodity_positions'],
        'to': [ECONOMIC_DATA_COL],
        'description': 'Links CFTC positions to macro economic conditions on same date'
    },
    'MARKET_ECONOMIC_CONTEXT': {
        'from': ['MarketData'],
        'to': [ECONOMIC_DATA_COL],
        'description': 'Links stock prices to macro economic conditions'
    },
    'PREDICTION_ECONOMIC_CONTEXT': {
        'from': ['prediction_markets_polymarket', 'prediction_markets_kalshi'],
        'to': [ECONOMIC_DATA_COL],
        'description': 'Links prediction markets to economic indicators'
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
    """Create EconomicData collection and edge definitions"""
    graph_name = os.getenv('ARANGO_GRAPH', 'FinanceGraph')

    # Create document collection
    if not db.has_collection(ECONOMIC_DATA_COL):
        db.create_collection(ECONOMIC_DATA_COL)
        print(f"  Created collection: {ECONOMIC_DATA_COL}")

    # Create edge collections
    for edge_name, config in EDGE_DEFINITIONS.items():
        if not db.has_collection(edge_name):
            db.create_collection(edge_name, edge=True)
            print(f"  Created edge collection: {edge_name}")

    # Add to graph
    if db.has_graph(graph_name):
        graph = db.graph(graph_name)
        existing_edges = [ed['edge_collection'] for ed in graph.edge_definitions()]

        for edge_name, config in EDGE_DEFINITIONS.items():
            if edge_name not in existing_edges:
                try:
                    graph.create_edge_definition(
                        edge_collection=edge_name,
                        from_vertex_collections=config['from'],
                        to_vertex_collections=config['to']
                    )
                    print(f"  Added {edge_name} to graph")
                except Exception as e:
                    print(f"  ⚠ Could not add {edge_name}: {e}")

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

def upsert_fred_data(db, df):
    """
    Upload FRED economic data to ArangoDB and create graph edges

    Args:
        db: ArangoDB connection
        df: DataFrame with economic indicators (wide format, one row per date)
    """
    if df.empty:
        return 0, 0, {}

    setup_collections(db)

    economic_col = db.collection(ECONOMIC_DATA_COL)

    inserted = 0
    updated = 0

    docs = []

    for _, row in df.iterrows():
        date = str(row.get('date', '')).strip()

        if not date:
            continue

        # Create unique key from date
        key = date.replace('-', '_')

        # Build document with all economic indicators
        doc = {
            "_key": key,
            "date": date,
            "data_source": "FRED",
            "ingested_at": datetime.utcnow().isoformat()
        }

        # Add all indicator fields with null/nan cleaning
        for col in df.columns:
            if col != 'date':
                val = clean_value(row.get(col))
                if val is not None:
                    doc[col] = val

        docs.append(doc)

    # Batch upsert documents
    if docs:
        print(f"  Total docs to upsert: {len(docs)}")
        for i in range(0, len(docs), 500):
            batch = docs[i:i+500]
            try:
                result = db.aql.execute(
                    f"FOR doc IN @docs UPSERT {{_key: doc._key}} INSERT doc UPDATE doc IN {ECONOMIC_DATA_COL} RETURN {{new: NEW, old: OLD}}",
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
    edge_counts = create_economic_edges(db, docs)

    return inserted, updated, edge_counts

def create_economic_edges(db, economic_docs):
    """
    Create edges linking economic data to:
    1. Commodity positions (same date)
    2. Market data (same date)
    3. Prediction markets (active during economic release)
    """
    edge_counts = {}

    if not economic_docs:
        return edge_counts

    print(f"  Creating economic context edges...")

    # Edge 1: EconomicData → commodity_positions (by date)
    try:
        edges = []
        for doc in economic_docs:
            date = doc['date']
            # Find commodity positions on same date
            query = """
            FOR pos IN commodity_positions
                FILTER pos.as_of_date == @date
                LIMIT 50
                RETURN pos._key
            """
            position_keys = list(db.aql.execute(query, bind_vars={'date': date}))

            for pos_key in position_keys:
                edge_key = f"{pos_key}_{doc['_key']}"
                edges.append({
                    "_key": edge_key,
                    "_from": f"commodity_positions/{pos_key}",
                    "_to": f"{ECONOMIC_DATA_COL}/{doc['_key']}"
                })

        if edges:
            for i in range(0, len(edges), 500):
                batch = edges[i:i+500]
                db.aql.execute(
                    "FOR edge IN @edges INSERT edge INTO HAS_ECONOMIC_CONTEXT OPTIONS {overwriteMode: 'ignore'}",
                    bind_vars={'edges': batch}
                )
            edge_counts['commodity_economic_edges'] = len(edges)
            print(f"    ✓ Created {len(edges)} commodity→economic edges")

    except Exception as e:
        print(f"    ✗ Error creating commodity edges: {e}")

    # Edge 2: EconomicData → MarketData (by date)
    try:
        edges = []
        for doc in economic_docs:
            date = doc['date']
            # Find market data on same date
            query = """
            FOR market IN MarketData
                FILTER market.date == @date
                LIMIT 50
                RETURN market._key
            """
            market_keys = list(db.aql.execute(query, bind_vars={'date': date}))

            for market_key in market_keys:
                edge_key = f"{market_key}_{doc['_key']}"
                edges.append({
                    "_key": edge_key,
                    "_from": f"MarketData/{market_key}",
                    "_to": f"{ECONOMIC_DATA_COL}/{doc['_key']}"
                })

        if edges:
            for i in range(0, len(edges), 500):
                batch = edges[i:i+500]
                db.aql.execute(
                    "FOR edge IN @edges INSERT edge INTO MARKET_ECONOMIC_CONTEXT OPTIONS {overwriteMode: 'ignore'}",
                    bind_vars={'edges': batch}
                )
            edge_counts['market_economic_edges'] = len(edges)
            print(f"    ✓ Created {len(edges)} market→economic edges")

    except Exception as e:
        print(f"    ✗ Error creating market edges: {e}")

    return edge_counts
