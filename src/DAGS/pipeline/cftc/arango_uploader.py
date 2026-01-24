"""
CFTC ArangoDB Uploader
"""
import os
from datetime import datetime
from arango import ArangoClient

COMMODITY_COL = "commodity_positions"
COMMODITY_EDGE = "HAS_COMMODITY_POSITION"

def get_arango_connection():
    """Connect to ArangoDB"""
    url = os.getenv('ARANGO_URL') or os.getenv('ARANGO_HOST')
    db_name = os.getenv('ARANGO_DATABASE') or os.getenv('ARANGO_DB')
    username = os.getenv('ARANGO_USERNAME', 'root')
    password = os.getenv('ARANGO_PASSWORD')

    client = ArangoClient(hosts=url)
    return client.db(db_name, username=username, password=password)

def setup_collections(db):
    """Ensure commodity collections exist"""
    graph_name = os.getenv('ARANGO_GRAPH', 'FinanceGraph')

    if not db.has_collection(COMMODITY_COL):
        db.create_collection(COMMODITY_COL)

    if not db.has_collection(COMMODITY_EDGE):
        db.create_collection(COMMODITY_EDGE, edge=True)

    if db.has_graph(graph_name):
        graph = db.graph(graph_name)
        edge_defs = [ed['edge_collection'] for ed in graph.edge_definitions()]
        if COMMODITY_EDGE not in edge_defs:
            try:
                graph.create_edge_definition(
                    edge_collection=COMMODITY_EDGE,
                    from_vertex_collections=['Company'],
                    to_vertex_collections=[COMMODITY_COL]
                )
            except:
                pass

def clean_column_name(col):
    """Convert column names to valid ArangoDB field names"""
    return (str(col).strip()
            .replace(' ', '_')
            .replace('/', '_')
            .replace('(', '')
            .replace(')', '')
            .replace('-', '_')
            .replace('%', 'Pct'))

def upsert_commodity_positions(db, df):
    """Upload commodity positions to ArangoDB"""
    import numpy as np

    setup_collections(db)

    commodity_col = db.collection(COMMODITY_COL)

    inserted = 0
    updated = 0

    docs = []

    for _, row in df.iterrows():
        commodity_code = str(row.get('CFTC Commodity Code', '')).strip()
        as_of_date = str(row.get('As of Date in Form YYYY-MM-DD', '')).strip()

        if not commodity_code or not as_of_date:
            continue

        # Create unique key
        key = f"{commodity_code}_{as_of_date}".replace('-', '_')

        # Build document
        doc = {
            "_key": key,
            "commodity_code": commodity_code,
            "as_of_date": as_of_date,
            "data_source": "CFTC_COT",
            "ingested_at": datetime.utcnow().isoformat()
        }

        # Add all other fields with cleaned names
        for col in df.columns:
            if col not in ['CFTC Commodity Code', 'As of Date in Form YYYY-MM-DD']:
                val = row.get(col)

                # Skip null/empty values
                if val is None or (isinstance(val, str) and str(val).strip() == ''):
                    continue

                # Clean NaN/infinity for numeric values
                if isinstance(val, (int, float)):
                    if np.isnan(val) or np.isinf(val):
                        continue
                    # Convert numpy types to Python types
                    if isinstance(val, (np.integer, np.int64, np.int32)):
                        val = int(val)
                    elif isinstance(val, (np.floating, np.float64, np.float32)):
                        val = float(val)

                clean_col = clean_column_name(col)
                doc[clean_col] = val

        docs.append(doc)

    # Batch upsert
    if docs:
        for i in range(0, len(docs), 500):
            batch = docs[i:i+500]
            result = db.aql.execute(
                "FOR doc IN @docs UPSERT {_key: doc._key} INSERT doc UPDATE doc IN commodity_positions RETURN {new: NEW, old: OLD}",
                bind_vars={'docs': batch}
            )
            for r in result:
                if r['old']:
                    updated += 1
                else:
                    inserted += 1

    return inserted, updated
