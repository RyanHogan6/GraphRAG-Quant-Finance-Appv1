"""
EIA ArangoDB Uploader with Graph Edge Creation
"""
import os
from datetime import datetime
from arango import ArangoClient
import numpy as np

# Collection definitions
COLLECTIONS = {
    'natgas_storage': {
        'collection': 'eia_natgas_storage',
        'edge': 'HAS_NATGAS_STORAGE_DATA',
        'commodity_codes': ['023651']  # CFTC Natural Gas code
    },
    'crude_inventory': {
        'collection': 'eia_crude_inventory',
        'edge': 'HAS_CRUDE_INVENTORY_DATA',
        'commodity_codes': ['067651']  # CFTC Crude Oil code
    },
    'lng_exports': {
        'collection': 'eia_lng_exports',
        'edge': 'HAS_LNG_EXPORT_DATA',
        'commodity_codes': ['023651']  # Links to nat gas
    },
    'natgas_production': {
        'collection': 'eia_natgas_production',
        'edge': 'HAS_NATGAS_PRODUCTION_DATA',
        'commodity_codes': ['023651']  # Links to nat gas
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
    """Create EIA collections and edge definitions"""
    graph_name = os.getenv('ARANGO_GRAPH', 'FinanceGraph')

    # Create document collections
    for config in COLLECTIONS.values():
        col_name = config['collection']
        edge_name = config['edge']

        if not db.has_collection(col_name):
            db.create_collection(col_name)
            print(f"  Created collection: {col_name}")

        if not db.has_collection(edge_name):
            db.create_collection(edge_name, edge=True)
            print(f"  Created edge collection: {edge_name}")

    # Add to graph
    if db.has_graph(graph_name):
        graph = db.graph(graph_name)
        existing_edges = [ed['edge_collection'] for ed in graph.edge_definitions()]

        for config in COLLECTIONS.values():
            edge_name = config['edge']
            col_name = config['collection']

            if edge_name not in existing_edges:
                try:
                    graph.create_edge_definition(
                        edge_collection=edge_name,
                        from_vertex_collections=['commodity_positions'],
                        to_vertex_collections=[col_name]
                    )
                    print(f"  Added {edge_name} to graph")
                except:
                    pass

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

def upsert_eia_dataset(db, df, dataset_key):
    """
    Upload EIA data to specific collection and create edges

    Args:
        db: ArangoDB connection
        df: DataFrame with EIA data
        dataset_key: Key in COLLECTIONS dict
    """
    if df.empty:
        return 0, 0, 0

    config = COLLECTIONS[dataset_key]
    collection_name = config['collection']
    edge_name = config['edge']
    commodity_codes = config['commodity_codes']

    collection = db.collection(collection_name)
    edge_collection = db.collection(edge_name)

    inserted = 0
    updated = 0
    edges_created = 0

    docs = []

    for _, row in df.iterrows():
        report_date = str(row.get('report_date', '')).strip()

        if not report_date:
            continue

        # Create unique key from date
        key = report_date.replace('-', '_')

        # Build document with cleaned values
        doc = {
            "_key": key,
            "report_date": report_date,
            "data_source": row.get('data_source', ''),
            "frequency": row.get('frequency', ''),
            "ingested_at": datetime.utcnow().isoformat()
        }

        # Add all other fields with null/nan cleaning
        for col in df.columns:
            if col not in ['report_date', 'data_source', 'frequency', 'period']:
                val = clean_value(row.get(col))
                if val is not None:
                    doc[col] = val

        docs.append(doc)

    # Batch upsert documents
    if docs:
        for i in range(0, len(docs), 500):
            batch = docs[i:i+500]
            result = db.aql.execute(
                f"FOR doc IN @docs UPSERT {{_key: doc._key}} INSERT doc UPDATE doc IN {collection_name} RETURN {{new: NEW, old: OLD}}",
                bind_vars={'docs': batch}
            )
            for r in result:
                if r['old']:
                    updated += 1
                else:
                    inserted += 1

    # Create graph edges: commodity_positions -> eia_data
    # Links CFTC positioning to EIA fundamentals
    if docs:
        print(f"  Creating edges to commodity_positions...")

        for commodity_code in commodity_codes:
            # Find recent commodity positions for this code
            query = """
            FOR pos IN commodity_positions
                FILTER pos.commodity_code == @code
                SORT pos.as_of_date DESC
                LIMIT 100
                RETURN pos._key
            """
            position_keys = list(db.aql.execute(query, bind_vars={'code': commodity_code}))

            if position_keys:
                edges = []
                for pos_key in position_keys:
                    for doc in docs:
                        edge_key = f"{pos_key}_{doc['_key']}"
                        edges.append({
                            "_key": edge_key,
                            "_from": f"commodity_positions/{pos_key}",
                            "_to": f"{collection_name}/{doc['_key']}"
                        })

                # Batch insert edges
                if edges:
                    for i in range(0, len(edges), 500):
                        batch = edges[i:i+500]
                        db.aql.execute(
                            f"FOR edge IN @edges INSERT edge INTO {edge_name} OPTIONS {{overwriteMode: 'ignore'}}",
                            bind_vars={'edges': batch}
                        )
                    edges_created = len(edges)

    return inserted, updated, edges_created

def upsert_all_eia_data(db, datasets):
    """
    Upload all EIA datasets and create graph edges

    Args:
        db: ArangoDB connection
        datasets: Dict of dataframes from fetch_all_eia_data()
    """
    setup_collections(db)

    results = {}

    for dataset_key, df in datasets.items():
        if not df.empty:
            print(f"\n  Uploading {dataset_key}...")
            inserted, updated, edges = upsert_eia_dataset(db, df, dataset_key)
            results[dataset_key] = {
                'inserted': inserted,
                'updated': updated,
                'edges': edges
            }
            print(f"    Inserted: {inserted}, Updated: {updated}, Edges: {edges}")

    return results
