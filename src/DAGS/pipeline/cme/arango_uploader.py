"""
CME Futures ArangoDB Uploader with Graph Edge Creation
Links futures prices to CFTC positioning, EIA inventory, economic data, and prediction markets
"""
import os
from datetime import datetime
from arango import ArangoClient
import numpy as np

FUTURES_COLLECTION = "futures_prices"

# Edge definitions for graph connections
EDGE_DEFINITIONS = {
    'POSITION_ON_COMMODITY': {
        'from': ['commodity_positions'],
        'to': [FUTURES_COLLECTION],
        'description': 'Links CFTC positioning data to futures prices (by commodity + date)'
    },
    'INVENTORY_AFFECTS_PRICE': {
        'from': ['eia_crude_inventory'],
        'to': [FUTURES_COLLECTION],
        'description': 'Links crude oil inventory to crude futures prices'
    },
    'STORAGE_AFFECTS_PRICE': {
        'from': ['eia_natgas_storage'],
        'to': [FUTURES_COLLECTION],
        'description': 'Links natural gas storage to NG futures prices'
    },
    'MACRO_IMPACTS_COMMODITY': {
        'from': ['EconomicData'],
        'to': [FUTURES_COLLECTION],
        'description': 'Links economic indicators to commodity prices (rates, inflation, etc.)'
    },
    'MARKET_PREDICTS_COMMODITY': {
        'from': ['prediction_markets_polymarket', 'prediction_markets_kalshi'],
        'to': [FUTURES_COLLECTION],
        'description': 'Links prediction markets to commodity futures (energy, metals markets)'
    }
}

# Mapping between CFTC commodity codes and futures commodities
CFTC_TO_FUTURES_MAP = {
    # Energy
    '067651': 'CRUDE_OIL',      # Crude Oil WTI
    '023651': 'NATURAL_GAS',    # Natural Gas
    '111659': 'GASOLINE',       # RBOB Gasoline
    '022651': 'HEATING_OIL',    # Heating Oil

    # Metals
    '088691': 'GOLD',           # Gold
    '084691': 'SILVER',         # Silver
    '085692': 'COPPER',         # Copper
    '076651': 'PLATINUM',       # Platinum

    # Agriculture
    '002602': 'CORN',           # Corn
    '001602': 'WHEAT',          # Wheat (Chicago)
    '005602': 'SOYBEANS',       # Soybeans
    '007601': 'SOYBEAN_OIL',    # Soybean Oil
    '033661': 'COTTON',         # Cotton
    '083731': 'COFFEE',         # Coffee
    '080732': 'SUGAR',          # Sugar #11

    # Livestock
    '057642': 'LIVE_CATTLE',    # Live Cattle
    '054642': 'LEAN_HOGS',      # Lean Hogs
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
    """Create futures_prices collection and edge definitions"""
    graph_name = os.getenv('ARANGO_GRAPH', 'FinanceGraph')

    # Create document collection
    if not db.has_collection(FUTURES_COLLECTION):
        db.create_collection(FUTURES_COLLECTION)
        print(f"  Created collection: {FUTURES_COLLECTION}")

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


def upsert_futures_data(db, df):
    """
    Upload futures prices to ArangoDB and create graph edges

    Args:
        db: ArangoDB connection
        df: DataFrame with futures data (wide format)

    Returns:
        Tuple of (inserted, updated, edge_counts)
    """
    if df.empty:
        return 0, 0, {}

    setup_collections(db)

    futures_col = db.collection(FUTURES_COLLECTION)

    inserted = 0
    updated = 0
    docs = []

    for _, row in df.iterrows():
        commodity = str(row.get('commodity', '')).strip()
        date = str(row.get('date', '')).strip()

        if not commodity or not date:
            continue

        # Create unique key from commodity + date
        key = f"{commodity}_{date.replace('-', '_')}"

        # Build document
        doc = {
            "_key": key,
            "commodity": commodity,
            "date": date
        }

        # Add all fields with null/nan cleaning
        for col in df.columns:
            if col not in ['commodity', 'date']:
                val = clean_value(row.get(col))
                if val is not None:
                    doc[col] = val

        docs.append(doc)

    # Batch upsert documents
    if docs:
        print(f"  Total futures docs to upsert: {len(docs)}")
        for i in range(0, len(docs), 500):
            batch = docs[i:i+500]
            try:
                result = db.aql.execute(
                    f"FOR doc IN @docs UPSERT {{_key: doc._key}} INSERT doc UPDATE doc IN {FUTURES_COLLECTION} RETURN {{new: NEW, old: OLD}}",
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
    edge_counts = create_futures_edges(db, docs)

    return inserted, updated, edge_counts


def create_futures_edges(db, futures_docs):
    """
    Create edges linking futures prices to:
    1. CFTC commodity positions (same commodity + date)
    2. EIA inventory data (crude oil)
    3. EIA storage data (natural gas)
    4. Economic data (same date, Fed policy impacts commodities)
    5. Prediction markets (energy/commodity markets)
    """
    edge_counts = {}

    if not futures_docs:
        return edge_counts

    print(f"  Creating futures price edges...")

    # Edge 1: CFTC positions → futures prices (by commodity + date)
    try:
        edges = []
        for future in futures_docs:
            commodity = future['commodity']
            date = future['date']

            # Find matching CFTC records
            # CFTC uses commodity codes, we need to reverse map
            cftc_codes = [code for code, comm in CFTC_TO_FUTURES_MAP.items() if comm == commodity]

            if cftc_codes:
                query = """
                FOR pos IN commodity_positions
                    FILTER pos.as_of_date == @date
                    FILTER pos.commodity_code IN @codes
                    LIMIT 10
                    RETURN pos._key
                """
                position_keys = list(db.aql.execute(query, bind_vars={'date': date, 'codes': cftc_codes}))

                for pos_key in position_keys:
                    edge_key = f"{pos_key}_{future['_key']}"
                    edges.append({
                        "_key": edge_key,
                        "_from": f"commodity_positions/{pos_key}",
                        "_to": f"{FUTURES_COLLECTION}/{future['_key']}",
                        "relationship": "positioning_on_price"
                    })

        if edges:
            for i in range(0, len(edges), 500):
                batch = edges[i:i+500]
                db.aql.execute(
                    "FOR edge IN @edges INSERT edge INTO POSITION_ON_COMMODITY OPTIONS {overwriteMode: 'ignore'}",
                    bind_vars={'edges': batch}
                )
            edge_counts['cftc_to_futures'] = len(edges)
            print(f"    ✓ Created {len(edges)} CFTC→futures edges")

    except Exception as e:
        print(f"    ✗ Error creating CFTC edges: {e}")

    # Edge 2: EIA crude inventory → crude futures prices
    try:
        edges = []
        for future in futures_docs:
            if future['commodity'] == 'CRUDE_OIL':
                date = future['date']

                # Find EIA inventory records (approximate date match)
                query = """
                FOR inv IN eia_crude_inventory
                    FILTER inv.report_date >= DATE_SUBTRACT(@date, 7, 'day')
                    FILTER inv.report_date <= DATE_ADD(@date, 7, 'day')
                    LIMIT 5
                    RETURN inv._key
                """
                inv_keys = list(db.aql.execute(query, bind_vars={'date': date}))

                for inv_key in inv_keys:
                    edge_key = f"{inv_key}_{future['_key']}"
                    edges.append({
                        "_key": edge_key,
                        "_from": f"eia_crude_inventory/{inv_key}",
                        "_to": f"{FUTURES_COLLECTION}/{future['_key']}",
                        "relationship": "inventory_affects_price"
                    })

        if edges:
            for i in range(0, len(edges), 500):
                batch = edges[i:i+500]
                db.aql.execute(
                    "FOR edge IN @edges INSERT edge INTO INVENTORY_AFFECTS_PRICE OPTIONS {overwriteMode: 'ignore'}",
                    bind_vars={'edges': batch}
                )
            edge_counts['eia_inventory_to_crude'] = len(edges)
            print(f"    ✓ Created {len(edges)} EIA inventory→crude edges")

    except Exception as e:
        print(f"    ✗ Error creating EIA inventory edges: {e}")

    # Edge 3: EIA natgas storage → natgas futures prices
    try:
        edges = []
        for future in futures_docs:
            if future['commodity'] == 'NATURAL_GAS':
                date = future['date']

                query = """
                FOR storage IN eia_natgas_storage
                    FILTER storage.report_date >= DATE_SUBTRACT(@date, 7, 'day')
                    FILTER storage.report_date <= DATE_ADD(@date, 7, 'day')
                    LIMIT 5
                    RETURN storage._key
                """
                storage_keys = list(db.aql.execute(query, bind_vars={'date': date}))

                for storage_key in storage_keys:
                    edge_key = f"{storage_key}_{future['_key']}"
                    edges.append({
                        "_key": edge_key,
                        "_from": f"eia_natgas_storage/{storage_key}",
                        "_to": f"{FUTURES_COLLECTION}/{future['_key']}",
                        "relationship": "storage_affects_price"
                    })

        if edges:
            for i in range(0, len(edges), 500):
                batch = edges[i:i+500]
                db.aql.execute(
                    "FOR edge IN @edges INSERT edge INTO STORAGE_AFFECTS_PRICE OPTIONS {overwriteMode: 'ignore'}",
                    bind_vars={'edges': batch}
                )
            edge_counts['eia_storage_to_natgas'] = len(edges)
            print(f"    ✓ Created {len(edges)} EIA storage→natgas edges")

    except Exception as e:
        print(f"    ✗ Error creating EIA storage edges: {e}")

    # Edge 4: Economic data → commodity prices (same date)
    try:
        edges = []
        for future in futures_docs:
            date = future['date']

            query = """
            FOR econ IN EconomicData
                FILTER econ.date == @date
                LIMIT 1
                RETURN econ._key
            """
            econ_keys = list(db.aql.execute(query, bind_vars={'date': date}))

            for econ_key in econ_keys:
                edge_key = f"{econ_key}_{future['_key']}"
                edges.append({
                    "_key": edge_key,
                    "_from": f"EconomicData/{econ_key}",
                    "_to": f"{FUTURES_COLLECTION}/{future['_key']}",
                    "relationship": "macro_impacts_commodity"
                })

        if edges:
            for i in range(0, len(edges), 500):
                batch = edges[i:i+500]
                db.aql.execute(
                    "FOR edge IN @edges INSERT edge INTO MACRO_IMPACTS_COMMODITY OPTIONS {overwriteMode: 'ignore'}",
                    bind_vars={'edges': batch}
                )
            edge_counts['economic_to_futures'] = len(edges)
            print(f"    ✓ Created {len(edges)} economic→futures edges")

    except Exception as e:
        print(f"    ✗ Error creating economic edges: {e}")

    return edge_counts
