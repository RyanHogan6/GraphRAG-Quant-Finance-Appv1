"""
Options Flow ArangoDB Uploader with Graph Edge Creation
Links unusual options activity to stocks, contracts, and SEC filings
"""
import os
from datetime import datetime, timedelta
from arango import ArangoClient
import numpy as np

OPTIONS_COLLECTION = "options_flow"

# Edge definitions for graph connections
EDGE_DEFINITIONS = {
    'HAS_OPTIONS_ACTIVITY': {
        'from': ['MarketData'],
        'to': [OPTIONS_COLLECTION],
        'description': 'Links stock prices to options activity on same date'
    },
    'COMPANY_HAS_OPTIONS': {
        'from': ['Company'],
        'to': [OPTIONS_COLLECTION],
        'description': 'Links company to options activity on their stock'
    },
    'OPTIONS_BEFORE_AWARD': {
        'from': [OPTIONS_COLLECTION],
        'to': ['Award'],
        'description': 'Links unusual options activity to awards announced shortly after'
    },
    'OPTIONS_BEFORE_FILING': {
        'from': [OPTIONS_COLLECTION],
        'to': ['sec_filings'],
        'description': 'Links unusual options activity to SEC filings shortly after (detect insider knowledge)'
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
    """Create options_flow collection and edge definitions"""
    graph_name = os.getenv('ARANGO_GRAPH', 'FinanceGraph')

    # Create document collection
    if not db.has_collection(OPTIONS_COLLECTION):
        db.create_collection(OPTIONS_COLLECTION)
        print(f"  Created collection: {OPTIONS_COLLECTION}")

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


def upsert_options_data(db, df):
    """
    Upload options flow data to ArangoDB and create graph edges

    Args:
        db: ArangoDB connection
        df: DataFrame with options data

    Returns:
        Tuple of (inserted, updated, edge_counts)
    """
    if df.empty:
        return 0, 0, {}

    setup_collections(db)

    options_col = db.collection(OPTIONS_COLLECTION)

    inserted = 0
    updated = 0
    docs = []

    for _, row in df.iterrows():
        ticker = str(row.get('ticker', '')).strip()
        date = str(row.get('date', '')).strip()

        if not ticker or not date:
            continue

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
        print(f"  Total options docs to upsert: {len(docs)}")
        for i in range(0, len(docs), 500):
            batch = docs[i:i+500]
            try:
                result = db.aql.execute(
                    f"FOR doc IN @docs UPSERT {{_key: doc._key}} INSERT doc UPDATE doc IN {OPTIONS_COLLECTION} RETURN {{new: NEW, old: OLD}}",
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
    edge_counts = create_options_edges(db, docs)

    return inserted, updated, edge_counts


def create_options_edges(db, options_docs):
    """
    Create edges linking options activity to:
    1. MarketData (same ticker + date)
    2. Company (by ticker)
    3. Awards (unusual activity before contract announcements)
    4. SEC filings (unusual activity before 8-K filings)
    """
    edge_counts = {}

    if not options_docs:
        return edge_counts

    print(f"  Creating options flow edges...")

    # Edge 1: MarketData → options_flow (same ticker + date)
    try:
        edges = []
        for opt in options_docs:
            ticker = opt['ticker']
            date = opt['date']

            # Find matching market data
            query = """
            FOR market IN MarketData
                FILTER market.ticker == @ticker
                FILTER market.date == @date
                LIMIT 1
                RETURN market._key
            """
            market_keys = list(db.aql.execute(query, bind_vars={'ticker': ticker, 'date': date}))

            for market_key in market_keys:
                edge_key = f"{market_key}_{opt['_key']}"
                edges.append({
                    "_key": edge_key,
                    "_from": f"MarketData/{market_key}",
                    "_to": f"{OPTIONS_COLLECTION}/{opt['_key']}",
                    "relationship": "has_options_activity"
                })

        if edges:
            for i in range(0, len(edges), 500):
                batch = edges[i:i+500]
                db.aql.execute(
                    "FOR edge IN @edges INSERT edge INTO HAS_OPTIONS_ACTIVITY OPTIONS {overwriteMode: 'ignore'}",
                    bind_vars={'edges': batch}
                )
            edge_counts['market_to_options'] = len(edges)
            print(f"    ✓ Created {len(edges)} market→options edges")

    except Exception as e:
        print(f"    ✗ Error creating market edges: {e}")

    # Edge 2: Company → options_flow (by ticker)
    try:
        edges = []
        for opt in options_docs:
            ticker = opt['ticker']

            # Find company
            query = """
            FOR company IN Company
                FILTER company.ticker == @ticker
                LIMIT 1
                RETURN company._key
            """
            company_keys = list(db.aql.execute(query, bind_vars={'ticker': ticker}))

            for company_key in company_keys:
                edge_key = f"{company_key}_{opt['_key']}"
                edges.append({
                    "_key": edge_key,
                    "_from": f"Company/{company_key}",
                    "_to": f"{OPTIONS_COLLECTION}/{opt['_key']}",
                    "relationship": "company_has_options"
                })

        if edges:
            # Remove duplicates (one edge per company, not per date)
            edges_dict = {e['_key']: e for e in edges}
            edges = list(edges_dict.values())

            for i in range(0, len(edges), 500):
                batch = edges[i:i+500]
                db.aql.execute(
                    "FOR edge IN @edges INSERT edge INTO COMPANY_HAS_OPTIONS OPTIONS {overwriteMode: 'ignore'}",
                    bind_vars={'edges': batch}
                )
            edge_counts['company_to_options'] = len(edges)
            print(f"    ✓ Created {len(edges)} company→options edges")

    except Exception as e:
        print(f"    ✗ Error creating company edges: {e}")

    # Edge 3: options_flow → Awards (unusual activity BEFORE awards)
    # Only create edges for UNUSUAL activity (potential insider knowledge)
    try:
        edges = []
        for opt in options_docs:
            # Only check if there's unusual activity
            if opt.get('unusual_total_activity') != 1 and opt.get('potential_call_sweep') != 1:
                continue

            ticker = opt['ticker']
            opt_date = opt['date']

            # Find awards 1-90 days AFTER this options activity
            query = """
            FOR award IN Award
                FILTER award.ticker == @ticker
                FILTER award.start_date > @opt_date
                FILTER DATE_DIFF(@opt_date, award.start_date, 'd') <= 90
                FILTER DATE_DIFF(@opt_date, award.start_date, 'd') >= 1
                LIMIT 10
                RETURN {key: award._key, start_date: award.start_date}
            """
            awards = list(db.aql.execute(query, bind_vars={'ticker': ticker, 'opt_date': opt_date}))

            for award in awards:
                # Calculate days before award
                from datetime import datetime as dt
                opt_dt = dt.strptime(opt_date, '%Y-%m-%d')
                award_dt = dt.strptime(award['start_date'], '%Y-%m-%d')
                days_before = (award_dt - opt_dt).days

                edge_key = f"{opt['_key']}_{award['key']}"
                edges.append({
                    "_key": edge_key,
                    "_from": f"{OPTIONS_COLLECTION}/{opt['_key']}",
                    "_to": f"Award/{award['key']}",
                    "relationship": "unusual_options_before_award",
                    "days_before": days_before,
                    "unusual_type": "call_sweep" if opt.get('potential_call_sweep') == 1 else "high_volume"
                })

        if edges:
            for i in range(0, len(edges), 500):
                batch = edges[i:i+500]
                db.aql.execute(
                    "FOR edge IN @edges INSERT edge INTO OPTIONS_BEFORE_AWARD OPTIONS {overwriteMode: 'ignore'}",
                    bind_vars={'edges': batch}
                )
            edge_counts['options_to_awards'] = len(edges)
            print(f"    ✓ Created {len(edges)} options→award edges (potential insider activity)")

    except Exception as e:
        print(f"    ✗ Error creating award edges: {e}")

    # Edge 4: options_flow → sec_filings (unusual activity BEFORE filings)
    try:
        edges = []
        for opt in options_docs:
            # Only check unusual activity
            if opt.get('unusual_total_activity') != 1:
                continue

            ticker = opt['ticker']
            opt_date = opt['date']

            # Find 8-K filings 1-30 days AFTER this options activity
            query = """
            FOR filing IN sec_filings
                FILTER filing.ticker == @ticker
                FILTER filing.type == "8-K"
                FILTER filing.filing_date > @opt_date
                FILTER DATE_DIFF(@opt_date, filing.filing_date, 'd') <= 30
                FILTER DATE_DIFF(@opt_date, filing.filing_date, 'd') >= 1
                LIMIT 5
                RETURN {key: filing._key, filing_date: filing.filing_date, sentiment: filing.avg_finbert}
            """
            filings = list(db.aql.execute(query, bind_vars={'ticker': ticker, 'opt_date': opt_date}))

            for filing in filings:
                from datetime import datetime as dt
                opt_dt = dt.strptime(opt_date, '%Y-%m-%d')
                filing_dt = dt.strptime(filing['filing_date'], '%Y-%m-%d')
                days_before = (filing_dt - opt_dt).days

                edge_key = f"{opt['_key']}_{filing['key']}"
                edges.append({
                    "_key": edge_key,
                    "_from": f"{OPTIONS_COLLECTION}/{opt['_key']}",
                    "_to": f"sec_filings/{filing['key']}",
                    "relationship": "unusual_options_before_filing",
                    "days_before": days_before,
                    "filing_sentiment": filing.get('sentiment')
                })

        if edges:
            for i in range(0, len(edges), 500):
                batch = edges[i:i+500]
                db.aql.execute(
                    "FOR edge IN @edges INSERT edge INTO OPTIONS_BEFORE_FILING OPTIONS {overwriteMode: 'ignore'}",
                    bind_vars={'edges': batch}
                )
            edge_counts['options_to_filings'] = len(edges)
            print(f"    ✓ Created {len(edges)} options→filing edges (potential insider activity)")

    except Exception as e:
        print(f"    ✗ Error creating filing edges: {e}")

    return edge_counts
