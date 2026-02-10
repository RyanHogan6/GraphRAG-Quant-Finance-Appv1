"""
Awards ArangoDB Uploader
"""
import os
from datetime import datetime
from arango import ArangoClient

def get_arango_connection():
    """Connect to ArangoDB"""
    url = os.getenv('ARANGO_URL') or os.getenv('ARANGO_HOST')
    db_name = os.getenv('ARANGO_DATABASE') or os.getenv('ARANGO_DB')
    username = os.getenv('ARANGO_USERNAME', 'root')
    password = os.getenv('ARANGO_PASSWORD')

    if not url or not url.strip():
        raise ValueError(
            "ArangoDB host not set. Set ARANGO_HOST (or ARANGO_URL) in your environment or .env (e.g. ARANGO_HOST=http://localhost:8529)."
        )
    if not db_name or not db_name.strip():
        raise ValueError(
            "ArangoDB database not set. Set ARANGO_DATABASE or ARANGO_DB in your environment or .env."
        )

    client = ArangoClient(hosts=url)
    return client.db(db_name, username=username, password=password)

def setup_collections(db):
    """Ensure Award collection and edges exist"""
    graph_name = os.getenv('ARANGO_GRAPH', 'FinanceGraph')

    if not db.has_collection('Award'):
        db.create_collection('Award')
    if not db.has_collection('HAS_AWARD'):
        db.create_collection('HAS_AWARD', edge=True)

    if db.has_graph(graph_name):
        graph = db.graph(graph_name)
        edge_defs = [ed['edge_collection'] for ed in graph.edge_definitions()]
        if 'HAS_AWARD' not in edge_defs:
            graph.create_edge_definition(
                edge_collection='HAS_AWARD',
                from_vertex_collections=['Company'],
                to_vertex_collections=['Award']
            )

def upsert_awards(db, df):
    """Upload awards to ArangoDB"""
    setup_collections(db)

    award_col = db.collection('Award')
    edge_col = db.collection('HAS_AWARD')

    inserted = 0
    updated = 0
    edges_created = 0

    docs = []
    edges = []

    for _, row in df.iterrows():
        award_id = row.get('Award ID', '')
        if not award_id:
            continue

        award_id = award_id.replace('/', '_').replace(' ', '_').replace(':', '_')[:254]

        # Parse amount
        amount_str = str(row.get('Award Amount', '')).replace(',', '').replace('$', '').strip()
        try:
            amount_float = float(amount_str) if amount_str else None
        except Exception:
            amount_float = None

        # Total potential value: Base and All Options Value (contract ceiling); fallback to award amount
        base_all_str = str(row.get('Base and All Options Value', '') or '').replace(',', '').replace('$', '').strip()
        try:
            total_potential_value = float(base_all_str) if base_all_str else None
        except Exception:
            total_potential_value = None
        if total_potential_value is None and amount_float is not None:
            total_potential_value = amount_float

        # Total obligations to date (from API or enriched); try multiple possible key names
        obligation_raw = (
            row.get('Total Obligation')
            or row.get('Action Obligation')
            or row.get('total_obligations_to_date')
            or row.get('total_obligation')
            or row.get('action_obligation')
        )
        if obligation_raw is not None and str(obligation_raw).strip() != '':
            try:
                total_obligations_to_date = float(str(obligation_raw).replace(',', '').replace('$', '').strip())
            except Exception:
                total_obligations_to_date = None
        else:
            total_obligations_to_date = None

        # Receivable proxy = total_potential_value - total_obligations_to_date (approximates outstanding contract receivables)
        receivable_proxy = None
        if total_potential_value is not None and total_obligations_to_date is not None:
            receivable_proxy = total_potential_value - total_obligations_to_date

        doc = {
            "_key": award_id,
            "recipient_name": row.get('Recipient Name', ''),
            "ticker": row.get('Ticker', ''),
            "start_date": row.get('Start Date', ''),
            "end_date": row.get('End Date', ''),
            "award_amount": row.get('Award Amount', ''),
            "award_amount_float": amount_float,
            "total_potential_value": total_potential_value,
            "total_obligations_to_date": total_obligations_to_date,
            "receivable_proxy": receivable_proxy,
            "awarding_agency": row.get('Awarding Agency', ''),
            "description": row.get('Description', ''),
            "description_embedding": row.get('description_embedding'),
            "ingested_at": datetime.utcnow().isoformat(),
        }

        doc = {k: v for k, v in doc.items() if v not in ['', None]}
        docs.append(doc)

        # Edge
        ticker = row.get('Ticker', '')
        if ticker:
            edges.append({
                "_key": f"{ticker}_{award_id}",
                "_from": f"Company/{ticker}",
                "_to": f"Award/{award_id}",
                "award_amount": amount_float
            })

    # Batch upsert
    if docs:
        for i in range(0, len(docs), 250):
            batch = docs[i:i+250]
            result = db.aql.execute(
                "FOR doc IN @docs UPSERT {_key: doc._key} INSERT doc UPDATE doc IN Award RETURN {new: NEW, old: OLD}",
                bind_vars={'docs': batch}
            )
            for r in result:
                if r['old']:
                    updated += 1
                else:
                    inserted += 1

    if edges:
        for i in range(0, len(edges), 250):
            batch = edges[i:i+250]
            db.aql.execute(
                "FOR edge IN @edges INSERT edge INTO HAS_AWARD OPTIONS {overwriteMode: 'ignore'}",
                bind_vars={'edges': batch}
            )
            edges_created += len(batch)

    return inserted, updated, edges_created
