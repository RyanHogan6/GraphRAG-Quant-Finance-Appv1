"""
Runtime schema introspection for dynamic query generation
Adapts from src/DAGS/pipeline/utils/introspect_schema.py with TTL caching
"""
from arango import ArangoClient
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sys

# Add parent directory to path for config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

# In-memory cache with TTL (longer TTL = fewer full introspects = faster queries)
_schema_cache = {
    'data': None,
    'timestamp': None,
    'ttl_minutes': 240  # Refresh every 4 hours (was 60; reduces load and prompt size churn)
}


def get_db():
    """Connect to ArangoDB using config (never displays credentials)"""
    try:
        client = ArangoClient(hosts=config.ARANGO_URL)
        return client.db(
            config.DB_NAME,
            username=config.USERNAME,
            password=config.PASSWORD
        )
    except Exception as e:
        print(f"[SCHEMA] Error connecting to DB: {e}")
        raise


def is_cache_stale() -> bool:
    """Check if cache exceeded TTL"""
    if _schema_cache['timestamp'] is None:
        return True
    age_minutes = (datetime.now() - _schema_cache['timestamp']).total_seconds() / 60
    return age_minutes > _schema_cache['ttl_minutes']


def get_cached_schema() -> Dict[str, Any]:
    """
    Get schema from cache or refresh if stale

    Returns:
        {
            'collections': {
                'Company': {
                    'fields': {'ticker': 'str', 'sector': 'str', ...},
                    'sample_values': {'ticker': ['AAPL', 'MSFT', ...], ...},
                    'count': 612,
                    'date_fields': ['founded_date', ...]
                },
                ...
            },
            'edges': {
                'HAS_MARKETDATA': {'from': ['Company'], 'to': ['MarketData']},
                ...
            },
            'introspection_time': '2026-02-01T10:30:00'
        }
    """
    if _schema_cache['data'] is None or is_cache_stale():
        print("[SCHEMA] Cache miss or stale, refreshing...")
        _schema_cache['data'] = introspect_full_schema()
        _schema_cache['timestamp'] = datetime.now()
        print(f"[SCHEMA] Cache refreshed at {_schema_cache['timestamp'].strftime('%H:%M:%S')}")
    else:
        age_minutes = (datetime.now() - _schema_cache['timestamp']).total_seconds() / 60
        print(f"[SCHEMA] Cache hit (age: {age_minutes:.1f} minutes)")

    return _schema_cache['data']


def introspect_collection(db, collection_name: str, sample_size: int = 10) -> Dict[str, Any]:
    """
    Sample documents from a collection to infer schema

    Args:
        db: ArangoDB connection
        collection_name: Name of collection to introspect
        sample_size: Number of documents to sample (default 10 for speed)

    Returns:
        dict: Field names, types, sample values, date fields
    """
    try:
        collection = db.collection(collection_name)
        count = collection.count()

        if count == 0:
            return {}

        # Sample documents (small sample for performance)
        query = f"""
        FOR doc IN {collection_name}
            LIMIT {sample_size}
            RETURN doc
        """

        samples = list(db.aql.execute(query))

        if not samples:
            return {}

        # Analyze field types
        field_types = defaultdict(set)
        sample_values = defaultdict(list)
        date_fields = []

        for doc in samples:
            for key, value in doc.items():
                if key.startswith('_'):  # Skip internal fields
                    continue

                if value is not None:
                    # Track type
                    value_type = type(value).__name__
                    field_types[key].add(value_type)

                    # Collect sample values (first 5 unique, excluding large objects)
                    if len(sample_values[key]) < 5:
                        if isinstance(value, (str, int, float, bool)):
                            if value not in sample_values[key]:
                                sample_values[key].append(value)
                        elif isinstance(value, list) and len(value) > 0:
                            # For arrays, just note it's a list
                            if 'array' not in sample_values[key]:
                                sample_values[key].append('array')

                    # Detect date fields
                    if 'date' in key.lower() or 'time' in key.lower():
                        if key not in date_fields:
                            date_fields.append(key)

        # Convert sets to strings (most common type)
        field_types_clean = {}
        for k, v in field_types.items():
            if len(v) == 1:
                field_types_clean[k] = list(v)[0]
            else:
                field_types_clean[k] = 'mixed'

        return {
            'fields': field_types_clean,
            'sample_values': dict(sample_values),
            'date_fields': date_fields,
            'count': count
        }

    except Exception as e:
        print(f"[SCHEMA] Error introspecting {collection_name}: {e}")
        return {}


def introspect_full_schema() -> Dict[str, Any]:
    """
    Discover all collections and their schemas at runtime

    Returns schema dict with collections, edges, and metadata
    """
    print("[SCHEMA] Starting full schema introspection...")
    db = get_db()

    result = {
        'collections': {},
        'edges': {},
        'introspection_time': datetime.now().isoformat()
    }

    # Collections to check (from pipeline/utils/introspect_schema.py)
    collections_to_check = [
        'Company',
        'MarketData',
        'Award',
        'EconomicData',
        'commodity_positions',
        'futures_prices',
        'options_flow',

        # EIA
        'eia_crude_inventory',
        'eia_natgas_storage',
        'eia_natgas_production',
        'eia_lng_exports',

        # SEC
        'sec_filings',
        'sec_sections',
        'sec_sentences',
        'sec_exhibits',
        'sec_xbrl_data',

        # Prediction Markets
        'prediction_markets_polymarket',
        'prediction_markets_kalshi',
        'polymarket_traders',
        'polymarket_positions',
        'polymarket_price_history',
        'congressional_trades',
    ]

    # Introspect each collection
    for coll_name in collections_to_check:
        try:
            if db.has_collection(coll_name):
                schema = introspect_collection(db, coll_name, sample_size=10)
                if schema:
                    result['collections'][coll_name] = schema
        except Exception as e:
            print(f"[SCHEMA] Error checking {coll_name}: {e}")
            continue

    # Get edge definitions from graph
    try:
        edge_query = """
        FOR g IN _graphs
          FOR edgeDef IN g.edgeDefinitions
            RETURN {
              edge: edgeDef.collection,
              from: edgeDef.from,
              to: edgeDef.to
            }
        """
        edges = list(db.aql.execute(edge_query))
        for edge in edges:
            result['edges'][edge['edge']] = {
                'from': edge['from'],
                'to': edge['to']
            }
    except Exception as e:
        print(f"[SCHEMA] Error getting edges: {e}")

    # Known edge collections used by the app that may not be in _graphs
    KNOWN_EDGES = {
        'COMPANY_HAS_OPTIONS': {'from': ['Company'], 'to': ['options_flow']},
        'OPTIONS_BEFORE_FILING': {'from': ['options_flow'], 'to': ['sec_filings']},
        'OPTIONS_BEFORE_AWARD': {'from': ['options_flow'], 'to': ['Award']},
        'HAS_OPTIONS_ACTIVITY': {'from': ['MarketData'], 'to': ['options_flow']},
        'has_exhibit': {'from': ['sec_filings'], 'to': ['sec_exhibits']},
        'has_xbrl_data': {'from': ['sec_filings'], 'to': ['sec_xbrl_data']},
        'INVENTORY_AFFECTS_PRICE': {'from': ['eia_crude_inventory'], 'to': ['futures_prices']},
        'STORAGE_AFFECTS_PRICE': {'from': ['eia_natgas_storage'], 'to': ['futures_prices']},
        'POSITION_ON_COMMODITY': {'from': ['commodity_positions'], 'to': ['futures_prices']},
        'MACRO_IMPACTS_COMMODITY': {'from': ['EconomicData'], 'to': ['futures_prices']},
        'COMPANY_TRADES_COMMODITY': {'from': ['Company'], 'to': ['futures_prices']},
        'CONGRESS_TRADES_COMPANY': {'from': ['congressional_trades'], 'to': ['Company']},
    }
    for edge_name, edge_def in KNOWN_EDGES.items():
        if edge_name not in result['edges']:
            result['edges'][edge_name] = edge_def

    print(f"[SCHEMA] Introspected {len(result['collections'])} collections, {len(result['edges'])} edges")
    return result


def get_collection_schema_dynamic(collection_name: str) -> Optional[Dict[str, Any]]:
    """Get schema for a specific collection from cache"""
    schema = get_cached_schema()
    return schema['collections'].get(collection_name)


def format_collection_for_prompt(collection_name: str, schema: Dict[str, Any]) -> str:
    """
    Format collection schema for LLM prompt

    Args:
        collection_name: Name of the collection
        schema: Schema dict from introspection

    Returns:
        Formatted string for prompt
    """
    if not schema or not schema.get('fields'):
        return f"**{collection_name}** - (No schema available)\n"

    # Build field list
    fields = schema.get('fields', {})
    field_list = []
    for field_name, field_type in sorted(fields.items()):
        field_list.append(f"{field_name} ({field_type})")

    # Add sample values for important fields
    sample_values = schema.get('sample_values', {})
    sample_text = ""
    if sample_values:
        important_samples = []
        for field in ['ticker', 'sector', 'category', 'type']:
            if field in sample_values:
                values = sample_values[field][:3]  # First 3 samples
                important_samples.append(f"{field}: {values}")

        if important_samples:
            sample_text = f"\nSample values: {', '.join(important_samples)}"

    # Note embedding fields (for semantic search)
    embedding_fields = [f for f in fields.keys() if 'embedding' in f.lower()]
    embedding_note = ""
    if embedding_fields:
        embedding_note = f"\n✅ HAS EMBEDDINGS: {', '.join(embedding_fields)} - use COSINE_SIMILARITY for semantic search"

    return f"""**{collection_name}**
Fields: {', '.join(field_list[:15])}{'...' if len(field_list) > 15 else ''}{sample_text}{embedding_note}
"""


def get_relevant_collections_dynamic(question: str) -> List[str]:
    """
    Detect which collections are relevant to user question
    Uses keyword matching + schema field matching

    Args:
        question: User's natural language question

    Returns:
        List of relevant collection names
    """
    schema = get_cached_schema()
    relevant = set()
    question_lower = question.lower()

    # Keyword-based detection (from planning.py logic)
    keyword_map = {
        'Company': ['company', 'ticker', 'stock', 'sector', 'industry'],
        'MarketData': ['price', 'stock', 'market', 'technical', 'sma', 'ema', 'macd', 'volume'],
        'Award': ['contract', 'award', 'government', 'federal', 'usaspending'],
        'sec_filings': ['sec', 'filing', '10-k', '10-q', 'sentiment', 'risk'],
        'sec_sentences': ['sentence', 'filing text'],
        'sec_xbrl_data': ['financial', 'balance sheet', 'income statement', 'cash flow'],
        'options_flow': ['options', 'calls', 'puts', 'unusual activity'],
        'prediction_markets_polymarket': ['polymarket', 'prediction', 'betting', 'odds', 'probability'],
        'prediction_markets_kalshi': ['kalshi', 'event contract'],
        'polymarket_traders': ['trader', 'whale', 'position'],
        'commodity_positions': ['commodity', 'cftc', 'futures'],
        'futures_prices': ['futures', 'crude', 'gold', 'copper', 'oil'],
        'EconomicData': ['economy', 'economic', 'fed', 'unemployment', 'gdp'],
        'eia_crude_inventory': ['crude', 'oil inventory', 'cushing'],
        'eia_natgas_storage': ['natural gas', 'gas storage'],
        'congressional_trades': ['congressional', 'congress', 'politician', 'stock act', 'disclosure'],
    }

    for coll_name, keywords in keyword_map.items():
        if any(keyword in question_lower for keyword in keywords):
            relevant.add(coll_name)

    # Field name matching - check if question mentions specific fields
    for coll_name, coll_schema in schema['collections'].items():
        if coll_name in relevant:
            continue  # Already added

        fields = coll_schema.get('fields', {})
        for field_name in fields.keys():
            # Skip common fields
            if field_name in ['_key', '_id', '_rev', 'ticker', 'date']:
                continue

            # Check if field name appears in question
            if field_name.lower() in question_lower:
                relevant.add(coll_name)
                break

    # Default fallback if no matches
    if not relevant:
        relevant = {'Company', 'MarketData'}

    return list(relevant)


def invalidate_cache():
    """Manually invalidate cache (for testing or after pipeline runs)"""
    global _schema_cache
    _schema_cache['data'] = None
    _schema_cache['timestamp'] = None
    print("[SCHEMA] Cache invalidated")


# For testing
if __name__ == '__main__':
    schema = get_cached_schema()
    print(f"\nIntrospected {len(schema['collections'])} collections:")
    for coll_name, coll_schema in schema['collections'].items():
        print(f"  {coll_name}: {coll_schema['count']} docs, {len(coll_schema['fields'])} fields")

    print(f"\nEdges: {len(schema['edges'])}")
    for edge_name in list(schema['edges'].keys())[:5]:
        print(f"  {edge_name}")
