"""
ArangoDB Schema Introspection Tool
Samples collections to understand document structure and field types
"""
from arango import ArangoClient
import json
import os
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()


def get_db():
    """Connect to ArangoDB using environment variables."""
    url = os.getenv('ARANGO_HOST', '')
    db_name = os.getenv('ARANGO_DB', 'QUANT_v3')
    username = os.getenv('ARANGO_USERNAME', 'root')
    password = os.getenv('ARANGO_PASSWORD', '')

    client = ArangoClient(hosts=url)
    return client.db(db_name, username=username, password=password)


def introspect_collection(db, collection_name, sample_size=5):
    """
    Sample documents from a collection to infer schema.

    Args:
        db: ArangoDB connection
        collection_name: Name of collection to introspect
        sample_size: Number of documents to sample

    Returns:
        dict: Field names and their types
    """
    print(f"\n{'='*80}")
    print(f"Collection: {collection_name}")
    print(f"{'='*80}")

    try:
        collection = db.collection(collection_name)
        count = collection.count()
        print(f"Total documents: {count:,}")

        if count == 0:
            print("  (empty collection)")
            return {}

        # Sample documents
        query = f"""
        FOR doc IN {collection_name}
            LIMIT {sample_size}
            RETURN doc
        """

        samples = list(db.aql.execute(query))

        if not samples:
            print("  (no samples retrieved)")
            return {}

        # Analyze field types
        field_types = defaultdict(set)

        for doc in samples:
            for key, value in doc.items():
                if value is not None:
                    field_types[key].add(type(value).__name__)

        # Print schema
        print("\nFields:")
        for field, types in sorted(field_types.items()):
            type_str = ', '.join(sorted(types))
            print(f"  {field:40} -> {type_str}")

        # Show sample document
        print("\nSample document:")
        print(json.dumps(samples[0], indent=2, default=str)[:1000])
        if len(json.dumps(samples[0], default=str)) > 1000:
            print("  ... (truncated)")

        return dict(field_types)

    except Exception as e:
        print(f"  Error: {e}")
        return {}


def introspect_schema():
    """Introspect all collections in the database."""
    print("Connecting to ArangoDB...")
    db = get_db()

    # Core collections
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

        # Prediction Markets
        'prediction_markets_polymarket',
        'prediction_markets_kalshi',
        'polymarket_traders',
        'polymarket_positions',
        'polymarket_price_history',
    ]

    schemas = {}

    for coll in collections_to_check:
        if db.has_collection(coll):
            schemas[coll] = introspect_collection(db, coll)
        else:
            print(f"\n{'='*80}")
            print(f"Collection: {coll}")
            print(f"{'='*80}")
            print("  (does not exist)")

    # Check edge collections
    print("\n" + "="*80)
    print("EDGE COLLECTIONS")
    print("="*80)

    edge_collections = [
        'HAS_MARKETDATA',
        'HAS_AWARD',
        'HAS_COMMODITY_POSITION',
        'HAS_FILING',
        'COMPANY_TRADES_COMMODITY',
        'COMPANY_HAS_OPTIONS',
        'HAS_OPTIONS_ACTIVITY',
        'OPTIONS_BEFORE_AWARD',
        'OPTIONS_BEFORE_FILING',
        'POSITION_ON_COMMODITY',
        'INVENTORY_AFFECTS_PRICE',
        'STORAGE_AFFECTS_PRICE',
        'MACRO_IMPACTS_COMMODITY',
        'has_section',
        'has_sentence',
        'market_mentions_company_polymarket',
        'market_related_to_sector_polymarket',
        'market_affects_company_polymarket',
        'market_mentions_company_kalshi',
        'market_related_to_sector_kalshi',
        'trader_has_position',
        'position_in_market',
    ]

    for edge_coll in edge_collections:
        if db.has_collection(edge_coll):
            collection = db.collection(edge_coll)
            count = collection.count()
            print(f"  {edge_coll:40} -> {count:,} edges")
        else:
            print(f"  {edge_coll:40} -> (does not exist)")

    print("\n" + "="*80)
    print("INTROSPECTION COMPLETE")
    print("="*80)

    return schemas


if __name__ == '__main__':
    introspect_schema()
