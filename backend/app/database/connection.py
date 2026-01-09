"""
Database connection and query execution for FastAPI
Ported from Streamlit database.py
"""
from arango import ArangoClient
from functools import lru_cache
import sys
import os

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

# Global database connection
_db_instance = None


def get_db():
    """Get or create cached ArangoDB connection"""
    global _db_instance
    if _db_instance is None:
        client = ArangoClient(hosts=config.ARANGO_URL)
        _db_instance = client.db(
            config.DB_NAME,
            username=config.USERNAME,
            password=config.PASSWORD
        )
    return _db_instance


def get_collection_stats(collection_name: str):
    """Get statistics for a collection"""
    db = get_db()
    try:
        collection = db.collection(collection_name)
        return {
            "count": collection.count(),
            "name": collection_name
        }
    except:
        return {"count": 0, "name": collection_name}


def get_collections_info():
    """Get all collections with their document counts"""
    collections = [
        "Company",
        "MarketData",
        "Award",
        "EconomicData",
        "commodity_positions",
        "prediction_markets_polymarket",
        "prediction_markets_kalshi",
        "sec_filings",
        "sec_sections",
        "sec_sentences"
    ]
    stats = []
    for col in collections:
        stats.append(get_collection_stats(col))
    return stats


def browse_collection(collection_name: str, limit: int = 50, filters: dict = None):
    """Browse documents in a collection with optional filters"""
    db = get_db()
    try:
        if filters and filters.get('field') and filters.get('value'):
            aql = f"""
            FOR doc IN {collection_name}
                FILTER doc.{filters['field']} == @value
                LIMIT @limit
                RETURN doc
            """
            bind_vars = {"value": filters['value'], "limit": limit}
        else:
            aql = f"""
            FOR doc IN {collection_name}
                LIMIT @limit
                RETURN doc
            """
            bind_vars = {"limit": limit}

        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        results = list(cursor)
        return results
    except Exception as e:
        print(f"Browse error: {str(e)}")
        return []


def execute_aql(aql_query: str, bind_vars: dict = None):
    """Execute AQL query and return results"""
    db = get_db()
    try:
        cursor = db.aql.execute(
            aql_query,
            bind_vars=bind_vars or {},
            ttl=config.QUERY_TIMEOUT
        )
        results = list(cursor)
        return results, None
    except Exception as e:
        error_msg = str(e)
        print(f"❌ AQL Error: {error_msg}")
        print(f"Query: {aql_query}")
        return [], error_msg


def fix_aql_query(query: str):
    """Fix common LLM mistakes in AQL queries"""

    # Fatal error: COSINE_SIMILARITY on SEC content
    if 'COSINE_SIMILARITY' in query and ('sec_sentences' in query or 'sec_sections' in query):
        print("❌ ERROR: SEC content has NO embeddings!")
        return None

    # Fatal errors: .content field
    if 'doc.content' in query or 'filing.content' in query:
        print("❌ ERROR: Query uses .content field which doesn't exist!")
        return None

    # INTO keyword doesn't exist
    if ' INTO ' in query.upper():
        print("❌ ERROR: INTO keyword not supported in AQL")
        return None

    # Collection name fixes
    replacements = {
        'SEC_Filings': 'sec_filings',
        'SEC_Sections': 'sec_sections',
        'SEC_Sentences': 'sec_sentences',
        'awards': 'Award',
        'Awards': 'Award',
        'companies': 'Company',
        'Companies': 'Company',
        'market_data': 'MarketData',
        'fred_data': 'EconomicData',
        'FREDData': 'EconomicData',
        'shares_outstanding': 'sharesOutstanding',
        'market_cap': 'marketCap',
        'employees': 'fullTimeEmployees',
        'full_time_employees': 'fullTimeEmployees',
    }

    fixed = query
    for wrong, correct in replacements.items():
        fixed = fixed.replace(wrong, correct)

    if fixed != query:
        print("🔧 Auto-corrected collection names")

    return fixed
