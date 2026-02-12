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
        # For ArangoDB Cloud, we typically don't need custom cert verification
        # The python-arango library handles SSL automatically for HTTPS URLs

        # Use a generous request_timeout for all connections to avoid "Read timed out (60s)" on slow queries or remote hosts
        request_timeout = config.ARANGO_REQUEST_TIMEOUT
        client = ArangoClient(
            hosts=config.ARANGO_URL,
            request_timeout=request_timeout
        )
        if 'arangodb.cloud' in config.ARANGO_URL or 'oasis' in config.ARANGO_URL:
            print(f"✓ Connecting to ArangoDB Cloud with SSL ({request_timeout}s timeout)")
        else:
            print(f"✓ Connecting to ArangoDB ({request_timeout}s timeout)")

        _db_instance = client.db(
            config.DB_NAME,
            username=config.USERNAME,
            password=config.PASSWORD
        )
        print(f"✓ Connected to ArangoDB: {config.DB_NAME}")
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


# All 21 document collections (matches introspect_schema / About)
DOCUMENT_COLLECTIONS = [
    "Company",
    "MarketData",
    "Award",
    "EconomicData",
    "commodity_positions",
    "futures_prices",
    "options_flow",
    "eia_crude_inventory",
    "eia_natgas_storage",
    "eia_natgas_production",
    "eia_lng_exports",
    "sec_filings",
    "sec_sections",
    "sec_sentences",
    "sec_exhibits",
    "sec_xbrl_data",
    "prediction_markets_polymarket",
    "prediction_markets_kalshi",
    "polymarket_traders",
    "polymarket_positions",
    "polymarket_price_history",
]


def get_collections_info():
    """Get all collections with their document counts"""
    stats = []
    for col in DOCUMENT_COLLECTIONS:
        stats.append(get_collection_stats(col))
    return stats


def browse_collection(collection_name: str, limit: int = 50, search: str = None, filters: dict = None, offset: int = 0):
    """Browse documents in a collection with optional filters and search"""
    db = get_db()
    try:
        # Detect date field for sorting
        date_fields = ['date', 'start_date', 'end_date', 'timestamp', 'filing_date', 'as_of_date']

        # Build base query
        filter_lines = []
        bind_vars = {"limit": limit}

        # Add search filter (searches across all fields)
        if search:
            filter_lines.append("FILTER CONTAINS(LOWER(TO_STRING(doc)), LOWER(@search))")
            bind_vars["search"] = search

        # Add specific field filters
        if filters and filters.get('field') and filters.get('value'):
            filter_lines.append(f"FILTER doc.{filters['field']} == @value")
            bind_vars["value"] = filters['value']

        # Try to sort by date field (most recent first)
        sort_clause = ""
        for date_field in date_fields:
            # We'll try each date field and if it exists, use it
            sort_clause = f"SORT doc.{date_field} DESC"
            # We can't easily check if field exists without a query, so we'll try the first common one
            # MarketData -> date, Award -> start_date, etc.
            if collection_name == "MarketData" or collection_name == "EconomicData":
                sort_clause = "SORT doc.date DESC"
                break
            elif collection_name == "Award":
                sort_clause = "SORT doc.start_date DESC"
                break
            elif collection_name == "sec_filings":
                sort_clause = "SORT doc.filing_date DESC"
                break
            elif collection_name == "commodity_positions":
                sort_clause = "SORT doc.as_of_date DESC"
                break
            elif collection_name == "prediction_markets_polymarket" or collection_name == "prediction_markets_kalshi":
                sort_clause = "SORT doc.end_date DESC"
                break
            elif collection_name in ("futures_prices", "options_flow", "eia_crude_inventory", "eia_natgas_storage",
                                     "eia_natgas_production", "eia_lng_exports", "polymarket_price_history"):
                sort_clause = "SORT doc.date DESC"
                break
            elif collection_name in ("sec_exhibits", "sec_xbrl_data"):
                sort_clause = "SORT doc.filing_date DESC"
                break
            elif collection_name in ("polymarket_positions", "polymarket_traders"):
                sort_clause = "SORT doc.date DESC"
                break

        if not sort_clause:
            sort_clause = "SORT doc.date DESC"

        filter_clause = "\n".join(filter_lines)

        bind_vars["limit"] = limit
        bind_vars["offset"] = offset
        
        aql = f"""
        FOR doc IN {collection_name}
            {filter_clause}
            {sort_clause}
            LIMIT @offset, @limit
            RETURN doc
        """

        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        results = list(cursor)
        return results
    except Exception as e:
        print(f"Browse error: {str(e)}")
        return []


def execute_aql(aql_query: str, bind_vars: dict = None):
    """Execute AQL query and return results"""
    db = get_db()
    
    # Ensure bind_vars is a dict
    bind_vars = bind_vars or {}
    
    # ARANGODB SAFETY: Remove bind parameters that aren't actually in the query
    # ArangoDB throws a 400 error if you provide a parameter but don't use it in the AQL
    if bind_vars:
        pruned_vars = {}
        for key, value in bind_vars.items():
            # Check if @key is in the query string
            parameter_string = f"@{key}"
            if parameter_string in aql_query:
                pruned_vars[key] = value
        bind_vars = pruned_vars

    try:
        cursor = db.aql.execute(
            aql_query,
            bind_vars=bind_vars,
            ttl=config.QUERY_TIMEOUT,
            max_runtime=config.QUERY_TIMEOUT,  # Also set max runtime
            batch_size=1000  # Stream results in batches for large datasets
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

    # Reject SQL-style INTO (INSERT INTO / SELECT INTO); allow AQL COLLECT ... INTO
    q = query.upper()
    if 'INSERT INTO ' in q or ('SELECT ' in q and ' INTO ' in q and 'COLLECT' not in q):
        print("❌ ERROR: SQL-style INTO not supported in AQL")
        return None

    # Fix incorrect clause order: RETURN ... LIMIT -> LIMIT ... RETURN
    import re
    # Pattern: RETURN <something> LIMIT <number>
    return_limit_pattern = r'(RETURN\s+[^\n]+?)\s+(LIMIT\s+\d+)'
    if re.search(return_limit_pattern, query, re.IGNORECASE):
        # Swap RETURN and LIMIT
        fixed_query = re.sub(
            return_limit_pattern,
            r'\2 \1',
            query,
            flags=re.IGNORECASE
        )
        print("🔧 Auto-corrected: Moved LIMIT before RETURN")
        query = fixed_query

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
