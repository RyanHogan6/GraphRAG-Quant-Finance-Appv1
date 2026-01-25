"""
User query logging for analytics and improvement
Stores all user queries with metadata to ArangoDB
"""
import hashlib
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from app.database.connection import get_db

# Collection name for user query logs
USER_QUERIES_COLLECTION = "user_queries"


def hash_ip(ip_address: str) -> str:
    """Hash IP address for privacy (GDPR compliant)"""
    return hashlib.sha256(f"{ip_address}-karga-salt".encode()).hexdigest()[:16]


def log_user_query(
    question: str,
    response_time: float,
    result_count: int,
    was_successful: bool,
    ip_address: str,
    query_intent: str = "unknown",
    error: Optional[str] = None,
    aql_query: Optional[str] = None,
    collections_used: Optional[List[str]] = None,
    api_cost: float = 0.0,
    from_cache: bool = False
) -> bool:
    """
    Log user query to ArangoDB for analytics

    Args:
        question: Natural language question asked
        response_time: Time taken to respond (seconds)
        result_count: Number of results returned
        was_successful: Whether query succeeded
        ip_address: User IP (will be hashed)
        query_intent: db_only, web_only, hybrid
        error: Error message if failed
        aql_query: Generated AQL query
        collections_used: List of collections queried
        api_cost: Estimated API cost ($)
        from_cache: Whether result came from cache

    Returns:
        True if logged successfully, False otherwise
    """
    try:
        db = get_db()

        # Ensure collection exists
        if not db.has_collection(USER_QUERIES_COLLECTION):
            print(f"[ANALYTICS] Creating {USER_QUERIES_COLLECTION} collection...")
            db.create_collection(USER_QUERIES_COLLECTION)
            print(f"[ANALYTICS] Collection created")

        collection = db.collection(USER_QUERIES_COLLECTION)

        # Create query log document
        doc = {
            "question": question[:500],  # Truncate long questions
            "timestamp": datetime.utcnow().isoformat(),
            "response_time": round(response_time, 3),
            "result_count": result_count,
            "was_successful": was_successful,
            "ip_hash": hash_ip(ip_address),
            "query_intent": query_intent,
            "error": error[:200] if error else None,
            "aql_query": aql_query[:1000] if aql_query else None,
            "collections_used": collections_used or [],
            "api_cost": round(api_cost, 4),
            "from_cache": from_cache,
        }

        collection.insert(doc)
        return True

    except Exception as e:
        print(f"[ANALYTICS ERROR] Failed to log query: {e}")
        # Don't fail the request if logging fails
        return False


def get_query_stats(hours: int = 24) -> Dict[str, Any]:
    """
    Get query statistics for the last N hours

    Returns:
        Dictionary with stats: total queries, success rate, avg response time, etc.
    """
    try:
        db = get_db()

        if not db.has_collection(USER_QUERIES_COLLECTION):
            return {"error": "No query logs found"}

        # Calculate timestamp threshold
        from datetime import timedelta
        threshold = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

        # Query for stats
        aql = """
        LET recent = (
            FOR q IN @@collection
                FILTER q.timestamp >= @threshold
                RETURN q
        )

        RETURN {
            total_queries: LENGTH(recent),
            successful_queries: LENGTH(FOR q IN recent FILTER q.was_successful RETURN 1),
            failed_queries: LENGTH(FOR q IN recent FILTER !q.was_successful RETURN 1),
            avg_response_time: AVG(recent[*].response_time),
            total_api_cost: SUM(recent[*].api_cost),
            cache_hit_rate: LENGTH(FOR q IN recent FILTER q.from_cache RETURN 1) / LENGTH(recent),
            top_collections: (
                FOR q IN recent
                    FOR coll IN q.collections_used
                        COLLECT collection = coll WITH COUNT INTO count
                        SORT count DESC
                        LIMIT 10
                        RETURN {collection, count}
            ),
            query_intents: (
                FOR q IN recent
                    COLLECT intent = q.query_intent WITH COUNT INTO count
                    RETURN {intent, count}
            )
        }
        """

        cursor = db.aql.execute(
            aql,
            bind_vars={
                "@collection": USER_QUERIES_COLLECTION,
                "threshold": threshold
            }
        )

        result = next(cursor)
        return {
            "time_window_hours": hours,
            "stats": result
        }

    except Exception as e:
        print(f"[ANALYTICS ERROR] Failed to get stats: {e}")
        return {"error": str(e)}


def get_daily_spend() -> float:
    """
    Get total API spend for today

    Returns:
        Total spend in dollars
    """
    try:
        db = get_db()

        if not db.has_collection(USER_QUERIES_COLLECTION):
            return 0.0

        # Get today's date at midnight UTC
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

        aql = """
        FOR q IN @@collection
            FILTER q.timestamp >= @today
            RETURN q.api_cost
        """

        cursor = db.aql.execute(
            aql,
            bind_vars={
                "@collection": USER_QUERIES_COLLECTION,
                "today": today
            }
        )

        costs = list(cursor)
        return sum(costs)

    except Exception as e:
        print(f"[ANALYTICS ERROR] Failed to get daily spend: {e}")
        return 0.0
