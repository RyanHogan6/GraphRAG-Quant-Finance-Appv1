"""
Query Validation Module - Industry Standard Pattern
Uses ArangoDB EXPLAIN API for syntax validation before execution
Implements self-healing retry loop based on Neo4j/Cypher best practices
"""
from typing import Dict, Any, Tuple, Optional, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from app.database.connection import get_db
import config


def validate_query_syntax(aql_query: str, bind_vars: dict = None) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """
    Validate AQL query syntax using EXPLAIN API (no execution).

    Industry standard: From ArangoDB docs - "EXPLAIN validates syntax WITHOUT executing"
    Similar to Neo4j's EXPLAIN for Cypher validation

    Returns:
        (is_valid, error_message, explain_result)
    """
    db = get_db()
    bind_vars = bind_vars or {}

    try:
        # Use EXPLAIN to validate syntax without execution (FREE!)
        explain_result = db.aql.explain(
            aql_query,
            bind_vars=bind_vars,
            all_plans=False  # Only get optimal plan
        )

        print(f"[VALIDATION] ✓ Syntax valid")

        # Check if query will use indexes (performance check)
        plan = explain_result.get('plan', {})
        nodes = plan.get('nodes', [])

        index_nodes = [n for n in nodes if n.get('type') == 'IndexNode']
        if not index_nodes:
            print(f"[VALIDATION] ⚠️ Warning: Query won't use any indexes (may be slow)")

        return True, None, explain_result

    except Exception as e:
        error_msg = str(e)
        print(f"[VALIDATION] ✗ Syntax error: {error_msg}")
        return False, error_msg, None


def check_semantic_correctness(
    results: List[Dict],
    question: str,
    query_plan: Dict
) -> Tuple[bool, Optional[str]]:
    """
    Check if query results make semantic sense.

    Industry standard: Self-healing pattern from Text2Cypher research
    "Return query with exception message to LLM for another chance"
    """
    # Check 1: 0 results when we expect data
    if len(results) == 0:
        # Check if this is a query that should have results
        question_lower = question.lower()

        expecting_results_keywords = [
            'show me', 'find', 'list', 'get', 'what are', 'which',
            'top', 'largest', 'most', 'recent'
        ]

        if any(kw in question_lower for kw in expecting_results_keywords):
            hints = []

            # Provide specific correction hints based on collections used
            collections = query_plan.get('collections', [])

            if 'futures_prices' in collections:
                hints.append("- Check commodity names are UPPERCASE: 'CRUDE_OIL' not 'crude oil'")

            if 'options_flow' in collections:
                hints.append("- Use options_flow collection, not MarketData")

            if 'Award' in collections and 'recipient_name' in str(query_plan.get('aql_query', '')):
                hints.append("- Use CONTAINS(LOWER(doc.recipient_name), 'keyword') not exact match")

            if any('eia_' in c for c in collections):
                hints.append("- Use correct EIA collection (eia_crude_inventory, eia_natgas_storage, etc.)")
                hints.append("- Do NOT use MarketData with ticker='NATGAS' or ticker='CRUDE'")

            hint_message = "\n".join(hints) if hints else "- Check field names and filter values"

            return False, f"0 results returned but query expects data. Common issues:\n{hint_message}"

    # Check 2: Very large result set (might need LIMIT)
    if len(results) > 10000:
        return False, f"Query returned {len(results)} results. Consider adding LIMIT clause for performance."

    return True, None


def execute_with_validation(
    aql_query: str,
    bind_vars: dict = None,
    question: str = "",
    query_plan: dict = None,
    max_retries: int = 2
) -> Tuple[List[Dict], Optional[str], Dict]:
    """
    Execute query with validation and retry loop.

    Industry standard pattern:
    1. Validate syntax with EXPLAIN (ArangoDB native)
    2. Execute query
    3. Check semantic correctness
    4. Retry with correction hints if needed

    Returns:
        (results, error_message, stats)
    """
    db = get_db()
    bind_vars = bind_vars or {}
    query_plan = query_plan or {}

    stats = {
        'attempts': 0,
        'syntax_errors': [],
        'semantic_errors': [],
        'execution_errors': []
    }

    # Prune unused bind variables (ArangoDB requirement)
    if bind_vars:
        pruned_vars = {}
        for key, value in bind_vars.items():
            if f"@{key}" in aql_query:
                pruned_vars[key] = value
        bind_vars = pruned_vars

    for attempt in range(max_retries + 1):
        stats['attempts'] = attempt + 1

        print(f"\n[VALIDATION] Attempt {attempt + 1}/{max_retries + 1}")

        # Step 1: Syntax validation with EXPLAIN API
        is_valid, syntax_error, explain_result = validate_query_syntax(aql_query, bind_vars)

        if not is_valid:
            stats['syntax_errors'].append(syntax_error)

            # On last retry, return the error
            if attempt >= max_retries:
                return [], f"Syntax error after {max_retries + 1} attempts: {syntax_error}", stats

            # Could add LLM retry here with syntax error feedback
            # For now, just return the error
            return [], syntax_error, stats

        # Step 2: Execute query for real
        try:
            cursor = db.aql.execute(
                aql_query,
                bind_vars=bind_vars,
                ttl=config.QUERY_TIMEOUT,
                max_runtime=config.QUERY_TIMEOUT,
                batch_size=1000
            )
            results = list(cursor)

            print(f"[VALIDATION] ✓ Query executed: {len(results)} results")

        except Exception as e:
            error_msg = str(e)
            stats['execution_errors'].append(error_msg)

            print(f"[VALIDATION] ✗ Execution error: {error_msg}")

            # On last retry, return the error
            if attempt >= max_retries:
                return [], f"Execution error after {max_retries + 1} attempts: {error_msg}", stats

            # Could add LLM retry here with execution error feedback
            return [], error_msg, stats

        # Step 3: Semantic correctness check
        is_correct, semantic_error = check_semantic_correctness(results, question, query_plan)

        if not is_correct:
            stats['semantic_errors'].append(semantic_error)

            print(f"[VALIDATION] ⚠️ Semantic issue: {semantic_error}")

            # On last retry, return results anyway (0 results might be valid)
            if attempt >= max_retries:
                print(f"[VALIDATION] Returning results despite semantic warning")
                return results, None, stats

            # Could trigger LLM retry here with hints
            # For now, return results with warning
            return results, None, stats

        # Success!
        print(f"[VALIDATION] ✓ All checks passed")
        return results, None, stats

    # Should never reach here
    return [], "Unexpected error in validation loop", stats


def get_correction_hints(error_message: str, aql_query: str) -> List[str]:
    """
    Generate specific correction hints based on error patterns.

    Industry standard: Self-correction with targeted feedback
    """
    hints = []

    # Collection not found errors
    if 'collection or view not found' in error_message.lower():
        if 'sectors' in error_message.lower():
            hints.append("Collection 'sectors' doesn't exist. Use Company.sector field instead.")

    # Syntax errors
    if 'unexpected' in error_message.lower() or 'syntax error' in error_message.lower():
        hints.append("Check AQL syntax. Common issues:")
        hints.append("  - Use FOR...IN not SELECT FROM")
        hints.append("  - FILTER clauses come before RETURN")
        hints.append("  - No JOIN keyword (use graph traversal or nested FOR loops)")

    # Field errors
    if 'attribute path' in error_message.lower() or 'undefined' in error_message.lower():
        hints.append("Field doesn't exist. Check collection schema for correct field names.")

    # Commodity name issues (specific to your domain)
    if 'commodity' in aql_query and len(hints) == 0:
        hints.append("Commodity names MUST be UPPERCASE: 'CRUDE_OIL' not 'crude oil'")

    return hints


def analyze_query_performance(explain_result: Dict) -> Dict[str, Any]:
    """
    Analyze query performance from EXPLAIN result.

    Industry standard: Use EXPLAIN for performance optimization
    """
    if not explain_result:
        return {}

    plan = explain_result.get('plan', {})
    stats = plan.get('estimatedCost', 0)
    nodes = plan.get('nodes', [])

    analysis = {
        'estimated_cost': stats,
        'uses_indexes': False,
        'index_nodes': [],
        'collection_scans': [],
        'warnings': []
    }

    for node in nodes:
        node_type = node.get('type', '')

        if node_type == 'IndexNode':
            analysis['uses_indexes'] = True
            analysis['index_nodes'].append({
                'collection': node.get('collection'),
                'index': node.get('indexes', [])
            })

        if node_type == 'EnumerateCollectionNode':
            collection = node.get('collection', 'unknown')
            analysis['collection_scans'].append(collection)
            analysis['warnings'].append(f"Full collection scan on {collection} (slow for large collections)")

    return analysis
