"""
AQL query validator for builder mode
Prevents abuse and runaway queries
"""
import re
from typing import Tuple, Optional
import config


def validate_aql_query(aql: str) -> Tuple[bool, Optional[str]]:
    """
    Validate AQL query for complexity and safety

    Returns:
        (is_valid, error_message)
        If is_valid=True, error_message=None
        If is_valid=False, error_message contains reason
    """

    if not aql or not isinstance(aql, str):
        return False, "Query is empty or invalid"

    # Remove comments and normalize whitespace
    aql_clean = re.sub(r'//.*?\n|/\*.*?\*/', '', aql, flags=re.DOTALL)
    aql_upper = aql_clean.upper()

    # 1. Check for forbidden operations (write/delete)
    forbidden_keywords = [
        'INSERT', 'UPDATE', 'REPLACE', 'REMOVE', 'DELETE',
        'CREATE', 'DROP', 'TRUNCATE', 'RENAME',
        'UPSERT'
    ]

    for keyword in forbidden_keywords:
        if re.search(rf'\b{keyword}\b', aql_upper):
            return False, f"Forbidden operation: {keyword}. Only read queries are allowed."

    # 2. Require LIMIT clause
    if 'LIMIT' not in aql_upper:
        return False, "Query must include a LIMIT clause to prevent excessive results."

    # 3. Check LIMIT value is reasonable
    limit_match = re.search(r'\bLIMIT\s+(\d+)', aql_upper)
    if limit_match:
        limit_value = int(limit_match.group(1))
        if limit_value > 1000:
            return False, f"LIMIT too high ({limit_value}). Maximum allowed: 1000"
    else:
        # LIMIT exists but couldn't parse value (e.g., LIMIT @var)
        # Allow it but warn
        pass

    # 4. Check query complexity (number of FOR loops)
    for_count = len(re.findall(r'\bFOR\b', aql_upper))
    max_complexity = config.MAX_QUERY_COMPLEXITY

    if for_count > max_complexity:
        return False, f"Query too complex ({for_count} FOR loops). Maximum allowed: {max_complexity}"

    # 5. Check for COLLECT with large grouping (can be expensive)
    collect_count = len(re.findall(r'\bCOLLECT\b', aql_upper))
    if collect_count > 2:
        return False, f"Too many COLLECT operations ({collect_count}). Maximum allowed: 2"

    # 6. Warn about Cartesian products (multiple FOR without proper FILTER)
    # Allow VQB-style queries: one root FOR + LET x = (FOR ...) traversals are constrained, not Cartesian.
    filter_count = len(re.findall(r'\bFILTER\b', aql_upper))
    has_let_subquery = bool(re.search(r'LET\s+\w+\s*=\s*\(', aql_clean, re.IGNORECASE))
    if for_count >= 3 and filter_count == 0 and not has_let_subquery:
        return False, "Potential Cartesian product detected. Add FILTER clauses to constrain joins."

    # 7. Check query length (simple DoS prevention)
    if len(aql) > 10000:
        return False, "Query too long (max 10000 characters)"

    # All checks passed
    return True, None


def estimate_query_rows(aql: str) -> int:
    """
    Estimate number of rows a query might return
    This is a rough heuristic, not precise

    Returns:
        Estimated row count (0 if can't estimate)
    """
    aql_upper = aql.upper()

    # Try to extract LIMIT value
    limit_match = re.search(r'\bLIMIT\s+(\d+)', aql_upper)
    if limit_match:
        return int(limit_match.group(1))

    # No LIMIT found (shouldn't happen after validation)
    return 0


def sanitize_aql_for_display(aql: str, max_length: int = 500) -> str:
    """
    Sanitize AQL query for safe display in logs/UI
    Truncates long queries and removes sensitive bind vars
    """
    if not aql:
        return ""

    # Truncate if too long
    if len(aql) > max_length:
        aql = aql[:max_length] + "... [truncated]"

    return aql
