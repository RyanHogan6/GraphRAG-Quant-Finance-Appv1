"""
Alpha signals derived from graph data (AQL-only, no GNN).
- Contract momentum: 90d rolling sum of award amounts by company, normalized rank.
- Options-filing convergence: unusual options within N days of filing (OPTIONS_BEFORE_FILING).
- Centrality: degree centrality on contract graph (HAS_AWARD count per company).
"""
from typing import List, Dict, Any, Optional
from app.database.connection import get_db, execute_aql


def get_contract_momentum_90d(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Companies with highest 90-day contract award sum (contract momentum).
    Uses graph: Company -> HAS_AWARD -> Award, filter by start_date, sum award_amount_float.
    """
    aql = """
    FOR company IN Company
      LET awards_90d = (
        FOR award IN OUTBOUND company HAS_AWARD
          FILTER award.start_date >= DATE_SUBTRACT(DATE_NOW(), 90, 'day')
          FILTER award.award_amount_float != null AND award.award_amount_float > 0
          COLLECT AGGREGATE total = SUM(award.award_amount_float), count = LENGTH(1)
          RETURN { total: total, count: count }
      )
      LET total_90d = awards_90d[0].total
      FILTER total_90d != null AND total_90d > 0
      SORT total_90d DESC
      LIMIT @limit
      RETURN {
        ticker: company.ticker,
        company: company.company,
        sector: company.sector,
        contract_momentum_90d: total_90d,
        award_count_90d: awards_90d[0].count
      }
    """
    results, err = execute_aql(aql, {"limit": limit})
    if err:
        return []
    # Normalized rank (1 = highest momentum)
    for i, row in enumerate(results or [], 1):
        row["momentum_rank"] = i
    return results or []


def get_options_filing_convergence(days_before: int = 14, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Options activity within N days before a filing (OPTIONS_BEFORE_FILING).
    Returns tickers with weighted signal: unusual activity + proximity to filing.
    """
    aql = """
    FOR opt IN options_flow
      FILTER opt.date != null
      LET filings_before = (
        FOR filing IN OUTBOUND opt OPTIONS_BEFORE_FILING
          FILTER filing.filing_date != null
          LET days_before = DATEDIFF(opt.date, filing.filing_date, 'd')
          FILTER days_before >= 0 AND days_before <= @days_before
          RETURN { filing_date: filing.filing_date, days_before: days_before }
      )
      FILTER LENGTH(filings_before) > 0
      LET unusual = (opt.unusual_ratio != null && opt.unusual_ratio > 1.5) ? 1 : 0
      COLLECT ticker = opt.ticker AGGREGATE
        events = LENGTH(filings_before),
        total_unusual = SUM(unusual),
        max_unusual_ratio = MAX(opt.unusual_ratio || 0)
      FILTER events > 0
      SORT total_unusual DESC, max_unusual_ratio DESC
      LIMIT @limit
      RETURN {
        ticker: ticker,
        options_filing_events: events,
        unusual_activity_count: total_unusual,
        max_unusual_ratio: max_unusual_ratio
      }
    """
    results, err = execute_aql(aql, {"days_before": days_before, "limit": limit})
    if err:
        return []
    return results or []


def get_contract_centrality(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Degree centrality on contract graph: count of HAS_AWARD edges per company.
    Companies that are hubs in the defense supply chain have more contract links.
    """
    aql = """
    FOR company IN Company
      LET award_count = (
        FOR award IN OUTBOUND company HAS_AWARD
          COLLECT WITH COUNT INTO c
          RETURN c
      )
      LET degree = award_count[0]
      FILTER degree != null AND degree > 0
      SORT degree DESC
      LIMIT @limit
      RETURN {
        ticker: company.ticker,
        company: company.company,
        sector: company.sector,
        contract_degree_centrality: degree
      }
    """
    results, err = execute_aql(aql, {"limit": limit})
    if err:
        return []
    return results or []


def get_all_signals(contract_limit: int = 50, options_limit: int = 30, centrality_limit: int = 50) -> Dict[str, Any]:
    """Return all alpha signals in one payload for dashboard use."""
    return {
        "contract_momentum_90d": get_contract_momentum_90d(limit=contract_limit),
        "options_filing_convergence": get_options_filing_convergence(limit=options_limit),
        "contract_centrality": get_contract_centrality(limit=centrality_limit),
    }
