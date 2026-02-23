"""
Research API - Prediction Market Research (Market Workup)
GET /api/research/market/{market_id}?platform=kalshi|polymarket
Optional: GET /api/research/markets (list markets for research)
POST /api/research/backtest (prediction market backtester)
"""
from fastapi import APIRouter, HTTPException, Query, Body
from typing import Optional, List, Dict, Any

from app.market_workup import run_market_workup
from app.database.connection import execute_aql

router = APIRouter()


@router.get("/market/{market_id}")
def get_market_workup(
    market_id: str,
    platform: str = Query("kalshi", description="Platform: kalshi or polymarket"),
):
    """
    Full Market Workup: market doc, theme, macro, options, SEC sentiment,
    contracts (if gov spending), news (Perplexity), cross-source synthesis.
    """
    if platform.lower() not in ("kalshi", "polymarket"):
        raise HTTPException(status_code=400, detail="platform must be kalshi or polymarket")
    result = run_market_workup(market_id, platform)
    if result.get("error"):
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/markets")
def list_research_markets(
    platform: str = Query("kalshi", description="kalshi or polymarket"),
    category: Optional[str] = Query(None, description="Filter by category/theme"),
    limit: int = Query(50, ge=1, le=200),
):
    """
    List markets suitable for research (e.g. Kalshi macro: Fed, CPI, unemployment, GDP, recession).
    """
    if platform.lower() == "kalshi":
        aql = """
        FOR m IN prediction_markets_kalshi
          FILTER m.status == "active" OR m.status == null
          FILTER m.open_interest != null AND m.open_interest > 0
          SORT m.open_interest DESC
          LIMIT @limit
          RETURN {
            id: m._key,
            question: m.event_title,
            yes_probability: m.yes_probability,
            volume: m.volume,
            open_interest: m.open_interest,
            category: m.category,
            end_date: m.close_time,
            platform: "kalshi"
          }
        """
    else:
        aql = """
        FOR m IN prediction_markets_polymarket
          FILTER m.closed == false
          FILTER m.liquidity != null AND m.liquidity > 0
          SORT m.liquidity DESC
          LIMIT @limit
          RETURN {
            id: m._key,
            question: m.question,
            yes_probability: m.yes_probability,
            volume: m.volume_24h,
            open_interest: null,
            category: m.category,
            end_date: m.end_date,
            platform: "polymarket"
          }
        """
    results, err = execute_aql(aql, {"limit": limit})
    if err:
        raise HTTPException(status_code=500, detail=err)
    # Optional server-side category filter (e.g. "Politics", "Economics")
    if category and results:
        cat_lower = category.lower()
        results = [r for r in results if r.get("category") and cat_lower in (r.get("category") or "").lower()]
    return {"markets": results or [], "platform": platform}


@router.post("/backtest")
def post_backtest(
    payload: dict = Body(..., description="platform, resolution_date, lookback_days, market_id?, probability_series?, signals?"),
):
    """
    Run prediction-market backtest: probability series over lookback window +
    overlay macro/options/SEC/contracts; return lead/lag and correlations.
    """
    from app.backtest_markets import run_backtest
    platform = payload.get("platform", "polymarket")
    resolution_date = payload.get("resolution_date")
    if not resolution_date:
        raise HTTPException(status_code=400, detail="resolution_date (YYYY-MM-DD) required")
    lookback_days = payload.get("lookback_days", 30)
    market_id = payload.get("market_id")
    probability_series = payload.get("probability_series")
    signals = payload.get("signals")
    theme = payload.get("theme", "other")
    result = run_backtest(
        platform=platform,
        resolution_date=resolution_date,
        lookback_days=lookback_days,
        market_id=market_id,
        probability_series=probability_series,
        signals=signals,
        theme=theme,
    )
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/congressional/{ticker}")
def get_congressional_by_ticker(
    ticker: str,
    days: int = Query(90, ge=1, le=365),
):
    """Recent congressional stock trading disclosures for a given ticker (for Company Workup card)."""
    from app.database.connection import execute_aql
    from datetime import datetime, timedelta
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    aql = """
    FOR t IN congressional_trades
      FILTER t.ticker == @ticker
      FILTER t.date >= @cutoff
      SORT t.date DESC
      LIMIT 50
      RETURN t
    """
    results, err = execute_aql(aql, {"ticker": ticker.upper(), "cutoff": cutoff})
    if err:
        raise HTTPException(status_code=500, detail=err)
    return {"ticker": ticker.upper(), "trades": results or []}
