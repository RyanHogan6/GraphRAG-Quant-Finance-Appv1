"""
Markets API routes - Polymarket and Kalshi prediction markets
"""
from fastapi import APIRouter, HTTPException, Query as QueryParam, Request
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.database.connection import get_db, execute_aql

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


class MarketResponse(BaseModel):
    question: str
    yes_prob: float
    volume_24h: float
    liquidity: Optional[float] = None
    category: Optional[str] = None
    end_date: Optional[str] = None


class WhaleTraderResponse(BaseModel):
    address: str
    volume: float
    profit: float
    trades: int
    activity: str
    profit_ratio: float


@router.get("/polymarket/sample")
def get_sample_market():
    """Get a sample market to see available fields - DEBUG ENDPOINT"""
    db = get_db()

    query = """
    FOR market IN prediction_markets_polymarket
        FILTER market.closed == false
        LIMIT 1
        RETURN market
    """

    try:
        results, error = execute_aql(query)
        if error or not results:
            return {"error": "No markets found"}
        return results[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/polymarket/categories")
@limiter.limit("30/minute")
def get_polymarket_categories(request: Request):
    """Get all Polymarket categories with counts"""
    db = get_db()

    query = """
    FOR m IN prediction_markets_polymarket
        FILTER m.closed == false AND m.category != null
        COLLECT category = m.category WITH COUNT INTO count
        SORT count DESC
        RETURN {category: category, count: count}
    """

    try:
        results, error = execute_aql(query)
        if error:
            raise HTTPException(status_code=500, detail=error)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/polymarket/market/{market_id}")
def get_market_detail(market_id: str):
    """Get detailed information for a specific market including trader count"""
    db = get_db()

    # Query with correct field names from polymarket_positions
    query = f"""
    FOR market IN prediction_markets_polymarket
        FILTER market._key == @market_id
        LIMIT 1

        // Count unique traders with positions in this market
        LET trader_count = LENGTH(
            FOR pos IN polymarket_positions
                FILTER pos.market_condition_id == market.condition_id OR
                       pos.market_key == market._key OR
                       CONTAINS(pos.market_condition_id, market.condition_id)
                RETURN DISTINCT pos.trader_address
        )

        RETURN MERGE(market, {{
            trader_count: trader_count > 0 ? trader_count : (
                market.num_traders != null ? market.num_traders : 0
            ),
            num_traders: trader_count > 0 ? trader_count : (
                market.num_traders != null ? market.num_traders : 0
            )
        }})
    """

    try:
        results, error = execute_aql(query, {"market_id": market_id})
        if error:
            # Fallback: return market without trader count
            fallback_query = """
            FOR market IN prediction_markets_polymarket
                FILTER market._key == @market_id
                LIMIT 1
                RETURN MERGE(market, {
                    trader_count: market.num_traders != null ? market.num_traders : 0,
                    num_traders: market.num_traders != null ? market.num_traders : 0
                })
            """
            results, error2 = execute_aql(fallback_query, {"market_id": market_id})
            if error2 or not results:
                raise HTTPException(status_code=404, detail="Market not found")

        if not results:
            raise HTTPException(status_code=404, detail="Market not found")

        return results[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/polymarket/featured")
@limiter.limit("30/minute")
def get_featured_markets(
    request: Request,
    limit: int = QueryParam(100)
):
    """
    Get high-quality featured markets - OPTIMIZED FOR SPEED v2.0

    Simplified query to avoid timeouts:
    - Direct filter and sort (no complex aggregations)
    - Min quality thresholds
    - Fast indexes on volume_24h and liquidity
    """

    print(f"🚀 OPTIMIZED QUERY v2.0 - Fetching {limit} markets (NEW CODE RUNNING)")

    query = f"""
    FOR market IN prediction_markets_polymarket
        FILTER market.closed == false
        FILTER market.volume_24h > 100
        FILTER market.liquidity > 50

        // Extract probabilities (simple, no nested logic)
        LET yes_prob_value = market.outcome_prices[0] != null ? market.outcome_prices[0] :
                             (market.yes_probability != null ? market.yes_probability : 0.5)
        LET yes_prob_pct = yes_prob_value <= 1 ? (yes_prob_value * 100) : yes_prob_value

        // Filter out resolved markets
        FILTER yes_prob_pct > 2 AND yes_prob_pct < 98

        // Sort by volume (most liquid markets first)
        SORT market.volume_24h DESC
        LIMIT {limit}

        // Calculate complementary values
        LET no_prob_value = market.outcome_prices[1] != null ? market.outcome_prices[1] : (1 - yes_prob_value)
        LET no_prob_pct = no_prob_value <= 1 ? (no_prob_value * 100) : no_prob_value

        // Get outcomes (simplified)
        LET outcomes_array = IS_ARRAY(market.outcomes) ? market.outcomes : []
        LET outcome_yes = outcomes_array[0] != null ? outcomes_array[0] : "Yes"
        LET outcome_no = outcomes_array[1] != null ? outcomes_array[1] : "No"

        RETURN {{
            id: market._key,
            question: market.question,
            yes_prob: ROUND(yes_prob_pct),
            no_prob: ROUND(no_prob_pct),
            outcome_yes: outcome_yes,
            outcome_no: outcome_no,
            volume_24h: market.volume_24h,
            liquidity: market.liquidity,
            category: market.category != null ? market.category : "Other",
            end_date: market.end_date,
            traders: 0
        }}
    """

    try:
        results, error = execute_aql(query)
        if error:
            raise HTTPException(status_code=500, detail=error)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/polymarket/markets")
def get_polymarket_markets(
    category: Optional[str] = QueryParam(None),
    min_volume: float = QueryParam(0),
    sort_by: str = QueryParam("volume_desc"),
    limit: int = QueryParam(20)
):
    """Get Polymarket markets with filtering (legacy endpoint - use /featured for main page)"""

    # Build filters
    category_filter = f"FILTER market.category == '{category}'" if category else ""
    volume_filter = f"FILTER market.volume_24h >= {min_volume}" if min_volume > 0 else ""

    # Sort mapping
    sort_mapping = {
        "volume_desc": ("market.volume_24h", "DESC"),
        "volume_asc": ("market.volume_24h", "ASC"),
        "probability_desc": ("market.yes_probability", "DESC"),
        "probability_asc": ("market.yes_probability", "ASC"),
    }
    sort_field, sort_dir = sort_mapping.get(sort_by, ("market.volume_24h", "DESC"))

    query = f"""
    FOR market IN prediction_markets_polymarket
        FILTER market.closed == false
        FILTER market.volume_24h > 0
        {category_filter}
        {volume_filter}

        // Extract probabilities from outcome_prices array
        LET yes_prob_value = (
            LENGTH(market.outcome_prices) >= 1 ? market.outcome_prices[0] :
            market.yes_probability != null ? market.yes_probability :
            market.yes_price != null ? market.yes_price :
            0.5
        )

        LET no_prob_value = (
            LENGTH(market.outcome_prices) >= 2 ? market.outcome_prices[1] :
            market.no_probability != null ? market.no_probability :
            (1 - yes_prob_value)
        )

        // Convert to percentage if it's a decimal (0-1 range)
        LET yes_prob_pct = yes_prob_value <= 1 ? (yes_prob_value * 100) : yes_prob_value
        LET no_prob_pct = no_prob_value <= 1 ? (no_prob_value * 100) : no_prob_value

        // Filter out resolved markets and markets with no real prices
        FILTER yes_prob_pct > 1 AND yes_prob_pct < 99
        FILTER market.liquidity > 0

        SORT {sort_field} {sort_dir}
        LIMIT {limit}

        // Count unique traders via graph traversal: market <- position <- trader
        LET trader_count = LENGTH(
            FOR position IN INBOUND market position_in_market
                FOR trader IN INBOUND position trader_has_position
                    RETURN DISTINCT trader._key
        )

        // Parse outcomes if it's a JSON string (use JSON_PARSE in ArangoDB)
        LET outcomes_array = (
            IS_STRING(market.outcomes) ? JSON_PARSE(market.outcomes) :
            IS_ARRAY(market.outcomes) ? market.outcomes :
            []
        )

        // Parse outcome_prices if it's a JSON string
        LET outcome_prices_array = (
            IS_STRING(market.outcome_prices) ? JSON_PARSE(market.outcome_prices) :
            IS_ARRAY(market.outcome_prices) ? market.outcome_prices :
            []
        )

        // Get outcome names if available (for sports, multiple choice)
        LET outcome_yes = LENGTH(outcomes_array) >= 1 ? outcomes_array[0] : "Yes"
        LET outcome_no = LENGTH(outcomes_array) >= 2 ? outcomes_array[1] : "No"

        RETURN {{
            id: market._key,
            market_id: market.market_id,
            condition_id: market.condition_id,
            question: market.question,
            yes_prob: ROUND(yes_prob_pct),
            no_prob: ROUND(no_prob_pct),
            outcome_yes: outcome_yes,
            outcome_no: outcome_no,
            volume_24h: market.volume_24h,
            liquidity: market.liquidity,
            category: market.category,
            end_date: market.end_date,
            description: market.description,
            traders: trader_count,
            outcomes: outcomes_array,
            outcome_prices: outcome_prices_array
        }}
    """

    try:
        results, error = execute_aql(query)
        if error:
            raise HTTPException(status_code=500, detail=error)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/polymarket/whales")
def get_whale_traders(
    limit: int = QueryParam(20)
):
    """Get top whale traders by volume"""
    db = get_db()

    query = f"""
    FOR trader IN polymarket_traders
        FILTER trader.is_whale == true
        SORT trader.total_volume DESC
        LIMIT {limit}
        RETURN {{
            address: trader.address,
            volume: trader.total_volume,
            profit: trader.total_profit,
            trades: trader.total_trades,
            activity: trader.activity_level != null ? trader.activity_level : "Unknown",
            profit_ratio: trader.profit_ratio,
            win_rate: trader.is_profitable * 100
        }}
    """

    try:
        results, error = execute_aql(query)
        if error:
            # If collection doesn't exist, return empty array
            return []
        return results
    except Exception as e:
        # If collection doesn't exist, return empty array instead of error
        return []


@router.get("/kalshi/markets")
def get_kalshi_markets(
    limit: int = QueryParam(20)
):
    """Get Kalshi markets"""
    db = get_db()

    query = f"""
    FOR market IN prediction_markets_kalshi
        FILTER market.status == "active"
        FILTER market.volume_24h > 0
        SORT market.volume_24h DESC
        LIMIT {limit}
        RETURN {{
            id: market._key,
            question: market.title,
            yes_prob: FLOOR(market.yes_price * 100),
            no_prob: FLOOR((1 - market.yes_price) * 100),
            volume_24h: market.volume_24h,
            category: market.category,
            close_time: market.close_time
        }}
    """

    try:
        results, error = execute_aql(query)
        if error:
            raise HTTPException(status_code=500, detail=error)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
