"""
Markets API routes - Polymarket and Kalshi prediction markets
"""
from fastapi import APIRouter, HTTPException, Query as QueryParam
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from app.database.connection import get_db, execute_aql

router = APIRouter()


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
def get_polymarket_categories():
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

    # First check if polymarket_positions collection exists
    query = f"""
    FOR market IN prediction_markets_polymarket
        FILTER market._key == @market_id
        LIMIT 1

        // Try to count traders - fallback to stored value if collection doesn't exist
        LET trader_count = (
            LENGTH(
                FOR pos IN polymarket_positions
                    FILTER pos.market_id == market.market_id OR pos.condition_id == market.condition_id
                    RETURN DISTINCT pos.trader_key
            ) > 0 ? LENGTH(
                FOR pos IN polymarket_positions
                    FILTER pos.market_id == market.market_id OR pos.condition_id == market.condition_id
                    RETURN DISTINCT pos.trader_key
            ) : (
                market.num_traders != null ? market.num_traders :
                market.trader_count != null ? market.trader_count :
                market.traders != null ? market.traders : 0
            )
        )

        RETURN MERGE(market, {{
            trader_count: trader_count,
            num_traders: trader_count
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


@router.get("/polymarket/markets")
def get_polymarket_markets(
    category: Optional[str] = QueryParam(None),
    min_volume: float = QueryParam(0),
    sort_by: str = QueryParam("volume_desc"),
    limit: int = QueryParam(20)
):
    """Get Polymarket markets with filtering"""

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

        // Handle different probability field names and formats
        LET yes_prob_value = (
            market.yes_probability != null ? market.yes_probability :
            market.yes_price != null ? market.yes_price :
            market.probability != null ? market.probability :
            0.5
        )

        // Convert to percentage if it's a decimal (0-1 range)
        LET yes_prob_pct = yes_prob_value <= 1 ? (yes_prob_value * 100) : yes_prob_value

        // Filter out resolved markets (where probability is exactly 0 or 100)
        FILTER yes_prob_pct > 1 AND yes_prob_pct < 99

        SORT {sort_field} {sort_dir}
        LIMIT {limit}

        // Get trader count from any available field
        LET trader_count = (
            market.num_traders != null ? market.num_traders :
            market.trader_count != null ? market.trader_count :
            market.traders != null ? market.traders :
            market.unique_traders != null ? market.unique_traders : 0
        )

        RETURN {{
            id: market._key,
            market_id: market.market_id,
            condition_id: market.condition_id,
            question: market.question,
            yes_prob: ROUND(yes_prob_pct),
            no_prob: ROUND(100 - yes_prob_pct),
            volume_24h: market.volume_24h,
            liquidity: market.liquidity,
            category: market.category,
            end_date: market.end_date,
            description: market.description,
            traders: trader_count
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
