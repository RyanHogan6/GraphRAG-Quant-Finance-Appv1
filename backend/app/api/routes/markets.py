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
        SORT {sort_field} {sort_dir}
        LIMIT {limit}
        RETURN {{
            id: market._key,
            question: market.question,
            yes_prob: FLOOR(market.yes_probability * 100),
            no_prob: FLOOR((1 - market.yes_probability) * 100),
            volume_24h: market.volume_24h,
            liquidity: market.liquidity,
            category: market.category,
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
