"""
Database API routes - Collection browsing and statistics
"""
from fastapi import APIRouter, HTTPException, Query as QueryParam
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from app.database.connection import get_db, get_collections_info, browse_collection, execute_aql

router = APIRouter()


class CollectionStat(BaseModel):
    name: str
    count: int


@router.get("/collections", response_model=List[CollectionStat])
def get_collections():
    """Get all collections with document counts"""
    try:
        stats = get_collections_info()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/browse/{collection_name}")
def browse(
    collection_name: str,
    limit: int = QueryParam(100, le=500),
    search: Optional[str] = QueryParam(None),
    offset: int = QueryParam(0, ge=0)
):
    """Browse documents in a collection with optional search"""
    try:
        # Validate collection name
        allowed_collections = [
            "Company", "MarketData", "Award", "EconomicData",
            "commodity_positions", "prediction_markets_polymarket",
            "prediction_markets_kalshi", "sec_filings", "sec_sections", "sec_sentences"
        ]

        if collection_name not in allowed_collections:
            raise HTTPException(status_code=400, detail="Invalid collection name")

        results = browse_collection(collection_name, limit=limit, search=search, offset=offset)
        return {
            "collection": collection_name,
            "count": len(results),
            "documents": results
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stock/{ticker}/overview")
def get_stock_overview(ticker: str):
    """Get comprehensive stock overview"""
    db = get_db()

    try:
        # Get company info
        company_query = """
        FOR company IN Company
            FILTER company.ticker == @ticker
            LIMIT 1
            RETURN company
        """
        company_results, _ = execute_aql(company_query, {"ticker": ticker})
        company = company_results[0] if company_results else None

        # Get latest market data
        market_query = """
        FOR doc IN MarketData
            FILTER doc.ticker == @ticker
            SORT doc.date DESC
            LIMIT 1
            RETURN doc
        """
        market_results, _ = execute_aql(market_query, {"ticker": ticker})
        latest_market = market_results[0] if market_results else None

        # Get awards
        awards_query = """
        FOR doc IN Award
            FILTER doc.ticker == @ticker
            SORT doc.start_date DESC
            LIMIT 10
            RETURN doc
        """
        awards_results, _ = execute_aql(awards_query, {"ticker": ticker})

        # Get SEC filings with top sentences (sentiment analysis)
        sec_query = """
        FOR company IN Company
            FILTER company.ticker == @ticker
            FOR filing IN OUTBOUND company HAS_FILING
                SORT filing.filing_date DESC
                LIMIT 20
                LET top_sentences = (
                    FOR section IN OUTBOUND filing has_section
                        FOR sentence IN OUTBOUND section has_sentence
                            FILTER sentence.finbertscore != null
                            SORT ABS(sentence.finbertscore) DESC
                            LIMIT 5
                            RETURN {
                                text: sentence.text,
                                score: sentence.finbertscore
                            }
                )
                RETURN MERGE(filing, {
                    top_sentences: top_sentences
                })
        """
        sec_results, _ = execute_aql(sec_query, {"ticker": ticker})

        return {
            "ticker": ticker,
            "company": company,
            "latest_market_data": latest_market,
            "awards": awards_results,
            "sec_filings": sec_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stock/{ticker}/market-data")
def get_stock_market_data(
    ticker: str,
    days: int = QueryParam(30, le=365)
):
    """Get historical market data for a stock"""
    db = get_db()

    query = f"""
    FOR doc IN MarketData
        FILTER doc.ticker == @ticker
        FILTER doc.date >= DATE_SUBTRACT(DATE_NOW(), @days, "day")
        SORT doc.date DESC
        LIMIT @days
        RETURN doc
    """

    try:
        results, error = execute_aql(query, {"ticker": ticker, "days": days})
        if error:
            raise HTTPException(status_code=500, detail=error)

        return {
            "ticker": ticker,
            "days": days,
            "count": len(results),
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
