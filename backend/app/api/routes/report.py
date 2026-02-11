"""
Report Generation API - GraphExplorer Execute Query
Generates comprehensive investment reports from graph queries
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import time
import json
from datetime import datetime

from app.database.connection import get_db
from app.llm.report_generator import generate_investment_report
from app.utils.query_validator import validate_aql_query
from slowapi import Limiter
from slowapi.util import get_remote_address

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


class ReportRequest(BaseModel):
    aql_query: str
    description: str  # English description from GraphExplorer
    collections: List[str]  # Collections involved in query
    ticker: Optional[str] = None  # If querying specific company


class ReportResponse(BaseModel):
    report: Dict[str, Any]  # Structured report sections
    raw_results: List[Dict[str, Any]]  # Raw query results
    execution_time: float
    query_metadata: Dict[str, Any]


@router.post("/generate", response_model=ReportResponse)
@limiter.limit("10/minute")  # Conservative limit for expensive operations
def generate_report(request: Request, body: ReportRequest):
    """
    Execute graph query and generate comprehensive investment report

    This is the "secret sauce" - turning graph traversals into actionable intelligence
    """
    start_time = time.time()

    is_valid, validation_error = validate_aql_query(body.aql_query)
    if not is_valid:
        raise HTTPException(status_code=400, detail=validation_error or "Query validation failed")

    try:
        db = get_db()
        print(f"[REPORT] Executing query: {body.aql_query[:100]}...")

        cursor = db.aql.execute(body.aql_query)
        results = list(cursor)

        print(f"[REPORT] Query returned {len(results)} results")

        if not results:
            return ReportResponse(
                report={
                    "executive_summary": "No data found for this query.",
                    "sections": []
                },
                raw_results=[],
                execution_time=time.time() - start_time,
                query_metadata={
                    "collections": body.collections,
                    "result_count": 0
                }
            )

        # Analyze what type of data we have
        query_metadata = {
            "collections": body.collections,
            "result_count": len(results),
            "date_range": extract_date_range(results),
            "has_company_data": "Company" in body.collections or "company" in body.aql_query.lower(),
            "has_options": "options" in body.aql_query.lower(),
            "has_sec": "sec" in body.aql_query.lower(),
            "has_awards": "award" in body.aql_query.lower(),
            "has_commodities": "futures" in body.aql_query.lower() or "eia_" in body.aql_query.lower(),
            "has_markets": "prediction" in body.aql_query.lower() or "polymarket" in body.aql_query.lower()
        }

        # Generate intelligent report using Claude
        report = generate_investment_report(
            query_description=body.description,
            query_results=results,
            metadata=query_metadata,
            ticker=body.ticker
        )

        execution_time = time.time() - start_time

        return ReportResponse(
            report=report,
            raw_results=results[:100],  # Limit raw results to first 100
            execution_time=execution_time,
            query_metadata=query_metadata
        )

    except Exception as e:
        import traceback
        print(f"[REPORT ERROR] {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


def extract_date_range(results: List[Dict]) -> Optional[Dict[str, str]]:
    """Extract date range from results"""
    dates = []

    for result in results:
        # Look for common date fields
        for field in ['date', 'filing_date', 'start_date', 'timestamp']:
            if field in result and result[field]:
                dates.append(str(result[field]))

    if not dates:
        return None

    dates_sorted = sorted(dates)
    return {
        "earliest": dates_sorted[0],
        "latest": dates_sorted[-1]
    }
