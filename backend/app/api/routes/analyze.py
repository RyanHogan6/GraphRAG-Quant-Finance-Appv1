"""
Analysis API - correlation and statistical endpoints.
"""
import math
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

from app.database.connection import get_db, execute_aql

router = APIRouter(prefix="/analyze", tags=["analyze"])


# Collection -> default date field for time series
DATE_FIELD_BY_COLLECTION = {
    "MarketData": "date",
    "EconomicData": "date",
    "options_flow": "date",
    "futures_prices": "date",
    "eia_crude_inventory": "date",
    "eia_natgas_storage": "date",
    "Award": "start_date",
    "sec_filings": "filing_date",
}


class SeriesSpec(BaseModel):
    collection: str = Field(..., min_length=1, max_length=64)
    field: str = Field(..., min_length=1, max_length=64)
    date_field: Optional[str] = None
    filter: Optional[Dict[str, Any]] = None


class CorrelationRequest(BaseModel):
    series_a: SeriesSpec
    series_b: SeriesSpec
    window_days: int = Field(90, ge=7, le=730)
    method: str = Field("pearson", pattern="^pearson$")


def _fetch_series(collection: str, field: str, date_field: Optional[str], filter_dict: Optional[Dict]) -> List[tuple]:
    """Fetch (date_str, value) list sorted by date. Returns list of (date_str, float)."""
    db = get_db()
    date_f = date_field or DATE_FIELD_BY_COLLECTION.get(collection, "date")
    # Sanitize collection name for AQL (no user input in FROM - we validate against known list)
    allowed = set(DATE_FIELD_BY_COLLECTION.keys()) | {"commodity_positions", "eia_natgas_production", "eia_lng_exports"}
    if collection not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported collection for series: {collection}")
    filter_lines = []
    bind_vars = {}
    if filter_dict:
        for k, v in filter_dict.items():
            if k and isinstance(v, (str, int, float, bool)):
                filter_lines.append(f"FILTER doc.{k} == @filter_{k}")
                bind_vars[f"filter_{k}"] = v
    filter_clause = "\n      ".join(filter_lines)
    aql = f"""
    FOR doc IN {collection}
      {filter_clause}
      FILTER doc.{date_f} != null AND doc.{field} != null
      SORT doc.{date_f} ASC
      RETURN [doc.{date_f}, doc.{field}]
    """
    results, err = execute_aql(aql, bind_vars)
    if err:
        raise HTTPException(status_code=400, detail=f"Series fetch failed: {err}")
    out = []
    for row in results or []:
        if isinstance(row, list) and len(row) >= 2:
            try:
                val = float(row[1]) if row[1] is not None else None
                if val is not None:
                    out.append((str(row[0])[:10], val))
            except (TypeError, ValueError):
                continue
    return out


def _pearson_and_pvalue(x: List[float], y: List[float]) -> tuple:
    """Return (correlation, p_value). Uses t-approximation for p-value."""
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    mx = sum(x) / n
    my = sum(y) / n
    ssx = sum((a - mx) ** 2 for a in x)
    ssy = sum((b - my) ** 2 for b in y)
    sp = sum((a - mx) * (b - my) for a, b in zip(x, y))
    if ssx * ssy <= 0:
        return 0.0, 1.0
    r = sp / math.sqrt(ssx * ssy)
    r = max(-1.0, min(1.0, r))
    # Two-tailed p-value from t = r * sqrt(n-2) / sqrt(1 - r^2)
    if abs(r) >= 1.0:
        p = 0.0
    else:
        t = r * math.sqrt(n - 2) / math.sqrt(1 - r * r) if (1 - r * r) > 0 else 0
        # Approximate p-value using normal for large n
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))) if n > 2 else 1.0
    return round(r, 6), round(p, 6)


@router.post("/correlation")
def correlation(body: CorrelationRequest):
    """
    Compute pairwise correlation between two time series from the graph.
    Series are aligned by date; only dates present in both are used.
    Optionally limited to the most recent window_days.
    """
    a = _fetch_series(
        body.series_a.collection,
        body.series_a.field,
        body.series_a.date_field,
        body.series_a.filter,
    )
    b = _fetch_series(
        body.series_b.collection,
        body.series_b.field,
        body.series_b.date_field,
        body.series_b.filter,
    )
    if not a or not b:
        raise HTTPException(status_code=400, detail="One or both series have no data")
    # Align by date (inner join)
    b_by_date = {d: v for d, v in b}
    aligned = [(da, va, b_by_date[da]) for da, va in a if da in b_by_date]
    if body.window_days and body.window_days < 730:
        from datetime import datetime, timedelta
        max_d = max(t[0] for t in aligned)
        try:
            cutoff = (datetime.strptime(max_d[:10], "%Y-%m-%d") - timedelta(days=body.window_days)).strftime("%Y-%m-%d")
            aligned = [t for t in aligned if t[0] >= cutoff]
        except Exception:
            pass
    if len(aligned) < 3:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient overlapping observations: {len(aligned)} (need at least 3)",
        )
    x = [t[1] for t in aligned]
    y = [t[2] for t in aligned]
    r, p = _pearson_and_pvalue(x, y)
    return {
        "correlation": r,
        "p_value": p,
        "n_observations": len(aligned),
        "method": body.method,
        "date_range": {"min": aligned[0][0], "max": aligned[-1][0]},
    }
