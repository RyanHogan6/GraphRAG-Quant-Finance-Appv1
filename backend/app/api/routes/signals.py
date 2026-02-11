"""
Alpha signals API - contract momentum, options-filing convergence, centrality, backtest.
"""
from fastapi import APIRouter, Query

from app.alpha_signals import (
    get_contract_momentum_90d,
    get_options_filing_convergence,
    get_contract_centrality,
    get_all_signals,
)

router = APIRouter(prefix="/signals", tags=["signals"])


@router.get("/contract-momentum")
def contract_momentum(limit: int = Query(100, ge=1, le=500)):
    """90-day contract award sum by company (momentum signal), sorted descending."""
    return {"signal": "contract_momentum_90d", "data": get_contract_momentum_90d(limit=limit)}


@router.get("/options-filing")
def options_filing(
    days_before: int = Query(14, ge=1, le=60),
    limit: int = Query(50, ge=1, le=200),
):
    """Options activity within N days before SEC filings (OPTIONS_BEFORE_FILING)."""
    return {
        "signal": "options_filing_convergence",
        "params": {"days_before": days_before},
        "data": get_options_filing_convergence(days_before=days_before, limit=limit),
    }


@router.get("/centrality")
def centrality(limit: int = Query(100, ge=1, le=500)):
    """Contract graph degree centrality (HAS_AWARD count per company)."""
    return {"signal": "contract_degree_centrality", "data": get_contract_centrality(limit=limit)}


@router.get("/all")
def all_signals(
    contract_limit: int = Query(50, ge=1, le=200),
    options_limit: int = Query(30, ge=1, le=100),
    centrality_limit: int = Query(50, ge=1, le=200),
):
    """All alpha signals in one response (for dashboard)."""
    return get_all_signals(
        contract_limit=contract_limit,
        options_limit=options_limit,
        centrality_limit=centrality_limit,
    )


@router.get("/backtest")
def backtest(
    start_date: str = Query(..., description="YYYY-MM-DD"),
    end_date: str = Query(..., description="YYYY-MM-DD"),
    rebalance_freq_days: int = Query(21, ge=5, le=63),
    forward_days: int = Query(21, ge=5, le=63),
):
    """Lightweight backtest of contract momentum signal (point-in-time, forward returns)."""
    from app.backtest import run_backtest
    return run_backtest(
        start_date=start_date,
        end_date=end_date,
        rebalance_freq_days=rebalance_freq_days,
        forward_days=forward_days,
    )
