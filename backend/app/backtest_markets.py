"""
Prediction-market backtester: overlay macro, options, SEC, contracts on market probability
over a lookback window; compute lead/lag and correlations.
Phase 1: Polymarket probability from polymarket_price_history; Kalshi can pass probability_series in request.
"""
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import math

from app.database.connection import execute_aql
from app.market_workup import fetch_macro, fetch_options, fetch_sec_sentiment, fetch_contracts, _classify_theme, get_market


def get_polymarket_probability_series(market_id: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """
    Fetch yes_price (probability) time series from polymarket_price_history.
    Returns list of { date: YYYY-MM-DD, probability: 0-1 } sorted by date.
    """
    aql = """
    FOR h IN polymarket_price_history
      FILTER h.market_id == @market_id
      FILTER h.datetime >= @start_date AND h.datetime <= @end_date
      SORT h.timestamp ASC
      RETURN {
        datetime: h.datetime,
        timestamp: h.timestamp,
        yes_price: h.yes_price,
        volume: h.volume
      }
    """
    # end_date should be inclusive; use ISO date for start of day and end of day
    start_ts = datetime.strptime(start_date, "%Y-%m-%d").replace(hour=0, minute=0, second=0)
    end_ts = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
    results, err = execute_aql(aql, {
        "market_id": str(market_id),
        "start_date": start_ts.isoformat() + "Z",
        "end_date": end_ts.isoformat() + "Z",
    })
    if err or not results:
        return []

    # Aggregate to daily: one row per calendar day (use last yes_price of day or average)
    by_date: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        dt = r.get("datetime") or ""
        if "T" in dt:
            day = dt.split("T")[0]
        else:
            day = dt[:10] if len(dt) >= 10 else ""
        if not day:
            continue
        p = r.get("yes_price")
        if p is not None:
            by_date[day].append(float(p))

    out = []
    for day in sorted(by_date.keys()):
        vals = by_date[day]
        prob = sum(vals) / len(vals) if vals else None
        if prob is not None:
            out.append({"date": day, "probability": round(prob, 4)})
    return out


def _align_signal_to_dates(
    signal_rows: List[Dict],
    date_field: str,
    value_field: str,
    value_agg: str = "last",
) -> Dict[str, float]:
    """Convert list of { date, value } (or similar) to dict date -> value. value_agg: last or mean."""
    by_date: Dict[str, List[float]] = defaultdict(list)
    for r in signal_rows:
        d = r.get(date_field)
        if not d:
            continue
        if isinstance(d, str) and len(d) >= 10:
            day = d[:10]
        else:
            day = str(d)
        v = r.get(value_field)
        if v is not None:
            try:
                by_date[day].append(float(v))
            except (TypeError, ValueError):
                pass
    out = {}
    for day, vals in by_date.items():
        if value_agg == "mean":
            out[day] = sum(vals) / len(vals)
        else:
            out[day] = vals[-1]
    return out


def _macro_series_for_window(start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """EconomicData over [start_date, end_date]; return list of { date, federal_funds_rate, sandp_500_index, ... }."""
    aql = """
    FOR doc IN EconomicData
      FILTER doc.date >= @start_date AND doc.date <= @end_date
      SORT doc.date ASC
      RETURN doc
    """
    results, err = execute_aql(aql, {"start_date": start_date, "end_date": end_date})
    if err:
        return []
    return results or []


def _options_series_for_window(start_date: str, end_date: str, tickers: List[str]) -> List[Dict[str, Any]]:
    """options_flow over window for tickers; aggregate to daily (e.g. avg put_call_volume_ratio)."""
    aql = """
    FOR o IN options_flow
      FILTER o.ticker IN @tickers
      FILTER o.date >= @start_date AND o.date <= @end_date
      SORT o.date ASC
      RETURN { date: o.date, put_call_volume_ratio: o.put_call_volume_ratio, unusual_total_activity: o.unusual_total_activity }
    """
    results, err = execute_aql(aql, {"tickers": tickers, "start_date": start_date, "end_date": end_date})
    if err:
        return []
    return results or []


def _contracts_series_for_window(start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """Award daily/weekly total value over window."""
    aql = """
    FOR a IN Award
      FILTER a.start_date >= @start_date AND a.start_date <= @end_date
      FILTER a.award_amount_float != null AND a.award_amount_float > 0
      COLLECT day = a.start_date WITH total = SUM(a.award_amount_float), n = COUNT(a)
      RETURN { date: day, total_value: total, award_count: n }
    """
    results, err = execute_aql(aql, {"start_date": start_date, "end_date": end_date})
    if err:
        return []
    return results or []


def _correlation_lead_lag(
    prob_dates: List[str],
    prob_values: List[float],
    signal_values_by_date: Dict[str, float],
    max_lag_days: int = 5,
) -> Dict[str, Any]:
    """
    For each lag in [-max_lag, +max_lag], compute correlation between signal and probability.
    Positive lag = signal at t, prob at t+lag (signal leads). Negative lag = signal lags.
    Returns { lag_days: int, correlation: float } list and best_lead_lag (lag with max abs corr).
    """
    if not prob_dates or not prob_values or len(prob_dates) != len(prob_values):
        return {"correlations": [], "best_lead_lag": None, "summary": "Insufficient data"}

    n = len(prob_dates)
    correlations = []
    for lag in range(-max_lag_days, max_lag_days + 1):
        paired = []
        for i in range(n):
            if lag >= 0:
                # Signal leads: signal at i-lag, probability at i
                j_sig = i - lag
                if j_sig < 0:
                    continue
                sig_date = prob_dates[j_sig]
                prob_val = prob_values[i]
            else:
                # Signal lags: signal at i, probability at i + abs(lag)
                j_prob = i - lag
                if j_prob >= n:
                    continue
                sig_date = prob_dates[i]
                prob_val = prob_values[j_prob]
            sig_val = signal_values_by_date.get(sig_date)
            if sig_val is None:
                continue
            paired.append((sig_val, prob_val))
        if len(paired) < 5:
            correlations.append({"lag_days": lag, "correlation": None, "pairs": len(paired)})
            continue
        sig_vals = [x[0] for x in paired]
        prob_vals = [x[1] for x in paired]
        s_mean = sum(sig_vals) / len(sig_vals)
        p_mean = sum(prob_vals) / len(prob_vals)
        s_var = sum((s - s_mean) ** 2 for s in sig_vals) / len(paired)
        p_var = sum((p - p_mean) ** 2 for p in prob_vals) / len(paired)
        if s_var <= 0 or p_var <= 0:
            correlations.append({"lag_days": lag, "correlation": 0.0, "pairs": len(paired)})
            continue
        cov = sum((s - s_mean) * (p - p_mean) for s, p in zip(sig_vals, prob_vals)) / len(paired)
        corr = cov / (math.sqrt(s_var) * math.sqrt(p_var))
        correlations.append({"lag_days": lag, "correlation": round(corr, 4), "pairs": len(paired)})

    best = None
    best_abs = -1
    for c in correlations:
        if c.get("correlation") is not None and abs(c["correlation"]) > best_abs:
            best_abs = abs(c["correlation"])
            best = c["lag_days"]

    summary = ""
    if best is not None:
        if best > 0:
            summary = f"Signal leads probability by {best} day(s) (strongest correlation at +{best} lag)."
        elif best < 0:
            summary = f"Signal lags probability by {abs(best)} day(s)."
        else:
            summary = "Signal and probability move same-day (no clear lead/lag)."

    return {"correlations": correlations, "best_lead_lag": best, "summary": summary}


def run_backtest(
    platform: str,
    resolution_date: str,
    lookback_days: int = 30,
    market_id: Optional[str] = None,
    probability_series: Optional[List[Dict[str, Any]]] = None,
    signals: Optional[Dict[str, bool]] = None,
    theme: str = "other",
) -> Dict[str, Any]:
    """
    Run prediction-market backtest over [resolution_date - lookback_days, resolution_date].
    probability_series: optional list of { date, probability }; if not provided and platform==polymarket and market_id set, fetch from polymarket_price_history.
    signals: { macro: true, options: true, sec: false, contracts: false }.
    Returns probability_series, signal_series (macro, options, sec, contracts), lead_lag per signal, annotations.
    """
    try:
        end_dt = datetime.strptime(resolution_date, "%Y-%m-%d")
    except Exception:
        return {"error": "Invalid resolution_date (use YYYY-MM-DD)"}
    start_dt = end_dt - timedelta(days=lookback_days)
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = resolution_date

    signals = signals or {}
    use_macro = signals.get("macro", True)
    use_options = signals.get("options", True)
    use_sec = signals.get("sec", False)
    use_contracts = signals.get("contracts", False)

    # 1) Probability series
    prob_series = probability_series
    if not prob_series and platform.lower() == "polymarket" and market_id:
        prob_series = get_polymarket_probability_series(market_id, start_date, end_date)
    if not prob_series:
        prob_series = []  # Kalshi without provided series

    prob_dates = [p["date"] for p in prob_series]
    prob_values = [p["probability"] for p in prob_series]

    # 2) Fetch signal data over window
    macro_rows = _macro_series_for_window(start_date, end_date) if use_macro else []
    tickers = ["TLT", "JPM", "BAC", "KRE", "VNQ"]
    options_rows = _options_series_for_window(start_date, end_date, tickers) if use_options else []
    contract_rows = _contracts_series_for_window(start_date, end_date) if use_contracts else []

    # SEC: use market_workup theme-based sector sentiment over window (one value per sector, not daily); skip lead/lag or use single scalar
    sec_rows = []
    if use_sec:
        sec_rows = fetch_sec_sentiment(theme, lookback_days)

    # 3) Build daily signal series for alignment (normalize for chart)
    macro_by_date = _align_signal_to_dates(macro_rows, "date", "federal_funds_rate", "last") if macro_rows else {}
    options_by_date = _align_signal_to_dates(options_rows, "date", "put_call_volume_ratio", "mean") if options_rows else {}
    contracts_by_date = _align_signal_to_dates(contract_rows, "date", "total_value", "last") if contract_rows else {}

    # 4) Lead/lag for each signal
    lead_lag = {}
    if prob_dates and prob_values:
        if macro_by_date:
            lead_lag["macro"] = _correlation_lead_lag(prob_dates, prob_values, macro_by_date)
        if options_by_date:
            lead_lag["options"] = _correlation_lead_lag(prob_dates, prob_values, options_by_date)
        if contracts_by_date:
            lead_lag["contracts"] = _correlation_lead_lag(prob_dates, prob_values, contracts_by_date)

    # 5) Annotations: FOMC/CPI dates could be added from a small calendar; skip for Phase 1 or stub
    annotations = []

    return {
        "market_id": market_id,
        "platform": platform,
        "resolution_date": resolution_date,
        "lookback_days": lookback_days,
        "probability_series": prob_series,
        "signal_series": {
            "macro": [{"date": r.get("date"), "federal_funds_rate": r.get("federal_funds_rate"), "sandp_500_index": r.get("sandp_500_index")} for r in macro_rows],
            "options": options_rows,
            "contracts": contract_rows,
            "sec_sentiment": sec_rows,
        },
        "lead_lag": lead_lag,
        "annotations": annotations,
    }
