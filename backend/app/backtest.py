"""
Lightweight event-driven backtester for graph-derived signals.
Point-in-time: signals computed from data available as of each date; forward returns from MarketData.
Metrics: Rank IC, hit rate, simple Sharpe (long top quintile, short bottom).
"""
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from app.database.connection import get_db, execute_aql


def get_contract_momentum_as_of_date(as_of_date: str, limit: int = 300) -> List[Dict[str, Any]]:
    """
    Contract momentum (90d award sum) computed with data available as of as_of_date.
    Award-first: filter Award by date range, join to HAS_AWARD by _to, aggregate by _from (Company).
    """
    aql = """
    LET cutoff = SUBSTRING(DATE_SUBTRACT(DATE_ISO8601(@as_of_date), 90, 'day'), 0, 10)
    FOR award IN Award
      FILTER award.start_date >= cutoff AND award.start_date <= @as_of_date
      FILTER award.award_amount_float != null AND award.award_amount_float > 0
      FOR edge IN HAS_AWARD
        FILTER edge._to == award._id
        COLLECT companyId = edge._from
        AGGREGATE total = SUM(award.award_amount_float)
        LET comp = DOCUMENT(companyId)
        FILTER comp != null
        SORT total DESC
        LIMIT @limit
        RETURN { ticker: comp.ticker, contract_momentum_90d: total }
    """
    results, err = execute_aql(aql, {"as_of_date": as_of_date, "limit": limit})
    if err:
        return []
    return results or []


def get_forward_returns(ticker: str, from_date: str, to_date: str) -> Optional[float]:
    """
    Forward return from from_date to to_date: (close_to - close_from) / close_from.
    Returns None if insufficient data.
    """
    aql = """
    LET from_close = (
      FOR m IN MarketData
        FILTER m.ticker == @ticker AND m.date <= @from_date
        SORT m.date DESC
        LIMIT 1
        RETURN m.close
    )
    LET to_close = (
      FOR m IN MarketData
        FILTER m.ticker == @ticker AND m.date >= @to_date
        SORT m.date ASC
        LIMIT 1
        RETURN m.close
    )
    FILTER LENGTH(from_close) == 1 AND LENGTH(to_close) == 1 AND from_close[0] > 0
    RETURN (to_close[0] - from_close[0]) / from_close[0]
    """
    results, err = execute_aql(aql, {"ticker": ticker, "from_date": from_date, "to_date": to_date})
    if err or not results:
        return None
    return results[0]


def run_backtest(
    start_date: str,
    end_date: str,
    rebalance_freq_days: int = 21,
    forward_days: int = 21,
    top_quintile_only: bool = True,
) -> Dict[str, Any]:
    """
    Backtest contract momentum: each rebalance date, compute momentum rank as of that date,
    get forward returns, aggregate Rank IC and hit rate.
    start_date/end_date: YYYY-MM-DD. rebalance_freq_days: step between signal dates.
    """
    from datetime import datetime as dt

    try:
        s = dt.strptime(start_date, "%Y-%m-%d")
        e = dt.strptime(end_date, "%Y-%m-%d")
    except Exception:
        return {"error": "Invalid date format (use YYYY-MM-DD)"}

    signal_dates = []
    d = s
    while d <= e:
        signal_dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=rebalance_freq_days)

    all_ranks = []  # (ticker, rank, forward_return)
    for as_of in signal_dates:
        momentum = get_contract_momentum_as_of_date(as_of, limit=200)
        if not momentum:
            continue
        # Rank 1 = highest momentum
        ticker_to_rank = {r["ticker"]: i + 1 for i, r in enumerate(momentum)}
        to_d = dt.strptime(as_of, "%Y-%m-%d") + timedelta(days=forward_days + 10)
        to_date = to_d.strftime("%Y-%m-%d")
        for r in momentum:
            ticker = r["ticker"]
            rank = ticker_to_rank[ticker]
            ret = get_forward_returns(ticker, as_of, to_date)
            if ret is not None:
                all_ranks.append({"ticker": ticker, "rank": rank, "forward_return": ret, "date": as_of})

    if len(all_ranks) < 10:
        return {
            "error": "Insufficient data for backtest",
            "observations": len(all_ranks),
            "signal_dates": len(signal_dates),
        }

    # Rank IC: Spearman-like correlation of rank (1=high) with forward return
    import math
    n = len(all_ranks)
    ranks = [x["rank"] for x in all_ranks]
    returns = [x["forward_return"] for x in all_ranks]
    rank_mean = sum(ranks) / n
    ret_mean = sum(returns) / n
    sr = sum((r - rank_mean) * (ret - ret_mean) for r, ret in zip(ranks, returns))
    ss_r = sum((r - rank_mean) ** 2 for r in ranks) ** 0.5
    ss_ret = sum((ret - ret_mean) ** 2 for ret in returns) ** 0.5
    rank_ic = (sr / (ss_r * ss_ret)) if (ss_r * ss_ret) > 0 else 0

    # Hit rate: % of observations where higher rank (lower rank number = higher momentum) has positive return
    hits = sum(1 for x in all_ranks if x["forward_return"] > 0)
    hit_rate = hits / n

    # Top quintile vs bottom: average return of top 20% rank vs bottom 20%
    sorted_by_rank = sorted(all_ranks, key=lambda x: x["rank"])
    q = max(1, n // 5)
    top_ret = sum(x["forward_return"] for x in sorted_by_rank[:q]) / q
    bot_ret = sum(x["forward_return"] for x in sorted_by_rank[-q:]) / q
    spread = top_ret - bot_ret
    ret_std = (sum((ret - ret_mean) ** 2 for ret in returns) / n) ** 0.5
    sharpe_like = (spread / ret_std * (252 / (forward_days or 1)) ** 0.5) if ret_std > 0 else 0

    return {
        "signal": "contract_momentum_90d",
        "start_date": start_date,
        "end_date": end_date,
        "observations": n,
        "rebalance_dates": len(signal_dates),
        "rank_ic": round(rank_ic, 4),
        "hit_rate": round(hit_rate, 4),
        "top_quintile_avg_return": round(top_ret, 4),
        "bottom_quintile_avg_return": round(bot_ret, 4),
        "spread": round(spread, 4),
        "sharpe_like": round(sharpe_like, 4),
        "point_in_time": "Award.start_date and MarketData.date used as observation dates",
    }
