"""
Market Workup orchestration: given a prediction market (Kalshi or Polymarket),
fetch connected data (macro, options, SEC sentiment, contracts, news) and synthesize.
Fed FEDS 2026-010 north star: validate signals and do market research.
"""
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import re

from app.database.connection import execute_aql

# Theme → EconomicData: we query by date; EconomicData has date + series columns (e.g. federal_funds_rate, unemployment_rate)
THEME_TO_MACRO_SERIES = {
    "fed_rates": ["federal_funds_rate", "sandp_500_index"],
    "cpi": ["sandp_500_index"],  # Add CPI series if we have them in EconomicData
    "inflation": ["sandp_500_index"],
    "gdp": ["sandp_500_index"],
    "unemployment": ["unemployment_rate", "federal_funds_rate"],
    "recession": ["unemployment_rate", "federal_funds_rate", "sandp_500_index"],
    "government_spending": ["federal_funds_rate"],
    "other": ["federal_funds_rate", "sandp_500_index", "unemployment_rate"],
}

# Theme → ticker basket for options flow (rate-sensitive, etc.)
THEME_TO_TICKERS = {
    "fed_rates": ["TLT", "JPM", "BAC", "KRE", "VNQ"],
    "cpi": ["TLT", "SPY"],
    "inflation": ["TLT", "SPY"],
    "gdp": ["SPY", "JPM", "BAC"],
    "unemployment": ["JPM", "BAC", "KRE"],
    "recession": ["TLT", "SPY", "JPM"],
    "government_spending": ["LMT", "RTX", "NOC", "BA", "GD"],
    "other": ["TLT", "JPM", "SPY"],
}


def _classify_theme(question: str, category: Optional[str]) -> str:
    """Map market to theme: fed_rates, cpi, gdp, unemployment, recession, government_spending, other."""
    text = (question or "") + " " + (category or "")
    text_lower = text.lower()
    if re.search(r"\bfed\b|fomc|rate cut|rate hike|federal funds|interest rate", text_lower):
        return "fed_rates"
    if re.search(r"\bcpi\b|inflation|consumer price", text_lower):
        return "cpi"
    if re.search(r"\bgdp\b|growth|recession", text_lower):
        if "recession" in text_lower:
            return "recession"
        return "gdp"
    if re.search(r"unemployment|payroll|jobs|labor", text_lower):
        return "unemployment"
    if re.search(r"government spending|defense|contract|cut.*spending", text_lower):
        return "government_spending"
    return "other"


def get_market(market_id: str, platform: str) -> Optional[Dict[str, Any]]:
    """Load market by _key from prediction_markets_kalshi or prediction_markets_polymarket."""
    col = "prediction_markets_kalshi" if platform.lower() == "kalshi" else "prediction_markets_polymarket"
    aql = f"""
    FOR m IN {col}
      FILTER m._key == @market_id
      LIMIT 1
      RETURN m
    """
    results, err = execute_aql(aql, {"market_id": market_id})
    if err or not results:
        return None
    m = results[0]
    question = m.get("event_title") or m.get("question") or m.get("title") or ""
    yes_prob = m.get("yes_probability")
    if yes_prob is None and "yes_prob" in m:
        yes_prob = m["yes_prob"] / 100.0 if m["yes_prob"] else None
    return {
        "id": m.get("_key"),
        "question": question,
        "yes_probability": yes_prob,
        "volume": m.get("volume") or m.get("volume_24h") or 0,
        "open_interest": m.get("open_interest"),
        "category": m.get("category"),
        "end_date": m.get("close_time") or m.get("end_date"),
        "platform": platform,
    }


def fetch_macro(theme: str, days: int = 90) -> List[Dict[str, Any]]:
    """Fetch EconomicData for last N days. Returns list of { date, ...series }."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    series = THEME_TO_MACRO_SERIES.get(theme, THEME_TO_MACRO_SERIES["other"])
    # EconomicData may have date + multiple fields per doc
    aql = """
    FOR doc IN EconomicData
      FILTER doc.date >= @cutoff
      SORT doc.date DESC
      LIMIT 500
      RETURN doc
    """
    results, err = execute_aql(aql, {"cutoff": cutoff})
    if err:
        return []
    return results or []


def fetch_options(theme: str, days: int = 30) -> List[Dict[str, Any]]:
    """Fetch options_flow for theme basket, last N days."""
    tickers = THEME_TO_TICKERS.get(theme, THEME_TO_TICKERS["other"])
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    aql = """
    FOR o IN options_flow
      FILTER o.ticker IN @tickers
      FILTER o.date >= @cutoff
      SORT o.date DESC
      LIMIT 300
      RETURN {
        ticker: o.ticker,
        date: o.date,
        put_call_volume_ratio: o.put_call_volume_ratio,
        call_volume_unusual: o.call_volume_unusual,
        unusual_total_activity: o.unusual_total_activity,
        unusual_call_activity: o.unusual_call_activity
      }
    """
    results, err = execute_aql(aql, {"tickers": tickers, "cutoff": cutoff})
    if err:
        return []
    return results or []


def fetch_sec_sentiment(theme: str, days: int = 60) -> List[Dict[str, Any]]:
    """Aggregate SEC (sec_sentences) FinBERT by sector for theme-relevant sectors."""
    sectors = ["Financials", "Real Estate"] if theme in ("fed_rates", "unemployment", "recession") else ["Industrials", "Information Technology"]
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    aql = """
    FOR c IN Company
      FILTER c.sector != null AND c.sector IN @sectors
      FOR s IN sec_sentences
        FILTER s.ticker == c.ticker AND s.filing_date >= @cutoff AND s.finbert_score != null
        COLLECT sector = c.sector AGGREGATE avg_finbert = AVG(s.finbert_score), sentence_count = COUNT(s)
        RETURN { sector: sector, avg_finbert: avg_finbert, sentence_count: sentence_count }
    """
    results, err = execute_aql(aql, {"sectors": sectors, "cutoff": cutoff})
    if err:
        return []
    return results or []


def fetch_contracts(theme: str, days: int = 90) -> List[Dict[str, Any]]:
    """Award volume trend (e.g. by week) for government_spending / defense."""
    if theme != "government_spending":
        return []
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    aql = """
    FOR a IN Award
      FILTER a.start_date >= @cutoff AND a.award_amount_float != null AND a.award_amount_float > 0
      COLLECT week = SUBSTRING(a.start_date, 0, 10) WITH COUNT INTO n, total = SUM(a.award_amount_float)
      SORT week DESC
      LIMIT 20
      RETURN { week: week, award_count: n, total_value: total }
    """
    results, err = execute_aql(aql, {"cutoff": cutoff})
    if err:
        return []
    return results or []


def fetch_congressional(theme: str, days: int = 90) -> List[Dict[str, Any]]:
    """Recent congressional trades for theme-relevant tickers (e.g. defense for government_spending)."""
    if theme != "government_spending":
        return []
    tickers = ["LMT", "RTX", "NOC", "BA", "GD"]
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    aql = """
    FOR t IN congressional_trades
      FILTER t.ticker IN @tickers
      FILTER t.date >= @cutoff
      SORT t.date DESC
      LIMIT 100
      RETURN { politician_name: t.politician_name, chamber: t.chamber, date: t.date, ticker: t.ticker, transaction_type: t.transaction_type, amount_range: t.amount_range }
    """
    results, err = execute_aql(aql, {"tickers": tickers, "cutoff": cutoff})
    if err:
        return []
    return results or []


def fetch_news(question: str) -> Dict[str, Any]:
    """Recent news via Perplexity for the market question."""
    try:
        from app.llm.web_search import search_web_context
        out = search_web_context(
            f"Recent news and developments that could affect this outcome: {question}. Focus on the last 1-2 weeks."
        )
        return {"summary": out.get("summary", ""), "sources": out.get("sources", []), "citations": out.get("citations", [])}
    except Exception as e:
        print(f"[market_workup] Perplexity error: {e}")
        return {"summary": "", "sources": [], "citations": [], "error": str(e)}


def synthesize_cross_source_insight(
    market_question: str,
    yes_probability: Optional[float],
    macro_data: List[Dict],
    options_data: List[Dict],
    sec_data: List[Dict],
    contract_data: List[Dict],
    news_context: Dict,
) -> str:
    """One short narrative: what does the data imply vs what the market is pricing?"""
    try:
        from app.llm.planning import get_openai_client
        import config
        client = get_openai_client()
    except Exception as e:
        print(f"[market_workup] OpenAI client error: {e}")
        return "Synthesis unavailable (LLM not configured)."

    def _summarize(name: str, data: Any, max_len: int = 600) -> str:
        if not data:
            return f"{name}: No data."
        import json
        s = json.dumps(data[:15] if isinstance(data, list) else data, default=str)
        return f"{name}: {s[:max_len]}" + ("..." if len(s) > max_len else "")

    macro_s = _summarize("Macro", macro_data)
    options_s = _summarize("Options flow", options_data)
    sec_s = _summarize("SEC sentiment", sec_data)
    contract_s = _summarize("Contracts", contract_data)
    news_s = (news_context.get("summary") or "")[:800]

    prompt = f"""You are a prediction market research analyst. Write 1-2 short paragraphs (3-5 sentences total) that answer: What does the connected data (macro, options, SEC, contracts, news) collectively imply versus what the market is currently pricing?

Market: {market_question}
Current yes probability: {yes_probability * 100 if yes_probability is not None else 'N/A'}%

Data summaries:
{macro_s}
{options_s}
{sec_s}
{contract_s}
Recent news: {news_s}

Write in plain English. Be concise. Do not use bullet points. Focus on one main insight (agreement or divergence between data and market)."""

    try:
        resp = client.chat.completions.create(
            model=getattr(config, "LLM_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[market_workup] Synthesis error: {e}")
        return f"Synthesis failed: {e}"


def run_market_workup(market_id: str, platform: str = "kalshi") -> Dict[str, Any]:
    """
    Full orchestration: get market, classify theme, fetch all panels, synthesize.
    Returns payload for GET /api/research/market/{market_id}.
    """
    market = get_market(market_id, platform)
    if not market:
        return {"error": "Market not found", "market_id": market_id, "platform": platform}

    question = market.get("question") or ""
    theme = _classify_theme(question, market.get("category"))

    macro_data = fetch_macro(theme, 90)
    options_data = fetch_options(theme, 30)
    sec_data = fetch_sec_sentiment(theme, 60)
    contract_data = fetch_contracts(theme, 90)
    congressional_data = fetch_congressional(theme, 90)
    news_context = fetch_news(question)

    cross_source_insight = synthesize_cross_source_insight(
        question,
        market.get("yes_probability"),
        macro_data,
        options_data,
        sec_data,
        contract_data,
        news_context,
    )

    return {
        "market": market,
        "theme": theme,
        "macro_data": macro_data,
        "options_signal": options_data,
        "sec_sentiment": sec_data,
        "contract_signal": contract_data,
        "congressional_trades": congressional_data,
        "news_context": news_context,
        "cross_source_insight": cross_source_insight,
    }
