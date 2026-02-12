"""
Query decomposition for narrative/why/explain questions.
Splits "Why is RTX up 12%?" into sub-intents (contracts, SEC, options, macro)
and returns sub-questions for parallel AQL execution + merged context.
"""
import re
from typing import List, Dict, Any, Optional

# Ticker extraction shared with planning
TICKER_TYPO_CORRECTIONS = {
    "APPL": "AAPL", "APL": "AAPL", "APPLE": "AAPL",
    "GOOG": "GOOGL", "AMAZON": "AMZN", "MICROSOFT": "MSFT",
    "TESLA": "TSLA", "NVIDIA": "NVDA", "META": "META", "ALPHABET": "GOOGL",
}


def normalize_ticker(raw: str) -> str:
    if not raw or len(raw) < 2:
        return raw
    return TICKER_TYPO_CORRECTIONS.get(raw.upper().strip(), raw.upper().strip())


# Phrases that indicate a narrative/explanation request (relationship-centric)
NARRATIVE_TRIGGERS = [
    "why is", "why are", "why did", "why has", "why have",
    "what drove", "what's driving", "what is driving",
    "explain", "explain why", "what caused", "what's the story",
    "what happened with", "what's going on with", "breakdown",
    "reasons for", "reason for", "catalyst", "catalysts",
    "what led to", "what led", "drivers", "driver behind",
]


def _normalize_question(question: str) -> str:
    """Normalize for matching: strip and replace curly/smart quotes."""
    if not question:
        return ""
    q = question.strip()
    for old, new in [("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"')]:
        q = q.replace(old, new)
    return q


def is_narrative_question(question: str) -> bool:
    """
    True if the question asks for explanation/why/what drove (narrative) rather than
    a single retrieval (entity-centric).
    """
    if not question or len(question.strip()) < 10:
        return False
    q = _normalize_question(question).lower()
    return any(trigger in q for trigger in NARRATIVE_TRIGGERS)


# Words that must never be treated as a ticker (common in "explain why...", "what is...")
NOT_A_TICKER = frozenset({
    "SEC", "ETF", "IPO", "EPS", "GDP", "CPI", "RSI", "MACD", "P/E", "THE",
    "WHY", "WHAT", "HOW", "WHEN", "OIL", "GAS", "EIA", "API",
})


def extract_ticker_from_question(question: str) -> Optional[str]:
    """Extract likely ticker from question (e.g. 'Why is RTX up', 'Explain PLTR', \"What drove PLTR's success\")."""
    q = _normalize_question(question)
    # After "what drove" / "why is" / "explain": ticker may be followed by 's (e.g. PLTR's) — exclude "why"
    m = re.search(r"\b(why is|explain|what drove|what's driving)\s+([A-Za-z]{2,5})'?s?", q, re.IGNORECASE)
    if m:
        tok = m.group(2).upper()
        if tok not in NOT_A_TICKER:
            return normalize_ticker(m.group(2))
    m = re.search(r"\b([A-Za-z]{2,5})'?s?\s+(?:stock\s+)?(?:up|down|rally|drop|success|performance)", q, re.IGNORECASE)
    if m:
        tok = m.group(1).upper()
        if tok not in NOT_A_TICKER:
            return normalize_ticker(m.group(1))
    # First 2-5 letter token (allow lowercase for PLTR in "PLTR's")
    m = re.search(r"\b([A-Za-z]{2,5})\b", q)
    if m:
        tok = m.group(1).upper()
        if tok not in NOT_A_TICKER:
            return normalize_ticker(tok)
    return None


# Sub-intent definitions: label, natural sub-question template (use @ticker or company name)
SUB_INTENTS = [
    {
        "label": "contracts",
        "sub_question_template": "Recent government contract awards for {ticker} in the last 90 days, sorted by amount",
        "collections": ["Company", "Award"],
        "edge_path": ["HAS_AWARD"],
    },
    {
        "label": "sec_filings",
        "sub_question_template": "Recent SEC filings and revenue or risk guidance for {ticker}",
        "collections": ["Company", "sec_filings", "sec_sections", "sec_sentences"],
        "edge_path": ["HAS_FILING", "has_section", "has_sentence"],
    },
    {
        "label": "options_flow",
        "sub_question_template": "Unusual options activity and put/call ratio for {ticker} in the last 30 days",
        "collections": ["Company", "options_flow"],
        "edge_path": ["COMPANY_HAS_OPTIONS"],
    },
    {
        "label": "market_data",
        "sub_question_template": "Stock price and key technical indicators for {ticker} over the last 90 days",
        "collections": ["Company", "MarketData"],
        "edge_path": ["HAS_MARKETDATA"],
    },
    {
        "label": "macro",
        "sub_question_template": "Relevant macro or defense spending trends from economic data",
        "collections": ["EconomicData"],
        "edge_path": [],
    },
]


def is_commodity_eia_question(question: str) -> bool:
    """True if question is about commodities/EIA/inventory, not a company — use single-query path."""
    q = _normalize_question(question).lower()
    return any(
        w in q for w in [
            "oil price", "oil prices", "crude oil", "eia", "inventory", "inventories",
            "natural gas", "commodity", "commodities", "futures price", "barrel", "gasoline",
        ]
    )


def decompose_question(question: str) -> List[Dict[str, Any]]:
    """
    Split a narrative question into 2-4 sub-questions for parallel execution.
    Returns list of { "label", "sub_question", "collections", "edge_path" }.
    For "what drove X's success" / "why is X up" we always include market_data, contracts, sec_filings, options (when ticker present).
    Returns [] for commodity/EIA questions so single-query path handles them (EIA → futures).
    """
    if is_commodity_eia_question(question):
        return []
    ticker = extract_ticker_from_question(question)
    ticker_str = ticker if ticker else "the company in question"
    placeholders = {"ticker": ticker_str}
    q_lower = _normalize_question(question).lower()
    out = []

    # Narrative with ticker: include market_data, contracts, sec_filings, options (full story)
    if ticker:
        for label in ["market_data", "contracts", "sec_filings", "options_flow"]:
            for defn in SUB_INTENTS:
                if defn["label"] == label:
                    out.append({
                        "label": defn["label"],
                        "sub_question": defn["sub_question_template"].format(**placeholders),
                        "collections": defn["collections"],
                        "edge_path": defn["edge_path"],
                        "bind_ticker": ticker if label != "macro" else None,
                    })
                    break
    # Macro: add when no ticker or to round out narrative
    if not ticker or len(out) < 4:
        for defn in SUB_INTENTS:
            if defn["label"] == "macro":
                out.append({
                    "label": defn["label"],
                    "sub_question": defn["sub_question_template"].format(**placeholders),
                    "collections": defn["collections"],
                    "edge_path": defn["edge_path"],
                    "bind_ticker": None,
                })
                break

    seen = set()
    deduped = []
    for item in out:
        if item["label"] not in seen:
            seen.add(item["label"])
            deduped.append(item)
    return deduped[:5]  # cap at 5 (market_data, contracts, sec_filings, options_flow, macro)
