"""
LLM query planning for FastAPI
Ported from Streamlit llm.py
"""
from openai import OpenAI
import json
import re
from datetime import datetime
import sys
import os
from typing import Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from app.llm.prompts import CRITICAL_AQL_RULES

# Initialize OpenAI client lazily to avoid import-time errors
def get_openai_client():
    """Get or create OpenAI client"""
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not configured")

    # Temporarily remove proxy environment variables to avoid httpx conflicts
    old_http_proxy = os.environ.pop('HTTP_PROXY', None)
    old_https_proxy = os.environ.pop('HTTPS_PROXY', None)
    old_http_proxy_lower = os.environ.pop('http_proxy', None)
    old_https_proxy_lower = os.environ.pop('https_proxy', None)

    try:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        return client
    finally:
        # Restore proxy settings
        if old_http_proxy:
            os.environ['HTTP_PROXY'] = old_http_proxy
        if old_https_proxy:
            os.environ['HTTPS_PROXY'] = old_https_proxy
        if old_http_proxy_lower:
            os.environ['http_proxy'] = old_http_proxy_lower
        if old_https_proxy_lower:
            os.environ['https_proxy'] = old_https_proxy_lower

# Query cache for similarity checking (in-memory for now)
# In production, consider Redis or proper caching layer
_query_cache = []
MAX_CACHE_SIZE = 50  # Keep last 50 queries


def get_relevant_schema(question, intent):
    """
    Return only relevant schema sections based on question.
    Ported from working Streamlit version - CRITICAL for correct query generation!
    """
    schemas = {
        "Award": {
            "description": "Government contract awards with semantic search",
            "key_fields": [
                "ticker", "award_amount_float", "start_date", "end_date",
                "awarding_agency", "recipient_name", "description",
                "description_embedding (for semantic search)",
                "contract_year", "award_id"
            ],
            "critical_notes": "✅ HAS description_embedding - use COSINE_SIMILARITY for semantic search",
            "sample_query": "FOR doc IN Award FILTER doc.ticker == @ticker SORT doc.award_amount_float DESC LIMIT 20 RETURN doc"
        },
        "Company": {
            "description": "Company master data - CRITICAL: camelCase field names! - ACTUAL DB FIELDS",
            "key_fields": [
                "ticker (str)", "company (str)", "sector (str)", "industry (str)",
                "country (str)", "city (str)", "cik (str)", "website (str)",
                "sharesOutstanding (int - camelCase!)", "marketCap (int - camelCase!)",
                "fullTimeEmployees (int - camelCase!)",
                "sp500_member (bool - snake_case!)",
                "lastUpdated (str)", "recordCount (int)"
            ],
            "critical_notes": "⚠️ Mix of camelCase (financial fields) and snake_case (sp500_member)",
            "sample_query": "FOR c IN Company FILTER c.ticker == @ticker RETURN c"
        },
        "MarketData": {
            "description": "Daily OHLCV + technical/fundamental indicators",
            "key_fields": [
                "ticker", "date", "open", "high", "low", "close", "volume",
                "sma_20, sma_50, sma_200 (snake_case)",
                "ema_12, ema_26",
                "macd, macd_signal, macd_histogram",
                "golden_cross, death_cross",
                "above_sma20, above_sma50",
                "targetMeanPrice (camelCase)",
                "forwardEps, trailingPE (camelCase)",
                "beta, dividendYield"
            ],
            "critical_notes": "Mix of snake_case (technical) and camelCase (fundamentals)",
            "sample_query": "FOR m IN MarketData FILTER m.ticker == @ticker AND m.date >= DATE_SUBTRACT(DATE_NOW(), 180, 'day') SORT m.date DESC LIMIT 100 RETURN m"
        },
        "EconomicData": {
            "description": "Macroeconomic indicators (all snake_case)",
            "key_fields": [
                "date", "sandp_500_index", "federal_funds_rate",
                "unemployment_rate", "10y_2y_treasury_spread",
                "yield_curve_inverted", "vix_volatility_index"
            ],
            "critical_notes": "All fields use snake_case",
            "sample_query": "FOR e IN EconomicData FILTER e.date >= @start_date SORT e.date DESC RETURN e"
        },
        "sec_filings": {
            "description": "SEC filing metadata with aggregated sentiment (NO embeddings!)",
            "key_fields": [
                "ticker", "type", "filing_date", "fiscal_year",
                "avg_finbert", "avg_uncertainty", "avg_negative",
                "sentence_count", "form_type"
            ],
            "critical_notes": "❌ NO embeddings - use text filters only",
            "sample_query": "FOR f IN sec_filings FILTER f.ticker == @ticker AND f.type == '10-K' SORT f.filing_date DESC LIMIT 20 RETURN f"
        },
        "sec_sentences": {
            "description": "Individual SEC sentences with sentiment (NO embeddings!)",
            "key_fields": [
                "text", "finbert_score", "finbert_probs",
                "negative_per_1k", "positive_per_1k",
                "uncertainty_per_1k", "section_id"
            ],
            "critical_notes": "❌ NO embeddings - use CONTAINS(LOWER(doc.text), 'keyword')",
            "sample_query": "FOR doc IN sec_sentences FILTER CONTAINS(LOWER(doc.text), @keyword) AND doc.finbert_score < -0.3 LIMIT 20 RETURN doc"
        },
        "prediction_markets_polymarket": {
            "description": "Polymarket prediction market data with semantic search - ACTUAL DB FIELDS",
            "key_fields": [
                "_key (str)", "condition_id (str)", "question (str)", "description (str)",
                "market_slug (str)", "end_date (str)", "category (str)",
                "volume (float)", "volume_24h (int - NOTE: integer!)", "liquidity (float)",
                "closed (bool)", "outcomes (list)", "outcome_prices (list)",
                "yes_probability (float)", "no_probability (float)",
                "question_embedding (array[1536] - for semantic search!)",
                "fetched_at (str)"
            ],
            "critical_notes": "✅ HAS question_embedding - use COSINE_SIMILARITY(doc.question_embedding, @query_vector) for semantic | Keyword search: CONTAINS(LOWER(doc.question), 'keyword') | volume_24h is INT not float!",
            "sample_query": "FOR m IN prediction_markets_polymarket FILTER m.closed == false AND m.category == @category SORT m.volume_24h DESC LIMIT 20 RETURN m"
        },
        "polymarket_traders": {
            "description": "Polymarket trader data - ACTUAL DB FIELDS",
            "key_fields": [
                "_key (str)", "address (str)", "trader_key (str)",
                "total_volume (float)", "total_trades (int)", "total_profit (float)",
                "is_whale (bool)", "volume_rank (int)", "avg_position_size (float)",
                "activity_level (str)", "profit_ratio (float)", "is_profitable (int)",
                "fetched_at (str)", "updated_at (str)"
            ],
            "critical_notes": "Use for trader analysis, whales, top performers",
            "sample_query": "FOR t IN polymarket_traders FILTER t.is_whale == true SORT t.total_volume DESC LIMIT 20 RETURN t"
        },
        "polymarket_positions": {
            "description": "Polymarket trader positions - ACTUAL DB FIELDS",
            "key_fields": [
                "_key (str)", "position_id (str)", "trader_address (str)", "trader_key (str)",
                "market_condition_id (str)", "market_key (str)", "market_question (str)",
                "outcome_index (int)", "size (float)", "average_price (float)",
                "realized_profit (float)", "unrealized_profit (int - NOTE: integer!)",
                "fetched_at (str)", "updated_at (str)"
            ],
            "critical_notes": "Links traders to markets, use for position analysis | unrealized_profit is INT",
            "sample_query": "FOR p IN polymarket_positions FILTER p.trader_key == @trader_key SORT p.size DESC LIMIT 20 RETURN p"
        },
        "polymarket_price_history": {
            "description": "Polymarket historical price snapshots - ACTUAL DB FIELDS",
            "key_fields": [
                "_key (str)", "market_id (str)", "condition_id (str)",
                "timestamp (int)", "datetime (str)",
                "yes_price (float)", "no_price (float)",
                "volume (float)", "volume_24h (float)", "liquidity (float)"
            ],
            "critical_notes": "Time-series data for market probability changes",
            "sample_query": "FOR h IN polymarket_price_history FILTER h.market_id == @market_id SORT h.timestamp DESC LIMIT 100 RETURN h"
        },
        "prediction_markets_kalshi": {
            "description": "Kalshi prediction market data with semantic search",
            "key_fields": [
                "title", "yes_price", "no_price", "volume",
                "status", "category", "close_time",
                "title_embedding (array[1536] - for semantic search!)"
            ],
            "critical_notes": "✅ HAS title_embedding - use COSINE_SIMILARITY(doc.title_embedding, @query_vector) for semantic | Keyword: CONTAINS(LOWER(doc.title), 'keyword')",
            "sample_query": "FOR m IN prediction_markets_kalshi FILTER m.status == 'active' SORT m.volume DESC LIMIT 20 RETURN m"
        },
        "commodity_positions": {
            "description": "CFTC Commitments of Traders data",
            "key_fields": [
                "ticker", "Market_and_Exchange_Names (Capital M!)",
                "as_of_date", "Noncommercial_Positions_Long_All",
                "Noncommercial_Positions_Short_All"
            ],
            "critical_notes": "⚠️ Some fields have Capital letters!",
            "sample_query": "FOR c IN commodity_positions FILTER c.ticker == @ticker SORT c.as_of_date DESC LIMIT 50 RETURN c"
        }
    }

    # Smart detection: which collections are relevant to this question?
    question_lower = question.lower()
    relevant_schemas = []

    # Intent-based selection
    if intent and intent.get("type") == "ticker":
        # For ticker queries, include Company + MarketData (NOT Award unless explicitly mentioned)
        relevant_schemas.extend(["Company", "MarketData"])
        print(f"[SCHEMA SELECTION] Ticker query detected: {intent.get('value')}")

    # Keyword-based detection for government contracts
    if any(word in question_lower for word in ['contract', 'award', 'government', 'federal', 'usaspending']):
        if "Award" not in relevant_schemas:
            relevant_schemas.append("Award")
            print(f"[SCHEMA SELECTION] Award collection added - contract keyword detected")

    # Geopolitical queries - search government contracts with semantic embeddings
    geopolitical_keywords = [
        'iran', 'china', 'russia', 'north korea', 'syria', 'ukraine', 'taiwan',
        'conflict', 'war', 'military', 'defense', 'geopolitical', 'sanctions',
        'bomb', 'strike', 'invasion', 'attack', 'troops', 'deployment'
    ]
    if any(word in question_lower for word in geopolitical_keywords):
        if "Award" not in relevant_schemas:
            relevant_schemas.append("Award")
            print(f"[SCHEMA SELECTION] Award collection added - geopolitical keyword detected")
        if "prediction_markets_polymarket" not in relevant_schemas:
            relevant_schemas.append("prediction_markets_polymarket")
            print(f"[SCHEMA SELECTION] Polymarket added - geopolitical topic detected")
        if "sec_filings" not in relevant_schemas:
            relevant_schemas.append("sec_filings")
            print(f"[SCHEMA SELECTION] SEC filings added - defense contractors may mention topic")

    # Trader-specific queries
    if any(word in question_lower for word in ['trader', 'whale', 'position', 'top trader', 'biggest bet']):
        for coll in ["polymarket_traders", "polymarket_positions"]:
            if coll not in relevant_schemas:
                relevant_schemas.append(coll)

    if any(word in question_lower for word in ['stock', 'price', 'market data', 'technical', 'sma', 'ema', 'macd']):
        if "MarketData" not in relevant_schemas:
            relevant_schemas.append("MarketData")

    if any(word in question_lower for word in ['sec', 'filing', '10-k', '10-q', 'sentiment', 'risk', 'uncertainty']):
        relevant_schemas.extend([s for s in ["sec_filings", "sec_sentences"] if s not in relevant_schemas])

    if any(word in question_lower for word in ['polymarket', 'prediction market', 'betting', 'odds', 'probability']):
        if "prediction_markets_polymarket" not in relevant_schemas:
            relevant_schemas.append("prediction_markets_polymarket")

    # Semantic prediction market queries (concepts, not specific keywords)
    if any(phrase in question_lower for phrase in ['markets about', 'predictions about', 'betting on', 'markets related to', 'markets concerning']):
        if "prediction_markets_polymarket" not in relevant_schemas:
            relevant_schemas.append("prediction_markets_polymarket")

    if any(word in question_lower for word in ['kalshi', 'event contract']):
        if "prediction_markets_kalshi" not in relevant_schemas:
            relevant_schemas.append("prediction_markets_kalshi")

    if any(word in question_lower for word in ['commodity', 'cftc', 'futures', 'copper', 'gold', 'oil']):
        if "commodity_positions" not in relevant_schemas:
            relevant_schemas.append("commodity_positions")

    if any(word in question_lower for word in ['economy', 'economic', 'fed', 'unemployment', 's&p 500', 'gdp']):
        if "EconomicData" not in relevant_schemas:
            relevant_schemas.append("EconomicData")

    # If no matches, return MINIMAL core collections (no Award fallback!)
    if not relevant_schemas:
        relevant_schemas = ["Company", "MarketData"]
        print("[SCHEMA SELECTION] No keywords matched, using minimal default: Company + MarketData")

    # Format schema output
    schema_text = "RELEVANT COLLECTIONS:\n\n"
    for coll_name in relevant_schemas:
        if coll_name in schemas:
            schema = schemas[coll_name]
            schema_text += f"**{coll_name}** - {schema['description']}\n"
            schema_text += f"Fields: {', '.join(schema['key_fields'])}\n"
            schema_text += f"⚠️ {schema['critical_notes']}\n"
            schema_text += f"Example: {schema['sample_query']}\n\n"

    return schema_text


def check_similar_previous_question(question: str) -> Optional[dict]:
    """
    Check if this question is very similar to a recent one.
    Ported from working Streamlit - saves LLM API calls!

    Returns cached query plan if similarity > 95%, otherwise None
    """
    if not _query_cache:
        return None

    try:
        # Get embedding for current question
        current_embedding = get_query_embedding(question)
        if not current_embedding:
            return None

        # Check last 10 queries for similarity
        recent_queries = _query_cache[-10:] if len(_query_cache) > 10 else _query_cache

        for cached_item in reversed(recent_queries):
            if 'question' not in cached_item or 'plan' not in cached_item:
                continue

            past_question = cached_item['question']

            # Skip if exact same question
            if past_question.lower() == question.lower():
                continue

            # Get embedding for past question
            past_embedding = get_query_embedding(past_question)
            if not past_embedding:
                continue

            # Calculate cosine similarity
            try:
                import numpy as np
                similarity = np.dot(current_embedding, past_embedding) / (
                    np.linalg.norm(current_embedding) * np.linalg.norm(past_embedding)
                )

                # If >95% similar, reuse the plan
                if similarity > 0.95:
                    print(f"💡 Query cache HIT! Similar to: \"{past_question}\" (similarity: {similarity:.2%})")
                    return cached_item['plan']

            except ImportError:
                # Numpy not available, skip similarity check
                break

    except Exception as e:
        print(f"Similarity check error: {str(e)}")
        pass

    return None


def add_to_query_cache(question: str, plan: dict):
    """Add successful query plan to cache for future similarity checks"""
    global _query_cache

    _query_cache.append({
        'question': question,
        'plan': plan,
        'timestamp': datetime.now().isoformat()
    })

    # Keep cache size manageable
    if len(_query_cache) > MAX_CACHE_SIZE:
        _query_cache = _query_cache[-MAX_CACHE_SIZE:]


def preprocess_query(question: str) -> Optional[dict]:
    """
    Handle simple queries without LLM - rule-based preprocessing.
    Ported from working Streamlit for performance optimization.

    Returns query plan if handled by rules, otherwise None
    """
    question_lower = question.lower()

    # Pattern 1: Simple ticker lookup
    ticker_match = re.search(r'\b([A-Z]{2,5})\b', question)
    if ticker_match and len(question.split()) <= 6:
        ticker = ticker_match.group(1)

        # Awards for ticker
        if any(word in question_lower for word in ['award', 'contract', 'government']):
            return {
                "intent": "ticker_awards",
                "collections": ["Award"],
                "requires_embedding": False,
                "aql_query": "FOR doc IN Award FILTER doc.ticker == @ticker SORT doc.award_amount_float DESC LIMIT 20 RETURN doc",
                "bind_vars": {"ticker": ticker},
                "explanation": f"Simple ticker awards lookup for {ticker}"
            }

        # Market data for ticker
        if any(word in question_lower for word in ['price', 'stock', 'market']):
            return {
                "intent": "ticker_market_data",
                "collections": ["MarketData"],
                "requires_embedding": False,
                "aql_query": "FOR doc IN MarketData FILTER doc.ticker == @ticker SORT doc.date DESC LIMIT 100 RETURN doc",
                "bind_vars": {"ticker": ticker},
                "explanation": f"Simple market data lookup for {ticker}"
            }

    # Pattern 2: Count queries
    if question_lower.startswith("how many"):
        if "award" in question_lower or "contract" in question_lower:
            return {
                "intent": "count_awards",
                "collections": ["Award"],
                "requires_embedding": False,
                "aql_query": "RETURN LENGTH(Award)",
                "bind_vars": {},
                "explanation": "Count total awards in database"
            }

        if "market" in question_lower and "polymarket" in question_lower:
            return {
                "intent": "count_markets",
                "collections": ["prediction_markets_polymarket"],
                "requires_embedding": False,
                "aql_query": "FOR m IN prediction_markets_polymarket FILTER m.closed == false COLLECT WITH COUNT INTO count RETURN count",
                "bind_vars": {},
                "explanation": "Count active polymarket markets"
            }

    # Pattern 3: Latest/recent data
    if any(word in question_lower for word in ['latest', 'recent', 'newest', 'last']):
        if "award" in question_lower or "contract" in question_lower:
            return {
                "intent": "recent_awards",
                "collections": ["Award"],
                "requires_embedding": False,
                "aql_query": "FOR doc IN Award SORT doc.start_date DESC LIMIT 20 RETURN doc",
                "bind_vars": {},
                "explanation": "Fetch most recent awards"
            }

        if "polymarket" in question_lower or "prediction" in question_lower:
            return {
                "intent": "recent_markets",
                "collections": ["prediction_markets_polymarket"],
                "requires_embedding": False,
                "aql_query": "FOR m IN prediction_markets_polymarket FILTER m.closed == false SORT m.volume_24h DESC LIMIT 20 RETURN m",
                "bind_vars": {},
                "explanation": "Fetch most active prediction markets"
            }

    # Pattern 4: Category-based polymarket queries
    if "polymarket" in question_lower or "prediction market" in question_lower:
        for category in ['sports', 'politics', 'crypto', 'business']:
            if category in question_lower:
                return {
                    "intent": "polymarket_by_category",
                    "collections": ["prediction_markets_polymarket"],
                    "requires_embedding": False,
                    "aql_query": f"FOR m IN prediction_markets_polymarket FILTER m.closed == false AND LOWER(m.category) == '{category}' SORT m.volume_24h DESC LIMIT 20 RETURN m",
                    "bind_vars": {},
                    "explanation": f"Fetch active {category} prediction markets"
                }

    # Pattern 5: Top traders / whales
    if any(phrase in question_lower for phrase in ['top trader', 'whale', 'biggest trader', 'most profitable trader']):
        return {
            "intent": "top_traders",
            "collections": ["polymarket_traders"],
            "requires_embedding": False,
            "aql_query": "FOR t IN polymarket_traders FILTER t.is_whale == true SORT t.total_volume DESC LIMIT 20 RETURN t",
            "bind_vars": {},
            "explanation": "Fetch top traders by volume"
        }

    return None


def quick_intent_check(question: str):
    """Quick LLM call to determine if ticker or semantic query"""
    check_prompt = f"""Question: "{question}"

Is this asking about a TICKER SYMBOL or a CONCEPT?

TICKER: Question mentions a specific stock ticker (2-5 uppercase letters like AAPL, CMI, TSLA, FCX)
CONCEPT: Question asks about a topic/theme (AI, cybersecurity, renewable energy, copper, gold, etc.)

Examples:
- "CMI awards" → TICKER (CMI is Cummins stock ticker)
- "awards related to AI" → CONCEPT (AI = artificial intelligence topic)
- "TSLA in 2024" → TICKER
- "renewable energy contracts" → CONCEPT

Return JSON: {{"type": "ticker", "value": "CMI"}} or {{"type": "concept", "value": "artificial intelligence"}}
"""

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": check_prompt}],
            max_tokens=100,
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Intent check error: {str(e)}")
        return {"type": "unknown"}


def plan_query_with_llm(question: str, intent_hint=None, conversation_history: list = None):
    """
    Generate query plan from natural language question.
    RESTORED from working Streamlit version with full optimizations:
    - Rule-based preprocessing for simple queries
    - Query similarity caching
    - Full schema context
    - Comprehensive validation
    - Conversation history context (NEW)
    """
    # TRY 1: Rule-based preprocessing (fastest - no LLM call)
    rule_result = preprocess_query(question)
    if rule_result:
        print("✓ Handled by rule-based system (no LLM call)")
        add_to_query_cache(question, rule_result)  # Cache for future similarity checks
        return rule_result

    # TRY 2: Check similarity cache (fast - 1 LLM call for embedding)
    similar_plan = check_similar_previous_question(question)
    if similar_plan:
        print("✓ Reusing similar query plan from cache")
        return similar_plan

    # TRY 3: Full LLM planning (slowest - full query generation)
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Build conversation context with entity extraction
    history_context = ""
    extracted_entities = {"tickers": [], "companies": [], "collections": []}

    if conversation_history and len(conversation_history) > 0:
        history_context = "\n\n**📜 CONVERSATION CONTEXT (Recent messages):**\n"
        # Take last 3 exchanges (6 messages) for context
        recent_history = conversation_history[-6:]

        for msg in recent_history:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            metadata = msg.get('metadata', {})

            # Collect entities from metadata
            if metadata:
                if metadata.get('tickers'):
                    extracted_entities['tickers'].extend(metadata['tickers'])
                if metadata.get('companies'):
                    extracted_entities['companies'].extend(metadata['companies'])
                if metadata.get('collections'):
                    extracted_entities['collections'].extend(metadata['collections'])

            # Truncate long messages
            if len(content) > 300:
                content = content[:300] + "..."

            history_context += f"- {role.upper()}: {content}\n"
            if metadata.get('tickers'):
                history_context += f"  └─ Tickers mentioned: {', '.join(metadata['tickers'][:5])}\n"

        # Deduplicate entities
        extracted_entities['tickers'] = list(set(extracted_entities['tickers']))
        extracted_entities['companies'] = list(set(extracted_entities['companies']))
        extracted_entities['collections'] = list(set(extracted_entities['collections']))

        history_context += "\n⚠️ **ENTITY CONTEXT** (extracted from previous results):\n"
        if extracted_entities['tickers']:
            history_context += f"- Known tickers: {', '.join(extracted_entities['tickers'][:10])}\n"
        if extracted_entities['companies']:
            history_context += f"- Known companies: {', '.join(extracted_entities['companies'][:10])}\n"
        if extracted_entities['collections']:
            history_context += f"- Previously queried collections: {', '.join(extracted_entities['collections'])}\n"

        history_context += "\n⚠️ **REFERENCE DETECTION**:\n"
        history_context += "- If user says 'that company/ticker', 'them', 'it', 'same one' → use entities above\n"
        history_context += "- If user says 'compare with X', use first ticker from previous results as base\n"
        history_context += "- If user asks follow-up without entity → reuse previous ticker/company\n"

    # Add intent hint to prompt
    hint_text = ""
    if intent_hint:
        if intent_hint.get("type") == "ticker":
            hint_text = f"\n\n🎯 CONFIRMED: This is a TICKER query for '{intent_hint.get('value')}'. Use doc.ticker == @ticker"
        elif intent_hint.get("type") == "concept":
            hint_text = f"\n\n🎯 CONFIRMED: This is a CONCEPT/SEMANTIC query about '{intent_hint.get('value')}'. Use semantic search with embeddings."

    # CRITICAL FIX: Get relevant schema based on question content!
    relevant_schema = get_relevant_schema(question, intent_hint)

    planning_prompt = f"""You are a database query planner for ArangoDB.

{relevant_schema}

{CRITICAL_AQL_RULES}

CRITICAL: AQL CLAUSE ORDER MUST BE:
FOR → FILTER → SORT → LIMIT → RETURN
⚠️ LIMIT always comes BEFORE RETURN!

PROVEN WORKING EXAMPLES (USE THESE PATTERNS):

Example 1: "Show me the top 10 largest government contracts"
FOR doc IN Award
  FILTER doc.award_amount_float != null
  SORT doc.award_amount_float DESC
  LIMIT 10
  RETURN {{
    recipient: doc.recipient_name,
    amount: doc.award_amount_float,
    agency: doc.awarding_agency,
    description: SUBSTRING(doc.description, 0, 200)
  }}

Example 2: "What are the most active Polymarket prediction markets?"
FOR market IN prediction_markets_polymarket
  FILTER market.volume_24h > 0
  FILTER market.closed == false
  SORT market.volume_24h DESC
  LIMIT 10
  RETURN {{
    question: market.question,
    volume_24h: market.volume_24h,
    yes_probability: market.yes_probability,
    liquidity: market.liquidity
  }}

Example 3: "Find contracts related to artificial intelligence"
(REQUIRES EMBEDDING - set requires_embedding: true)
FOR doc IN Award
  FILTER doc.description_embedding != null
  LET similarity = COSINE_SIMILARITY(doc.description_embedding, @query_vector)
  FILTER similarity >= 0.70
  SORT similarity DESC
  LIMIT 10
  RETURN {{
    recipient: doc.recipient_name,
    amount: doc.award_amount_float,
    description: SUBSTRING(doc.description, 0, 300),
    similarity: similarity
  }}

Example 4: "What markets are whale traders betting on?"
FOR trader IN polymarket_traders
  FILTER trader.is_whale == true
  FOR position IN OUTBOUND trader trader_has_position
    FILTER position.size > 100
    FOR market IN OUTBOUND position position_in_market
      FILTER market.closed == false
      SORT position.size DESC
      LIMIT 20
      RETURN DISTINCT {{
        market_question: market.question,
        yes_probability: market.yes_probability,
        position_size: position.size,
        trader_volume: trader.total_volume
      }}

Example 5: "Show me Apple stock data for the last 30 days"
⚠️ CRITICAL: For time-series queries, LIMIT must match the requested days (30 days = LIMIT 30, 60 days = LIMIT 60, etc.)
FOR doc IN MarketData
  FILTER doc.ticker == @ticker
  FILTER doc.date >= DATE_SUBTRACT(DATE_NOW(), 30, "day")
  SORT doc.date DESC
  LIMIT 30  # ⚠️ MUST match the number of days requested!
  RETURN {{
    date: doc.date,
    close: doc.close,
    volume: doc.volume,
    sma_20: doc.sma_20
  }}

Example 6: "Find SEC filings with negative sentiment about cybersecurity"
FOR doc IN sec_sentences
  FILTER CONTAINS(LOWER(doc.text), "cybersecurity")
  FILTER doc.finbert_score < -0.3
  SORT doc.finbert_score ASC
  LIMIT 20
  RETURN {{
    text: SUBSTRING(doc.text, 0, 400),
    sentiment: doc.finbert_score,
    negative_per_1k: doc.negative_per_1k
  }}

{history_context}

USER QUESTION: "{question}"{hint_text}

Current Date: {current_date}

⚠️ **CRITICAL: DATE RANGE PARSING**
When user mentions a specific month/year or time period, convert to EXACT date ranges:

Examples:
- "October 2020" → start_date: "2020-10-01", end_date: "2020-11-01" (first day of next month)
- "Q1 2021" → start_date: "2021-01-01", end_date: "2021-04-01"
- "January 2024" → start_date: "2024-01-01", end_date: "2024-02-01"
- "2023" → start_date: "2023-01-01", end_date: "2024-01-01"
- "last 30 days" → use DATE_SUBTRACT(DATE_NOW(), 30, "day")

⚠️ NEVER use DATE_NOW() for historical queries! User asking about "October 2020" wants data from 2020, NOT recent data!

**INSTRUCTIONS:**
1. Match the user question to the closest example above
2. Adapt that example's pattern for the user's specific needs
3. Keep the same structure: FOR → FILTER → SORT → LIMIT → RETURN
4. Use EXACT field names from examples (award_amount_float, volume_24h, etc.)
5. For ticker queries, use bind variable: @ticker
6. For semantic search, MUST set requires_embedding: true
7. **For date ranges:** Parse natural language dates into specific YYYY-MM-DD strings

**QUERY OUTPUT STRATEGY:**
For time series queries, choose the right output format:

1. **Summary Format** (date range > 7 days OR user says "performance/summary"):
   - Return aggregated metrics (open, close, high, low, % change)
   - Single row with summary statistics
   - Use Example 9b pattern

2. **Daily Format** (date range <= 7 days OR user says "daily/detailed"):
   - Return individual rows per day
   - Use Example 9 pattern
   - Limit to reasonable number (30-50 rows max)

Generate a JSON response with:
- "intent": classification (e.g., "top_contracts", "active_markets", "semantic_awards", "whale_positions", "stock_data", "sec_sentiment")
- "collections": array of collection names used
- "requires_embedding": boolean (true ONLY if doing semantic/similarity search on Award or Polymarket)
- "embedding_text": text to embed (if requires_embedding is true)
- "aql_query": valid AQL query (MUST follow proven examples above)
- "bind_vars": object with bind variables (e.g., {{"ticker": "AAPL"}})
- "explanation": brief strategy explanation

CRITICAL VALIDATION CHECKLIST:
✅ Collection names EXACT: Award, prediction_markets_polymarket, polymarket_traders, MarketData, sec_sentences
✅ Field names EXACT: award_amount_float, volume_24h, yes_probability, finbert_score, closed
✅ NEVER use: award_amount (wrong), market.volume (wrong), markets (wrong)
✅ Date functions: DATE_SUBTRACT(DATE_NOW(), 30, "day") - NOT DATE_SUB()
✅ Boolean filters: market.closed == false (NOT market.closed = false)
✅ Text search: CONTAINS(LOWER(doc.text), "keyword") - always LOWER()
✅ Order MUST be: FOR → FILTER → SORT → LIMIT → RETURN
✅ LIMIT is REQUIRED and comes BEFORE RETURN
✅ For ticker queries, use @ticker bind variable
✅ Semantic search: requires_embedding: true, similarity >= 0.70 for Award

Return ONLY valid JSON, no markdown formatting.

Response:"""

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": planning_prompt}],
            max_tokens=config.MAX_TOKENS,
            temperature=config.TEMPERATURE,
            response_format={"type": "json_object"}
        )

        plan = json.loads(response.choices[0].message.content)

        # CRITICAL FIX: Validate and auto-fix AQL syntax before returning!
        if 'aql_query' in plan:
            try:
                corrected_query, errors, bind_params = validate_aql_syntax(plan['aql_query'], question)
                plan['aql_query'] = corrected_query  # Use corrected version
                if errors:
                    plan['validation_warnings'] = errors  # Add warnings to plan
                    print(f"Query validation: {len(errors)} warnings/fixes")
                    for error in errors:
                        print(f"  - {error}")
            except ValueError as ve:
                # Critical validation error - query is invalid
                print(f"CRITICAL validation error: {ve}")
                return None

        # Add successful plan to cache for future similarity checks
        add_to_query_cache(question, plan)

        return plan

    except Exception as e:
        print(f"\n{'='*80}")
        print("[ERROR] LLM QUERY PLANNING FAILED!")
        print(f"{'='*80}")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error message: {str(e)}")
        print(f"  Question: '{question}'")
        print(f"  Intent hint: {intent_hint}")
        print(f"{'='*80}\n")

        # Return structured error - let API layer handle user-facing message
        return {
            "error": True,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "question": question,
            "suggestion": "Try rephrasing your question or being more specific about what data you want"
        }


def get_query_embedding(text: str):
    """Generate embedding vector for semantic search"""
    try:
        client = get_openai_client()
        response = client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding generation error: {str(e)}")
        return None


def detect_time_series_query(results: list, query_plan: dict):
    """Detect if this is a time series query (stock prices, etc.)"""
    if not results:
        return False

    # Check if results have time series characteristics
    first_result = results[0]
    has_date = 'date' in first_result
    has_price_fields = any(field in first_result for field in ['close', 'open', 'high', 'low', 'price', 'volume'])

    # Check query plan collections
    collections = query_plan.get('collections', [])
    is_market_data = 'MarketData' in collections

    return has_date and has_price_fields and is_market_data


def format_time_series_analysis(user_question: str, results: list, query_plan: dict):
    """Format time series data with statistics and chart-ready format"""

    if not results:
        return "No data found for the specified time period."

    # Sort by date
    sorted_results = sorted(results, key=lambda x: x.get('date', ''))

    # Extract time series data
    dates = [r.get('date', '') for r in sorted_results]
    closes = [float(r.get('close', 0)) for r in sorted_results]
    volumes = [float(r.get('volume', 0)) for r in sorted_results]

    # Calculate statistics
    if closes:
        first_close = closes[0]
        last_close = closes[-1]
        price_change = last_close - first_close
        price_change_pct = (price_change / first_close * 100) if first_close > 0 else 0
        high_price = max(closes)
        low_price = min(closes)
        avg_price = sum(closes) / len(closes)
        avg_volume = sum(volumes) / len(volumes) if volumes else 0

        # Format summary
        ticker = sorted_results[0].get('ticker', 'Unknown')
        period_start = dates[0] if dates else 'N/A'
        period_end = dates[-1] if dates else 'N/A'

        change_indicator = "📈" if price_change >= 0 else "📉"

        analysis = f"""## {ticker} Stock Performance

**Period:** {period_start} to {period_end} ({len(sorted_results)} trading days)

### Price Summary
- **Starting Price:** ${first_close:.2f}
- **Ending Price:** ${last_close:.2f}
- **Change:** {change_indicator} ${price_change:+.2f} ({price_change_pct:+.2f}%)
- **High:** ${high_price:.2f}
- **Low:** ${low_price:.2f}
- **Average:** ${avg_price:.2f}

### Trading Activity
- **Average Daily Volume:** {avg_volume:,.0f} shares
- **Total Days:** {len(sorted_results)}

### Key Insights
"""

        # Add insights based on data
        if price_change_pct > 10:
            analysis += f"- Strong upward trend with {price_change_pct:.1f}% gain over the period\n"
        elif price_change_pct < -10:
            analysis += f"- Significant decline of {price_change_pct:.1f}% over the period\n"
        else:
            analysis += f"- Relatively stable price movement ({price_change_pct:+.1f}% change)\n"

        volatility = (high_price - low_price) / avg_price * 100
        if volatility > 15:
            analysis += f"- High volatility with {volatility:.1f}% price range\n"
        elif volatility < 5:
            analysis += f"- Low volatility with {volatility:.1f}% price range\n"

        # Add moving average if available
        if 'sma_20' in sorted_results[-1]:
            sma_20 = sorted_results[-1].get('sma_20')
            if sma_20:
                if last_close > sma_20:
                    analysis += f"- Currently trading above 20-day moving average (${sma_20:.2f})\n"
                else:
                    analysis += f"- Currently trading below 20-day moving average (${sma_20:.2f})\n"

        return analysis

    return "Unable to analyze price data."


def analyze_results_with_llm(user_question: str, results: list, query_plan: dict):
    """Analyze query results and generate natural language response"""

    print("\n" + "="*80)
    print("[LLM ANALYSIS] Starting analysis")
    print("="*80)
    print(f"Question: {user_question}")
    print(f"Result count: {len(results)}")

    if not results:
        print("[LLM ANALYSIS] No results found, using fallback response")
        return generate_no_results_response(user_question, query_plan)

    # Check if this is a time series query (stock prices)
    if detect_time_series_query(results, query_plan):
        print("[LLM ANALYSIS] Detected time series query - using specialized formatter")
        return format_time_series_analysis(user_question, results, query_plan)

    # Limit results sample for LLM analysis (avoid token limits)
    results_sample = results[:10] if len(results) > 10 else results
    result_count = len(results)

    analysis_prompt = f"""You are a financial data analyst. The user asked: "{user_question}"

We queried the database and found {result_count} results. Here's a sample:

{json.dumps(results_sample, indent=2)}

Please provide a clear, concise analysis in markdown format:

**CRITICAL: You MUST present the data in a markdown table format.**

Guidelines:
1. **Always start with a markdown table** showing the most relevant fields from the results
2. Include the top 10 rows (or all if less than 10)
3. Choose the most important columns (max 5-7 columns) - exclude internal fields like _id, _key, _rev
4. Format numbers with proper units ($, %, dates, etc.)
5. After the table, provide 2-3 key insights or observations
6. Keep the analysis professional and focused on answering the user's question

Markdown table format example:
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Value 1  | Value 2  | Value 3  |

Do NOT mention technical details like query execution or database operations.

Response:"""

    print(f"\n[LLM ANALYSIS] Prompt length: {len(analysis_prompt)} chars")
    print(f"[LLM ANALYSIS] Using model: {config.LLM_MODEL}")
    print(f"[LLM ANALYSIS] OpenAI API key set: {bool(config.OPENAI_API_KEY)}")
    print(f"[LLM ANALYSIS] Calling OpenAI...")

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": analysis_prompt}],
            max_tokens=1500,  # Increased for table generation
            temperature=0.3
        )

        analysis_result = response.choices[0].message.content.strip()

        print(f"\n[LLM ANALYSIS] ✓ OpenAI call successful!")
        print(f"[LLM ANALYSIS] Response length: {len(analysis_result)} chars")
        print(f"[LLM ANALYSIS] Response preview (first 200 chars):")
        print(f"  {analysis_result[:200]}...")
        print(f"[LLM ANALYSIS] Contains table markers: {'|' in analysis_result}")
        print("="*80 + "\n")

        return analysis_result

    except Exception as e:
        print(f"\n[LLM ANALYSIS] ✗ ERROR calling OpenAI!")
        print(f"[LLM ANALYSIS] Error type: {type(e).__name__}")
        print(f"[LLM ANALYSIS] Error message: {str(e)}")
        print(f"[LLM ANALYSIS] Falling back to simple JSON formatting")
        print("="*80 + "\n")
        # Fallback to simple formatting
        return f"Found {result_count} results. Here's a summary:\n\n" + "\n".join([f"- {json.dumps(r)[:100]}..." for r in results_sample[:5]])


def generate_no_results_response(user_question: str, query_plan: dict):
    """
    Generate helpful response when no results are found.
    ENHANCED from working Streamlit with more context and suggestions.
    """

    collections = query_plan.get('collections', [])
    intent = query_plan.get('intent', '')
    aql_query = query_plan.get('aql_query', '')

    # Extract any filter values from query for better diagnostics
    bind_vars = query_plan.get('bind_vars', {})
    filter_info = ""
    if bind_vars:
        filter_info = f"\n\nFilters applied: {json.dumps(bind_vars, indent=2)}"

    fallback_prompt = f"""The user asked: "{user_question}"

We searched our financial database but found no matching results.

**Query Details:**
- Collections searched: {', '.join(collections) if collections else 'unknown'}
- Intent: {intent}
{filter_info}

**Available data in our system:**
- **Market Data**: Stock prices, technical indicators, fundamentals for S&P 500 companies
  - Coverage: 2015-present
  - 40+ technical indicators (SMA, EMA, MACD, etc.)

- **Government Contracts**: Federal awards from USASpending.gov
  - Coverage: 2015-present
  - Includes contract amounts, agencies, recipients
  - Semantic search available

- **SEC Filings**: 10-K and 10-Q filings with sentiment analysis
  - Coverage: Recent years
  - FinBERT sentiment scores
  - Risk and uncertainty metrics

- **Prediction Markets**: Polymarket and Kalshi
  - Live prediction market data
  - Categories: Politics, Sports, Crypto, World Events, Business
  - Probabilities, volume, liquidity

- **Economic Data**: Macroeconomic indicators from FRED
  - S&P 500 index, Fed funds rate, unemployment
  - Treasury yields, yield curve

- **Commodities**: CFTC Commitments of Traders
  - Copper, gold, oil, etc.
  - Trader positions (long/short)

**Please provide a helpful response that:**
1. Explains what the user was likely looking for
2. Diagnoses why no results were found:
   - Wrong ticker symbol? (e.g., "TSLA" not "Tesla")
   - Data not in coverage period?
   - Collection doesn't have that type of data?
   - Query too restrictive (date range, filters)?
3. Provides 2-3 **specific** alternative queries they could try:
   - Suggest actual ticker symbols if relevant
   - Suggest broadening filters
   - Suggest different collections
4. If appropriate, provide general context about their question using your knowledge
5. Be encouraging and helpful, not dismissive

**Important:**
- If they asked about a company by name, suggest the ticker symbol
- If they asked about recent data, note our coverage dates
- If they asked about prediction markets, mention available categories
- If query was too specific, suggest broadening it

Keep it conversational, helpful, and professional. Use markdown formatting with clear sections.

Response:"""

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": fallback_prompt}],
            max_tokens=1000,
            temperature=0.6
        )

        return "**No results found in database.**\n\n" + response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Fallback response error: {str(e)}")

        # Enhanced fallback if LLM fails
        ticker_hint = ""
        if bind_vars.get('ticker'):
            ticker_hint = f"\n- You searched for ticker: **{bind_vars['ticker']}** - Make sure this is the correct stock symbol"

        return f"""**No results found in database.**

**What we searched:**
- Collections: {', '.join(collections) if collections else 'database'}
- Intent: {intent}
{ticker_hint}

**Common issues:**
1. **Ticker symbols** must be exact (e.g., "AAPL" not "Apple")
2. **Date ranges** might be too restrictive (try last 6 months instead of last week)
3. **Company names** - use ticker symbols instead
4. **Prediction markets** - try broader categories (Sports, Politics, Crypto)

**Try these alternatives:**
- 📊 Browse available prediction markets by category
- 💼 Search for S&P 500 companies by ticker
- 📈 Check market data for popular stocks (AAPL, MSFT, GOOGL, TSLA)
- 🏛️ Search government contracts by company name (not ticker)

**Need help?** Ask me to show you available data or suggest a related query!"""


def validate_aql_syntax(aql_query: str, question: str = ""):
    """
    Comprehensive AQL syntax validation - ported from working Streamlit version.
    Returns: (corrected_query, errors_list, bind_params_set)
    """
    errors = []

    # CRITICAL ERROR: Check for UNION (not supported in AQL)
    if 'UNION' in aql_query.upper():
        errors.append("❌ CRITICAL: AQL does not support UNION! Use nested FOR loops instead.")
        raise ValueError("AQL does not support UNION syntax. Use nested FOR loops or MERGE() to combine data.")

    # CRITICAL ERROR: Check for JOIN (not supported in AQL)
    if re.search(r'\bJOIN\b', aql_query, re.IGNORECASE):
        errors.append("❌ CRITICAL: AQL does not support JOIN! Use nested FOR loops instead.")
        raise ValueError("AQL does not support JOIN syntax. Use nested FOR loops to connect data.")

    # CRITICAL ERROR: Check for embeddings on wrong collections
    # Award, prediction_markets_polymarket, and prediction_markets_kalshi have embeddings
    collections_without_embeddings = [
        'sec_filings', 'sec_sections', 'sec_sentences',
        'Company', 'MarketData', 'EconomicData', 'commodity_positions',
        'polymarket_traders', 'polymarket_positions', 'polymarket_price_history',
        'trader_has_position', 'position_in_market',
        'market_mentions_company_polymarket', 'market_related_to_sector_polymarket',
        'market_affects_company_polymarket'
    ]

    for collection in collections_without_embeddings:
        if collection in aql_query and 'COSINE_SIMILARITY' in aql_query:
            errors.append(f"❌ CRITICAL: {collection} does NOT have embeddings!")
            raise ValueError(f"Collection '{collection}' does not have embeddings. Only Award (description_embedding) and prediction_markets_polymarket (question_embedding) have embeddings. Use CONTAINS(LOWER(doc.field), 'keyword') for text search.")

    # Auto-fix: Common field name mistakes
    field_fixes = {
        r'\.award_amount\b(?!_float)': '.award_amount_float',  # award_amount → award_amount_float
        r'\.volume\b(?!_24h)': '.volume_24h',  # volume → volume_24h (for markets)
        r'market\.closed\s*=\s*false': 'market.closed == false',  # = → ==
        r'doc\.closed\s*=\s*false': 'doc.closed == false',  # = → ==
        r'DATE_SUB\(': 'DATE_SUBTRACT(',  # DATE_SUB → DATE_SUBTRACT
    }

    for wrong_pattern, correct in field_fixes.items():
        if re.search(wrong_pattern, aql_query):
            aql_query = re.sub(wrong_pattern, correct, aql_query)
            errors.append(f"✅ Auto-fixed field/syntax: {wrong_pattern} → {correct}")

    # Auto-fix: Collection name mistakes (with word boundaries to prevent double-replacement)
    replacements = {
        r'\bawards\b': 'Award',
        r'\bAwards\b': 'Award',
        r'\bCompanies\b': 'Company',
        r'\bcompanies\b': 'Company',
        r'\bmarket_data\b': 'MarketData',
        r'\bMarketDatas\b': 'MarketData',
        r'\bfred_data\b': 'EconomicData',
        r'\bFREDData\b': 'EconomicData',
        r'\bcommodity_position\b': 'commodity_positions',
        r'\bsec_filing\b': 'sec_filings',
        r'\bsec_section\b': 'sec_sections',
        r'\bsec_sentence\b': 'sec_sentences',
        r'\bpolymarket\b(?!_)': 'prediction_markets_polymarket',
        r'\bkalshi\b(?!_)': 'prediction_markets_kalshi',
    }

    for wrong_pattern, correct in replacements.items():
        if re.search(wrong_pattern, aql_query, flags=re.IGNORECASE):
            aql_query = re.sub(wrong_pattern, correct, aql_query, flags=re.IGNORECASE)
            wrong_word = wrong_pattern.replace(r'\b', '').replace(r'(?!_)', '')
            errors.append(f"✅ Auto-fixed: '{wrong_word}' → '{correct}'")

    # Extract bind parameters
    bind_params = set(re.findall(r'@(\w+)', aql_query))

    return aql_query, errors, bind_params


def generate_smart_suggestions(results: list, query_plan: dict, user_question: str) -> list:
    """
    Generate smart contextual suggestions based on query results
    Analyzes collections, entities found, and suggests related queries
    """
    if not results or len(results) == 0:
        return []

    suggestions = []
    collections = query_plan.get('collections', [])
    bind_vars = query_plan.get('bind_vars', {})

    # Extract entities from results
    tickers = set()
    companies = set()

    for result in results[:10]:  # Check first 10 results
        if result.get('ticker'):
            tickers.add(result['ticker'])
        if result.get('recipient_name'):
            companies.add(result['recipient_name'])
        if result.get('company'):
            companies.add(result['company'])

    tickers = list(tickers)[:3]  # Top 3 tickers
    companies = list(companies)[:2]  # Top 2 companies

    # Collection-based suggestions
    if 'Award' in collections:
        if tickers:
            suggestions.append(f"Show prediction markets about {tickers[0]}")
            suggestions.append(f"What's the stock performance of {tickers[0]}?")
        suggestions.append("Find contracts related to cybersecurity")

    if 'MarketData' in collections:
        if tickers:
            other_tickers = ['MSFT', 'GOOGL', 'AMZN']
            for t in other_tickers:
                if t not in tickers:
                    suggestions.append(f"Compare with {t}")
                    break
            suggestions.append(f"Show government contracts for {tickers[0]}")

    if 'prediction_markets_polymarket' in collections:
        suggestions.append("What markets are whale traders betting on?")
        if tickers:
            suggestions.append(f"Show {tickers[0]} stock data")

    if 'polymarket_traders' in collections or 'polymarket_positions' in collections:
        suggestions.append("Show the most active Polymarket markets")
        suggestions.append("Find prediction markets with high volume")

    if 'sec_sentences' in collections or 'sec_filings' in collections:
        if tickers:
            suggestions.append(f"Show {tickers[0]} stock performance")
        suggestions.append("Find companies with negative earnings sentiment")

    # Trim to 3-4 suggestions
    return suggestions[:4]


def generate_follow_up_questions(user_question: str, results: list, query_plan: dict):
    """
    Generate dynamic, contextual follow-up questions using LLM.
    Like ChatGPT/Perplexity - relevant to what was just shown.
    """
    # Skip if no results
    if not results or len(results) == 0:
        return []

    # First try smart rule-based suggestions
    smart_suggestions = generate_smart_suggestions(results, query_plan, user_question)
    if smart_suggestions:
        return smart_suggestions

    collections = query_plan.get('collections', [])

    # Prepare context for LLM
    result_summary = f"{len(results)} results from {', '.join(collections)}"

    # Sample of results (first 3 to avoid token limits)
    result_sample = results[:3]

    # Get field names from results
    available_fields = []
    if results and isinstance(results[0], dict):
        available_fields = list(results[0].keys())[:10]  # First 10 fields

    prompt = f"""You are a financial data analyst helping users explore data.

User just asked: "{user_question}"

Results returned: {result_summary}
Available data fields: {', '.join(available_fields)}
Sample data: {json.dumps(result_sample, indent=2)[:500]}

Generate 3-4 natural follow-up questions that:
1. **Deepen the analysis** (e.g., "How does this compare to industry average?")
2. **Expand time range** (e.g., "Show me this over the past year")
3. **Compare entities** (e.g., "Compare AAPL with MSFT and GOOGL")
4. **Cross-reference data** (e.g., "What do prediction markets say about this?")

Rules:
- Be specific (use actual tickers/entities from results)
- Be actionable (user can directly ask your suggestion)
- Be relevant (based on what data is available)
- NO generic questions like "Tell me more"
- NO emojis
- Each question should be 8-15 words

Return JSON array of 3-4 questions:
["question 1", "question 2", "question 3"]
"""

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Fast model for follow-ups
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        # Extract questions from various possible response formats
        if 'questions' in result:
            follow_ups = result['questions'][:4]
        elif isinstance(result, list):
            follow_ups = result[:4]
        else:
            # Fallback if response format is unexpected
            follow_ups = list(result.values())[:4] if result else []

        print(f"[FOLLOW-UPS] Generated {len(follow_ups)} contextual questions")
        return follow_ups

    except Exception as e:
        print(f"[FOLLOW-UPS] Error generating questions: {e}")
        print(f"[FOLLOW-UPS] Question: '{user_question}'")
        print(f"[FOLLOW-UPS] Results count: {len(results)}")
        import traceback
        print(f"[FOLLOW-UPS] Traceback: {traceback.format_exc()}")
        # Return empty list on error
        return []
