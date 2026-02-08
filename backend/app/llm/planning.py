"""
LLM query planning for FastAPI
Ported from Streamlit llm.py
"""
from openai import OpenAI
import json
import re
import copy
from datetime import datetime
import sys
import os
from typing import Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from app.llm.prompts import CRITICAL_AQL_RULES, build_json_intent_prompt
from app.llm.json_to_aql import json_to_aql
from app.llm.response_synthesis import get_enhanced_analysis_prompt
# Schema grounding kept for potential future use; two-step flow uses get_cached_schema + build_json_intent_prompt
from app.llm.query_validation import execute_with_validation
from app.llm.schema_introspection import (
    get_cached_schema,
    get_collection_schema_dynamic,
    format_collection_for_prompt,
    get_relevant_collections_dynamic
)

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
    NOW USES DYNAMIC RUNTIME INTROSPECTION - no hard-coded schemas!
    """
    # Get relevant collections using dynamic detection
    relevant_collections = get_relevant_collections_dynamic(question)

    # Add intent-based collections
    if intent and intent.get("type") == "ticker":
        # For ticker queries, ensure Company + MarketData are included
        if "Company" not in relevant_collections:
            relevant_collections.insert(0, "Company")
        if "MarketData" not in relevant_collections:
            relevant_collections.insert(1, "MarketData")
        print(f"[SCHEMA SELECTION] Ticker query detected: {intent.get('value')}")

    # Geopolitical queries - add additional collections
    geopolitical_keywords = [
        'iran', 'china', 'russia', 'north korea', 'syria', 'ukraine', 'taiwan',
        'conflict', 'war', 'military', 'defense', 'geopolitical', 'sanctions',
        'bomb', 'strike', 'invasion', 'attack', 'troops', 'deployment'
    ]
    question_lower = question.lower()
    if any(word in question_lower for word in geopolitical_keywords):
        for coll in ["Award", "prediction_markets_polymarket", "sec_filings"]:
            if coll not in relevant_collections:
                relevant_collections.append(coll)
        print(f"[SCHEMA SELECTION] Geopolitical query - added Award, Polymarket, SEC filings")

    print(f"[SCHEMA SELECTION] Selected collections: {relevant_collections}")

    # Get cached schema
    full_schema = get_cached_schema()

    # Format schema output
    schema_text = "RELEVANT COLLECTIONS:\n\n"
    for coll_name in relevant_collections:
        coll_schema = full_schema['collections'].get(coll_name)
        if coll_schema:
            schema_text += format_collection_for_prompt(coll_name, coll_schema)

            # Add sample query if available (for common collections)
            sample_queries = {
                "Company": "FOR c IN Company FILTER c.ticker == @ticker RETURN c",
                "MarketData": "FOR m IN MarketData FILTER m.ticker == @ticker AND m.date >= DATE_SUBTRACT(DATE_NOW(), 180, 'day') SORT m.date DESC LIMIT 100 RETURN m",
                "Award": "FOR doc IN Award FILTER doc.ticker == @ticker SORT doc.award_amount_float DESC LIMIT 20 RETURN doc",
                "sec_filings": "FOR f IN sec_filings FILTER f.ticker == @ticker AND f.type == '10-K' SORT f.filing_date DESC LIMIT 20 RETURN f"
            }
            if coll_name in sample_queries:
                schema_text += f"Example: {sample_queries[coll_name]}\n"

            schema_text += "\n"

    # If no collections found, use default
    if not relevant_collections:
        schema_text += "Using default: Company, MarketData\n"
        schema_text += format_collection_for_prompt("Company", full_schema['collections'].get("Company", {}))
        schema_text += format_collection_for_prompt("MarketData", full_schema['collections'].get("MarketData", {}))

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


def _schema_for_json_intent(full_schema: dict) -> dict:
    """Convert get_cached_schema() format to JSON intent prompt format (collections with fields list, edges with from/to strings)."""
    collections_out = {}
    for coll_name, coll_data in full_schema.get("collections", {}).items():
        fields = coll_data.get("fields", {})
        if isinstance(fields, dict):
            fields = list(fields.keys())
        collections_out[coll_name] = {
            "fields": fields,
            "description": f"{coll_name} collection"
        }
    edges_out = {}
    for edge_name, edge_data in full_schema.get("edges", {}).items():
        from_coll = edge_data.get("from")
        to_coll = edge_data.get("to")
        if isinstance(from_coll, list):
            from_coll = from_coll[0] if from_coll else ""
        if isinstance(to_coll, list):
            to_coll = to_coll[0] if to_coll else ""
        edges_out[edge_name] = {"from": from_coll, "to": to_coll}
    return {"collections": collections_out, "edges": edges_out}


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


# Company workup AQL: single-company with MarketData, sec_filings, sec_xbrl_data, etc.
# Use when user asks for "X financials" / "X's financials" to avoid JSON plan producing date+average aggregation.
COMPANY_WORKUP_AQL = """
FOR company IN Company
  FILTER company.ticker == @ticker
  LIMIT 1

  LET market_data = (
    FOR m IN OUTBOUND company HAS_MARKETDATA
      SORT m.date DESC
      LIMIT 365
      RETURN m
  )

  LET sec_filings = (
    FOR filing IN OUTBOUND company HAS_FILING
      SORT filing.filing_date DESC
      LIMIT 20
      LET top_sentences = (
        FOR section IN OUTBOUND filing has_section
          FOR sentence IN OUTBOUND section has_sentence
            FILTER sentence.finbert_score != null
            SORT ABS(sentence.finbert_score) DESC
            LIMIT 10
            RETURN {
              text: sentence.text,
              score: sentence.finbert_score
            }
      )
      RETURN MERGE(filing, { top_sentences: top_sentences })
  )

  LET sec_exhibits = (
    FOR filing IN OUTBOUND company HAS_FILING
      FOR exhibit IN OUTBOUND filing has_exhibit
        SORT exhibit.filing_date DESC
        LIMIT 20
        RETURN exhibit
  )

  LET sec_xbrl_data = (
    FOR filing IN OUTBOUND company HAS_FILING
      FOR xbrl IN OUTBOUND filing has_xbrl_data
        SORT xbrl.filing_date DESC
        LIMIT 20
        RETURN xbrl
  )

  LET awards = (
    FOR award IN OUTBOUND company HAS_AWARD
      SORT award.start_date DESC
      LIMIT 20
      RETURN award
  )

  LET options_flow = (
    FOR opt IN OUTBOUND company COMPANY_HAS_OPTIONS
      SORT opt.date DESC
      LIMIT 20
      RETURN opt
  )

  RETURN MERGE(company, {
    MarketData: market_data,
    sec_filings: sec_filings,
    sec_exhibits: sec_exhibits,
    sec_xbrl_data: sec_xbrl_data,
    Award: awards,
    options_flow: options_flow
  })
"""


def preprocess_query(question: str) -> Optional[dict]:
    """
    Handle simple queries without LLM - rule-based preprocessing.
    Ported from working Streamlit for performance optimization.

    Returns query plan if handled by rules, otherwise None
    """
    question_lower = question.lower().strip()

    # Pattern 0: Company financials (balance sheet, income statement, 10-K, "X's financials")
    # Bypass JSON->AQL so we never get wrong aggregation (date + average). Use full workup AQL.
    financial_triggers = [
        "financial", "financials", "balance sheet", "income statement",
        "financial statements", "10-k", "10k", "10-k financials", "cash flow"
    ]
    if any(trigger in question_lower for trigger in financial_triggers):
        ticker = None
        # "PPGs financials" or "PPG's financials" -> capture word before financial, strip 's
        m = re.search(r"(\w{2,5})'?s?\s+financial", question_lower, re.IGNORECASE)
        if m:
            raw = re.sub(r"'?s?$", "", m.group(1)).upper()
            if len(raw) >= 2 and raw.isalpha():
                ticker = raw
        if not ticker:
            # "Show me PPG financials" -> first 2-5 letter all-caps token
            m = re.search(r"\b([A-Z]{2,5})\b", question)
            if m:
                ticker = m.group(1)
        if ticker:
            return {
                "intent": "company_comprehensive_workup",
                "collections": ["Company", "MarketData", "sec_filings", "sec_exhibits", "sec_xbrl_data", "Award", "options_flow"],
                "requires_embedding": False,
                "aql_query": COMPANY_WORKUP_AQL.strip(),
                "bind_vars": {"ticker": ticker},
                "explanation": f"Company financials/workup for {ticker} (rule-based bypass)"
            }

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
        # Re-extract ticker from current question so we don't return wrong-company data (e.g. LDOS vs LHX)
        ticker_match = re.search(r'\b([A-Z]{2,5})\b', question)
        if ticker_match:
            extracted_ticker = ticker_match.group(1)
            plan_bind_vars = similar_plan.get('bind_vars') or {}
            cached_ticker = plan_bind_vars.get('ticker')
            if cached_ticker is not None and str(cached_ticker).upper() != str(extracted_ticker).upper():
                similar_plan = copy.deepcopy(similar_plan)
                similar_plan['bind_vars'] = { **similar_plan.get('bind_vars', {}), 'ticker': extracted_ticker }
                print(f"✓ Reusing similar query plan from cache (ticker overridden to {extracted_ticker})")
            else:
                print("✓ Reusing similar query plan from cache")
        else:
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
            ticker_val = intent_hint.get("value", "")
            hint_text = f"\n\n🎯 CONFIRMED: This is a TICKER query for '{ticker_val}'. You MUST include in filters: \"Company.ticker\": {{\"operator\": \"==\", \"value\": \"{ticker_val}\"}}. Without this the query would return data for all companies."
        elif intent_hint.get("type") == "concept":
            hint_text = f"\n\n🎯 CONFIRMED: This is a CONCEPT/SEMANTIC query about '{intent_hint.get('value')}'. Use semantic search with embeddings."

    # TWO-STEP FLOW: NL -> JSON intent -> deterministic AQL (replaces hard-coded AQL prompts)
    print("[PLANNING] Two-step flow: NL -> JSON intent -> AQL")
    full_schema = get_cached_schema()
    schema_for_intent = _schema_for_json_intent(full_schema)
    planning_prompt = build_json_intent_prompt(schema_for_intent, question, hint=hint_text)

    prompt_chars = len(planning_prompt)
    estimated_tokens = prompt_chars // 4
    print(f"[TOKEN CHECK] Prompt size: {prompt_chars:,} chars (~{estimated_tokens:,} tokens)")

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": planning_prompt}],
            max_tokens=config.MAX_TOKENS,
            temperature=config.TEMPERATURE,
            response_format={"type": "json_object"}
        )

        json_plan = json.loads(response.choices[0].message.content)

        # Deterministic JSON -> AQL (no second LLM call)
        aql_query = json_to_aql(json_plan, log=print)

        if not aql_query or aql_query.startswith("//"):
            print("[PLANNING] json_to_aql produced no valid AQL")
            return None

        # Build plan in the shape the rest of the backend expects
        primary = json_plan.get("primary_collection", "Company")
        traversals = json_plan.get("traversals", [])
        collections = [primary]
        for t in traversals:
            to_coll = t.get("to_collection")
            if to_coll and to_coll not in collections:
                collections.append(to_coll)

        bind_vars = dict(json_plan.get("bind_vars") or {})
        if intent_hint and intent_hint.get("type") == "ticker" and intent_hint.get("value"):
            bind_vars["ticker"] = intent_hint["value"]
        plan = {
            "intent": json_plan.get("intent", "json_intent"),
            "collections": collections,
            "requires_embedding": False,
            "aql_query": aql_query,
            "bind_vars": bind_vars,
            "explanation": f"Two-step: {json_plan.get('intent', '')}"
        }

        # Optional: run existing AQL validation/correction on generated query
        try:
            corrected_query, errors, bind_params = validate_aql_syntax(plan["aql_query"], question)
            plan["aql_query"] = corrected_query
            if errors:
                plan["validation_warnings"] = errors
                for error in errors:
                    print(f"  - {error}")
        except ValueError as ve:
            print(f"CRITICAL validation error: {ve}")
            return None

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

    first_result = results[0]

    # CRITICAL: If this is a company overview with nested structure, it's NOT a time series query
    # Company overview has nested arrays: MarketData, sec_filings, Award, options_flow
    has_nested_market_data = isinstance(first_result.get('MarketData'), list)
    has_nested_filings = isinstance(first_result.get('sec_filings'), list)
    has_company_info = 'ticker' in first_result and 'company' in first_result

    if has_nested_market_data or (has_nested_filings and has_company_info):
        # This is a comprehensive company workup, not a simple time series
        return False

    # Check if results have time series characteristics (flat MarketData records)
    has_date = 'date' in first_result
    has_price_fields = any(field in first_result for field in ['close', 'open', 'high', 'low', 'price', 'volume'])

    # Check query plan collections
    collections = query_plan.get('collections', [])
    is_market_data = 'MarketData' in collections

    return has_date and has_price_fields and is_market_data


def extract_year_from_question(question: str) -> Optional[int]:
    """Extract a calendar year when the question implies financials/results for that year (e.g. '2025 financials')."""
    if not question or not question.strip():
        return None
    q = question.strip().lower()
    # When user asks about financials/results/performance for a year, use that full year
    if 'financial' in q or 'fy ' in q or 'fy' in q or 'result' in q or 'performance' in q:
        years = re.findall(r'\b(20[0-3]\d|19[9]\d)\b', q)
        if years:
            return int(years[0])
    return None


def format_time_series_analysis(user_question: str, results: list, query_plan: dict):
    """Format time series data with statistics and chart-ready format"""

    if not results:
        return "No data found for the specified time period."

    # If user asked for a specific year (e.g. "2025 financials"), filter to that full year
    requested_year = extract_year_from_question(user_question)
    if requested_year:
        year_start = f"{requested_year}-01-01"
        year_end = f"{requested_year}-12-31"
        results = [r for r in results if year_start <= (r.get('date') or '') <= year_end]
        if not results:
            return f"No market data found for {requested_year}. Try a different year or check the date range."

    # Sort by date
    sorted_results = sorted(results, key=lambda x: x.get('date', ''))

    # Extract time series data
    dates = [r.get('date', '') for r in sorted_results]
    closes = [float(r.get('close') or 0) for r in sorted_results]
    volumes = [float(r.get('volume') or 0) for r in sorted_results]

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

        # Format summary (prefer bind_vars.ticker when result rows lack ticker)
        ticker = (
            sorted_results[0].get('ticker')
            or (query_plan.get('bind_vars') or {}).get('ticker')
            or 'Unknown'
        )
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


def trim_results_for_llm(results: list, max_results: int = 10, max_tokens: int = 3000) -> list:
    """
    Aggressively trim query results to fit within token limits.

    Strategy:
    1. Limit number of results to 1 (for company overview queries)
    2. Remove/truncate large nested structures
    3. Keep only essential fields for analysis
    4. Force truncate if still too large

    Note: Default max_tokens reduced to 3000 to stay well under OpenAI's 25k total limit
    """
    if not results:
        return results

    # Step 1: Limit number of results - ONLY 1 result for company overview
    trimmed = results[:min(max_results, 1)]

    # Step 2: Aggressively trim each result
    cleaned_results = []
    for result in trimmed:
        cleaned = {}

        for key, value in result.items():
            # Skip internal fields
            if key.startswith('_'):
                continue

            # Handle nested arrays (MarketData, sec_filings, etc.)
            if isinstance(value, list):
                if len(value) > 0:
                    # Keep ONLY FIRST item from nested arrays
                    trimmed_array = []
                    item = value[0]
                    if isinstance(item, dict):
                        # Keep only essential numeric/date fields
                        small_item = {}
                        for k, v in item.items():
                            if k.startswith('_'):
                                continue
                            # Skip ALL nested lists
                            if isinstance(v, list):
                                continue
                            # Skip long strings
                            if isinstance(v, str) and len(v) > 50:
                                small_item[k] = v[:50] + "..."
                            # Only keep numbers, dates, bools, short strings
                            elif isinstance(v, (int, float, bool)) or (isinstance(v, str) and len(v) <= 50):
                                small_item[k] = v
                        trimmed_array.append(small_item)
                    else:
                        trimmed_array.append(item)

                    cleaned[key] = trimmed_array
                    # Add count to show how much data exists
                    cleaned[f"{key}_count"] = len(value)
                else:
                    cleaned[key] = []

            # Handle large text fields - VERY aggressive truncation
            elif isinstance(value, str) and len(value) > 100:
                cleaned[key] = value[:100] + "..."

            # Handle normal fields
            else:
                cleaned[key] = value

        cleaned_results.append(cleaned)

    # Step 3: Better token estimation (1 token ≈ 3 chars for JSON)
    result_json = json.dumps(cleaned_results)
    estimated_tokens = len(result_json) / 3

    print(f"[TRIM] Results: {len(results)} → {len(cleaned_results)}, Data tokens: {estimated_tokens:.0f}")

    # Step 4: If STILL too large, force truncate the JSON
    if estimated_tokens > max_tokens:
        max_chars = int(max_tokens * 3)
        result_json = result_json[:max_chars]
        print(f"[TRIM] 🚨 Force truncated JSON to {max_tokens} tokens")
        # Try to parse truncated JSON (might fail, but that's ok)
        try:
            cleaned_results = json.loads(result_json)
        except:
            # If truncation broke JSON, just use first result with minimal data
            cleaned_results = [{'error': 'Data truncated due to size', 'count': len(results)}]
            print(f"[TRIM] JSON truncation broke structure, using minimal fallback")

    return cleaned_results


def strip_markdown_tables(text: str) -> str:
    """
    Remove markdown tables from text.
    Handles both full tables and partial table fragments.
    """
    import re

    lines = text.split('\n')
    cleaned_lines = []
    in_table = False
    table_header = None

    for i, line in enumerate(lines):
        # Check if this line is part of a markdown table
        if '|' in line:
            # Could be a table row
            # Check if next line has --- separators (table header marker)
            if i + 1 < len(lines) and re.match(r'^[\s|:-]+$', lines[i + 1].replace('-', '').replace(':', '')):
                # This is a table header, skip it and the separator
                in_table = True
                table_header = line
                continue
            elif in_table:
                # We're inside a table, skip this row
                continue
            elif re.match(r'^[\s|:-]+$', line.replace('-', '').replace(':', '')):
                # This is a separator line, skip it
                continue
            else:
                # Might be a table row without proper header detected
                # Check if line has multiple | characters (likely a table)
                if line.count('|') >= 2:
                    in_table = True
                    continue
        else:
            # Not a table line
            if in_table:
                # Just exited a table
                in_table = False
                table_header = None
            cleaned_lines.append(line)

    result = '\n'.join(cleaned_lines)

    # Additional cleanup: remove any "Apple Stock Data" or similar headers that precede tables
    result = re.sub(r'#+\s*\w+\s+Stock Data[^\n]*\n', '', result, flags=re.IGNORECASE)
    result = re.sub(r'Apple Stock Data \(Database Results\)', '', result, flags=re.IGNORECASE)

    # Remove excessive blank lines
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result.strip()


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
    # HARD CAP: OpenAI gpt-4o limit is 25,000 tokens per request
    # Start VERY conservative: max 1 result, max 3k tokens for results data
    results_sample = trim_results_for_llm(results, max_results=1, max_tokens=3000)
    result_count = len(results)

    # Use context-aware response synthesis for intelligent analysis
    print("[LLM ANALYSIS] Using context-aware synthesis engine")
    print(f"[LLM ANALYSIS] Trimmed {len(results)} results to {len(results_sample)} for analysis")
    analysis_prompt = get_enhanced_analysis_prompt(user_question, results_sample, query_plan)

    # Better token estimation (1 token ≈ 3 chars for structured content)
    prompt_tokens = len(analysis_prompt) / 3
    print(f"\n[LLM ANALYSIS] Prompt length: {len(analysis_prompt)} chars (~{prompt_tokens:.0f} tokens)")

    # HARD CAP at 25,000 tokens - truncate prompt if needed
    MAX_PROMPT_TOKENS = 25000

    if prompt_tokens > MAX_PROMPT_TOKENS:
        print(f"[LLM ANALYSIS] 🚨 Prompt at {prompt_tokens:.0f} tokens, FORCE TRUNCATING to stay under {MAX_PROMPT_TOKENS}")
        # Calculate max chars for 25k tokens
        max_chars = int(MAX_PROMPT_TOKENS * 3)
        analysis_prompt = analysis_prompt[:max_chars]
        prompt_tokens = len(analysis_prompt) / 3
        print(f"[LLM ANALYSIS] Force truncated to {prompt_tokens:.0f} tokens ({len(analysis_prompt)} chars)")

    print(f"[LLM ANALYSIS] Using model: {config.LLM_MODEL}")
    print(f"[LLM ANALYSIS] OpenAI API key set: {bool(config.OPENAI_API_KEY)}")
    print(f"[LLM ANALYSIS] Calling OpenAI...")

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": analysis_prompt}],
            max_tokens=3000,  # Increased for comprehensive analysis with SEC/insider signals
            temperature=0.3
        )

        analysis_result = response.choices[0].message.content.strip()

        print(f"\n[LLM ANALYSIS] ✓ OpenAI call successful!")
        print(f"[LLM ANALYSIS] Response length: {len(analysis_result)} chars")
        print(f"[LLM ANALYSIS] Response preview (first 200 chars):")
        print(f"  {analysis_result[:200]}...")
        print(f"[LLM ANALYSIS] Contains table markers: {'|' in analysis_result}")

        # CRITICAL: For company overview queries, strip any tables the LLM created
        if detect_time_series_query(results, query_plan) == False and len(results) > 0:
            first_result = results[0]
            # Check if this is a comprehensive company workup (nested structure)
            has_nested_data = isinstance(first_result.get('MarketData'), list) or isinstance(first_result.get('sec_filings'), list)

            if has_nested_data and '|' in analysis_result:
                print(f"[LLM ANALYSIS] ⚠️ Company overview with table detected - stripping tables!")
                analysis_result = strip_markdown_tables(analysis_result)
                print(f"[LLM ANALYSIS] ✓ Tables removed, new length: {len(analysis_result)} chars")

        print("="*80 + "\n")

        return analysis_result

    except Exception as e:
        print(f"\n[LLM ANALYSIS] ✗ ERROR calling OpenAI!")
        print(f"[LLM ANALYSIS] Error type: {type(e).__name__}")
        print(f"[LLM ANALYSIS] Error message: {str(e)}")

        # Check if it's a rate limit error (429)
        error_str = str(e).lower()
        if '429' in error_str or 'rate_limit' in error_str or 'tokens per min' in error_str:
            print(f"[LLM ANALYSIS] Rate limit error detected - query returned too much data")
            return f"""## Query Results ({result_count} records found)

**⚠️ Results too large for AI analysis**

This query returned {result_count} results with detailed data that exceeded OpenAI's token limit (30,000 tokens per request).

**What happened:**
The query successfully retrieved data from the database, but the response was too large to analyze with AI. This typically happens with queries that return:
- Many records (>100)
- Records with nested data (company details, market data arrays, etc.)
- Long text fields (descriptions, contract details)

**Suggestions:**
1. **Narrow your search** - Add more filters (date range, amount threshold, specific companies)
2. **Request specific fields** - Instead of "with company details", ask for specific info
3. **Use pagination** - Ask for "top 10" or "first 20" results
4. **Try simpler queries** - Break complex queries into smaller parts

**Example refined queries:**
- "Show me the top 10 largest government contracts over $100M in 2025"
- "Find contracts over $100M for defense sector companies"
- "List contracts over $100M awarded to Lockheed Martin"

**Raw result count:** {result_count} contracts found"""

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
            max_tokens=3000,
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

    # Skip rule-based suggestions - use LLM for more conversational follow-ups
    # smart_suggestions = generate_smart_suggestions(results, query_plan, user_question)
    # if smart_suggestions:
    #     return smart_suggestions

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
