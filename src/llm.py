"""LLM query planning, analysis, and validation"""

import openai
import json
from config import *
from prompts import CRITICAL_AQL_RULES  # Import only what we need (not full SCHEMA_DESCRIPTION)
import streamlit as st
import re
from datetime import datetime
import database as arango_db
import config as cfg
import hashlib 


def preprocess_query(question):
    """Handle simple queries without LLM"""
    # ticker_match = re.search(r'\b([A-Z]{2,5})\b', question)
    # if ticker_match and len(question.split()) <= 5:
    #     ticker = ticker_match.group(1)
    #     if any(word in question.lower() for word in ['show', 'get', 'find']):
    #         return {
    #             "intent": "ticker_lookup",
    #             "aql_query": "FOR doc IN Award FILTER doc.ticker == @ticker SORT doc.startdate DESC LIMIT 20 RETURN doc",  # FIXED
    #             "bind_vars": {"ticker": ticker},
    #             "requires_embedding": False,
    #             "explanation": f"Simple ticker lookup for {ticker}"
    #         }
    
    # if question.lower().startswith("how many"):
    #     if "awards" in question.lower():
    #         return {
    #             "intent": "count_awards",
    #             "aql_query": "RETURN LENGTH(Award)",
    #             "bind_vars": {},
    #             "requires_embedding": False,
    #             "explanation": "Count total awards"
    #         }
    
    # if any(word in question.lower() for word in ['latest', 'recent', 'last']):
    #     return {
    #         "intent": "recent_data",
    #         "aql_query": "FOR doc IN Award SORT doc.startdate DESC LIMIT 10 RETURN doc",  # FIXED
    #         "bind_vars": {},
    #         "requires_embedding": False,
    #         "explanation": "Fetch most recent awards"
    #     }
    
    return None



def get_query_embedding(text):
    """Generate embedding vector for semantic search"""
    try:
        response = openai.embeddings.create(
            model=cfg.EMBEDDING_MODEL,  # Use config constant
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Embedding generation error: {str(e)}")
        return None


def check_similar_previous_question(question):
    """Check if this question is very similar to a recent one"""
    if 'query_history' not in st.session_state or not st.session_state.query_history:
        return None

    # Check last 5 queries
    recent_queries = st.session_state.query_history[-5:]

    try:
        # Get embedding for current question
        current_embedding = get_query_embedding(question)
        if not current_embedding:
            return None

        # Check similarity with recent queries
        for past_query in reversed(recent_queries):
            if 'question' not in past_query or 'plan' not in past_query:
                continue

            past_question = past_query['question']
            past_embedding = get_query_embedding(past_question)

            if not past_embedding:
                continue

            # Calculate cosine similarity
            from numpy import dot
            from numpy.linalg import norm

            similarity = dot(current_embedding, past_embedding) / (norm(current_embedding) * norm(past_embedding))

            # If >95% similar, reuse the plan
            if similarity > 0.95:
                st.info(f"💡 This question is very similar to: \"{past_question}\"")
                st.caption("Reusing previous query plan...")
                return past_query['plan']

    except Exception as e:
        # If similarity check fails, just proceed normally
        pass

    return None


def plan_query_with_llm(question, intent_hint=None, use_local=False):
    """Generate query plan from natural language question"""
    # Try rules first
    rule_result = preprocess_query(question)
    if rule_result:
        st.info("✓ Handled by rule (no LLM call)")
        return rule_result

    # Check if similar question was asked recently
    similar_plan = check_similar_previous_question(question)
    if similar_plan:
        return similar_plan

    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Add intent hint to prompt
    hint_text = ""
    if intent_hint:
        if intent_hint.get("type") == "ticker":
            hint_text = f"\n\n🎯 CONFIRMED: This is a TICKER query for '{intent_hint.get('value')}'. Use doc.ticker == @ticker"
        elif intent_hint.get("type") == "concept":
            hint_text = f"\n\n🎯 CONFIRMED: This is a CONCEPT/SEMANTIC query about '{intent_hint.get('value')}'. Use semantic search with embeddings."
    
    relevant_schema = get_relevant_schema(question, intent_hint)

    planning_prompt = f"""You are a database query planner for ArangoDB.

RELEVANT SCHEMA:
{relevant_schema}

{CRITICAL_AQL_RULES}

EXAMPLE QUERIES (for reference):
- Ticker query: FOR doc IN Award FILTER doc.ticker == @ticker SORT doc.award_amount_float DESC LIMIT 10 RETURN doc
- Semantic query: FOR doc IN Award FILTER doc.description_embedding != null LET sim = COSINE_SIMILARITY(doc.description_embedding, @query_vector) FILTER sim >= 0.7 SORT sim DESC LIMIT 10 RETURN doc
- SEC sentiment: FOR doc IN sec_sentences FILTER CONTAINS(LOWER(doc.text), @keyword) AND doc.finbert_score < -0.3 LIMIT 20 RETURN doc
- Date range: FOR doc IN MarketData FILTER doc.ticker == @ticker AND doc.date >= DATE_SUBTRACT(DATE_NOW(), 180, "day") SORT doc.date DESC LIMIT 100 RETURN doc
- Multi-collection (SEC + Company): FOR filing IN sec_filings FILTER filing.ticker IN @tickers FOR company IN Company FILTER company.ticker == filing.ticker RETURN MERGE(filing, {{marketCap: company.marketCap, employees: company.fullTimeEmployees}}) LIMIT 50

USER QUESTION: "{question}"{hint_text}

Current Date: {current_date}

Generate a JSON response with:
- "intent": classification (e.g., "ticker_awards", "semantic_awards", "sec_sentiment", "market_data")
- "collections": array of collection names
- "requires_embedding": boolean (true ONLY for Award semantic search)
- "embedding_text": text to embed (if semantic search)
- "aql_query": valid AQL query
- "bind_vars": object with bind variables
- "explanation": brief strategy explanation

CRITICAL CHECKLIST:
✅ Collection names: Award (not awards), sec_filings (not SEC_Filings)
✅ Field names: award_amount_float, sharesOutstanding, sandp_500_index
✅ Date functions: DATE_SUBTRACT(DATE_NOW(), N, "day")
✅ Order: FOR → FILTER → SORT → LIMIT → RETURN
✅ Embeddings: Only Award collection has description_embedding
✅ Always include LIMIT

⚠️ EMBEDDING RULES:
- Set requires_embedding = true ONLY if:
  1. Query asks for semantic/concept search
  2. AND uses Award collection
  3. AND uses description_embedding field

- For SEC sentiment queries about "risks" or concepts:
  - Set requires_embedding = false
  - Use text filters: CONTAINS(LOWER(doc.text), 'risk')
  - Use sentiment: FILTER doc.finbert_score < -0.3

EXAMPLES OF CORRECT QUERIES:

/* SEC sentiment query - NO embeddings */
{{
  "intent": "sec_sentiment",
  "requires_embedding": false,
  "aql_query": "FOR doc IN sec_sentences FILTER doc.finbert_score < -0.3 AND CONTAINS(LOWER(doc.text), 'cybersecurity') LIMIT 20 RETURN doc"
}}

/* Award semantic search - YES embeddings */
{{
  "intent": "award_semantic",
  "requires_embedding": true,
  "embedding_text": "artificial intelligence machine learning",
  "aql_query": "FOR doc IN Award FILTER doc.description_embedding != null LET sim = COSINE_SIMILARITY(doc.description_embedding, @query_vector) FILTER sim >= 0.7 LIMIT 10 RETURN doc"
}}

Return ONLY valid JSON.

Response:"""
    
    try:
        response = openai.chat.completions.create(
            model=cfg.LLM_MODEL,
            messages=[{"role": "user", "content": planning_prompt}],
            max_tokens=cfg.MAX_TOKENS,
            temperature=cfg.TEMPERATURE,
            response_format={"type": "json_object"}
        )
        
        plan = json.loads(response.choices[0].message.content)
        
        # Auto-fix common mistakes
        plan = arango_db.fix_ticker_confusion(plan, question)
        
        return plan
        
    except Exception as e:
        st.error(f"Query planning error: {str(e)}")
        return None


def quick_intent_check(question, use_local=False):
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
- "FCX commodity positions" → TICKER (FCX is Freeport-McMoRan ticker)
- "copper exposure" → CONCEPT (copper is a commodity, not a ticker)

Return JSON: {{"type": "ticker", "value": "CMI"}} or {{"type": "concept", "value": "artificial intelligence"}}
"""
    
    try:
        response = openai.chat.completions.create(
            model=cfg.LLM_MODEL,
            messages=[{"role": "user", "content": check_prompt}],
            max_tokens=100,
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except:
        return {"type": "unknown"}


def generate_follow_up_questions(user_question, results, query_plan):
    """Generate contextual follow-up questions based on results"""
    intent = query_plan.get('intent', '')
    collections = query_plan.get('collections', [])
    follow_ups = []
    
    # Pattern 1: Temporal expansion
    if 'date' in str(results).lower() or any(c in collections for c in ['MarketData', 'EconomicData']):
        follow_ups.append(f"📈 How has this changed over the past year?")
        follow_ups.append(f"📊 Show me the trend for the last 5 years")
    
    # Pattern 2: Entity expansion (if tickers found)
    if results and 'ticker' in str(results[0]):
        tickers = [r.get('ticker') for r in results[:3] if r.get('ticker')]
        if tickers:
            follow_ups.append(f"💼 Compare {', '.join(tickers[:3])} financial metrics")
            follow_ups.append(f"🔍 What are the biggest risks for {tickers[0]}?")
    
    # Pattern 3: Cross-collection expansion
    if 'Award' in collections:
        follow_ups.append(f"📉 What's the stock performance for these companies?")
        follow_ups.append(f"🔮 What do prediction markets say about these companies?")  # UPDATED
    
    if 'sec_filings' in collections or 'sec_sentences' in collections:
        follow_ups.append(f"💰 Show me government contracts for these companies")
        follow_ups.append(f"📊 What are their financial metrics?")
    
    if 'MarketData' in collections:
        follow_ups.append(f"🏛️ Have these companies received government contracts?")
        follow_ups.append(f"⚠️ What risks do they mention in SEC filings?")
    
    # Pattern 4: Prediction markets (NEW)
    if any(c in collections for c in ['prediction_markets_polymarket', 'prediction_markets_kalshi']):
        follow_ups.append(f"📈 How does this compare to their stock performance?")
        follow_ups.append(f"📦 Do they have commodity exposure?")
    
    # Pattern 5: Commodity positions (NEW)
    if 'commodity_positions' in collections:
        follow_ups.append(f"📊 How does this correlate with commodity prices?")
        follow_ups.append(f"💼 Show me their recent financial performance")
    
    # Pattern 6: Depth expansion
    if len(results) > 5:
        follow_ups.append(f"🎯 Show me only the top 3 results with more detail")
        follow_ups.append(f"📋 Export this data to CSV")
    
    # Pattern 7: Comparative analysis
    if len(results) >= 2:
        follow_ups.append(f"⚖️ Compare the top 3 companies side-by-side")
    
    return follow_ups[:4]  # Return top 4


def validate_aql_syntax(aql_query):
    """Basic syntax validation before execution"""
    errors = []

    # Check for UNION (not supported in AQL) - CRITICAL ERROR
    if 'UNION' in aql_query.upper():
        errors.append("❌ CRITICAL: AQL does not support UNION! Use nested FOR loops instead.")
        raise ValueError("AQL does not support UNION syntax. Use nested FOR loops or MERGE() to combine data from multiple collections.")

    # Check for JOIN (not supported in AQL) - CRITICAL ERROR
    if re.search(r'\bJOIN\b', aql_query, re.IGNORECASE):
        errors.append("❌ CRITICAL: AQL does not support JOIN! Use nested FOR loops instead.")
        raise ValueError("AQL does not support JOIN syntax. Use nested FOR loops to connect data.")

    # Check for common typos
    if "compan." in aql_query and "company" in aql_query:
        errors.append("Typo detected: 'compan.' should be 'company.'")
        aql_query = aql_query.replace("compan.", "company.")

    # Fix: INTO keyword (doesn't exist in AQL)
    if ' INTO ' in aql_query.upper():
        errors.append("⚠️ AQL syntax error: INTO keyword is not supported")
        if 'RETURN' in aql_query and 'INTO' in aql_query:
            st.warning("🔧 Detected invalid INTO syntax. Attempting to rewrite as LET...")
            errors.append("Cannot auto-fix: Query needs manual rewrite using LET or subquery")
            errors.append("Recommendation: Use LET variable = (subquery) instead of RETURN ... INTO")

    # Fix: Common collection name mistakes
    replacements = {
        'awards': 'Award',
        'Companies': 'Company',
        'companies': 'Company',
        'market_data': 'MarketData',
        'MarketData_edge': 'HAS_MARKETDATA',  # FIXED
        'fred_data': 'EconomicData',
        'FREDData': 'EconomicData',
        'polymarket': 'prediction_markets_polymarket',
        'kalshi': 'prediction_markets_kalshi',
        'CommodityPosition': 'commodity_positions',
    }
    
    for wrong, correct in replacements.items():
        if wrong in aql_query:
            aql_query = aql_query.replace(wrong, correct)
            errors.append(f"Auto-fixed: '{wrong}' → '{correct}'")
    
    # Check for undeclared variables
    declared_vars = set(re.findall(r'FOR\s+(\w+)\s+IN', aql_query, re.IGNORECASE))
    used_vars = set(re.findall(r'(\w+)\.', aql_query))
    
    # Check if any used vars aren't declared
    undeclared = used_vars - declared_vars - {'doc', 'data', 'edge', 'market', 'company', 'award'}
    if undeclared:
        errors.append(f"Undeclared variables: {undeclared}")
    
    # Check bind variables
    bind_params = set(re.findall(r'@(\w+)', aql_query))
    
    return aql_query, errors, bind_params


def get_relevant_schema(question, intent):
    schemas = {
        "Award": {
            "description": "Government contract awards",
            "key_fields": ["ticker", "award_amount_float", "start_date", "awarding_agency", "recipient_name", "description", "description_embedding", "contract_year"],
            "sample_query": "FOR doc IN Award FILTER doc.ticker == @ticker SORT doc.start_date DESC LIMIT 20 RETURN doc"
        },
        "Company": {
            "description": "Company master data",
            "key_fields": [
                "ticker", "company", "sector", "industry", "country", "cik",
                "sharesOutstanding", "marketCap", "fullTimeEmployees",  # EXACT names
                "sp500_member", "website", "lastUpdated"
            ],
            "sample_query": "FOR c IN Company FILTER c.ticker == @ticker RETURN c"
        },
        "MarketData": {
            "description": "Daily OHLCV + 40+ technical/fundamental indicators",
            "key_fields": [
                "ticker", "date", "open", "high", "low", "close", "volume",
                "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
                "ema_12", "ema_26",
                "macd", "macd_signal", "macd_histogram",
                "dist_from_sma20", "dist_from_sma50", "dist_from_sma200",
                "golden_cross", "death_cross",
                "above_sma20", "above_sma50", "above_sma200",
                "targetMeanPrice", "forwardEps", "trailingPE", "beta"
            ],
            "sample_query": "FOR m IN MarketData FILTER m.ticker == @ticker AND m.date >= @start_date SORT m.date DESC RETURN m"
        },
        "EconomicData": {
            "description": "Macroeconomic indicators from FRED",
            "key_fields": ["date", "sandp_500_index", "federal_funds_rate", "unemployment_rate", "yield_curve_inverted", "10y_2y_treasury_spread", "vix_volatility_index"],
            "sample_query": "FOR e IN EconomicData FILTER e.date == @date RETURN e"
        },
        "commodity_positions": {
            "description": "CFTC Commitments of Traders data",
            "key_fields": ["ticker", "Market_and_Exchange_Names", "as_of_date", "Noncommercial_Positions_Long_All", "Noncommercial_Positions_Short_All"],
            "sample_query": "FOR c IN commodity_positions FILTER c.Market_and_Exchange_Names LIKE @commodity RETURN c"
        },
        "prediction_markets_polymarket": {
            "description": "Polymarket prediction market data",
            "key_fields": ["question", "yes_probability", "no_probability", "volume_24h", "closed", "end_date", "market_slug"],
            "sample_query": "FOR m IN prediction_markets_polymarket FILTER m.closed == false SORT m.volume_24h DESC LIMIT 20 RETURN m"
        },
        "sec_filings": {
            "description": "SEC filing metadata with aggregated sentiment",
            "key_fields": ["ticker", "type", "filing_date", "fiscal_year", "avg_finbert", "avg_uncertainty", "avg_negative", "avg_positive", "sentence_count"],
            "sample_query": "FOR f IN sec_filings FILTER f.ticker == @ticker AND f.type == '10-K' SORT f.filing_date DESC RETURN f"
        },
        "sec_sentences": {
            "description": "Individual SEC sentences with sentiment (NO embeddings - use text search)",
            "key_fields": ["text", "finbert_score", "finbert_probs", "negative_per_1k", "positive_per_1k", "uncertainty_per_1k", "litigious_per_1k", "section_id"],
            "sample_query": "FOR s IN sec_sentences FILTER s.finbert_score < -0.3 AND CONTAINS(LOWER(s.text), @keyword) LIMIT 20 RETURN s"
        }
    }
   
    
    # Keyword mapping
    relevant_collections = []
    
    if any(word in question.lower() for word in ['award', 'contract', 'government']):
        relevant_collections.append("Award")
    
    if any(word in question.lower() for word in ['price', 'stock', 'market', 'volume', 'rsi', 'macd', 'beta']):
        relevant_collections.append("MarketData")
    
    if any(word in question.lower() for word in ['filing', '10-k', '10-q', 'sec', 'sentiment', 'risk']):
        relevant_collections.extend(["sec_filings", "sec_sentences"])
    
    if any(word in question.lower() for word in ['economic', 'macro', 'unemployment', 'gdp', 'inflation', 'fed', 's&p', 'yield curve']):
        relevant_collections.append("EconomicData")
    
    if any(word in question.lower() for word in ['commodity', 'crude', 'oil', 'gold', 'copper', 'cftc', 'position']):
        relevant_collections.append("commodity_positions")
    
    if any(word in question.lower() for word in ['polymarket', 'prediction', 'market sentiment']):
        relevant_collections.append("prediction_markets_polymarket")
    
    if any(word in question.lower() for word in ['kalshi', 'betting market']):
        relevant_collections.append("prediction_markets_kalshi")
    
    # Always include Company for context
    if len(relevant_collections) > 0 and "Company" not in relevant_collections:
        relevant_collections.append("Company")
    
    # If nothing matched, include Award (default)
    if not relevant_collections:
        relevant_collections = ["Award", "Company"]
    
    # Build focused schema description
    focused_schema = "\n\n".join([
        f"Collection: {coll}\n" +
        f"Description: {schemas[coll]['description']}\n" +
        f"Key Fields: {', '.join(schemas[coll]['key_fields'])}\n" +
        f"Example: {schemas[coll]['sample_query']}"
        for coll in relevant_collections if coll in schemas
    ])
    
    return focused_schema


def execute_with_retry(plan, max_retries=2):
    """Execute query with self-correction"""
    for attempt in range(max_retries + 1):
        try:
            results = execute_planned_query(plan, raise_on_error=True)  # Enable exception raising
            return results, None  # Success
        except Exception as e:
            error_msg = str(e)
            if attempt == max_retries:
                return [], error_msg  # Give up
            
            # Self-correct
            st.warning(f"⚠️ Attempt {attempt + 1} failed: {error_msg}")
            st.info("🔄 Asking LLM to fix query...")
            
            # Re-prompt with error context
            correction_prompt = f"""The following AQL query failed:

QUERY:
{plan['aql_query']}

BIND VARS:
{plan['bind_vars']}

ERROR:
{error_msg}

Generate a CORRECTED query that fixes this error.
Use correct collection names (Award not awards, Company not companies).
Return JSON with same format: {{"aql_query": "...", "bind_vars": {{...}}}}
"""
            
            try:
                response = openai.chat.completions.create(
                    model=cfg.LLM_MODEL,
                    messages=[{"role": "user", "content": correction_prompt}],
                    max_tokens=800,
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                
                corrected = json.loads(response.choices[0].message.content)
                # Update plan with correction
                plan['aql_query'] = corrected.get('aql_query', plan['aql_query'])
                plan['bind_vars'] = corrected.get('bind_vars', plan['bind_vars'])
                st.success("✓ Query corrected, retrying...")
                
            except Exception as correction_error:
                st.error(f"Self-correction failed: {correction_error}")
                return [], error_msg
    
    return [], "Max retries exceeded"


def estimate_query_cost(aql_query):
    """Estimate if query will timeout and identify issues"""
    issues = []
    cost_score = 0
    
    # High cost: Full collection scan without early LIMIT
    query_before_return = aql_query.split('RETURN')[0] if 'RETURN' in aql_query else aql_query
    
    if 'FOR' in aql_query and 'LIMIT' not in query_before_return:
        issues.append("No LIMIT before RETURN (will scan entire collection)")
        cost_score += 50
    
    # High cost: Text search on large collections without index
    if 'CONTAINS' in aql_query and 'sec_sentences' in aql_query:
        issues.append("CONTAINS() on sec_sentences (slow - use FULLTEXT index)")
        cost_score += 40
    
    if 'CONTAINS' in aql_query and 'sec_sections' in aql_query:
        issues.append("CONTAINS() on sec_sections (slow)")
        cost_score += 35
    
    # High cost: Multiple nested FOR loops
    nested_fors = aql_query.count('FOR')
    if nested_fors > 2:
        issues.append(f"Multiple nested FOR loops ({nested_fors}) - use COLLECT")
        cost_score += 25 * (nested_fors - 1)
    
    # High cost: Cross-collection join without proper indexing
    if 'MarketData' in aql_query and any(coll in aql_query for coll in ['sec_sentences', 'sec_sections', 'Award']):
        if 'COLLECT' not in aql_query:
            issues.append("Cross-collection join without COLLECT (expensive)")
            cost_score += 30
    
    # Very high cost: Nested FOR without LIMIT in subquery
    if 'LET' in aql_query and 'FOR' in aql_query.split('LET')[1]:
        subquery_text = aql_query.split('LET')[1].split(')')[0] if ')' in aql_query.split('LET')[1] else ''
        if 'LIMIT' not in subquery_text and 'FIRST' not in subquery_text:
            issues.append("Subquery without LIMIT or FIRST() (will process all rows)")
            cost_score += 40
    
    return cost_score, issues


def optimize_expensive_query(aql_query):
    """Apply automatic optimizations to prevent timeout"""
    import re
    optimizations_applied = []
    
    # Optimization 1: Replace CONTAINS with FULLTEXT for single keywords
    if 'CONTAINS(LOWER(' in aql_query and 'sec_sentences' in aql_query:
        # Pattern: CONTAINS(LOWER(var.text), 'keyword')
        pattern = r"CONTAINS\(LOWER\((\w+)\.text\),\s*['\"]([^'\"]+)['\"]\)"
        matches = re.findall(pattern, aql_query)
        
        replaced_any = False
        for var_name, keyword in matches:
            # Only optimize single-word keywords
            if ' ' not in keyword.strip():
                # Replace CONTAINS with FULLTEXT
                old_pattern = f"CONTAINS(LOWER({var_name}.text), '{keyword}')"
                
                # Need to restructure: FOR sentence IN FULLTEXT(...)
                # This is complex, so just flag it for now
                optimizations_applied.append(
                    f"💡 Performance: '{keyword}' search could use FULLTEXT() (10x faster)"
                )
                optimizations_applied.append(
                    f"   Change: FOR {var_name} IN FULLTEXT(sec_sentences, 'text', '{keyword}')"
                )
            else:
                optimizations_applied.append(
                    f"✓ Using CONTAINS for phrase '{keyword}' (correct choice)"
                )
    
    # Optimization 2: Add LIMIT if missing before first RETURN
    query_before_return = aql_query.split('RETURN')[0] if 'RETURN' in aql_query else ''
    
    if 'FOR' in aql_query and 'LIMIT' not in query_before_return:
        # Add LIMIT before RETURN
        aql_query = aql_query.replace('RETURN', 'LIMIT 100\n  RETURN', 1)
        optimizations_applied.append("✅ Added LIMIT 100 before RETURN")
    
    # Optimization 3: Add sentiment filter if searching SEC without it
    if ('sec_sentences' in aql_query or 'sec_sections' in aql_query) and \
       'finbert_score' not in aql_query and \
       'FILTER' in aql_query:
        optimizations_applied.append(
            "⚠️ Missing sentiment filter - add: FILTER doc.finbert_score < -0.3"
        )
    
    # Optimization 4: Check for missing indexes on join fields
    if 'section.filing_id' in aql_query or 'sentence.section_id' in aql_query:
        optimizations_applied.append(
            "💡 Ensure indexes exist on filing_id and section_id for fast joins"
        )
    
    # Optimization 5: Detect inefficient nested loops
    for_count = aql_query.count('FOR')
    if for_count > 3:
        optimizations_applied.append(
            f"⚠️ Query has {for_count} nested loops - consider using COLLECT"
        )
    
    # Optimization 6: Check for full collection scans
    if 'FOR doc IN sec_sentences' in aql_query and 'FILTER' not in aql_query.split('FOR doc IN sec_sentences')[1].split('RETURN')[0]:
        optimizations_applied.append(
            "❌ WARNING: Full scan of 4.8M sentences without filter!"
        )
    
    # Optimization 7: Suggest subquery LIMIT
    if 'LET' in aql_query and 'FOR' in aql_query.split('LET')[1]:
        # Check if subqueries have LIMIT
        subquery_pattern = r'LET \w+ = \((.*?)\)'
        subqueries = re.findall(subquery_pattern, aql_query, re.DOTALL)
        
        for i, subquery in enumerate(subqueries):
            if 'FOR' in subquery and 'LIMIT' not in subquery:
                optimizations_applied.append(
                    f"⚠️ Subquery #{i+1} has no LIMIT - could process many rows"
                )
    
    # Optimization 8: Date filter positioning
    if 'filing_date' in aql_query and 'sec_filings' in aql_query:
        # Check if date filter comes early
        filing_index = aql_query.index('sec_filings')
        date_index = aql_query.index('filing_date')
        
        if date_index - filing_index > 200:  # Date filter is far from collection
            optimizations_applied.append(
                "⚠️ Date filter should be immediately after FOR filing IN sec_filings"
            )

    if 'FOR' in aql_query and 'LIMIT' not in aql_query:
        # Find the last RETURN that's not in a subquery
        lines = aql_query.split('\n')
        
        # Simple approach: add LIMIT before the last RETURN at depth 0
        # Count parentheses to find depth
        depth = 0
        insert_position = -1
        
        for i, line in enumerate(lines):
            depth += line.count('(') - line.count(')')
            
            if 'RETURN' in line and depth == 0:
                insert_position = i
        
        if insert_position > 0:
            # Insert LIMIT before this RETURN
            lines.insert(insert_position, '  LIMIT 100')
            aql_query = '\n'.join(lines)
            optimizations_applied.append("✅ Added LIMIT 100 before RETURN")
    
    return aql_query, optimizations_applied


def optimize_expensive_query_aggressive(aql_query):
    """Aggressive optimization that actually rewrites queries"""
    import re
    optimizations_applied = []
    original_query = aql_query
    
    # 1. Replace CONTAINS with FULLTEXT (actual replacement)
    if 'CONTAINS(LOWER(' in aql_query and 'sec_sentences' in aql_query:
        pattern = r"FOR (\w+) IN sec_sentences\s+FILTER[^F]*CONTAINS\(LOWER\(\1\.text\),\s*['\"](\w+)['\"]\)"
        
        def replace_with_fulltext(match):
            var_name = match.group(1)
            keyword = match.group(2)
            
            # Get the rest of the filters
            rest = match.group(0).split('CONTAINS')[1].split('\n')[1:]
            
            optimizations_applied.append(f"✅ Replaced CONTAINS with FULLTEXT for '{keyword}'")
            
            return f"FOR {var_name} IN FULLTEXT(sec_sentences, 'text', '{keyword}')\n  FILTER"
        
        aql_query = re.sub(pattern, replace_with_fulltext, aql_query)
    
    # 2. Add LIMIT to unlimited loops
    if 'FOR' in aql_query and 'LIMIT' not in aql_query.split('RETURN')[0]:
        aql_query = aql_query.replace('RETURN', 'LIMIT 100\n  RETURN', 1)
        optimizations_applied.append("✅ Added LIMIT 100")
    
    # 3. Move sentiment filters earlier
    if 'finbert_score' in aql_query:
        # Try to move sentiment filter up
        lines = aql_query.split('\n')
        sentiment_line = None
        for_line = None
        
        for i, line in enumerate(lines):
            if 'FOR' in line and 'sec_sentences' in line:
                for_line = i
            if 'finbert_score' in line and for_line:
                sentiment_line = i
                break
        
        if sentiment_line and for_line and sentiment_line > for_line + 2:
            optimizations_applied.append("💡 Consider moving sentiment filter closer to FOR loop")
    
    if aql_query != original_query:
        return aql_query, optimizations_applied
    else:
        # No aggressive changes made, return suggestions only
        return original_query, ["No automatic optimizations applied - query structure is good"]


def apply_fulltext_conversion(aql_query):
    """Automatically convert CONTAINS to FULLTEXT for sec_sentences/sec_sections queries"""
    import re

    # Only convert for SEC collections (large collections benefit most)
    if 'sec_sentences' not in aql_query and 'sec_sections' not in aql_query:
        return aql_query

    # Pattern: FOR var IN sec_sentences FILTER ... CONTAINS(LOWER(var.text), 'keyword')
    # Convert to: FOR var IN FULLTEXT(sec_sentences, 'text', 'keyword') FILTER ...

    # Match CONTAINS pattern with single-word keywords
    pattern = r"FOR\s+(\w+)\s+IN\s+(sec_sentences|sec_sections)\s+FILTER\s+(.*?)CONTAINS\(LOWER\(\1\.text\),\s*['\"]([^'\"]+)['\"]\)"

    def replace_with_fulltext(match):
        var_name = match.group(1)
        collection = match.group(2)
        other_filters = match.group(3).strip()
        keyword = match.group(4).strip()

        # Only convert single-word keywords (FULLTEXT works best with these)
        if ' ' in keyword:
            return match.group(0)  # Keep original for multi-word

        # Build replacement
        # Remove trailing AND/OR from other_filters if present
        other_filters = re.sub(r'\s+(AND|OR)\s*$', '', other_filters, flags=re.IGNORECASE)

        if other_filters:
            # Keep other filters after FULLTEXT
            return f"FOR {var_name} IN FULLTEXT({collection}, 'text', '{keyword}') FILTER {other_filters}"
        else:
            # No other filters, just FULLTEXT
            return f"FOR {var_name} IN FULLTEXT({collection}, 'text', '{keyword}')"

    converted = re.sub(pattern, replace_with_fulltext, aql_query, flags=re.DOTALL | re.IGNORECASE)

    # Log if conversion happened
    if converted != aql_query:
        st.info("⚡ Auto-optimized: Converted CONTAINS to FULLTEXT (10x faster for SEC queries)")

    return converted


def suggest_fulltext_conversion(aql_query):
    """Suggest how to convert CONTAINS to FULLTEXT (legacy function, now replaced by apply_fulltext_conversion)"""
    import re

    suggestions = []

    # Find all CONTAINS patterns
    pattern = r"FOR (\w+) IN (sec_sentences|sec_sections)\s+(.*?)CONTAINS\(LOWER\(\1\.text\),\s*['\"]([^'\"]+)['\"]\)"
    matches = re.findall(pattern, aql_query, re.DOTALL)

    for var_name, collection, filters_between, keyword in matches:
        if ' ' not in keyword.strip():
            # Single word - can use FULLTEXT
            suggestion = f"""
💡 FULLTEXT Optimization Available:

Current (slow):
  FOR {var_name} IN {collection}
  {filters_between.strip()}CONTAINS(LOWER({var_name}.text), '{keyword}')

Optimized (10x faster):
  FOR {var_name} IN FULLTEXT({collection}, 'text', '{keyword}')
  {filters_between.strip()}/* CONTAINS removed - FULLTEXT handles it */

Requirement: Create fulltext index first:
  db.collection('{collection}').add_fulltext_index(fields=['text'], min_length=3)
"""
            suggestions.append(suggestion)

    return suggestions


@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def _execute_query_cached(aql_query, bind_vars_json, raise_on_error=False):
    """Internal cached query execution - caches by query + bind vars"""
    # Deserialize bind vars
    bind_vars = json.loads(bind_vars_json)

    # Reconstruct plan for actual execution
    plan = {
        "aql_query": aql_query,
        "bind_vars": bind_vars
    }

    # Execute without caching to avoid recursion
    return _execute_query_internal(plan, raise_on_error)


def _execute_query_internal(plan, raise_on_error=False):
    """Internal query execution (no caching) - used by cached wrapper"""
    if not plan or 'aql_query' not in plan:
        if raise_on_error:
            raise ValueError("Invalid query plan: missing aql_query")
        return []

    db = arango_db.get_arango_connection()
    if not db:
        if raise_on_error:
            raise ConnectionError("Failed to connect to ArangoDB")
        return []

    try:
        aql_query = plan.get("aql_query", "")
        bind_vars = plan.get("bind_vars", {})

        # Step 0: Quick validation of bind variables (fail fast before expensive operations)
        # Extract required bind variables from query (simple regex check)
        required_bind_vars_quick = set(re.findall(r'@(\w+)', aql_query))
        # Exclude query_vector since it's handled specially later
        required_bind_vars_quick.discard('query_vector')

        missing_vars_quick = required_bind_vars_quick - set(bind_vars.keys())
        if missing_vars_quick and not plan.get("requires_embedding"):
            # Fail fast if basic bind vars are missing (before cost estimation)
            st.error(f"❌ Missing required bind variables: {missing_vars_quick}")
            st.caption("Query plan is incomplete. This is likely an LLM generation error.")
            return []

        # Step 1: Estimate query cost
        cost_score, cost_issues = estimate_query_cost(aql_query)

        # COMPLEXITY GATE: Reject queries that are too expensive even after optimization
        MAX_ACCEPTABLE_COST = 85

        if cost_score > MAX_ACCEPTABLE_COST:
            # Try optimization first
            st.warning(f"⚠️ Very expensive query detected (cost score: {cost_score}/150)")
            st.info("🔧 Attempting automatic optimization...")

            aql_query_optimized, optimizations = optimize_expensive_query(aql_query)
            aql_query_optimized = apply_fulltext_conversion(aql_query_optimized)

            # Re-estimate cost after optimization
            cost_after_optimization, _ = estimate_query_cost(aql_query_optimized)

            if cost_after_optimization > MAX_ACCEPTABLE_COST:
                # Still too expensive, reject and provide guidance
                if raise_on_error:
                    raise ValueError(f"Query too complex (cost: {cost_after_optimization})")

                st.error("❌ Query too complex to execute safely")
                with st.expander("💡 How to Simplify Your Query"):
                    st.markdown("""
**Your query is too expensive and may timeout. Please try:**

1. **Add a specific date range:**
   - Instead of: "all awards"
   - Try: "awards in the last 6 months"

2. **Filter by ticker:**
   - Instead of: "companies with negative sentiment"
   - Try: "AAPL, MSFT, GOOGL with negative sentiment"

3. **Limit result count:**
   - Add "top 10" or "show me 20 results"

4. **Break into smaller questions:**
   - Instead of: "cybersecurity risks + financials + sentiment"
   - Try each separately

**Cost score:** {cost_after_optimization}/150 (max: {MAX_ACCEPTABLE_COST})
                    """)

                with st.expander("🐛 Query Details"):
                    st.code(aql_query, language="sql")
                    if cost_issues:
                        st.write("**Issues found:**")
                        for issue in cost_issues:
                            st.caption(f"  - {issue}")

                return []

            # Optimization succeeded, use optimized query
            aql_query = aql_query_optimized
            st.success(f"✅ Optimized to acceptable cost ({cost_after_optimization}/150)")
            plan["aql_query"] = aql_query
            cost_score = cost_after_optimization

        elif cost_score > 70:
            st.warning(f"⚠️ Expensive query detected (cost score: {cost_score}/150)")

            # Apply automatic optimizations
            st.info("🔧 Analyzing optimizations...")
            aql_query, optimizations = optimize_expensive_query(aql_query)

            # Note: FULLTEXT conversion now happens automatically in Step 2.5 (no manual suggestion needed)

            plan["aql_query"] = aql_query
        
        # Step 2: Validate and fix syntax
        aql_query, syntax_errors, required_bind_vars = validate_aql_syntax(aql_query)

        if syntax_errors:
            pass
            # st.warning(f"⚠️ Query issues detected and auto-fixed:")
            # for error in syntax_errors:
            #     st.caption(f"  - {error}")

        # Step 2.5: Auto-apply FULLTEXT optimization for SEC queries
        aql_query = apply_fulltext_conversion(aql_query)

        # Update the plan with fixed query
        plan["aql_query"] = aql_query
        
        # Step 3: Check required bind variables
        missing_vars = required_bind_vars - set(bind_vars.keys())
        
        # Handle embeddings
        if "query_vector" in required_bind_vars:
            if plan.get("requires_embedding") and plan.get("embedding_text"):
                embedding = get_query_embedding(plan["embedding_text"])
                if embedding:
                    bind_vars["query_vector"] = embedding
                    missing_vars.discard("query_vector")
                else:
                    st.error("Failed to generate embedding for semantic search")
                    return []
            else:
                st.error("Query requires @query_vector but no embedding_text provided")
                return []
        
        # Remove requires_embedding from bind_vars if it was added
        if "requires_embedding" in bind_vars:
            del bind_vars["requires_embedding"]
        
        if missing_vars:
            st.error(f"❌ Missing bind variables: {missing_vars}")
            with st.expander("🐛 Debug"):
                st.write(f"Required: {required_bind_vars}")
                st.write(f"Provided: {set(bind_vars.keys())}")
            return []
        
        # Step 4: Execute with timeout protection
        # Calculate timeout based on query cost (higher cost = longer timeout)
        timeout = min(60 + (cost_score // 10), cfg.QUERY_TIMEOUT)  # Uses config.QUERY_TIMEOUT (360s)

        with st.spinner(f"Executing query... (timeout: {timeout}s)"):
            cursor = db.aql.execute(
                aql_query,
                bind_vars=bind_vars,
                ttl=timeout + 10,  # TTL slightly longer than max_runtime
                max_runtime=float(timeout),  # Use calculated timeout instead of hardcoded 300.0
                batch_size=5000,
                optimizer_rules=["+all"]
            )
            results = list(cursor)
        
        # Show performance warning if query was expensive
        if cost_score > 70:
            if len(results) > 0:
                st.success(f"✅ Retrieved {len(results)} results (query was expensive but completed)")
            else:
                st.warning("Query completed but returned no results. Try adding more specific filters.")
        
        return results
        
    except Exception as e:
        error_msg = str(e)

        # If raise_on_error=True, re-raise for retry logic (don't show UI errors yet)
        if raise_on_error:
            raise  # Re-raise the exception for execute_with_retry to handle

        # Otherwise, show UI error messages and return empty list
        # Check if timeout error
        if "Read timed out" in error_msg or "timeout" in error_msg.lower():
            st.error("⏱️ Query timed out (took longer than 60 seconds)")
#             with st.expander("💡 How to Fix Timeout Issues"):
#                 st.markdown("""
# **Your query is too expensive. Try these fixes:**

# 1. **Add more specific filters:**
#    - Instead of: "all cybersecurity risks"
#    - Try: "cybersecurity risks in 2024" or "cybersecurity risks for AAPL"

# 2. **Limit results:**
#    - Add "top 10" or "show me 20" to your question

# 3. **Use tickers instead of concepts:**
#    - Instead of: "tech companies with negative sentiment"
#    - Try: "AAPL, MSFT, GOOGL with negative sentiment"

# 4. **Break into simpler questions:**
#    - Instead of: "cybersecurity risks + cash flow + EPS"
#    - Try: "Which companies mention cybersecurity risks?" (then ask about financials separately)
#                 """)
#             with st.expander("🐛 Debug Query"):
#                 st.code(plan.get("aql_query", ""), language="sql")
#                 st.json(plan.get("bind_vars", {}))

#             # Show cost analysis
#             cost, issues = estimate_query_cost(plan.get("aql_query", ""))
#             st.write(f"**Query Cost Score:** {cost}/150")
#             if issues:
#                 st.write("**Issues:**")
#                 for issue in issues:
#                     st.caption(f"  - {issue}")
#         else:
#             # Other error
#             st.error(f"Query execution error: {error_msg}")
#             with st.expander("🐛 Debug Query"):
#                 st.code(plan.get("aql_query", ""), language="sql")
#                 st.json(plan.get("bind_vars", {}))

        return []


def execute_planned_query(plan, raise_on_error=False):
    """Execute query with caching and timeout protection

    Args:
        plan: Query plan dict with aql_query and bind_vars
        raise_on_error: If True, re-raises exceptions for retry logic

    Returns:
        List of query results (or empty list on error)
    """
    aql_query = plan.get("aql_query", "")
    bind_vars = plan.get("bind_vars", {})

    # Serialize bind_vars for caching (must be JSON-serializable)
    # Remove non-serializable items (like query_vector which is a list)
    serializable_bind_vars = {}
    for k, v in bind_vars.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            serializable_bind_vars[k] = v
        elif isinstance(v, list) and k != "query_vector":
            serializable_bind_vars[k] = v

    bind_vars_json = json.dumps(serializable_bind_vars, sort_keys=True)

    # Check if we can use cache (only for non-embedding queries)
    if plan.get("requires_embedding") or "query_vector" in bind_vars:
        # Skip cache for semantic search (embedding vectors not serializable)
        return _execute_query_internal(plan, raise_on_error)

    # Use cached execution
    try:
        # Show cache hit indicator
        cache_key = hashlib.md5(f"{aql_query}{bind_vars_json}".encode()).hexdigest()[:8]
        results = _execute_query_cached(aql_query, bind_vars_json, raise_on_error)

        # Indicate cache hit (only if results exist)
        if results:
            st.caption(f"⚡ Results retrieved from cache (key: {cache_key})")

        return results
    except Exception as e:
        # If caching fails, fall back to direct execution
        st.warning(f"Cache error, executing directly: {str(e)}")
        return _execute_query_internal(plan, raise_on_error)


def format_results_for_llm(results, query_plan=None):
    """
    Format query results for LLM consumption
    Handles both dictionary objects and scalar values (counts, aggregates, etc.)
    """
    # Handle empty results
    if not results:
        return "No results found."
    
    # Handle None
    if results is None:
        return "Query returned no data."
    
    formatted = []
    for doc in results:
        # Case 1: Dictionary (normal document)
        if isinstance(doc, dict):
            # Remove internal fields (_id, _key, _rev)
            clean_doc = {k: v for k, v in doc.items()
                        if not k.startswith('_')}
            formatted.append(clean_doc)
        
        # Case 2: List or tuple
        elif isinstance(doc, (list, tuple)):
            formatted.append(list(doc))
        
        # Case 3: Simple value (int, float, str, bool, None)
        else:
            formatted.append(doc)
    
    # If single scalar value, format nicely
    if len(formatted) == 1:
        value = formatted[0]
        # Single number
        if isinstance(value, (int, float)):
            return f"Result: {value:,}" if isinstance(value, int) else f"Result: {value:.2f}"
        # Single dict
        elif isinstance(value, dict):
            return formatted
        # Single string/other
        else:
            return f"Result: {value}"
    
    # Multiple results
    return formatted


def create_analysis_prompt(question, formatted_context, plan):
    """
    Create prompt for final LLM analysis with domain expertise
    """
    prompt = f"""You are a quantitative financial analyst providing insights from a multi-source graph database containing:
- Stock market data (OHLCV + 40+ indicators)
- Government contracts (USASpending.gov)
- Macroeconomic indicators (FRED)
- SEC filings with sentiment analysis
- Commodity positions (CFTC data)
- Prediction markets (Polymarket & Kalshi)

DATABASE QUERY EXECUTED:
Intent: {plan.get('intent', 'Unknown')}
Strategy: {plan.get('explanation', 'Data retrieved from graph database')}

RETRIEVED DATA:
{formatted_context}

USER QUESTION: {question}

ANALYSIS INSTRUCTIONS:
1. Answer the question directly and concisely using ONLY the provided data
2. For financial data: highlight trends, anomalies, patterns, or notable values
3. For government awards: focus on amounts, agencies, recipients, timing
4. For macroeconomic data: provide context and interpretation
5. For commodity data: explain positions and market indicators
6. For prediction markets: explain probabilities and trading volumes
7. If multiple results exist, format as a Markdown table with relevant columns
8. Cite specific data points using [1], [2], etc. corresponding to result numbers
9. Provide quantitative summary when applicable (totals, averages, ranges, changes)
10. If data is incomplete or missing, explicitly state what's absent
11. For semantic searches: explain why results are relevant
12. Keep response concise and data-focused (avoid unnecessary preamble)

FORMAT GUIDELINES:
- Single values: Direct answer with citation
- Multiple items: Markdown table with key columns
- Time-series: Show trends and notable changes
- Comparisons: Highlight differences and similarities

ANSWER:"""
    
    return prompt


def get_llm_analysis(prompt, use_local=False):
    """Get analysis from model with explicit logging"""
    print(f"🔍 get_llm_analysis called with use_local={use_local}")  # DEBUG
    
    use_local = False  # Force OpenAI for now
    
    if use_local:
        print("🟢 Attempting local model...")  # DEBUG
        try:
            from local_llm import get_local_llm
            llm = get_local_llm()
            print("✅ Local model loaded")  # DEBUG
            
            result = llm.generate(prompt, max_tokens=512, temperature=0.1)
            
            # Extract response
            if "assistant<|end_header_id|>" in result:
                response = result.split("assistant<|end_header_id|>")[-1].strip()
            else:
                response = result
            
            response = response.replace("<|eot_id|>", "").strip()
            print(f"✅ Local model generated {len(response)} chars")  # DEBUG
            return response
            
        except Exception as e:
            print(f"❌ Local model failed: {e}")  # DEBUG
            import traceback
            traceback.print_exc()
            # DON'T fall back - raise error to see what's wrong
            raise Exception(f"Local model failed: {e}")
    
    # OpenAI path
    print("🔵 Using OpenAI...")  # DEBUG
    import openai
    
    response = openai.chat.completions.create(
        model=cfg.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=cfg.MAX_TOKENS,
        temperature=0.2,
    )
    
    return response.choices[0].message.content


# """LLM query planning, analysis, and validation"""
# import openai
# import json
# from config import *
# from prompts import *
# import streamlit as st
# import re
# import openai
# from datetime import datetime
# import json
# import database as arango_db

# def preprocess_query(question):
#     """Handle simple queries without LLM"""
    
#     # Pattern 1: Single ticker mention
#     ticker_match = re.search(r'\b([A-Z]{2,5})\b', question)
#     if ticker_match and len(question.split()) <= 5:
#         ticker = ticker_match.group(1)
        
#         # "AAPL" or "show me AAPL"
#         if any(word in question.lower() for word in ['show', 'get', 'find']):
#             return {
#                 "intent": "ticker_lookup",
#                 "aql_query": "FOR doc IN awards FILTER doc.ticker == @ticker SORT doc.Action_Date DESC LIMIT 20 RETURN doc",
#                 "bind_vars": {"ticker": ticker},
#                 "requires_embedding": False,
#                 "explanation": f"Simple ticker lookup for {ticker}"
#             }
    
#     # Pattern 2: Count queries
#     if question.lower().startswith("how many"):
#         if "awards" in question.lower():
#             return {
#                 "intent": "count_awards",
#                 "aql_query": "RETURN LENGTH(awards)",
#                 "bind_vars": {},
#                 "requires_embedding": False,
#                 "explanation": "Count total awards"
#             }
    
#     # Pattern 3: Latest/recent queries
#     if any(word in question.lower() for word in ['latest', 'recent', 'last']):
#         return {
#             "intent": "recent_data",
#             "aql_query": "FOR doc IN awards SORT doc.Action_Date DESC LIMIT 10 RETURN doc",
#             "bind_vars": {},
#             "requires_embedding": False,
#             "explanation": "Fetch most recent awards"
#         }
    
#     # No rule matched → send to LLM
#     return None


# def get_query_embedding(text):
#     """Generate embedding vector for semantic search"""
#     try:
#         response = openai.embeddings.create(
#             model="text-embedding-3-small",
#             input=text
#         )
#         return response.data[0].embedding
#     except Exception as e:
#         st.error(f"Embedding generation error: {str(e)}")
#         return None

# def plan_query_with_llm(question, intent_hint=None, use_local=False):
#     # Try rules first
#     rule_result = preprocess_query(question)
#     if rule_result:
#         st.info("✓ Handled by rule (no LLM call)")
#         return rule_result
    
#     current_date = datetime.now().strftime("%Y-%m-%d")
    
#     # Add intent hint to prompt
#     hint_text = ""
#     if intent_hint:
#         if intent_hint.get("type") == "ticker":
#             hint_text = f"\n\n🎯 CONFIRMED: This is a TICKER query for '{intent_hint.get('value')}'. Use doc.ticker == @ticker"
#         elif intent_hint.get("type") == "concept":
#             hint_text = f"\n\n🎯 CONFIRMED: This is a CONCEPT/SEMANTIC query about '{intent_hint.get('value')}'. Use semantic search with embeddings."
    
#     relevant_schema = get_relevant_schema(question, intent_hint)
#     planning_prompt = f"""You are a database query planner for ArangoDB.

# {relevant_schema}

# {SCHEMA_DESCRIPTION}

# {FEW_SHOT_EXAMPLES}

# USER QUESTION: "{question}"{hint_text}
# Current Date: {current_date}

# Generate a JSON response with:
# - "intent": classification
# - "collections": array
# - "requires_embedding": boolean (true for concept queries)
# - "embedding_text": text (if semantic)
# - "aql_query": AQL query
# - "bind_vars": object
# - "explanation": strategy explanation

# Return ONLY valid JSON.

# Response:"""

#     # Rest of your existing function...
#     try:
#         response = openai.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": planning_prompt}],
#             max_tokens=1200,
#             temperature=0.1,
#             response_format={"type": "json_object"}
#         )
#         return json.loads(response.choices[0].message.content)
#     except Exception as e:
#         st.error(f"Query planning error: {str(e)}")
#         return None


# def quick_intent_check(question, use_local=False):
#     """Quick LLM call to determine if ticker or semantic query"""
    
#     check_prompt = f"""Question: "{question}"

# Is this asking about a TICKER SYMBOL or a CONCEPT?

# TICKER: Question mentions a specific stock ticker (2-5 uppercase letters like AAPL, CMI, TSLA)
# CONCEPT: Question asks about a topic/theme (AI, cybersecurity, renewable energy, etc.)

# Examples:
# - "CMI awards" → TICKER (CMI is Cummins stock ticker)
# - "awards related to AI" → CONCEPT (AI = artificial intelligence topic)
# - "TSLA in 2024" → TICKER
# - "renewable energy contracts" → CONCEPT

# Return JSON: {{"type": "ticker", "value": "CMI"}} or {{"type": "concept", "value": "artificial intelligence"}}
# """

#     try:
#         response = openai.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": check_prompt}],
#             max_tokens=100,
#             temperature=0,
#             response_format={"type": "json_object"}
#         )
#         return json.loads(response.choices[0].message.content)
#     except:
#         return {"type": "unknown"}

# def generate_follow_up_questions(user_question, results, query_plan):
#     """Generate contextual follow-up questions based on results"""
    
#     intent = query_plan.get('intent', '')
#     collections = query_plan.get('collections', [])
    
#     follow_ups = []
    
#     # Pattern 1: Temporal expansion
#     if 'date' in str(results).lower() or any(c in collections for c in ['MarketData', 'EconomicData']):
#         follow_ups.append(f"📈 How has this changed over the past year?")
#         follow_ups.append(f"📊 Show me the trend for the last 5 years")
    
#     # Pattern 2: Entity expansion (if tickers found)
#     if results and 'ticker' in str(results[0]):
#         tickers = [r.get('ticker') for r in results[:3] if r.get('ticker')]
#         if tickers:
#             follow_ups.append(f"💼 Compare {', '.join(tickers[:3])} financial metrics")
#             follow_ups.append(f"🔍 What are the biggest risks for {tickers[0]}?")
    
#     # Pattern 3: Cross-collection expansion
#     if 'Award' in collections:
#         follow_ups.append(f"📉 What's the stock performance for these companies?")
#         follow_ups.append(f"😟 Do any of these companies have negative SEC sentiment?")
    
#     if 'sec_filings' in collections or 'sec_sentences' in collections:
#         follow_ups.append(f"💰 Show me government contracts for these companies")
#         follow_ups.append(f"📊 What are their financial metrics?")
    
#     if 'MarketData' in collections:
#         follow_ups.append(f"🏛️ Have these companies received government contracts?")
#         follow_ups.append(f"⚠️ What risks do they mention in SEC filings?")
    
#     # Pattern 4: Depth expansion
#     if len(results) > 5:
#         follow_ups.append(f"🎯 Show me only the top 3 results with more detail")
#         follow_ups.append(f"📋 Export this data to CSV")
    
#     # Pattern 5: Comparative analysis
#     if len(results) >= 2:
#         follow_ups.append(f"⚖️ Compare the top 3 companies side-by-side")
    
#     # Pattern 6: "Rabbit in the hat" - unexpected connections
#     follow_ups.append(f"🎩 Find surprising correlations in this data")
    
#     return follow_ups[:4]  # Return top 4
    
# def validate_aql_syntax(aql_query):
#     """Basic syntax validation before execution"""
#     errors = []
    
#     # Check for common typos
#     if "compan." in aql_query and "company" in aql_query:
#         errors.append("Typo detected: 'compan.' should be 'company.'")
#         aql_query = aql_query.replace("compan.", "company.")
    
#     # Fix: INTO keyword (doesn't exist in AQL)
#     if ' INTO ' in aql_query.upper():
#         errors.append("⚠️ AQL syntax error: INTO keyword is not supported")
        
#         # Try to auto-fix: Convert INTO pattern to LET
#         # Pattern: RETURN x INTO var → LET var = (RETURN x)
        
#         # Check if it's a multi-statement query with INTO
#         if 'RETURN' in aql_query and 'INTO' in aql_query:
#             st.warning("🔧 Detected invalid INTO syntax. Attempting to rewrite as LET...")
            
#             # This is complex to auto-fix, so just flag it
#             errors.append("Cannot auto-fix: Query needs manual rewrite using LET or subquery")
#             errors.append("Recommendation: Use LET variable = (subquery) instead of RETURN ... INTO")
#     # Check for undeclared variables
#     # Find all variable declarations (FOR var IN ...)
#     declared_vars = set(re.findall(r'FOR\s+(\w+)\s+IN', aql_query, re.IGNORECASE))
    
#     # Find all variable uses (var.field)
#     used_vars = set(re.findall(r'(\w+)\.', aql_query))
    
#     # Check if any used vars aren't declared
#     undeclared = used_vars - declared_vars - {'doc', 'data'}  # doc and data are common
#     if undeclared:
#         errors.append(f"Undeclared variables: {undeclared}")
    
#     # Check bind variables
#     bind_params = set(re.findall(r'@(\w+)', aql_query))
    
#     return aql_query, errors, bind_params


# def get_relevant_schema(question, intent):
#     """Return only relevant schema for this query"""
    
#     # Collection descriptions
#     schemas = {
#         "awards": {
#             "description": "Government contract awards",
#             "key_fields": ["ticker", "Award_Amount", "Action_Date", "Awarding_Agency", "description"],
#             "sample_query": "FOR doc IN awards FILTER doc.ticker == @ticker RETURN doc"
#         },
#         "companies": {
#             "description": "Company master data",
#             "key_fields": ["ticker", "name", "sector", "industry"],
#             "sample_query": "FOR c IN companies FILTER c.ticker == @ticker RETURN c"
#         },
#         "market_data": {
#             "description": "Daily OHLCV + indicators",
#             "key_fields": ["ticker", "date", "close", "volume", "rsi_14", "macd"],
#             "sample_query": "FOR m IN market_data FILTER m.ticker == @ticker AND m.date >= @start_date RETURN m"
#         },
#         "sec_filings": {
#             "description": "10-K/10-Q embeddings + sentiment",
#             "key_fields": ["ticker", "filing_date", "embedding", "sentiment_score"],
#             "sample_query": "FOR doc IN sec_filings FILTER COSINE_SIMILARITY(doc.embedding, @query_vector) > 0.7 RETURN doc"
#         }
#     }
    
#     # Keyword mapping
#     relevant_collections = []
    
#     if any(word in question.lower() for word in ['award', 'contract', 'government']):
#         relevant_collections.append("awards")
    
#     if any(word in question.lower() for word in ['price', 'stock', 'market', 'volume', 'rsi', 'macd']):
#         relevant_collections.append("market_data")
    
#     if any(word in question.lower() for word in ['filing', '10-k', '10-q', 'sec', 'sentiment']):
#         relevant_collections.append("sec_filings")
    
#     # Always include companies for joins
#     if len(relevant_collections) > 0:
#         relevant_collections.append("companies")
    
#     # If nothing matched, include awards (default)
#     if not relevant_collections:
#         relevant_collections = ["awards", "companies"]
    
#     # Build focused schema description
#     focused_schema = "\n\n".join([
#         f"Collection: {coll}\n" +
#         f"Description: {schemas[coll]['description']}\n" +
#         f"Key Fields: {', '.join(schemas[coll]['key_fields'])}\n" +
#         f"Example: {schemas[coll]['sample_query']}"
#         for coll in relevant_collections if coll in schemas
#     ])
    
#     return focused_schema

# def execute_with_retry(plan, max_retries=2):
#     """Execute query with self-correction"""
    
#     for attempt in range(max_retries + 1):
#         try:
#             results = execute_planned_query(plan)
#             return results, None  # Success
            
#         except Exception as e:
#             error_msg = str(e)
            
#             if attempt == max_retries:
#                 return [], error_msg  # Give up
            
#             # Self-correct
#             st.warning(f"⚠️ Attempt {attempt + 1} failed: {error_msg}")
#             st.info("🔄 Asking LLM to fix query...")
            
#             # Re-prompt with error context
#             correction_prompt = f"""The following AQL query failed:

#             QUERY:
#             {plan['aql_query']}

#             BIND VARS:
#             {plan['bind_vars']}

#             ERROR:
#             {error_msg}

#             Generate a CORRECTED query that fixes this error.
#             Return JSON with same format: {{"aql_query": "...", "bind_vars": {{...}}}}
#             """
                        
#             try:
#                 response = openai.chat.completions.create(
#                     model="gpt-4o-mini",
#                     messages=[{"role": "user", "content": correction_prompt}],
#                     max_tokens=800,
#                     temperature=0,
#                     response_format={"type": "json_object"}
#                 )
                
#                 corrected = json.loads(response.choices[0].message.content)
                
#                 # Update plan with correction
#                 plan['aql_query'] = corrected.get('aql_query', plan['aql_query'])
#                 plan['bind_vars'] = corrected.get('bind_vars', plan['bind_vars'])
                
#                 st.success("✓ Query corrected, retrying...")
                
#             except Exception as correction_error:
#                 st.error(f"Self-correction failed: {correction_error}")
#                 return [], error_msg
            
#         return [], "Max retries exceeded"

#     # Replace execute_planned_query call with:
#     results, error = execute_with_retry(plan)
#     if error:
#         st.error(f"Query failed after retries: {error}")


# def estimate_query_cost(aql_query):
#     """Estimate if query will timeout and identify issues"""
    
#     issues = []
#     cost_score = 0
    
#     # High cost: Full collection scan without early LIMIT
#     query_before_return = aql_query.split('RETURN')[0] if 'RETURN' in aql_query else aql_query
#     if 'FOR' in aql_query and 'LIMIT' not in query_before_return:
#         issues.append("No LIMIT before RETURN (will scan entire collection)")
#         cost_score += 50
    
#     # High cost: Text search on large collections without index
#     if 'CONTAINS' in aql_query and 'sec_sentences' in aql_query:
#         issues.append("CONTAINS() on sec_sentences (slow - use FULLTEXT index)")
#         cost_score += 40
    
#     if 'CONTAINS' in aql_query and 'sec_sections' in aql_query:
#         issues.append("CONTAINS() on sec_sections (slow)")
#         cost_score += 35
    
#     # High cost: Multiple nested FOR loops
#     nested_fors = aql_query.count('FOR')
#     if nested_fors > 2:
#         issues.append(f"Multiple nested FOR loops ({nested_fors}) - use COLLECT")
#         cost_score += 25 * (nested_fors - 1)
    
#     # High cost: Cross-collection join without proper indexing
#     if 'MarketData' in aql_query and any(coll in aql_query for coll in ['sec_sentences', 'sec_sections', 'Award']):
#         if 'COLLECT' not in aql_query:
#             issues.append("Cross-collection join without COLLECT (expensive)")
#             cost_score += 30
    
#     # Very high cost: Nested FOR without LIMIT in subquery
#     if 'LET' in aql_query and 'FOR' in aql_query.split('LET')[1]:
#         subquery_text = aql_query.split('LET')[1].split(')')[0] if ')' in aql_query.split('LET')[1] else ''
#         if 'LIMIT' not in subquery_text and 'FIRST' not in subquery_text:
#             issues.append("Subquery without LIMIT or FIRST() (will process all rows)")
#             cost_score += 40
    
#     return cost_score, issues


# def optimize_expensive_query(aql_query):
#     """Apply automatic optimizations to prevent timeout"""
    
#     optimizations_applied = []
    
#     # Optimization 1: Add LIMIT if missing before first RETURN
#     query_before_return = aql_query.split('RETURN')[0] if 'RETURN' in aql_query else ''
#     if 'FOR' in aql_query and 'LIMIT' not in query_before_return:
#         # Find position to insert LIMIT (after SORT if exists, otherwise before RETURN)
#         if 'SORT' in query_before_return:
#             # Add LIMIT after last SORT
#             parts = aql_query.split('SORT')
#             if len(parts) > 1:
#                 # Find end of SORT clause (next line break or RETURN)
#                 sort_end = parts[-1].find('\n')
#                 if sort_end != -1:
#                     aql_query = aql_query[:aql_query.rfind('SORT')] + \
#                                 'SORT' + parts[-1][:sort_end] + \
#                                 '\n  LIMIT 100' + parts[-1][sort_end:]
#                     optimizations_applied.append("Added LIMIT 100 after SORT")
#         else:
#             # Add LIMIT before RETURN
#             aql_query = aql_query.replace('RETURN', 'LIMIT 100\n  RETURN', 1)
#             optimizations_applied.append("Added LIMIT 100 before RETURN")
    
#     # Optimization 2: Replace CONTAINS with suggestion (can't auto-fix)
#     if 'CONTAINS(LOWER(' in aql_query and 'sec_sentences' in aql_query:
#         optimizations_applied.append("⚠️ Recommend: Use FULLTEXT() instead of CONTAINS() for better performance")
    
#     # Optimization 3: Add sentiment filter if searching SEC content without it
#     if ('sec_sentences' in aql_query or 'sec_sections' in aql_query) and \
#        'finbert_score' not in aql_query and \
#        'FILTER' in aql_query:
#         optimizations_applied.append("⚠️ Recommend: Add sentiment filter (e.g., FILTER doc.finbert_score < -0.3)")
    
#     return aql_query, optimizations_applied


# def execute_planned_query(plan):
#     """Execute query with timeout protection and optimization"""
    
#     if not plan or 'aql_query' not in plan:
#         return []
    
#     db = arango_db.get_arango_connection()
#     if not db:
#         return []
    
#     try:
#         aql_query = plan.get("aql_query", "")
#         bind_vars = plan.get("bind_vars", {})
        
#         # Step 1: Estimate query cost BEFORE validation
#         cost_score, cost_issues = estimate_query_cost(aql_query)
        
#         if cost_score > 70:
#             st.warning(f"⚠️ Expensive query detected (cost score: {cost_score}/150)")
#             with st.expander("💡 Performance Issues Detected"):
#                 for issue in cost_issues:
#                     st.caption(f"  - {issue}")
            
#             # Apply automatic optimizations
#             st.info("🔧 Applying automatic optimizations...")
#             aql_query, optimizations = optimize_expensive_query(aql_query)
            
#             if optimizations:
#                 for opt in optimizations:
#                     st.success(f"✓ {opt}")
            
#             plan["aql_query"] = aql_query
        
#         # Step 2: Validate and fix syntax
#         aql_query, syntax_errors, required_bind_vars = validate_aql_syntax(aql_query)
        
#         if syntax_errors:
#             st.warning(f"⚠️ Query issues detected and auto-fixed:")
#             for error in syntax_errors:
#                 st.caption(f"  - {error}")
        
#         # Update the plan with fixed query
#         plan["aql_query"] = aql_query
        
#         # Step 3: Check required bind variables
#         missing_vars = required_bind_vars - set(bind_vars.keys())
        
#         # Handle embeddings
#         if "@query_vector" in required_bind_vars:
#             if plan.get("requires_embedding") and plan.get("embedding_text"):
#                 embedding = get_query_embedding(plan["embedding_text"])
#                 if embedding:
#                     bind_vars["query_vector"] = embedding
#                     missing_vars.discard("query_vector")
#                 else:
#                     st.error("Failed to generate embedding for semantic search")
#                     return []
#             else:
#                 st.error("Query requires @query_vector but no embedding_text provided")
#                 return []
        
#         # Remove requires_embedding from bind_vars if it was added
#         if "requires_embedding" in bind_vars:
#             del bind_vars["requires_embedding"]
        
#         if missing_vars:
#             st.error(f"❌ Missing bind variables: {missing_vars}")
#             with st.expander("🐛 Debug"):
#                 st.write(f"Required: {required_bind_vars}")
#                 st.write(f"Provided: {set(bind_vars.keys())}")
#             return []
        
#         # Step 4: Execute with timeout protection
#         # Increase timeout for complex queries, but cap at 90s
#         timeout = min(60 + (cost_score // 10), 90)
        
#         with st.spinner(f"Executing query... (timeout: {timeout}s)"):
#             cursor = db.aql.execute(
#                 aql_query, 
#                 bind_vars=bind_vars,
#                 ttl=timeout,
#                 batch_size=1000,
#                 optimizer_rules=["+all"]
#             )
            
#             results = list(cursor)
        
#         # Show performance warning if query was expensive
#         if cost_score > 70:
#             if len(results) > 0:
#                 st.success(f"✅ Retrieved {len(results)} results (query was expensive but completed)")
#             else:
#                 st.warning("Query completed but returned no results. Try adding more specific filters.")
        
#         return results
        
#     except Exception as e:
#         error_msg = str(e)
        
#         # Check if timeout error
#         if "Read timed out" in error_msg or "timeout" in error_msg.lower():
#             st.error("⏱️ Query timed out (took longer than 60 seconds)")
            
#             with st.expander("💡 How to Fix Timeout Issues"):
#                 st.markdown("""
#                 **Your query is too expensive. Try these fixes:**
                
#                 1. **Add more specific filters:**
#                    - Instead of: "all cybersecurity risks"
#                    - Try: "cybersecurity risks in 2024" or "cybersecurity risks for AAPL"
                
#                 2. **Limit results:**
#                    - Add "top 10" or "show me 20" to your question
                
#                 3. **Use tickers instead of concepts:**
#                    - Instead of: "tech companies with negative sentiment"
#                    - Try: "AAPL, MSFT, GOOGL with negative sentiment"
                
#                 4. **Break into simpler questions:**
#                    - Instead of: "cybersecurity risks + cash flow + EPS"
#                    - Try: "Which companies mention cybersecurity risks?" (then ask about financials separately)
#                 """)
            
#             with st.expander("🐛 Debug Query"):
#                 st.code(plan.get("aql_query", ""), language="sql")
#                 st.json(plan.get("bind_vars", {}))
                
#                 # Show cost analysis
#                 cost, issues = estimate_query_cost(plan.get("aql_query", ""))
#                 st.write(f"**Query Cost Score:** {cost}/150")
#                 if issues:
#                     st.write("**Issues:**")
#                     for issue in issues:
#                         st.caption(f"  - {issue}")
#         else:
#             # Other error
#             st.error(f"Query execution error: {error_msg}")
            
#             with st.expander("🐛 Debug Query"):
#                 st.code(plan.get("aql_query", ""), language="sql")
#                 st.json(plan.get("bind_vars", {}))
        
#         return []




# def format_results_for_llm(results, query_plan=None):
#     """
#     Format query results for LLM consumption
#     Handles both dictionary objects and scalar values (counts, aggregates, etc.)
#     """
    
#     # Handle empty results
#     if not results:
#         return "No results found."
    
#     # Handle None
#     if results is None:
#         return "Query returned no data."
    
#     formatted = []
    
#     for doc in results:
#         # Case 1: Dictionary (normal document)
#         if isinstance(doc, dict):
#             # Remove internal fields (_id, _key, _rev)
#             clean_doc = {k: v for k, v in doc.items() 
#                         if not k.startswith('_')}
#             formatted.append(clean_doc)
        
#         # Case 2: List or tuple
#         elif isinstance(doc, (list, tuple)):
#             formatted.append(list(doc))
        
#         # Case 3: Simple value (int, float, str, bool, None)
#         else:
#             formatted.append(doc)
    
#     # If single scalar value, format nicely
#     if len(formatted) == 1:
#         value = formatted[0]
        
#         # Single number
#         if isinstance(value, (int, float)):
#             return f"Result: {value:,}" if isinstance(value, int) else f"Result: {value:.2f}"
        
#         # Single dict
#         elif isinstance(value, dict):
#             return formatted
        
#         # Single string/other
#         else:
#             return f"Result: {value}"
    
#     # Multiple results
#     return formatted



# def create_analysis_prompt(question, formatted_context, plan):
#     """
#     Create prompt for final LLM analysis with domain expertise
#     """
#     prompt = f"""You are a quantitative financial analyst providing insights from a multi-source graph database containing market data, government contracts, macroeconomic indicators, and commodity positions.

# DATABASE QUERY EXECUTED:
# Intent: {plan.get('intent', 'Unknown')}
# Strategy: {plan.get('explanation', 'Data retrieved from graph database')}

# RETRIEVED DATA:
# {formatted_context}

# USER QUESTION: {question}

# ANALYSIS INSTRUCTIONS:
# 1. Answer the question directly and concisely using ONLY the provided data
# 2. For financial data: highlight trends, anomalies, patterns, or notable values
# 3. For government awards: focus on amounts, agencies, recipients, timing
# 4. For macroeconomic data: provide context and interpretation
# 5. For commodity data: explain positions and market indicators
# 6. If multiple results exist, format as a Markdown table with relevant columns
# 7. Cite specific data points using [1], [2], etc. corresponding to result numbers
# 8. Provide quantitative summary when applicable (totals, averages, ranges, changes)
# 9. If data is incomplete or missing, explicitly state what's absent
# 10. For semantic searches: explain why results are relevant
# 11. Keep response concise and data-focused (avoid unnecessary preamble)

# FORMAT GUIDELINES:
# - Single values: Direct answer with citation
# - Multiple items: Markdown table with key columns
# - Time-series: Show trends and notable changes
# - Comparisons: Highlight differences and similarities

# ANSWER:"""
    
#     return prompt


# # def get_llm_analysis(prompt):
# #     """Get analysis from OpenAI"""
# #     try:
# #         response = openai.chat.completions.create(
# #             model="gpt-4o-mini",
# #             messages=[{"role": "user", "content": prompt}],
# #             max_tokens=1500,
# #             temperature=0.2,
# #         )
# #         return response.choices[0].message.content
# #     except Exception as e:
# #         return f"OpenAI API error: {str(e)}"

# #chema context for the model
# def get_llm_analysis(prompt, use_local=False):
#     """Get analysis from model with explicit logging"""
    
#     print(f"🔍 get_llm_analysis called with use_local={use_local}")  # DEBUG
    
#     use_local = False
#     if use_local:
#         print("🟢 Attempting local model...")  # DEBUG
#         try:
#             from local_llm import get_local_llm
            
#             llm = get_local_llm()
#             print("✅ Local model loaded")  # DEBUG
            
#             result = llm.generate(prompt, max_tokens=512, temperature=0.1)
            
#             # Extract response
#             if "assistant<|end_header_id|>" in result:
#                 response = result.split("assistant<|end_header_id|>")[-1].strip()
#             else:
#                 response = result
            
#             response = response.replace("<|eot_id|>", "").strip()
            
#             print(f"✅ Local model generated {len(response)} chars")  # DEBUG
#             return response
            
#         except Exception as e:
#             print(f"❌ Local model failed: {e}")  # DEBUG
#             import traceback
#             traceback.print_exc()
#             # DON'T fall back - raise error to see what's wrong
#             raise Exception(f"Local model failed: {e}")
    
#     # OpenAI path
#     print("🔵 Using OpenAI...")  # DEBUG
#     import openai
    
#     response = openai.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}],
#         max_tokens=1500,
#         temperature=0.2,
#     )
#     return response.choices[0].message.content
