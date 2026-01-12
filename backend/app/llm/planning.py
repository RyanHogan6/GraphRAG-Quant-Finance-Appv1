"""
LLM query planning for FastAPI
Ported from Streamlit llm.py
"""
import openai
import json
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from app.llm.prompts import CRITICAL_AQL_RULES

# Set OpenAI API key
openai.api_key = config.OPENAI_API_KEY


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
        response = openai.chat.completions.create(
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


def plan_query_with_llm(question: str, intent_hint=None):
    """Generate query plan from natural language question"""
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Add intent hint to prompt
    hint_text = ""
    if intent_hint:
        if intent_hint.get("type") == "ticker":
            hint_text = f"\n\n🎯 CONFIRMED: This is a TICKER query for '{intent_hint.get('value')}'. Use doc.ticker == @ticker"
        elif intent_hint.get("type") == "concept":
            hint_text = f"\n\n🎯 CONFIRMED: This is a CONCEPT/SEMANTIC query about '{intent_hint.get('value')}'. Use semantic search with embeddings."

    planning_prompt = f"""You are a database query planner for ArangoDB.

{CRITICAL_AQL_RULES}

CRITICAL: AQL CLAUSE ORDER MUST BE:
FOR → FILTER → SORT → LIMIT → RETURN
⚠️ LIMIT always comes BEFORE RETURN!

EXAMPLE QUERIES (for reference):
- Ticker query: FOR doc IN Award FILTER doc.ticker == @ticker SORT doc.award_amount_float DESC LIMIT 10 RETURN doc
- Semantic query: FOR doc IN Award FILTER doc.description_embedding != null LET sim = COSINE_SIMILARITY(doc.description_embedding, @query_vector) FILTER sim >= 0.7 SORT sim DESC LIMIT 10 RETURN doc
- SEC sentiment: FOR doc IN sec_sentences FILTER CONTAINS(LOWER(doc.text), @keyword) AND doc.finbert_score < -0.3 LIMIT 20 RETURN doc
- Date range: FOR doc IN MarketData FILTER doc.ticker == @ticker AND doc.date >= DATE_SUBTRACT(DATE_NOW(), 180, "day") SORT doc.date DESC LIMIT 100 RETURN doc
- Simple query: FOR doc IN EconomicData FILTER doc.federal_funds_rate != null LIMIT 1 RETURN doc

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

Return ONLY valid JSON.

Response:"""

    try:
        response = openai.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": planning_prompt}],
            max_tokens=config.MAX_TOKENS,
            temperature=config.TEMPERATURE,
            response_format={"type": "json_object"}
        )

        plan = json.loads(response.choices[0].message.content)
        return plan

    except Exception as e:
        print(f"Query planning error: {str(e)}")
        return None


def get_query_embedding(text: str):
    """Generate embedding vector for semantic search"""
    try:
        response = openai.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding generation error: {str(e)}")
        return None


def analyze_results_with_llm(user_question: str, results: list, query_plan: dict):
    """Analyze query results and generate natural language response"""

    if not results:
        # Intelligent fallback when no results found
        return generate_no_results_response(user_question, query_plan)

    # Limit results sample for LLM analysis (avoid token limits)
    results_sample = results[:10] if len(results) > 10 else results
    result_count = len(results)

    analysis_prompt = f"""You are a financial data analyst. The user asked: "{user_question}"

We queried the database and found {result_count} results. Here's a sample:

{json.dumps(results_sample, indent=2)}

Please provide a clear, concise analysis in markdown format:
1. Summarize the key findings
2. Highlight important numbers, dates, or trends
3. Present data in a table if appropriate (use markdown tables)
4. Keep it professional and easy to read

Focus on answering the user's question directly. Don't mention technical details like query execution.

Response:"""

    try:
        response = openai.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": analysis_prompt}],
            max_tokens=1000,
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Analysis error: {str(e)}")
        # Fallback to simple formatting
        return f"Found {result_count} results. Here's a summary:\n\n" + "\n".join([f"- {json.dumps(r)[:100]}..." for r in results_sample[:5]])


def generate_no_results_response(user_question: str, query_plan: dict):
    """Generate helpful response when no results are found"""

    collections = query_plan.get('collections', [])
    intent = query_plan.get('intent', '')

    fallback_prompt = f"""The user asked: "{user_question}"

We searched our financial database but found no matching results.

Available data in our system:
- **Market Data**: Stock prices, technical indicators, fundamentals (OHLCV, P/E ratios, market cap, etc.)
- **Government Contracts**: Federal awards with semantic search capabilities
- **SEC Filings**: Corporate filings with sentiment analysis
- **Prediction Markets**: Polymarket and Kalshi market data
- **Economic Data**: Fed rates, S&P 500, unemployment, yield curves
- **Commodities**: CFTC trader positions

The query attempted to search: {', '.join(collections) if collections else 'multiple collections'}

Please provide a helpful response that:
1. Explains what the user was likely looking for
2. Suggests why no results were found (wrong ticker symbol? data not available? incorrect time range?)
3. Provides 2-3 alternative queries they could try that might yield results
4. If appropriate, provide general context about their question using your knowledge

Keep it conversational, helpful, and professional. Use markdown formatting.

Response:"""

    try:
        response = openai.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": fallback_prompt}],
            max_tokens=800,
            temperature=0.5
        )

        return "**No results found in database.**\n\n" + response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Fallback response error: {str(e)}")
        return f"""**No results found.**

Your query searched: {', '.join(collections) if collections else 'the database'}

**Possible reasons:**
- The data you're looking for might not be in our database
- Try checking the ticker symbol or company name
- Adjust your date range or search criteria

**Suggestions:**
- Browse the Markets page to see available prediction markets
- Try searching for a different company or topic
- Use the Database page to explore available collections"""


def generate_follow_up_questions(user_question: str, results: list, query_plan: dict):
    """Generate contextual follow-up questions based on results"""
    intent = query_plan.get('intent', '')
    collections = query_plan.get('collections', [])
    follow_ups = []

    # Pattern 1: Temporal expansion
    if 'date' in str(results).lower() or any(c in collections for c in ['MarketData', 'EconomicData']):
        follow_ups.append(f"📈 How has this changed over the past year?")
        follow_ups.append(f"📊 Show me the trend for the last 5 years")

    # Pattern 2: Entity expansion
    if results and isinstance(results[0], dict) and 'ticker' in results[0]:
        tickers = [r.get('ticker') for r in results[:3] if r.get('ticker')]
        if tickers:
            follow_ups.append(f"💼 Compare {', '.join(tickers[:3])} financial metrics")

    # Pattern 3: Cross-collection expansion
    if 'Award' in collections:
        follow_ups.append(f"📉 What's the stock performance for these companies?")
        follow_ups.append(f"🔮 What do prediction markets say about these companies?")

    if 'MarketData' in collections:
        follow_ups.append(f"🏛️ Have these companies received government contracts?")

    return follow_ups[:4]  # Return top 4
