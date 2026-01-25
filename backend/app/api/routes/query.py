"""
Query API routes - Natural language to AQL query planning and execution
"""
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, validator, constr
from typing import Optional, Dict, List, Any
import time
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.database.connection import get_db, execute_aql, fix_aql_query
from app.llm.planning import plan_query_with_llm, quick_intent_check, get_query_embedding, generate_follow_up_questions, analyze_results_with_llm
from app.llm.web_search import classify_query_intent, search_web_context, synthesize_hybrid_response
from app.cache import query_cache
from app.analytics import log_user_query, get_daily_spend
from app.utils.cost_estimator import estimate_query_cost_simple
from app.utils.query_validator import validate_aql_query
import config

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


class QueryRequest(BaseModel):
    question: constr(min_length=1, max_length=1000)  # Limit query length
    conversation_history: Optional[List[Dict[str, Any]]] = []  # Conversation context
    forced_plan_aql: Optional[str] = None  # Direct AQL bypass for Query Builder

    @validator('question')
    def sanitize_question(cls, v):
        """Prevent AQL injection and malicious queries"""
        v = v.strip()

        # Forbidden patterns that could be AQL injection
        forbidden = [
            'DROP', 'DELETE', 'REMOVE', 'INSERT', 'UPDATE', 'REPLACE',
            'CREATE', 'RENAME', 'TRUNCATE', '/*', '*/', '--'
        ]

        # Check for forbidden keywords
        upper_query = v.upper()
        for forbidden_word in forbidden:
            if forbidden_word in upper_query:
                raise ValueError(f'Query contains forbidden keyword: {forbidden_word}')

        # Check for excessive special characters (potential injection)
        special_chars = sum(c in v for c in [';', '{', '}', '[', ']'])
        if special_chars > 5:
            raise ValueError('Query contains too many special characters')

        return v


class QueryPlanResponse(BaseModel):
    aql_query: str
    bind_vars: dict
    intent: str
    requires_embedding: bool
    execution_time: float


class QueryExecuteResponse(BaseModel):
    results: List[Dict[str, Any]]
    count: int
    execution_time: float
    query_plan: dict
    analysis: str
    follow_up_questions: Optional[List[str]] = None
    query_intent: Optional[str] = None  # db_only, web_only, hybrid
    web_context: Optional[Dict[str, Any]] = None  # Web search results if hybrid/web_only
    metadata: Optional[Dict[str, Any]] = None  # Query metadata (collections used, data types present)


@router.post("/intent")
@limiter.limit("60/minute")
def check_intent(request: Request, body: QueryRequest):
    """Classify query intent (ticker vs concept)"""
    try:
        intent = quick_intent_check(body.question)
        return {"intent": intent}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/plan", response_model=QueryPlanResponse)
@limiter.limit("30/minute")
def plan_query(request: Request, body: QueryRequest):
    """Generate AQL query from natural language"""
    start_time = time.time()

    try:
        # Check intent
        intent = quick_intent_check(body.question)

        # Plan query
        query_plan = plan_query_with_llm(body.question, intent_hint=intent)

        if not query_plan:
            raise HTTPException(status_code=500, detail="Failed to generate query plan")

        execution_time = time.time() - start_time

        return QueryPlanResponse(
            aql_query=query_plan.get("aql_query", ""),
            bind_vars=query_plan.get("bind_vars", {}),
            intent=query_plan.get("intent", "unknown"),
            requires_embedding=query_plan.get("requires_embedding", False),
            execution_time=execution_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def detect_presentation_type(aql_query: str, results: list, question: str) -> str:
    """Detect what kind of specialized presentation to use based on query pattern"""
    if not results:
        return 'table'

    q_lower = question.lower() if question else ''
    aql_lower = aql_query.lower() if aql_query else ''

    # Insider trading signals - options activity before filings/awards
    if ('options_before' in aql_lower or 'unusual' in q_lower) and ('filing' in aql_lower or 'award' in aql_lower):
        if any('days_before' in str(r) or 'options_date' in str(r) for r in results):
            return 'insider_trading'

    # Sentiment divergence - multiple sentiment sources
    if 'divergence' in q_lower or ('sentiment' in q_lower and ('sec' in q_lower or 'market' in q_lower)):
        if any(('sec_sentiment' in str(r) or 'market_sentiment' in str(r) or 'sentiment_change' in str(r)) for r in results):
            return 'sentiment_divergence'

    # CFTC positioning - speculator vs commercial
    if ('cftc' in q_lower or 'speculator' in q_lower or 'commercial' in q_lower) and 'position' in q_lower:
        return 'cftc_positioning'

    # Whale trader analysis
    if 'whale' in q_lower or ('trader' in q_lower and 'profit' in q_lower):
        return 'whale_analysis'

    return 'table'


def analyze_query_metadata(aql_query: str, results: List[Dict]) -> Dict[str, Any]:
    """
    Analyze AQL query and results to determine what data types are present.
    This enables frontend to conditionally render appropriate sections.

    Returns metadata with:
    - collections_used: List of collection names found in query
    - data_types: Boolean flags for each data type (has_options, has_sec_filings, etc.)
    """
    import re

    metadata = {
        "collections_used": [],
        "data_types": {
            "has_options": False,
            "has_sec_filings": False,
            "has_form4_insider": False,  # Specific for Form 4/5 insider trades
            "has_awards": False,
            "has_futures": False,
            "has_commodities": False,  # CFTC positions
            "has_eia_data": False,
            "has_prediction_markets": False,
            "has_market_data": False
        }
    }

    if not aql_query:
        return metadata

    # Extract collection names from AQL (FOR doc IN CollectionName)
    collection_pattern = r'FOR\s+\w+\s+IN\s+(\w+)'
    collections = re.findall(collection_pattern, aql_query, re.IGNORECASE)

    # Also check OUTBOUND/INBOUND traversals (might reference collections)
    traversal_pattern = r'(?:OUTBOUND|INBOUND)\s+\w+\s+(\w+)'
    collections.extend(re.findall(traversal_pattern, aql_query, re.IGNORECASE))

    # Remove duplicates and filter out common keywords
    collections = list(set(c for c in collections if c.upper() not in ['FILTER', 'SORT', 'LIMIT', 'RETURN']))
    metadata["collections_used"] = collections

    # Set data type flags based on collections found
    for coll in collections:
        coll_lower = coll.lower()

        if coll_lower == 'options_flow':
            metadata["data_types"]["has_options"] = True

        if coll_lower == 'sec_filings':
            metadata["data_types"]["has_sec_filings"] = True
            # Check if query filters for Form 4/5
            if 'type' in aql_query and ('"4"' in aql_query or '"5"' in aql_query or '["4"' in aql_query):
                metadata["data_types"]["has_form4_insider"] = True

        if coll_lower == 'award':
            metadata["data_types"]["has_awards"] = True

        if coll_lower == 'futures_prices':
            metadata["data_types"]["has_futures"] = True

        if coll_lower == 'commodity_positions':
            metadata["data_types"]["has_commodities"] = True

        if coll_lower in ['eia_crude_inventory', 'eia_natgas_storage', 'eia_natgas_production', 'eia_lng_exports']:
            metadata["data_types"]["has_eia_data"] = True

        if coll_lower in ['prediction_markets_polymarket', 'prediction_markets_kalshi', 'polymarket_traders']:
            metadata["data_types"]["has_prediction_markets"] = True

        if coll_lower == 'marketdata':
            metadata["data_types"]["has_market_data"] = True

    # Also check results structure for nested data
    if results:
        sample = results[0] if isinstance(results, list) else results
        if isinstance(sample, dict):
            # Check for nested data structures
            if 'options_flow' in sample or any('options' in str(k).lower() for k in sample.keys()):
                metadata["data_types"]["has_options"] = True

            if 'sec_filings' in sample or 'SEC_filings' in sample:
                metadata["data_types"]["has_sec_filings"] = True

            if 'awards' in sample or 'Award' in sample:
                metadata["data_types"]["has_awards"] = True

            if 'futures_prices' in sample:
                metadata["data_types"]["has_futures"] = True

    return metadata


def execute_db_query(question: str, conversation_history: list = None):
    """Execute database query in parallel thread"""
    try:
        db = get_db()
        intent = quick_intent_check(question)
        query_plan = plan_query_with_llm(question, intent_hint=intent, conversation_history=conversation_history)

        if not query_plan:
            return None, None, "Failed to generate query plan"

        # Check if planning returned an error
        if query_plan.get('error'):
            error_detail = f"{query_plan.get('error_type')}: {query_plan.get('error_message')}"
            print(f"[DB QUERY ERROR] {error_detail}")
            return None, None, error_detail

        aql_query = query_plan.get("aql_query")
        bind_vars = query_plan.get("bind_vars", {})

        if query_plan.get("requires_embedding"):
            embedding_text = query_plan.get("embedding_text", question)
            embedding_vector = get_query_embedding(embedding_text)

            if not embedding_vector:
                return None, None, "Failed to generate embedding"

            bind_vars["query_vector"] = embedding_vector

        fixed_query = fix_aql_query(aql_query)

        if fixed_query is None:
            return None, None, "Query contains unfixable errors"

        results, error = execute_aql(fixed_query, bind_vars)

        if error:
            return None, None, f"Query execution error: {error}"

        return results, query_plan, None

    except Exception as e:
        return None, None, str(e)


def execute_web_search(question: str):
    """Execute web search in parallel thread"""
    try:
        print(f"[WEB SEARCH] Starting web search for: '{question[:50]}...'")
        result = search_web_context(question)
        print(f"[WEB SEARCH] Completed. Sources: {len(result.get('sources', []))}, Citations: {len(result.get('citations', []))}")
        return result, None
    except Exception as e:
        print(f"[WARNING] Web search failed: {e}")
        import traceback
        print(f"[WARNING] Web search traceback: {traceback.format_exc()}")
        return {
            'summary': f"Web search unavailable: {str(e)}",
            'sources': []
        }, str(e)


def enrich_single_company_results(results: List[Dict], query_plan: Dict) -> List[Dict]:
    """
    Detect if results are for a single company and enrich with workup data.
    
    This enables the CompanyWorkup component to display for natural language queries
    like "Show me apple stock performance", matching the advanced query builder experience.
    
    Returns:
        - Original results if not single-company or too large
        - Enriched workup structure if single-company detected
    """
    # Skip enrichment for empty, large, or already-enriched results
    if not results:
        print("[ENRICH] No results to enrich")
        return results
    
    if len(results) > 100:
        print(f"[ENRICH] Too many results ({len(results)}), skipping enrichment")
        return results
    
    # Check if already enriched (has nested MarketData, sec_filings, etc.)
    if results and any(isinstance(r.get('MarketData'), list) for r in results):
        print("[ENRICH] Results already enriched, skipping")
        return results
    
    # Debug: Log first result structure
    if results:
        print(f"[ENRICH DEBUG] First result keys: {list(results[0].keys())}")
        print(f"[ENRICH DEBUG] First result ticker field: {results[0].get('ticker')}")
    
    # Check if all results have the same ticker
    tickers = {r.get('ticker') for r in results if r.get('ticker')}
    print(f"[ENRICH DEBUG] Found {len(tickers)} unique tickers: {tickers}")
    
    ticker = None
    
    # Strategy 1: Find ticker in results
    if len(tickers) == 1:
        ticker = list(tickers)[0]
    # Strategy 2: If no ticker in results (e.g. only date/close selected), check query plan
    elif len(tickers) == 0 and query_plan:
        print("[ENRICH DEBUG] No ticker in results, checking query plan bind vars...")
        bind_vars = query_plan.get('bind_vars', {})
        # Look for bind vars that might contain ticker
        # Common patterns: @ticker, @symbol, @company
        for key, value in bind_vars.items():
            if key in ['ticker', 'symbol', 'company', 'TICKER', '0'] and isinstance(value, str):
                ticker = value.upper()
                print(f"[ENRICH DEBUG] Found ticker in bind vars: {ticker}")
                break
    
    if not ticker:
        print(f"[ENRICH] Could not determine unique ticker. Tickers found: {tickers}, skipping enrichment")
        return results
    
    if len(tickers) > 1:
        print(f"[ENRICH] Multi-company detected ({len(tickers)} tickers), skipping enrichment")
        return results
    print(f"[ENRICH] Single company detected: {ticker}, enriching with workup data...")
    
    try:
        # Fetch enrichment data in parallel for better performance
        from concurrent.futures import ThreadPoolExecutor
        
        def fetch_market_data():
            query = """
            FOR m IN MarketData
              FILTER m.ticker == @ticker
              SORT m.date DESC
              LIMIT 500
              RETURN m
            """
            data, _ = execute_aql(query, {"ticker": ticker})
            return data or []
        
        def fetch_sec_filings():
            query = """
            FOR s IN sec_filings
              FILTER s.ticker == @ticker
              SORT s.filing_date DESC
              LIMIT 10
              RETURN s
            """
            data, _ = execute_aql(query, {"ticker": ticker})
            return data or []
        
        def fetch_awards():
            query = """
            FOR a IN Award
              FILTER a.ticker == @ticker
              SORT a.start_date DESC
              LIMIT 20
              RETURN a
            """
            data, _ = execute_aql(query, {"ticker": ticker})
            return data or []
        
        def fetch_company_info():
            query = """
            FOR c IN Company
              FILTER c.ticker == @ticker
              LIMIT 1
              RETURN c
            """
            data, _ = execute_aql(query, {"ticker": ticker})
            return data[0] if data else {}
        
        # Execute all queries in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            market_future = executor.submit(fetch_market_data)
            sec_future = executor.submit(fetch_sec_filings)
            award_future = executor.submit(fetch_awards)
            company_future = executor.submit(fetch_company_info)
            
            market_data = market_future.result()
            sec_data = sec_future.result()
            award_data = award_future.result()
            company_info = company_future.result()
        
        # Build enriched structure matching CompanyWorkup expectations
        # Merge all company fields (sector, industry, etc.) into enriched result
        enriched = {
            **company_info,  # Spread all company fields (company, sector, industry, city, country, etc.)
            "ticker": ticker,
            "MarketData": market_data,
            "sec_filings": sec_data,
            "Award": award_data,
            "prediction_markets_polymarket": []  # Optional: can add later
        }
        
        print(f"[ENRICH] Enriched {ticker} with {len(market_data)} market records, {len(sec_data)} filings, {len(award_data)} awards")
        return [enriched]
        
    except Exception as e:
        print(f"[ENRICH ERROR] Failed to enrich {ticker}: {e}")
        import traceback
        print(f"[ENRICH ERROR] Traceback: {traceback.format_exc()}")
        return results  # Return original results on error


@router.post("/execute", response_model=QueryExecuteResponse)
@limiter.limit("20/minute")  # Stricter limit for expensive operations
def execute_query(request: Request, body: QueryRequest):
    """
    Execute natural language query with PARALLEL DB + Web search

    Flow:
    1. Launch DB query + Web search + Intent classification ALL IN PARALLEL
    2. Wait for all to complete
    3. Synthesize hybrid response combining both sources

    Benefits:
    - Faster (2-3s instead of 3-5s)
    - Always have web context for richer answers
    - Never miss current events even for DB-heavy queries
    """
    start_time = time.time()

    try:
        # Run DB query, web search, and intent classification IN PARALLEL
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all tasks with conversation history
            db_future = executor.submit(execute_db_query, body.question, body.conversation_history)
            web_future = executor.submit(execute_web_search, body.question)
            intent_future = executor.submit(classify_query_intent, body.question)

            # Wait for all to complete
            results, query_plan, db_error = db_future.result()
            web_context_data, web_error = web_future.result()
            intent_classification = intent_future.result()

        query_intent = intent_classification.get('intent', 'hybrid')

        # Handle DB errors
        if db_error:
            print(f"[WARNING] DB query failed: {db_error}")
            results = []
            query_plan = {}

        # Ensure we have valid data structures
        if results is None:
            results = []
        if query_plan is None:
            query_plan = {}
        if web_context_data is None:
            web_context_data = {'summary': 'No web context available', 'sources': []}

        execution_time = time.time() - start_time

        # Synthesize response based on what data we have
        if results and web_context_data.get('summary'):
            # Hybrid: combine both DB and web
            analysis = synthesize_hybrid_response(
                question=body.question,
                db_results={'data': results, 'count': len(results)},
                web_context=web_context_data
            )
        elif web_context_data.get('summary'):
            # Web-only (DB failed or empty)
            analysis = web_context_data['summary']
        elif results:
            # DB-only (web failed)
            analysis = analyze_results_with_llm(body.question, results, query_plan)
        else:
            # Both failed
            analysis = "I couldn't retrieve information from either the database or web sources. Please try rephrasing your question."

        # Generate follow-up questions
        print(f"[EXECUTE] Generating follow-ups for {len(results)} results")
        follow_ups = generate_follow_up_questions(body.question, results, query_plan)
        print(f"[EXECUTE] Generated {len(follow_ups)} follow-up questions")

        # Check if this is a time series query and add chart metadata
        if results and query_plan:
            from app.llm.planning import detect_time_series_query
            if detect_time_series_query(results, query_plan):
                # Sort results by date for chart
                sorted_results = sorted(results, key=lambda x: x.get('date', ''))
                # Add chart metadata to query_plan
                query_plan['is_time_series'] = True
                query_plan['chart_data'] = {
                    'type': 'line',
                    'dates': [r.get('date', '') for r in sorted_results],
                    'values': [float(r.get('close', 0)) for r in sorted_results],
                    'label': f"{sorted_results[0].get('ticker', 'Stock')} Close Price",
                    'ticker': sorted_results[0].get('ticker', 'Unknown')
                }
                print(f"[EXECUTE] Added chart metadata for time series query")

        # Log web context status
        has_sources = bool(web_context_data and (web_context_data.get('sources') or web_context_data.get('citations')))
        print(f"[EXECUTE] Web context has sources: {has_sources}")

        # Enrich single-company results for CompanyWorkup display
        if results:
            results = enrich_single_company_results(results, query_plan)

        # Generate query metadata for frontend conditional rendering
        aql_query = query_plan.get('aql_query', '') if query_plan else ''
        query_metadata = analyze_query_metadata(aql_query, results)
        print(f"[EXECUTE] Query metadata: {query_metadata}")

        return QueryExecuteResponse(
            results=results,
            count=len(results),
            execution_time=execution_time,
            query_plan=query_plan,
            analysis=analysis,
            follow_up_questions=follow_ups,
            query_intent=query_intent,
            web_context=web_context_data,
            metadata=query_metadata
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute-stream")
@limiter.limit("20/minute")
async def execute_query_stream(request: Request, body: QueryRequest):
    """
    STREAMING version: Execute query with real-time progress updates

    Sends Server-Sent Events (SSE) with:
    - Progress updates (searching DB, web, analyzing)
    - Streamed analysis content (token by token)
    - Final results payload
    """

    # Check daily API budget BEFORE processing query
    try:
        current_spend = get_daily_spend()
        if current_spend >= config.DAILY_API_BUDGET:
            raise HTTPException(
                status_code=503,
                detail=f"Daily API budget exceeded (${current_spend:.2f} / ${config.DAILY_API_BUDGET:.2f}). Please try again tomorrow or contact support."
            )

        # Warn if close to limit (90%)
        if current_spend >= config.DAILY_API_BUDGET * 0.9:
            print(f"[WARNING] Daily API budget at {(current_spend/config.DAILY_API_BUDGET)*100:.1f}% (${current_spend:.2f} / ${config.DAILY_API_BUDGET:.2f})")
    except HTTPException:
        raise  # Re-raise budget exceeded error
    except Exception as budget_error:
        # Log error but don't block request if budget check fails
        print(f"[WARNING] Budget check failed: {budget_error}")

    async def generate():
        try:
            start_time = time.time()

            # Step 0: Check cache first
            cached_result = query_cache.get(body.question)
            if cached_result:
                # Cache hit! Return immediately
                yield f"data: {json.dumps({'type': 'progress', 'stage': 'cache_hit', 'message': '⚡ Retrieved from cache (instant!)'})}\n\n"
                await asyncio.sleep(0.1)

                # Stream cached analysis
                yield f"data: {json.dumps({'type': 'content_start', 'message': 'Streaming cached analysis...'})}\n\n"

                analysis = cached_result.get('analysis', '')
                words = analysis.split(' ')
                chunk_size = 8  # Faster streaming for cached results
                for i in range(0, len(words), chunk_size):
                    chunk = ' '.join(words[i:i+chunk_size])
                    if i + chunk_size < len(words):
                        chunk += ' '
                    yield f"data: {json.dumps({'type': 'content_chunk', 'chunk': chunk})}\n\n"
                    await asyncio.sleep(0.01)

                # Send cached complete payload
                final_payload = {
                    'type': 'complete',
                    'results': cached_result.get('results', [])[:100],
                    'count': cached_result.get('count', 0),
                    'execution_time': time.time() - start_time,
                    'query_plan': cached_result.get('query_plan', {}),
                    'follow_up_questions': cached_result.get('follow_up_questions', []),
                    'query_intent': cached_result.get('query_intent', ''),
                    'web_context': cached_result.get('web_context', {}),
                    'metadata': cached_result.get('metadata', {}),
                    'from_cache': True
                }

                # Log cache hit (no API cost, instant response)
                try:
                    log_user_query(
                        question=body.question,
                        response_time=time.time() - start_time,
                        result_count=cached_result.get('count', 0),
                        was_successful=True,
                        ip_address=request.client.host if request.client else "unknown",
                        query_intent=cached_result.get('query_intent', 'unknown'),
                        error=None,
                        aql_query=cached_result.get('query_plan', {}).get('aql_query'),
                        collections_used=cached_result.get('metadata', {}).get('collections_used', []),
                        api_cost=0.0,  # No cost for cache hits
                        from_cache=True
                    )
                except Exception as log_error:
                    print(f"[WARNING] Cache hit logging failed: {log_error}")

                yield f"data: {json.dumps(final_payload)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Step 1: Analyzing question (cache miss)
            if body.forced_plan_aql:
                # VALIDATE builder query for safety and complexity
                is_valid, validation_error = validate_aql_query(body.forced_plan_aql)
                if not is_valid:
                    error_payload = {
                        'type': 'error',
                        'message': f'Query validation failed: {validation_error}',
                        'details': 'Your manual query does not meet safety requirements. Please adjust and try again.'
                    }
                    yield f"data: {json.dumps(error_payload)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                yield f"data: {json.dumps({'type': 'progress', 'stage': 'analyzing', 'message': 'Executing Builder Query...', 'details': 'Query validated ✓'})}\n\n"
                await asyncio.sleep(0.1)

                # Run AQL in thread to avoid blocking event loop
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    results, db_error = await loop.run_in_executor(executor, execute_aql, body.forced_plan_aql)
                
                # Mock query plan for builder execution
                query_plan = {
                    "aql_query": body.forced_plan_aql,
                    "intent": "builder_execution",
                    "explanation": "Manual Query Builder Execution"
                }
                web_context_data = {'summary': '', 'sources': []}
                intent_class = {'intent': 'db_only'}
                query_intent = 'db_only'
                
            else:
                yield f"data: {json.dumps({'type': 'progress', 'stage': 'analyzing', 'message': 'Understanding your question...'})}\n\n"
                await asyncio.sleep(0.1)  # Small delay for UX

                # Step 2: Execute parallel searches
                yield f"data: {json.dumps({'type': 'progress', 'stage': 'searching', 'message': 'Searching database and web...', 'details': 'Running 3 parallel tasks'})}\n\n"

                # Run in thread pool (FastAPI handles async/sync mixing)
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=3) as executor:
                    db_future = loop.run_in_executor(executor, execute_db_query, body.question, body.conversation_history)
                    web_future = loop.run_in_executor(executor, execute_web_search, body.question)
                    intent_future = loop.run_in_executor(executor, classify_query_intent, body.question)

                    # Wait for all
                    results, query_plan, db_error = await db_future
                    web_context_data, web_error = await web_future
                    intent_classification = await intent_future

                query_intent = intent_classification.get('intent', 'hybrid')

            # Handle errors
            if db_error:
                print(f"[STREAM WARNING] DB query failed: {db_error}")
                results = []
                query_plan = {}

            if results is None:
                results = []
            if query_plan is None:
                query_plan = {}
            if web_context_data is None:
                web_context_data = {'summary': 'No web context available', 'sources': []}

            execution_time = time.time() - start_time

            # Step 3: Synthesizing response
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'synthesizing', 'message': 'Generating analysis...', 'details': f'Found {len(results)} results'})}\n\n"

            # Generate analysis
            if query_intent == 'db_only' and body.forced_plan_aql:
                print(f"[ANALYSIS] Using concise builder-mode analysis")
                analysis = f"I've executed your manual query. Found {len(results)} records matching your criteria. You can explore the structured results in the table below."
            elif results and web_context_data.get('summary'):
                print(f"[ANALYSIS] Using hybrid synthesis (DB + Web)")
                analysis = synthesize_hybrid_response(
                    question=body.question,
                    db_results={'data': results, 'count': len(results)},
                    web_context=web_context_data
                )
            elif web_context_data.get('summary'):
                print(f"[ANALYSIS] Using web-only summary")
                analysis = web_context_data['summary']
            elif results:
                print(f"[ANALYSIS] Using DB-only LLM analysis")
                analysis = analyze_results_with_llm(body.question, results, query_plan)
            else:
                print(f"[ANALYSIS] No results - using fallback message")
                analysis = "I couldn't retrieve information from either the database or web sources. Please try rephrasing your question."

            print(f"[ANALYSIS] Generated {len(analysis)} characters")

            # Step 4: Stream analysis content in chunks
            yield f"data: {json.dumps({'type': 'content_start', 'message': 'Streaming analysis...'})}\n\n"

            # Stream in word chunks for smooth effect
            words = analysis.split(' ')
            chunk_size = 5  # 5 words at a time
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i+chunk_size])
                if i + chunk_size < len(words):
                    chunk += ' '  # Add space between chunks

                yield f"data: {json.dumps({'type': 'content_chunk', 'chunk': chunk})}\n\n"
                await asyncio.sleep(0.02)  # Small delay for streaming effect

            # Step 5: Generate follow-up questions (silently, don't send progress that overwrites content)
            print(f"[STREAM] Generating follow-up questions...")

            follow_ups = []
            if results:
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    follow_ups = await loop.run_in_executor(
                        executor,
                        generate_follow_up_questions,
                        body.question,
                        results,
                        query_plan
                    )

            # Check if this is a time series query and add chart metadata
            if results and query_plan:
                from app.llm.planning import detect_time_series_query
                if detect_time_series_query(results, query_plan):
                    # Sort results by date for chart
                    sorted_results = sorted(results, key=lambda x: x.get('date', ''))
                    # Add chart metadata to query_plan
                    query_plan['is_time_series'] = True
                    query_plan['chart_data'] = {
                        'type': 'line',
                        'dates': [r.get('date', '') for r in sorted_results],
                        'values': [float(r.get('close', 0)) for r in sorted_results],
                        'label': f"{sorted_results[0].get('ticker', 'Stock')} Close Price",
                        'ticker': sorted_results[0].get('ticker', 'Unknown')
                    }
                    print(f"[STREAM] Added chart metadata for time series query")

            # Enrich single-company results for CompanyWorkup display
            if results:
                results = enrich_single_company_results(results, query_plan)

            # Generate query metadata for frontend conditional rendering
            aql_query = query_plan.get('aql_query', '') if query_plan else ''
            query_metadata = analyze_query_metadata(aql_query, results)
            print(f"[STREAM] Query metadata: {query_metadata}")

            # Detect specialized presentation type
            presentation_type = detect_presentation_type(aql_query, results, body.question)
            print(f"[STREAM] Presentation type: {presentation_type}")

            # Step 6: Send complete payload
            final_payload = {
                'type': 'complete',
                'results': results[:100],  # Limit results in stream for performance
                'count': len(results),
                'execution_time': execution_time,
                'query_plan': query_plan,
                'follow_up_questions': follow_ups,
                'query_intent': query_intent,
                'presentation_type': presentation_type,
                'web_context': {
                    'sources': web_context_data.get('sources', []),
                    'citations': web_context_data.get('citations', [])
                },
                'metadata': query_metadata
            }

            # Cache the complete result for future queries
            cache_data = {
                'results': results,
                'count': len(results),
                'query_plan': query_plan,
                'analysis': analysis,
                'follow_up_questions': follow_ups,
                'query_intent': query_intent,
                'web_context': {
                    'sources': web_context_data.get('sources', []),
                    'citations': web_context_data.get('citations', [])
                },
                'metadata': query_metadata
            }
            query_cache.set(body.question, cache_data)

            # Log query for analytics (async, won't block stream)
            try:
                estimated_cost = estimate_query_cost_simple(
                    question_length=len(body.question),
                    result_count=len(results),
                    has_web_search=(query_intent in ['hybrid', 'web_only']),
                    model="gpt-4o-mini"
                )

                log_user_query(
                    question=body.question,
                    response_time=execution_time,
                    result_count=len(results),
                    was_successful=(not db_error),
                    ip_address=request.client.host if request.client else "unknown",
                    query_intent=query_intent,
                    error=db_error,
                    aql_query=query_plan.get('aql_query') if query_plan else None,
                    collections_used=query_metadata.get('collections_used', []),
                    api_cost=estimated_cost,
                    from_cache=False
                )
            except Exception as log_error:
                print(f"[WARNING] Query logging failed: {log_error}")

            yield f"data: {json.dumps(final_payload)}\n\n"

            # End stream
            yield "data: [DONE]\n\n"

        except Exception as e:
            error_payload = {
                'type': 'error',
                'message': f'Error: {str(e)}',
                'details': str(e)
            }
            yield f"data: {json.dumps(error_payload)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@router.get("/cache/stats")
def get_cache_stats():
    """Get query cache statistics"""
    return query_cache.get_stats()


@router.post("/cache/clear")
def clear_cache():
    """Clear query cache (admin only in production)"""
    query_cache.clear()
    return {"message": "Cache cleared successfully"}


@router.get("/analytics/stats")
def get_analytics_stats(hours: int = 24):
    """
    Get query analytics for the last N hours
    Shows: total queries, success rate, costs, popular collections
    """
    from app.analytics import get_query_stats
    return get_query_stats(hours=hours)


@router.get("/analytics/budget")
def get_budget_status():
    """Get current daily API spend vs budget"""
    current_spend = get_daily_spend()
    budget = config.DAILY_API_BUDGET
    return {
        "current_spend": round(current_spend, 2),
        "daily_budget": budget,
        "remaining": round(budget - current_spend, 2),
        "percent_used": round((current_spend / budget) * 100, 1) if budget > 0 else 0,
        "is_near_limit": current_spend >= budget * 0.9,
        "is_over_budget": current_spend >= budget
    }
