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

        return QueryExecuteResponse(
            results=results,
            count=len(results),
            execution_time=execution_time,
            query_plan=query_plan,
            analysis=analysis,
            follow_up_questions=follow_ups,
            query_intent=query_intent,
            web_context=web_context_data
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
                    'from_cache': True
                }
                yield f"data: {json.dumps(final_payload)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Step 1: Analyzing question (cache miss)
            if body.forced_plan_aql:
                yield f"data: {json.dumps({'type': 'progress', 'stage': 'analyzing', 'message': 'Executing Builder Query...', 'details': 'Bypassing LLM Planner'})}\n\n"
                await asyncio.sleep(0.1)
                
                # Mock query plan for builder execution
                results, db_error = execute_aql(body.forced_plan_aql)
                query_plan = {
                    "aql_query": body.forced_plan_aql,
                    "intent": "builder_execution",
                    "explanation": "Manual Query Builder Execution"
                }
                web_context_data = {'summary': '', 'sources': []}
                intent_class = {'intent': 'db_only'}
                
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
            if results and web_context_data.get('summary'):
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

            # Step 6: Send complete payload
            final_payload = {
                'type': 'complete',
                'results': results[:100],  # Limit results in stream for performance
                'count': len(results),
                'execution_time': execution_time,
                'query_plan': query_plan,
                'follow_up_questions': follow_ups,
                'query_intent': query_intent,
                'web_context': {
                    'sources': web_context_data.get('sources', []),
                    'citations': web_context_data.get('citations', [])
                }
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
                }
            }
            query_cache.set(body.question, cache_data)

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
