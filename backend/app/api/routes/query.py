"""
Query API routes - Natural language to AQL query planning and execution
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.database.connection import get_db, execute_aql, fix_aql_query
from app.llm.planning import plan_query_with_llm, quick_intent_check, get_query_embedding, generate_follow_up_questions, analyze_results_with_llm
from app.llm.web_search import classify_query_intent, search_web_context, synthesize_hybrid_response

router = APIRouter()


class QueryRequest(BaseModel):
    question: str


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
def check_intent(request: QueryRequest):
    """Classify query intent (ticker vs concept)"""
    try:
        intent = quick_intent_check(request.question)
        return {"intent": intent}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/plan", response_model=QueryPlanResponse)
def plan_query(request: QueryRequest):
    """Generate AQL query from natural language"""
    start_time = time.time()

    try:
        # Check intent
        intent = quick_intent_check(request.question)

        # Plan query
        query_plan = plan_query_with_llm(request.question, intent_hint=intent)

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


def execute_db_query(question: str):
    """Execute database query in parallel thread"""
    try:
        db = get_db()
        intent = quick_intent_check(question)
        query_plan = plan_query_with_llm(question, intent_hint=intent)

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
        return search_web_context(question), None
    except Exception as e:
        print(f"[WARNING] Web search failed: {e}")
        return {
            'summary': f"Web search unavailable: {str(e)}",
            'sources': []
        }, str(e)


@router.post("/execute", response_model=QueryExecuteResponse)
def execute_query(request: QueryRequest):
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
            # Submit all tasks
            db_future = executor.submit(execute_db_query, request.question)
            web_future = executor.submit(execute_web_search, request.question)
            intent_future = executor.submit(classify_query_intent, request.question)

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
                question=request.question,
                db_results={'data': results, 'count': len(results)},
                web_context=web_context_data
            )
        elif web_context_data.get('summary'):
            # Web-only (DB failed or empty)
            analysis = web_context_data['summary']
        elif results:
            # DB-only (web failed)
            analysis = analyze_results_with_llm(request.question, results, query_plan)
        else:
            # Both failed
            analysis = "I couldn't retrieve information from either the database or web sources. Please try rephrasing your question."

        # Generate follow-up questions
        follow_ups = generate_follow_up_questions(request.question, results, query_plan)

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
