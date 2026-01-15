"""
Query API routes - Natural language to AQL query planning and execution
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import time

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


@router.post("/execute", response_model=QueryExecuteResponse)
def execute_query(request: QueryRequest):
    """
    Execute natural language query end-to-end with hybrid DB + Web search

    Flow:
    1. Classify intent (db_only, web_only, hybrid)
    2. Execute DB query if needed
    3. Fetch web context if needed
    4. Synthesize hybrid response combining both sources
    """
    start_time = time.time()

    try:
        # Step 1: Classify query intent (db_only, web_only, hybrid)
        intent_classification = classify_query_intent(request.question)
        query_intent = intent_classification['intent']
        requires_web = query_intent in ['web_only', 'hybrid']
        requires_db = query_intent in ['db_only', 'hybrid']

        results = []
        query_plan = {}
        web_context_data = None

        # Step 2: Execute DB query if needed
        if requires_db:
            db = get_db()

            # Check intent (ticker vs concept)
            intent = quick_intent_check(request.question)

            # Plan query
            query_plan = plan_query_with_llm(request.question, intent_hint=intent)

            if not query_plan:
                raise HTTPException(status_code=500, detail="Failed to generate query plan")

            # Handle embeddings if needed
            aql_query = query_plan.get("aql_query")
            bind_vars = query_plan.get("bind_vars", {})

            if query_plan.get("requires_embedding"):
                embedding_text = query_plan.get("embedding_text", request.question)
                embedding_vector = get_query_embedding(embedding_text)

                if not embedding_vector:
                    raise HTTPException(status_code=500, detail="Failed to generate embedding")

                bind_vars["query_vector"] = embedding_vector

            # Fix and validate query
            fixed_query = fix_aql_query(aql_query)

            if fixed_query is None:
                raise HTTPException(status_code=400, detail="Query contains unfixable errors")

            # Execute
            results, error = execute_aql(fixed_query, bind_vars)

            if error:
                raise HTTPException(status_code=500, detail=f"Query execution error: {error}")

        # Step 3: Fetch web context if needed
        if requires_web:
            try:
                web_context_data = search_web_context(request.question)
            except Exception as e:
                print(f"[WARNING] Web search failed: {e}")
                # Continue without web context rather than failing entire query
                web_context_data = {
                    'summary': f"Web search unavailable: {str(e)}",
                    'sources': []
                }

        execution_time = time.time() - start_time

        # Step 4: Generate analysis
        if query_intent == 'hybrid' and web_context_data:
            # Synthesize hybrid response combining DB + Web
            analysis = synthesize_hybrid_response(
                question=request.question,
                db_results={'data': results, 'count': len(results)},
                web_context=web_context_data
            )
        elif query_intent == 'web_only' and web_context_data:
            # Web-only response
            analysis = web_context_data['summary']
        else:
            # DB-only response (original behavior)
            analysis = analyze_results_with_llm(request.question, results, query_plan)

        # Step 5: Generate follow-up questions
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

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
