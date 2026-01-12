"""
Query API routes - Natural language to AQL query planning and execution
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import time

from app.database.connection import get_db, execute_aql, fix_aql_query
from app.llm.planning import plan_query_with_llm, quick_intent_check, get_query_embedding, generate_follow_up_questions, analyze_results_with_llm

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
    """Execute natural language query end-to-end"""
    start_time = time.time()

    try:
        db = get_db()

        # Step 1: Check intent
        intent = quick_intent_check(request.question)

        # Step 2: Plan query
        query_plan = plan_query_with_llm(request.question, intent_hint=intent)

        if not query_plan:
            raise HTTPException(status_code=500, detail="Failed to generate query plan")

        # Step 3: Handle embeddings if needed
        aql_query = query_plan.get("aql_query")
        bind_vars = query_plan.get("bind_vars", {})

        if query_plan.get("requires_embedding"):
            embedding_text = query_plan.get("embedding_text", request.question)
            embedding_vector = get_query_embedding(embedding_text)

            if not embedding_vector:
                raise HTTPException(status_code=500, detail="Failed to generate embedding")

            bind_vars["query_vector"] = embedding_vector

        # Step 4: Fix and validate query
        fixed_query = fix_aql_query(aql_query)

        if fixed_query is None:
            raise HTTPException(status_code=400, detail="Query contains unfixable errors")

        # Step 5: Execute
        results, error = execute_aql(fixed_query, bind_vars)

        if error:
            raise HTTPException(status_code=500, detail=f"Query execution error: {error}")

        execution_time = time.time() - start_time

        # Step 6: Analyze results with LLM
        analysis = analyze_results_with_llm(request.question, results, query_plan)

        # Step 7: Generate follow-up questions
        follow_ups = generate_follow_up_questions(request.question, results, query_plan)

        return QueryExecuteResponse(
            results=results,
            count=len(results),
            execution_time=execution_time,
            query_plan=query_plan,
            analysis=analysis,
            follow_up_questions=follow_ups
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
