"""
Saved workspaces API - session-scoped save/load/re-run of queries.
No auth; uses X-Session-Id header (or generated session) for association.
"""
from fastapi import APIRouter, HTTPException, Request, Header
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from slowapi import Limiter
from slowapi.util import get_remote_address
import uuid
import time

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

COLLECTION_NAME = "saved_workspaces"


def _get_session_id(x_session_id: Optional[str] = Header(None, alias="X-Session-Id")) -> str:
    """Resolve session id from header or generate a new one."""
    if x_session_id and isinstance(x_session_id, str) and len(x_session_id.strip()) > 0:
        return x_session_id.strip()[:128]
    return str(uuid.uuid4())


def _ensure_collection():
    from app.database.connection import get_db
    db = get_db()
    if not db.has_collection(COLLECTION_NAME):
        db.create_collection(COLLECTION_NAME)
    return db.collection(COLLECTION_NAME)


class WorkspaceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    type: str = Field(..., pattern="^(nl|builder)$")
    question: str = Field(..., min_length=1, max_length=2000)
    forced_plan_aql: Optional[str] = None
    watchlist: Optional[List[str]] = None


class WorkspaceUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    question: Optional[str] = Field(None, min_length=1, max_length=2000)
    forced_plan_aql: Optional[str] = None
    watchlist: Optional[List[str]] = None


@router.get("", response_model=List[dict])
@limiter.limit("60/minute")
def list_workspaces(
    request: Request,
    x_session_id: Optional[str] = Header(None, alias="X-Session-Id"),
):
    """List workspaces for the current session."""
    from app.database.connection import get_db
    session_id = _get_session_id(x_session_id)
    _ensure_collection()  # ensure collection exists before querying
    db = get_db()
    cursor = db.aql.execute(
        """
        FOR doc IN saved_workspaces
            FILTER doc.session_id == @session_id
            SORT doc.updated_at DESC
            RETURN { id: doc._key, name: doc.name, type: doc.type, question: doc.question,
                     forced_plan_aql: doc.forced_plan_aql, watchlist: doc.watchlist,
                     created_at: doc.created_at, updated_at: doc.updated_at }
        """,
        bind_vars={"session_id": session_id},
    )
    return list(cursor)


@router.post("", status_code=201)
@limiter.limit("30/minute")
def create_workspace(
    request: Request,
    body: WorkspaceCreate,
    x_session_id: Optional[str] = Header(None, alias="X-Session-Id"),
):
    """Create a saved workspace."""
    session_id = _get_session_id(x_session_id)
    col = _ensure_collection()
    now = time.time()
    key = str(uuid.uuid4()).replace("-", "")[:20]
    doc = {
        "_key": key,
        "session_id": session_id,
        "name": body.name,
        "type": body.type,
        "question": body.question,
        "forced_plan_aql": body.forced_plan_aql,
        "watchlist": body.watchlist or [],
        "created_at": now,
        "updated_at": now,
    }
    col.insert(doc)
    return {"id": key, "created_at": now, "updated_at": now}


@router.get("/{workspace_id}")
@limiter.limit("60/minute")
def get_workspace(
    request: Request,
    workspace_id: str,
    x_session_id: Optional[str] = Header(None, alias="X-Session-Id"),
):
    """Get a single workspace by id."""
    session_id = _get_session_id(x_session_id)
    col = _ensure_collection()
    try:
        doc = col.get(workspace_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Workspace not found")
    if doc.get("session_id") != session_id:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return {
        "id": doc["_key"],
        "name": doc["name"],
        "type": doc["type"],
        "question": doc["question"],
        "forced_plan_aql": doc.get("forced_plan_aql"),
        "watchlist": doc.get("watchlist") or [],
        "created_at": doc["created_at"],
        "updated_at": doc["updated_at"],
    }


@router.put("/{workspace_id}")
@limiter.limit("30/minute")
def update_workspace(
    request: Request,
    workspace_id: str,
    body: WorkspaceUpdate,
    x_session_id: Optional[str] = Header(None, alias="X-Session-Id"),
):
    """Update a workspace (partial)."""
    session_id = _get_session_id(x_session_id)
    col = _ensure_collection()
    try:
        doc = col.get(workspace_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Workspace not found")
    if doc.get("session_id") != session_id:
        raise HTTPException(status_code=404, detail="Workspace not found")
    updates = {"updated_at": time.time()}
    if body.name is not None:
        updates["name"] = body.name
    if body.question is not None:
        updates["question"] = body.question
    if body.forced_plan_aql is not None:
        updates["forced_plan_aql"] = body.forced_plan_aql
    if body.watchlist is not None:
        updates["watchlist"] = body.watchlist
    col.update(workspace_id, updates)
    return {"id": workspace_id, "updated_at": updates["updated_at"]}


@router.delete("/{workspace_id}", status_code=204)
@limiter.limit("30/minute")
def delete_workspace(
    request: Request,
    workspace_id: str,
    x_session_id: Optional[str] = Header(None, alias="X-Session-Id"),
):
    """Delete a workspace."""
    session_id = _get_session_id(x_session_id)
    col = _ensure_collection()
    try:
        doc = col.get(workspace_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Workspace not found")
    if doc.get("session_id") != session_id:
        raise HTTPException(status_code=404, detail="Workspace not found")
    col.delete(workspace_id)
    return None


@router.post("/{workspace_id}/run")
@limiter.limit("20/minute")
def run_workspace(
    request: Request,
    workspace_id: str,
    x_session_id: Optional[str] = Header(None, alias="X-Session-Id"),
):
    """Re-run the stored query and return the same shape as /api/query/execute."""
    session_id = _get_session_id(x_session_id)
    col = _ensure_collection()
    try:
        doc = col.get(workspace_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Workspace not found")
    if doc.get("session_id") != session_id:
        raise HTTPException(status_code=404, detail="Workspace not found")

    from app.database.connection import execute_aql
    from app.api.routes.query import (
        execute_db_query,
        enrich_single_company_results,
        analyze_query_metadata,
    )
    from app.llm.planning import analyze_results_with_llm, generate_follow_up_questions
    from app.utils.query_validator import validate_aql_query

    question = doc.get("question") or ""
    forced_aql = doc.get("forced_plan_aql")
    conversation_history = []
    results = []
    query_plan = {}

    if forced_aql:
        is_valid, validation_error = validate_aql_query(forced_aql)
        if not is_valid:
            raise HTTPException(status_code=400, detail=validation_error or "Query validation failed")
        res, db_error = execute_aql(forced_aql, {})
        if db_error:
            raise HTTPException(status_code=400, detail=db_error or "Query execution failed")
        results = res or []
        query_plan = {
            "aql_query": forced_aql,
            "intent": "builder_execution",
            "explanation": "Saved workspace (Builder)",
        }
    else:
        results, query_plan, db_error = execute_db_query(question, conversation_history)
        if results is None:
            results = []
        if query_plan is None:
            query_plan = {}
        if db_error and not results:
            raise HTTPException(status_code=400, detail=db_error or "Query execution failed")

    analysis = analyze_results_with_llm(question, results, query_plan)
    follow_ups = generate_follow_up_questions(question, results, query_plan)
    results = enrich_single_company_results(results, query_plan)
    aql_query = query_plan.get("aql_query", "") if query_plan else ""
    metadata = analyze_query_metadata(aql_query, results)

    return {
        "results": results,
        "count": len(results),
        "query_plan": query_plan,
        "analysis": analysis,
        "follow_up_questions": follow_ups,
        "metadata": metadata,
    }
