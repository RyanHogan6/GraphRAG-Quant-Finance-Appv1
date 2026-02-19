"""
Alerts API - session-scoped alert rules and notifications.
Alert = run a saved workspace on a schedule (or on-demand); notify when results exist.
"""
from fastapi import APIRouter, HTTPException, Request, Header
from pydantic import BaseModel, Field
from typing import Optional, List
from slowapi import Limiter
from slowapi.util import get_remote_address
import uuid
import time

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

ALERTS_COLLECTION = "user_alerts"
NOTIFICATIONS_COLLECTION = "user_notifications"


def _get_session_id(x_session_id: Optional[str] = Header(None, alias="X-Session-Id")) -> str:
    if x_session_id and isinstance(x_session_id, str) and len(x_session_id.strip()) > 0:
        return x_session_id.strip()[:128]
    return str(uuid.uuid4())


def _get_db():
    from app.database.connection import get_db
    return get_db()


def _ensure_collections():
    db = _get_db()
    for name in (ALERTS_COLLECTION, NOTIFICATIONS_COLLECTION):
        if not db.has_collection(name):
            db.create_collection(name)


class AlertCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    workspace_id: str = Field(..., min_length=1, max_length=64)


@router.get("", response_model=List[dict])
@limiter.limit("60/minute")
def list_alerts(
    request: Request,
    x_session_id: Optional[str] = Header(None, alias="X-Session-Id"),
):
    """List alert rules for the current session."""
    session_id = _get_session_id(x_session_id)
    _ensure_collections()
    db = _get_db()
    cursor = db.aql.execute(
        f"""
        FOR doc IN {ALERTS_COLLECTION}
            FILTER doc.session_id == @session_id
            SORT doc.created_at DESC
            RETURN {{ id: doc._key, name: doc.name, workspace_id: doc.workspace_id, created_at: doc.created_at }}
        """,
        bind_vars={"session_id": session_id},
    )
    return list(cursor)


@router.post("", status_code=201)
@limiter.limit("30/minute")
def create_alert(
    request: Request,
    body: AlertCreate,
    x_session_id: Optional[str] = Header(None, alias="X-Session-Id"),
):
    """Create an alert rule: when the given workspace is run and has results, a notification is created."""
    session_id = _get_session_id(x_session_id)
    _ensure_collections()
    db = _get_db()
    # Verify workspace belongs to this session
    try:
        col = db.collection("saved_workspaces")
        ws = col.get(body.workspace_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Workspace not found")
    if ws.get("session_id") != session_id:
        raise HTTPException(status_code=403, detail="Workspace not found")
    col = db.collection(ALERTS_COLLECTION)
    now = time.time()
    key = str(uuid.uuid4()).replace("-", "")[:20]
    doc = {
        "_key": key,
        "session_id": session_id,
        "name": body.name,
        "workspace_id": body.workspace_id,
        "created_at": now,
    }
    col.insert(doc)
    return {"id": key, "created_at": now}


@router.delete("/{alert_id}", status_code=204)
@limiter.limit("30/minute")
def delete_alert(
    request: Request,
    alert_id: str,
    x_session_id: Optional[str] = Header(None, alias="X-Session-Id"),
):
    """Delete an alert rule."""
    session_id = _get_session_id(x_session_id)
    _ensure_collections()
    db = _get_db()
    col = db.collection(ALERTS_COLLECTION)
    try:
        doc = col.get(alert_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Alert not found")
    if doc.get("session_id") != session_id:
        raise HTTPException(status_code=404, detail="Alert not found")
    col.delete(alert_id)
    return None


@router.get("/notifications", response_model=List[dict])
@limiter.limit("60/minute")
def list_notifications(
    request: Request,
    unread_only: bool = False,
    limit: int = 50,
    x_session_id: Optional[str] = Header(None, alias="X-Session-Id"),
):
    """List notifications for the current session."""
    session_id = _get_session_id(x_session_id)
    _ensure_collections()
    db = _get_db()
    filter_clause = "FILTER doc.session_id == @session_id" + (" AND doc.read == false" if unread_only else "")
    cursor = db.aql.execute(
        f"""
        FOR doc IN {NOTIFICATIONS_COLLECTION}
            {filter_clause}
            SORT doc.created_at DESC
            LIMIT @limit
            RETURN {{ id: doc._key, alert_id: doc.alert_id, title: doc.title, body: doc.body, created_at: doc.created_at, read: doc.read }}
        """,
        bind_vars={"session_id": session_id, "limit": limit},
    )
    return list(cursor)


@router.post("/notifications/{notification_id}/read", status_code=200)
@limiter.limit("60/minute")
def mark_notification_read(
    request: Request,
    notification_id: str,
    x_session_id: Optional[str] = Header(None, alias="X-Session-Id"),
):
    """Mark a notification as read."""
    session_id = _get_session_id(x_session_id)
    _ensure_collections()
    db = _get_db()
    col = db.collection(NOTIFICATIONS_COLLECTION)
    try:
        doc = col.get(notification_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Notification not found")
    if doc.get("session_id") != session_id:
        raise HTTPException(status_code=404, detail="Notification not found")
    col.update(notification_id, {"read": True})
    return {"ok": True}


@router.post("/evaluate", status_code=200)
@limiter.limit("20/minute")
def evaluate_alerts(
    request: Request,
    x_session_id: Optional[str] = Header(None, alias="X-Session-Id"),
):
    """Run all alerts for this session: for each alert, run the linked workspace; if it returns results, create a notification."""
    session_id = _get_session_id(x_session_id)
    _ensure_collections()
    db = _get_db()
    cursor = db.aql.execute(
        f"""
        FOR doc IN {ALERTS_COLLECTION}
            FILTER doc.session_id == @session_id
            RETURN doc
        """,
        bind_vars={"session_id": session_id},
    )
    alerts = list(cursor)
    created = 0
    for alert in alerts:
        workspace_id = alert.get("workspace_id")
        if not workspace_id:
            continue
        try:
            ws_col = db.collection("saved_workspaces")
            ws = ws_col.get(workspace_id)
        except Exception:
            continue
        if ws.get("session_id") != session_id:
            continue
        question = ws.get("question") or ""
        forced_aql = ws.get("forced_plan_aql")
        from app.database.connection import execute_aql
        from app.api.routes.query import (
            execute_db_query,
            enrich_single_company_results,
            analyze_results_with_llm,
            generate_follow_up_questions,
            analyze_query_metadata,
        )
        from app.utils.query_validator import validate_aql_query
        results = []
        if forced_aql:
            is_valid, _ = validate_aql_query(forced_aql)
            if is_valid:
                res, err = execute_aql(forced_aql, {})
                if not err and res:
                    results = res
        else:
            results, _, _ = execute_db_query(question, [])
            if results is None:
                results = []
        if not results or len(results) == 0:
            continue
        notif_col = db.collection(NOTIFICATIONS_COLLECTION)
        key = str(uuid.uuid4()).replace("-", "")[:20]
        title = alert.get("name") or "Alert"
        body = f"Workspace returned {len(results)} result(s)."
        notif_col.insert({
            "_key": key,
            "session_id": session_id,
            "alert_id": alert["_key"],
            "title": title,
            "body": body,
            "created_at": time.time(),
            "read": False,
        })
        created += 1
    return {"evaluated": len(alerts), "notifications_created": created}
