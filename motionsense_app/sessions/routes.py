from fastapi import APIRouter, Query
from fastapi.responses import Response

from motionsense_app.dependencies import EngineDependency, SessionRepositoryDependency
from motionsense_app.sessions import (
    AdvanceRequest,
    AdvanceResult,
    AttachRequest,
    ControlRequest,
    CreateSessionRequest,
    Session,
    SessionPage,
    SessionRowPage,
    SessionSummary,
)
from motionsense_app.sessions.export import export_session

router = APIRouter(prefix="/api/sessions")


@router.post("", response_model=Session, status_code=201)
def create_session(body: CreateSessionRequest, engine: EngineDependency) -> dict:
    return engine.create(body.model_dump())


@router.get("", response_model=SessionPage)
def list_sessions(
    repo: SessionRepositoryDependency, offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
) -> dict:
    return repo.list(offset, limit)


@router.get("/{id}", response_model=Session)
def get_session(id: str, repo: SessionRepositoryDependency) -> dict:
    return repo.get(id)


@router.post("/{id}/attach", response_model=Session)
def attach_session(id: str, body: AttachRequest, engine: EngineDependency) -> dict:
    return engine.attach(id, body.client_id)


@router.post("/{id}/control", response_model=Session)
def control_session(id: str, body: ControlRequest, engine: EngineDependency) -> dict:
    return engine.control(id, body.client_id, body.action, body.speed)


@router.post("/{id}/advance", response_model=AdvanceResult)
def advance_session(id: str, body: AdvanceRequest, engine: EngineDependency) -> dict:
    return engine.advance(id, body.client_id, body.expected_cursor)


@router.get("/{id}/rows", response_model=SessionRowPage)
def session_rows(
    id: str, repo: SessionRepositoryDependency, offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
) -> dict:
    return repo.rows(id, offset, limit)


@router.get("/{id}/summary", response_model=SessionSummary)
def session_summary(id: str, repo: SessionRepositoryDependency) -> dict:
    return repo.summary(id)


@router.get("/{id}/export")
def session_export(id: str, repo: SessionRepositoryDependency) -> Response:
    return Response(
        export_session(repo, id), media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="motionsense-session.csv"'},
    )
