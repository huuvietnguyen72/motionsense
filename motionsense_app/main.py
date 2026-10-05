import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from motionsense_app.data.routes import router as data_router
from motionsense_app.data.store import DatasetStore
from motionsense_app.errors import DomainError, api_error
from motionsense_app.models.jobs import TrainingJobs
from motionsense_app.models.registry import ModelRegistry
from motionsense_app.models.routes import jobs_router
from motionsense_app.models.routes import router as models_router
from motionsense_app.recognition.routes import router as recognition_router
from motionsense_app.sessions.engine import SessionEngine
from motionsense_app.sessions.repository import SessionRepository
from motionsense_app.sessions.routes import router as sessions_router
from motionsense_app.settings import Settings

logger = logging.getLogger(__name__)


async def session_watchdog(engine: SessionEngine) -> None:
    while True:
        await asyncio.sleep(0.5)
        try:
            await asyncio.to_thread(engine.expire_leases)
        except DomainError:
            logger.exception("Session watchdog could not persist state")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings.var_root.mkdir(parents=True, exist_ok=True)
    if "store" in app.state.service_overrides:
        app.state.store = app.state.service_overrides["store"]
    else:
        app.state.store = DatasetStore(app.state.settings.data_dir)
    if "registry" in app.state.service_overrides:
        app.state.registry = app.state.service_overrides["registry"]
    else:
        app.state.registry = ModelRegistry(app.state.settings.model_dir)
    engine = None
    jobs = None
    watchdog = None
    try:
        injected_jobs = "training_jobs" in app.state.service_overrides
        candidate = (app.state.service_overrides["training_jobs"] if injected_jobs else
                     TrainingJobs(app.state.settings.db_path, app.state.store, app.state.registry))
        try:
            await asyncio.to_thread(candidate.start_lifecycle)
        except Exception as exc:
            # A failed co-start must not close a borrowed, already-live service.
            if not (injected_jobs and isinstance(exc, DomainError) and exc.code == "TRAINING_BUSY"):
                await asyncio.to_thread(candidate.close)
            raise
        jobs = candidate
        app.state.training_jobs = jobs
        if "session_engine" in app.state.service_overrides:
            engine = app.state.service_overrides["session_engine"]
        else:
            repo = SessionRepository(app.state.settings.db_path)
            engine = SessionEngine(repo, app.state.store, app.state.registry)
        app.state.session_engine = engine
        try:
            await asyncio.to_thread(engine.repo.recover)
        finally:
            await asyncio.to_thread(jobs.recover)
        watchdog = asyncio.create_task(session_watchdog(engine))
        yield
    finally:
        try:
            if watchdog is not None:
                watchdog.cancel()
                with suppress(asyncio.CancelledError):
                    await watchdog
        finally:
            try:
                if engine is not None:
                    await asyncio.to_thread(engine.close)
            finally:
                if jobs is not None:
                    await asyncio.to_thread(jobs.close)


def create_app(
    settings: Settings | None = None,
    *,
    services: dict | None = None,
) -> FastAPI:
    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.settings = settings or Settings.default()
    app.state.service_overrides = dict(services or {})

    @app.exception_handler(DomainError)
    async def handle_domain_error(_request: Request, exc: DomainError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status,
            content=api_error(exc.code, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details = []
        for error in exc.errors():
            location = error.get("loc", ())
            column = str(location[-1]) if location else None
            details.append(
                {
                    "row": None,
                    "column": column,
                    "message": "Giá trị không hợp lệ.",
                }
            )
        return JSONResponse(
            status_code=422,
            content=api_error(
                "VALIDATION_ERROR",
                "Dữ liệu yêu cầu không hợp lệ.",
                details,
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(
        _request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        if exc.status_code == 404:
            code = "NOT_FOUND"
            message = "Không tìm thấy tài nguyên."
        else:
            code = "HTTP_ERROR"
            message = "Yêu cầu không thể được xử lý."
        details: list[dict[str, Any]] = []
        return JSONResponse(
            status_code=exc.status_code,
            content=api_error(code, message, details),
            headers=exc.headers,
        )

    @app.get("/api/health")
    def health(response: Response) -> dict[str, Any]:
        response.headers['X-MotionSense-Instance'] = app.state.settings.instance_marker
        store = getattr(app.state, "store", None)
        try:
            info = store.info() if store is not None else {}
            data_ready = bool(info.get("ready"))
        except (OSError, ValueError, TypeError, KeyError, AttributeError, DomainError):
            data_ready = False
        models_ready = False
        if data_ready and info.get("dataset_id"):
            try:
                models_ready = bool(app.state.registry.has_ready(info["dataset_id"]))
            except (OSError, ValueError, TypeError, KeyError, AttributeError, DomainError):
                pass
        return {
            "app": "motionsense",
            "status": "ok",
            "data_ready": data_ready,
            "models_ready": models_ready,
        }

    app.include_router(data_router)
    app.include_router(models_router)
    app.include_router(jobs_router)
    app.include_router(recognition_router)
    app.include_router(sessions_router)

    frontend = app.state.settings.frontend_dir
    if (frontend / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=frontend / "assets"), name="assets")

    @app.get("/", response_model=None)
    def frontend_index() -> FileResponse | HTMLResponse:
        if (frontend / "index.html").is_file():
            return FileResponse(frontend / "index.html", headers={"Cache-Control": "no-cache"})
        return HTMLResponse(
            '<!doctype html><html lang="vi"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>MotionSense · Cần thiết lập</title><body><h1>MotionSense</h1>'
            '<p>Giao diện chưa được build. Hãy chạy Cai_dat.bat để hoàn tất thiết lập, '
            'sau đó khởi động lại MotionSense.</p></body></html>',
            status_code=503,
        )

    return app
