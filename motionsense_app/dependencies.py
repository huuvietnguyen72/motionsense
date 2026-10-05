from typing import Annotated

from fastapi import Depends, Request

from motionsense_app.data.store import DatasetStore
from motionsense_app.errors import DomainError
from motionsense_app.models.jobs import TrainingJobs
from motionsense_app.models.registry import ModelRegistry
from motionsense_app.sessions.engine import SessionEngine
from motionsense_app.sessions.repository import SessionRepository


def get_store(request: Request) -> DatasetStore:
    return request.app.state.store


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


def get_training_jobs(request: Request) -> TrainingJobs:
    return request.app.state.training_jobs


def get_session_engine(request: Request) -> SessionEngine:
    return request.app.state.session_engine


def get_session_repository(request: Request) -> SessionRepository:
    return request.app.state.session_engine.repo


StoreDependency = Annotated[DatasetStore, Depends(get_store)]
RegistryDependency = Annotated[ModelRegistry, Depends(get_registry)]
TrainingJobsDependency = Annotated[TrainingJobs, Depends(get_training_jobs)]
EngineDependency = Annotated[SessionEngine, Depends(get_session_engine)]
SessionRepositoryDependency = Annotated[SessionRepository, Depends(get_session_repository)]


def require_dataset(store: DatasetStore) -> dict:
    info = store.info()
    if not info.get("ready"):
        raise DomainError(
            "DATA_UNAVAILABLE", "Dữ liệu chưa sẵn sàng. Hãy chạy Cai_dat.bat.", 503,
        )
    return info


def require_compatible_dataset(store: DatasetStore, artifact: dict) -> None:
    info = require_dataset(store)
    if info["dataset_id"] != artifact["manifest"]["dataset_id"]:
        raise DomainError(
            "DATASET_MISMATCH",
            "Mô hình không tương thích với phiên bản dữ liệu hiện tại. Hãy huấn luyện lại.",
            503,
        )
