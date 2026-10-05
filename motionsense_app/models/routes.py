from typing import Annotated, Literal

from fastapi import APIRouter
from pydantic import Field, field_validator

from motionsense_app.dependencies import RegistryDependency, TrainingJobsDependency
from motionsense_app.models import Contract, ModelInfo, ModelReport
from motionsense_app.models.jobs import Job

router = APIRouter(prefix="/api/models")
jobs_router = APIRouter(prefix="/api/jobs")


class ModelList(Contract):
    items: list[ModelInfo]


class TrainRequest(Contract):
    profile: Literal["full", "reduced"]
    # Literal integers alone accept equal-valued floats/bools; enforce strict int first.
    trees: Annotated[int, Field(strict=True)]

    @field_validator("trees")
    @classmethod
    def supported_trees(cls, value: int) -> int:
        if value not in (25, 50, 100, 150):
            raise ValueError("unsupported tree count")
        return value


@router.post("/train", response_model=Job, status_code=202)
def train(body: TrainRequest, jobs: TrainingJobsDependency) -> dict:
    return jobs.submit(body.profile, body.trees)


@jobs_router.get("/{job_id}", response_model=Job)
def job(job_id: str, jobs: TrainingJobsDependency) -> dict:
    return jobs.get(job_id)


@router.get("", response_model=ModelList)
def models(registry: RegistryDependency) -> dict:
    return {"items": registry.list()}


@router.get("/{model_id}/report", response_model=ModelReport)
def report(model_id: str, registry: RegistryDependency) -> dict:
    return registry.report(model_id)
