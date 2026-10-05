"""Public playback contracts; timing and browser ownership stay server-side."""

from typing import Literal

from pydantic import Field, FiniteFloat, field_validator

from motionsense_app.models import Contract, LabelId, ModelId, Prediction, Score, SourceHash


class CreateSessionRequest(Contract):
    dataset_id: SourceHash
    split: Literal["train", "test"]
    subject_id: int = Field(gt=0)
    model_id: str = Field(min_length=1)


class Session(Contract):
    session_id: str
    dataset_id: SourceHash
    split: Literal["train", "test"]
    subject_id: int
    model_id: ModelId
    status: Literal["ready", "running", "paused", "finished"]
    cursor: int = Field(ge=0)
    total: int = Field(gt=0)
    speed: Literal[0.5, 1.0, 2.0]
    created_at: str
    updated_at: str
    finish_reason: Literal["complete", "user"] | None
    last_prediction: Prediction | None


class SessionPage(Contract):
    items: list[Session]
    offset: int
    limit: int
    total: int


class AttachRequest(Contract):
    client_id: str = Field(min_length=1, pattern=r"\S")


class ControlRequest(AttachRequest):
    action: str = Field(min_length=1)
    speed: FiniteFloat | None = None

    @field_validator("speed")
    @classmethod
    def validate_supplied_speed(cls, value: float | None) -> float:
        # An omitted field keeps its default; an explicitly supplied null is invalid.
        # FiniteFloat + the strict contract rejects booleans and strings before this.
        if value not in (0.5, 1, 2):
            raise ValueError("unsupported playback speed")
        return value


class AdvanceRequest(AttachRequest):
    expected_cursor: int = Field(ge=0)


class SessionRow(Contract):
    ordinal: int = Field(ge=0)
    sample_id: str
    prediction: Prediction
    processed_at: str


class SessionRowPage(Contract):
    items: list[SessionRow]
    offset: int
    limit: int
    total: int


class AdvanceResult(Contract):
    session: Session
    row: SessionRow | None


class ActivityCount(Contract):
    label_id: LabelId
    count: int = Field(ge=0)


class SessionSummary(Contract):
    processed: int = Field(ge=0)
    labeled: int = Field(ge=0)
    correct: int = Field(ge=0)
    counts: list[ActivityCount]
    session_accuracy: Score | None
