"""Shared model contracts for locally trained MotionSense models."""

import math
import platform
from datetime import datetime
from typing import Annotated, Any, Literal

import joblib
import numpy as np
import scipy
import sklearn
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from motionsense_app.data.schema import ACTIVITY_NAMES

LABELS = [1, 2, 3, 4, 5, 6]
FEATURE_IDS = [f"f{i:03d}" for i in range(1, 562)]
ModelId = Annotated[str, Field(pattern=r"^model-[0-9a-f]{32}$")]
SourceHash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Score = Annotated[FiniteFloat, Field(ge=0, le=1)]
LabelId = Annotated[int, Field(ge=1, le=6)]


class Contract(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class ModelInfo(Contract):
    model_id: ModelId
    dataset_id: SourceHash
    profile: Literal["full", "reduced"]
    trees: Literal[25, 50, 100, 150]
    seed: Literal[42]
    feature_ids: list[str]
    created_at: str
    status: Literal["ready", "unavailable"]

    @model_validator(mode="after")
    def validate_features_and_time(self):
        if self.profile == "full":
            valid = self.feature_ids == FEATURE_IDS
        else:
            valid = (len(self.feature_ids) == len(set(self.feature_ids)) == 5
                     and set(self.feature_ids).issubset(FEATURE_IDS))
        timestamp = datetime.fromisoformat(self.created_at)
        if not valid or timestamp.utcoffset() is None or timestamp.utcoffset().total_seconds() != 0:
            raise ValueError("invalid feature order or UTC timestamp")
        return self


class ModelManifest(ModelInfo):
    schema_version: Literal[1]
    source_sha256: SourceHash
    labels: dict[str, str]
    versions: dict[str, str]
    config: dict[str, Any]
    file_sha256: dict[str, SourceHash] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_metadata(self):
        if self.source_sha256 != self.dataset_id:
            raise ValueError("source hash differs from snapshot")
        if self.labels != {str(k): v for k, v in ACTIVITY_NAMES.items()}:
            raise ValueError("invalid label mapping")
        if set(self.versions) != {"python", "sklearn", "numpy", "scipy", "joblib"}:
            raise ValueError("missing runtime versions")
        if any(not value.strip() for value in self.versions.values()):
            raise ValueError("invalid runtime versions")
        return self


class Probability(Contract):
    label_id: LabelId
    value: Score


class Prediction(Contract):
    sample_id: str | None = None
    model_id: ModelId
    predicted_label: LabelId
    probabilities: list[Probability]
    actual_label: LabelId | None = None

    @model_validator(mode="after")
    def validate_probabilities(self):
        if [p.label_id for p in self.probabilities] != LABELS or not math.isclose(
            sum(p.value for p in self.probabilities), 1.0, abs_tol=1e-12
        ):
            raise ValueError("invalid six-class probability distribution")
        return self


class ClassMetrics(Contract):
    label_id: LabelId
    precision: Score
    recall: Score
    f1: Score
    support: int = Field(ge=0)


class FeatureImportance(Contract):
    feature_id: str
    name: str = Field(min_length=1)
    value: Score


class ModelReport(Contract):
    model_id: ModelId
    labels: list[int]
    accuracy: Score
    macro_f1: Score
    train_accuracy: Score
    oob_score: Score | None
    oob_warning: str | None
    confusion_matrix: list[list[int]]
    per_class: list[ClassMetrics]
    feature_importance: list[FeatureImportance]
    test_count: int = Field(gt=0)
    split_strategy: Literal["uci_subject_split"]

    @model_validator(mode="after")
    def validate_metrics(self):
        matrix = self.confusion_matrix
        if (self.labels != LABELS or len(matrix) != 6
                or any(len(row) != 6 or any(value < 0 for value in row) for row in matrix)
                or [c.label_id for c in self.per_class] != LABELS):
            raise ValueError("invalid six-class report")
        if sum(map(sum, matrix)) != self.test_count:
            raise ValueError("invalid test count")
        for i, metrics in enumerate(self.per_class):
            support = sum(matrix[i])
            predicted_count = sum(row[i] for row in matrix)
            precision = matrix[i][i] / predicted_count if predicted_count else 0.0
            recall = matrix[i][i] / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            if metrics.support != support or any(not math.isclose(a, b, abs_tol=1e-12)
                                                for a, b in [(metrics.precision, precision),
                                                             (metrics.recall, recall),
                                                             (metrics.f1, f1)]):
                raise ValueError("class metrics differ from confusion matrix")
        if not math.isclose(self.accuracy, sum(matrix[i][i] for i in range(6)) / self.test_count,
                            abs_tol=1e-12) or not math.isclose(
                                self.macro_f1, sum(c.f1 for c in self.per_class) / 6,
                                abs_tol=1e-12):
            raise ValueError("summary metrics differ from confusion matrix")
        if (self.oob_score is None) != bool(self.oob_warning):
            raise ValueError("missing OOB warning")
        return self


def runtime_versions() -> dict[str, str]:
    return {"python": platform.python_version(), "sklearn": sklearn.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__}
