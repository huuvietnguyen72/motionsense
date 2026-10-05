from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, Field, FiniteFloat

from motionsense_app.errors import DomainError

SCHEMA_VERSION = 1
SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)
DOI = "10.24432/C54S4K"
LICENSE = "CC BY 4.0"
ACTIVITY_NAMES = {
    1: "Đi bộ",
    2: "Đi bộ lên cầu thang",
    3: "Đi bộ xuống cầu thang",
    4: "Ngồi",
    5: "Đứng",
    6: "Nằm",
}

SignalPoints = Annotated[list[FiniteFloat], Field(min_length=128, max_length=128)]


class Signal(BaseModel):
    time_seconds: SignalPoints
    x: SignalPoints
    y: SignalPoints
    z: SignalPoints
    unit: Literal["g"]


class Sample(BaseModel):
    sample_id: str
    dataset_id: str
    split: Literal["train", "test"]
    subject_id: int = Field(gt=0)
    features: dict[str, FiniteFloat]
    actual_label: int = Field(ge=1, le=6)
    signal: Signal


class SplitCounts(BaseModel):
    train: int = Field(ge=0)
    test: int = Field(ge=0)


class SubjectInfo(BaseModel):
    subject_id: int = Field(gt=0)
    split: Literal["train", "test"]
    sample_count: int = Field(ge=0)


class ActivityInfo(BaseModel):
    label_id: int = Field(ge=1, le=6)
    name_vi: str
    count: int = Field(ge=0)


class DatasetInfo(BaseModel):
    dataset_id: str | None
    ready: bool
    source_url: str
    license: str
    feature_count: int = Field(ge=0)
    split_counts: SplitCounts
    subjects: list[SubjectInfo]
    activities: list[ActivityInfo]


class FeatureInfo(BaseModel):
    feature_id: str
    name: str


class FeatureList(BaseModel):
    items: list[FeatureInfo]


class SampleSummary(BaseModel):
    sample_id: str
    subject_id: int = Field(gt=0)
    split: Literal["train", "test"]
    actual_label: int = Field(ge=1, le=6)


class SamplePage(BaseModel):
    items: list[SampleSummary]
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=200)
    total: int = Field(ge=0)


def validate_bundle(bundle: dict) -> dict[str, int]:
    """Validate the shared loader/publisher/store contract, including small fixtures."""
    def invalid(message: str) -> DomainError:
        return DomainError("DATA_INVALID", f"Dữ liệu UCI HAR không hợp lệ: {message}")

    features = bundle.get("features")
    labels = bundle.get("labels")
    if not isinstance(features, list) or len(features) != 561:
        raise invalid("phải có đúng 561 đặc trưng.")
    for index, feature in enumerate(features, 1):
        if (
            not isinstance(feature, dict)
            or feature.get("feature_id") != f"f{index:03d}"
            or not isinstance(feature.get("name"), str)
            or not feature["name"].strip()
        ):
            raise invalid("ID đặc trưng phải duy nhất theo thứ tự cột gốc.")
    if (
        not isinstance(labels, dict)
        or set(labels) != set(range(1, 7))
        or any(not isinstance(name, str) or not name.strip() for name in labels.values())
    ):
        raise invalid("schema nhãn phải định nghĩa đủ nhãn 1–6.")
    counts = {}
    subjects_by_split = {}
    for split in ("train", "test"):
        try:
            data = bundle[split]
            X, y, subjects, signal = data["X"], data["y"], data["subjects"], data["total_acc"]
        except (KeyError, TypeError) as exc:
            raise invalid(f"thiếu dữ liệu {split}.") from exc
        if not all(isinstance(array, np.ndarray) for array in (X, y, subjects, signal)):
            raise invalid(f"{split}: dữ liệu phải là mảng NumPy.")
        if X.ndim != 2 or X.shape[1] != 561:
            raise invalid(f"{split}: X phải có 561 cột.")
        n = len(X)
        if n == 0 or y.shape != (n,) or subjects.shape != (n,) or signal.shape != (n, 128, 3):
            raise invalid(f"{split}: kích thước X/y/subjects/tín hiệu không khớp hoặc rỗng.")
        if any(array.dtype.kind not in "fiu" for array in (X, y, subjects, signal)):
            raise invalid(f"{split}: dữ liệu phải là số.")
        if not all(np.isfinite(array).all() for array in (X, y, subjects, signal)):
            raise invalid(f"{split}: phải chứa giá trị hữu hạn.")
        if not np.equal(y, np.floor(y)).all() or np.any((y < 1) | (y > 6)):
            raise invalid(f"{split}: nhãn phải là số nguyên từ 1 đến 6.")
        if not np.equal(subjects, np.floor(subjects)).all() or np.any(subjects <= 0):
            raise invalid(f"{split}: người tham gia phải là số nguyên dương.")
        counts[split] = n
        subjects_by_split[split] = set(subjects.tolist())
    if subjects_by_split["train"] & subjects_by_split["test"]:
        raise DomainError("SUBJECT_OVERLAP", "Hai tập dữ liệu có người tham gia trùng nhau.")
    return counts
