import warnings
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from motionsense_app.data.schema import ACTIVITY_NAMES
from motionsense_app.data.store import DatasetStore
from motionsense_app.errors import DomainError
from motionsense_app.models import FEATURE_IDS, LABELS, ModelManifest, ModelReport, runtime_versions
from motionsense_app.models.prediction import deterministic_probabilities


def select_reduced_features(X: np.ndarray, y: np.ndarray, feature_ids: list[str]) -> list[str]:
    if (X.ndim != 2 or X.shape[1] != len(feature_ids) or X.shape[1] < 7
            or len(set(feature_ids)) != len(feature_ids) or not set(feature_ids).issubset(FEATURE_IDS)
            or not len(X) or y.shape != (len(X),) or not np.isfinite(X).all()
            or np.max(np.abs(X)) > float(np.finfo(np.float32).max)
            or not np.isin(y, LABELS).all() or set(y) != set(LABELS)):
        raise DomainError("TRAINING_INVALID", "Lược đồ đặc trưng huấn luyện không hợp lệ.")
    ranker = RandomForestClassifier(n_estimators=25, random_state=42, n_jobs=2)
    ranker.fit(X, y)
    order = np.lexsort((np.arange(X.shape[1]), -ranker.feature_importances_))
    return [feature_ids[i] for i in [*order[:4], order[6]]]


def train_model(
    store: DatasetStore,
    profile: str,
    trees: int = 50,
    on_stage: Callable[[str], None] = lambda stage: None,
) -> dict:
    on_stage("validating")
    if profile not in ("full", "reduced") or type(trees) is not int or trees not in (25, 50, 100, 150):
        raise DomainError("TRAINING_CONFIG", "Chọn cấu hình đầy đủ/rút gọn và 25, 50, 100 hoặc 150 cây.")
    info = store.info()
    if not info.get("ready"):
        raise DomainError("DATA_UNAVAILABLE", "Dữ liệu chưa sẵn sàng. Hãy chuẩn bị dữ liệu UCI HAR.", 503)
    features = store.features()
    ids = [f["feature_id"] for f in features]
    if ids != FEATURE_IDS:
        raise DomainError("TRAINING_INVALID", "Dữ liệu phải có đủ 561 đặc trưng theo thứ tự gốc.")
    train_X, train_y, train_subjects = store.arrays("train")
    test_X, test_y, test_subjects = store.arrays("test")
    for X, y, subjects in ((train_X, train_y, train_subjects), (test_X, test_y, test_subjects)):
        if (X.ndim != 2 or X.shape[1] != 561 or not len(X) or y.shape != (len(X),)
                or subjects.shape != y.shape or not np.isfinite(X).all()
                or np.max(np.abs(X)) > float(np.finfo(np.float32).max)
                or not np.isin(y, LABELS).all() or not np.isfinite(subjects).all()
                or not np.equal(subjects, np.floor(subjects)).all() or np.any(subjects <= 0)):
            raise DomainError("TRAINING_INVALID", "Dữ liệu huấn luyện/kiểm tra không hợp lệ.")
    if set(train_subjects) & set(test_subjects):
        raise DomainError("SUBJECT_OVERLAP", "Hai tập dữ liệu có người tham gia trùng nhau.")
    if set(train_y) != set(LABELS):
        raise DomainError("TRAINING_INVALID", "Tập huấn luyện phải có đủ sáu nhãn hoạt động.")

    on_stage("selecting_features")
    selected = ids if profile == "full" else select_reduced_features(train_X, train_y, ids)
    indices = [ids.index(fid) for fid in selected]
    train_frame = pd.DataFrame(train_X[:, indices], columns=selected)
    test_frame = pd.DataFrame(test_X[:, indices], columns=selected)
    on_stage("fitting")
    estimator = RandomForestClassifier(n_estimators=trees, random_state=42, n_jobs=2, oob_score=True)
    with warnings.catch_warnings():
        # Translate only sklearn's known incomplete-vote warning into the persisted report.
        warnings.filterwarnings("ignore", message="Some inputs do not have OOB scores.*",
                                category=UserWarning)
        estimator.fit(train_frame, train_y)
    on_stage("evaluating")
    predicted = estimator.classes_[np.argmax(deterministic_probabilities(estimator, test_frame), axis=1)]
    per_class = classification_report(test_y, predicted, labels=LABELS,
                                      output_dict=True, zero_division=0)
    votes = estimator.oob_decision_function_
    complete_oob = (np.isfinite(votes).all() and np.all(votes.sum(axis=1) > 0)
                    and np.isfinite(estimator.oob_score_))
    model_id = f"model-{uuid4().hex}"
    manifest = ModelManifest.model_validate({
        "schema_version": 1, "model_id": model_id, "dataset_id": info["dataset_id"],
        "source_sha256": info["dataset_id"], "profile": profile, "trees": trees, "seed": 42,
        "feature_ids": selected, "created_at": datetime.now(UTC).isoformat(), "status": "ready",
        "labels": {str(k): v for k, v in ACTIVITY_NAMES.items()}, "versions": runtime_versions(),
        "config": estimator.get_params(),
    }).model_dump()
    names = {f["feature_id"]: f["name"] for f in features}
    report = ModelReport.model_validate({
        "model_id": model_id, "labels": LABELS,
        "accuracy": float(accuracy_score(test_y, predicted)),
        "macro_f1": float(f1_score(test_y, predicted, labels=LABELS, average="macro", zero_division=0)),
        "train_accuracy": float(accuracy_score(train_y, estimator.classes_[np.argmax(
            deterministic_probabilities(estimator, train_frame), axis=1
        )])),
        "oob_score": float(estimator.oob_score_) if complete_oob else None,
        "oob_warning": None if complete_oob else
            "Một số mẫu huấn luyện thiếu phiếu OOB hợp lệ. Hãy huấn luyện lại với nhiều cây hơn.",
        "confusion_matrix": confusion_matrix(test_y, predicted, labels=LABELS).tolist(),
        "per_class": [{"label_id": label, "precision": float(per_class[str(label)]["precision"]),
                       "recall": float(per_class[str(label)]["recall"]),
                       "f1": float(per_class[str(label)]["f1-score"]),
                       "support": int(per_class[str(label)]["support"])} for label in LABELS],
        "feature_importance": [{"feature_id": fid, "name": names[fid], "value": float(value)}
                               for fid, value in zip(selected, estimator.feature_importances_)],
        "test_count": len(test_y), "split_strategy": "uci_subject_split",
    }).model_dump()
    # Verify at least one real test row per present class without persisting the entire dataset.
    verification_rows = [dict(zip(selected, map(float, test_frame.iloc[i])))
                         for i in np.unique(test_y, return_index=True)[1]]
    on_stage("saving")
    return {"estimator": estimator, "manifest": manifest, "report": report,
            "verification_rows": verification_rows}
