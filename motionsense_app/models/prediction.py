import copy
import math
from numbers import Real

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from motionsense_app.errors import DomainError
from motionsense_app.models import LABELS, ModelManifest, Prediction, runtime_versions


def deterministic_probabilities(estimator: RandomForestClassifier, frame: pd.DataFrame) -> np.ndarray:
    """Reduce tree probabilities in estimator order without changing shared fit metadata.

    The shallow copy owns its inference settings and shares only read-only fitted trees.
    A single worker avoids scheduling-dependent floating-point accumulation in sklearn.
    """
    inference = copy.copy(estimator)
    inference.n_jobs = 1
    return inference.predict_proba(frame)


def validate_artifact(artifact: dict) -> ModelManifest:
    """Bind the persisted column/class/configuration contract to the fitted estimator."""
    try:
        manifest = ModelManifest.model_validate(artifact["manifest"])
        runtime = runtime_versions()
        if any(manifest.versions[k] != runtime[k] for k in ("sklearn", "numpy", "scipy")):
            raise DomainError("MODEL_UNAVAILABLE", "Phiên bản thư viện đã thay đổi; hãy huấn luyện lại.", 503)
        estimator = artifact["estimator"]
        expected = RandomForestClassifier(n_estimators=manifest.trees, random_state=42,
                                          n_jobs=2, oob_score=True).get_params()
        if (manifest.status != "ready" or not isinstance(estimator, RandomForestClassifier)
                or estimator.get_params() != manifest.config or manifest.config != expected
                or list(estimator.feature_names_in_) != manifest.feature_ids
                or estimator.n_features_in_ != len(manifest.feature_ids)
                or len(estimator.estimators_) != manifest.trees
                or np.asarray(estimator.classes_).shape != (6,)
                or set(estimator.classes_) != set(LABELS)):
            raise ValueError("inconsistent estimator metadata")
        return manifest
    except DomainError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise DomainError("MODEL_UNAVAILABLE", "Mô hình không hợp lệ; hãy huấn luyện lại.", 503) from exc


def predict_rows(artifact: dict, rows: list[dict[str, float]]) -> list[dict]:
    manifest = validate_artifact(artifact)
    values = []
    if not isinstance(rows, list):
        raise DomainError("PREDICTION_INVALID", "Đầu vào nhận diện phải là danh sách hàng đặc trưng.")
    for ordinal, row in enumerate(rows, 1):
        selected = []
        for fid in manifest.feature_ids:
            value = row.get(fid) if isinstance(row, dict) else None
            numeric = None
            if not isinstance(value, (bool, np.bool_)) and isinstance(value, Real):
                try:
                    numeric = float(value)
                except (ValueError, TypeError, OverflowError):
                    pass
            if (numeric is None or not math.isfinite(numeric)
                    or abs(numeric) > float(np.finfo(np.float32).max)):
                message = "Đặc trưng bắt buộc phải là số hữu hạn trong miền float32, không thiếu giá trị."
                raise DomainError("PREDICTION_INVALID", "Dữ liệu nhận diện không hợp lệ.", details=[
                    {"row": ordinal, "column": fid, "message": message}
                ])
            selected.append(numeric)
        values.append(selected)
    if not values:
        return []
    try:
        estimator = artifact["estimator"]
        probabilities = deterministic_probabilities(
            estimator, pd.DataFrame(values, columns=manifest.feature_ids)
        )
        results = []
        for vector in probabilities:
            mapping = {int(label): float(value) for label, value in zip(estimator.classes_, vector)}
            results.append(Prediction.model_validate({
                "model_id": manifest.model_id,
                "predicted_label": int(estimator.classes_[np.argmax(vector)]),
                "probabilities": [{"label_id": label, "value": mapping[label]} for label in LABELS],
            }).model_dump())
        return results
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise DomainError("MODEL_UNAVAILABLE", "Không thể dự đoán bằng mô hình; hãy huấn luyện lại.", 503) from exc
