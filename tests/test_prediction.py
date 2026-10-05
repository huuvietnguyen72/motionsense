import copy
import threading
import time
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from motionsense_app.errors import DomainError


def test_selects_ids_ignores_metadata_and_maps_estimator_class_order(training_result, dataset_store):
    from motionsense_app.models.prediction import predict_rows

    sample = dataset_store.sample("test:000001")
    row = dict(reversed(list(sample["features"].items())))
    row.update(subject_id=999, actual_label=6, sample_id="test:000001")
    estimator = training_result["estimator"]
    estimator.classes_ = np.array([6, 3, 1, 5, 2, 4])
    ids = training_result["manifest"]["feature_ids"]
    frame = pd.DataFrame([[row[fid] for fid in ids]], columns=ids)
    expected = dict(zip(estimator.classes_, estimator.predict_proba(frame)[0]))
    prediction = predict_rows(training_result, [row])[0]
    assert prediction["sample_id"] is None and prediction["actual_label"] is None
    assert prediction["model_id"] == training_result["manifest"]["model_id"]
    assert prediction["predicted_label"] == int(estimator.predict(frame)[0])
    assert [p["label_id"] for p in prediction["probabilities"]] == [1, 2, 3, 4, 5, 6]
    assert {p["label_id"]: p["value"] for p in prediction["probabilities"]} == expected
    assert sum(p["value"] for p in prediction["probabilities"]) == pytest.approx(1)
    assert predict_rows(training_result, []) == []
    assert predict_rows(training_result, [row, row]) == [prediction, prediction]


@pytest.mark.parametrize("value", [None, "0.2", True, float("nan"), float("inf"),
                                   float("-inf"), [], 1e300, 10**1000])
def test_rejects_invalid_numeric_inputs(training_result, dataset_store, value):
    from motionsense_app.models.prediction import predict_rows

    row = dataset_store.sample("test:000001")["features"]
    fid = training_result["manifest"]["feature_ids"][0]
    row[fid] = value
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(DomainError) as error:
            predict_rows(training_result, [row])
    assert error.value.status == 422
    assert error.value.details[0]["column"] == fid
    assert error.value.details[0]["row"] == 1


@pytest.mark.parametrize("value", [10**30, np.int64(-9223372036854775808)])
def test_finite_numeric_integers_do_not_overflow_validation(training_result, dataset_store, value):
    from motionsense_app.models.prediction import predict_rows

    row = dataset_store.sample("test:000001")["features"]
    row[training_result["manifest"]["feature_ids"][0]] = value
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = predict_rows(training_result, [row])
    assert len(result) == 1
    assert sum(p["value"] for p in result[0]["probabilities"]) == pytest.approx(1)


def test_missing_required_feature_rejected(training_result):
    from motionsense_app.models.prediction import predict_rows

    with pytest.raises(DomainError):
        predict_rows(training_result, [{}])


@pytest.mark.parametrize("mutation", ["order", "duplicate", "unknown", "labels", "classes"])
def test_invalid_artifact_mapping_rejected(training_result, mutation):
    from motionsense_app.models.prediction import predict_rows

    artifact = copy.deepcopy(training_result)
    if mutation == "order":
        artifact["manifest"]["feature_ids"].reverse()
    elif mutation == "duplicate":
        artifact["manifest"]["feature_ids"][0] = artifact["manifest"]["feature_ids"][1]
    elif mutation == "unknown":
        artifact["manifest"]["feature_ids"][0] = "subject_id"
    elif mutation == "labels":
        artifact["manifest"]["labels"]["1"] = "Nằm"
    else:
        artifact["estimator"].classes_ = np.arange(6)
    with pytest.raises(DomainError):
        predict_rows(artifact, artifact["verification_rows"])


def test_mixed_leaf_inference_is_serial_without_mutation_and_reload_exact(
    dataset_store, tmp_path, monkeypatch
):
    from sklearn.ensemble import _forest

    from motionsense_app.models.prediction import predict_rows
    from motionsense_app.models.registry import ModelRegistry
    from motionsense_app.models.training import train_model

    # Real bootstrap trees with conflicting labels at identical feature vectors.
    dataset_store.arrays("train")[0][:] = 0
    dataset_store.arrays("test")[0][:] = 0
    result = train_model(dataset_store, "reduced")
    estimator = result["estimator"]
    ids = result["manifest"]["feature_ids"]
    rows = result["verification_rows"]
    frame = pd.DataFrame(rows, columns=ids)
    assert any(np.any((tree.tree_.value > 0) & (tree.tree_.value < 1))
               for tree in estimator.estimators_)
    serial = copy.copy(estimator)
    serial.n_jobs = 1
    expected = serial.predict_proba(frame)
    assert np.all((expected > 0) & (expected < 1))

    original_accumulate = _forest._accumulate_prediction
    scheduling = {"reverse": False}
    seeds = [tree.random_state for tree in estimator.estimators_]

    def varied_completion(predict, X, out, lock):
        # Delay real per-tree work in alternating patterns only for worker threads.
        # Outputs are never replaced; the genuine sklearn n_jobs=2 path is exercised.
        if threading.current_thread() is not threading.main_thread():
            index = seeds.index(predict.__self__.random_state)
            if bool(index % 2) == scheduling["reverse"]:
                time.sleep(0.002)
        return original_accumulate(predict, X, out, lock)

    monkeypatch.setattr(_forest, "_accumulate_prediction", varied_completion)
    for reverse in (False, True):
        scheduling["reverse"] = reverse
        parallel = estimator.predict_proba(frame)
        np.testing.assert_allclose(parallel, expected, rtol=0, atol=1e-15)

    original_predict_proba = RandomForestClassifier.predict_proba
    inference_workers = []

    def real_predict_proba(self, X):
        inference_workers.append(self.n_jobs)
        # Check during inference, not just afterwards: temporary mutation is unsafe too.
        assert estimator.n_jobs == 2
        return original_predict_proba(self, X)

    monkeypatch.setattr(RandomForestClassifier, "predict_proba", real_predict_proba)
    registry = ModelRegistry(tmp_path / "models")
    before = predict_rows(result, rows)
    np.testing.assert_array_equal(
        [[p["value"] for p in row["probabilities"]] for row in before], expected
    )
    assert inference_workers == [1]
    info = registry.publish(result)
    for reverse in (False, True, False):
        scheduling["reverse"] = reverse
        loaded = registry.load(info["model_id"])
        assert predict_rows(loaded, rows) == before
        assert loaded["estimator"].n_jobs == loaded["manifest"]["config"]["n_jobs"] == 2
        assert registry.list()[0]["status"] == "ready"
    assert set(inference_workers) == {1}
    assert estimator.get_params() == result["manifest"]["config"]
