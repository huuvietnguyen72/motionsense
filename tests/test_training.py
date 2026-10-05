import copy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from motionsense_app.errors import DomainError


def test_selection_exact_formula_repeatable_and_tie_order(dataset_store):
    from motionsense_app.models.training import select_reduced_features

    X, y, _ = dataset_store.arrays("train")
    ids = [f["feature_id"] for f in dataset_store.features()]
    ranker = RandomForestClassifier(n_estimators=25, random_state=42, n_jobs=2).fit(X, y)
    order = np.lexsort((np.arange(561), -ranker.feature_importances_))
    expected = [ids[i] for i in [*order[:4], order[6]]]
    assert select_reduced_features(X, y, ids) == expected
    assert select_reduced_features(X, y, ids) == expected
    assert expected != [ids[i] for i in order[:5]]
    # All importances zero: tie-break by original column order, still skip ranks 5/6.
    assert select_reduced_features(np.zeros_like(X), y, ids) == [
        "f001", "f002", "f003", "f004", "f007"
    ]


@pytest.mark.parametrize("profile,count", [("full", 561), ("reduced", 5)])
def test_training_stages_config_and_actual_report(dataset_store, profile, count):
    from motionsense_app.models.training import train_model

    stages = []
    result = train_model(dataset_store, profile, on_stage=stages.append)
    estimator, manifest, report = (result[k] for k in ("estimator", "manifest", "report"))
    assert stages == ["validating", "selecting_features", "fitting", "evaluating", "saving"]
    assert len(manifest["feature_ids"]) == count
    assert manifest["dataset_id"] == dataset_store.info()["dataset_id"]
    assert manifest["source_sha256"] == manifest["dataset_id"]
    assert set(manifest["versions"]) >= {"python", "sklearn", "numpy", "scipy", "joblib"}
    assert estimator.n_estimators == 50 and estimator.random_state == 42
    assert estimator.n_jobs == 2 and estimator.oob_score is True
    assert list(estimator.feature_names_in_) == manifest["feature_ids"]
    assert list(estimator.classes_) == [1, 2, 3, 4, 5, 6]
    X, y, _ = dataset_store.arrays("test")
    indices = [int(fid[1:]) - 1 for fid in manifest["feature_ids"]]
    predicted = estimator.predict(pd.DataFrame(X[:, indices], columns=manifest["feature_ids"]))
    assert report["accuracy"] == accuracy_score(y, predicted)
    assert report["macro_f1"] == f1_score(y, predicted, average="macro", zero_division=0)
    assert report["confusion_matrix"] == confusion_matrix(y, predicted, labels=range(1, 7)).tolist()
    expected = classification_report(y, predicted, labels=range(1, 7), output_dict=True,
                                     zero_division=0)
    for entry in report["per_class"]:
        metrics = expected[str(entry["label_id"])]
        assert entry == {"label_id": entry["label_id"], "precision": metrics["precision"],
                         "recall": metrics["recall"], "f1": metrics["f1-score"],
                         "support": int(metrics["support"])}
    X_train, y_train, _ = dataset_store.arrays("train")
    train_predicted = estimator.predict(pd.DataFrame(X_train[:, indices],
                                                    columns=manifest["feature_ids"]))
    assert report["train_accuracy"] == accuracy_score(y_train, train_predicted)
    assert report["oob_score"] == estimator.oob_score_
    assert report["oob_warning"] is None
    assert report["test_count"] == 48 and report["split_strategy"] == "uci_subject_split"
    assert report["labels"] == [1, 2, 3, 4, 5, 6]
    assert report["model_id"] == manifest["model_id"]
    assert [f["feature_id"] for f in report["feature_importance"]] == manifest["feature_ids"]
    assert all(f["name"] == "duplicate-name" for f in report["feature_importance"])
    assert [f["value"] for f in report["feature_importance"]] == list(estimator.feature_importances_)
    assert result["verification_rows"]
    json.dumps(report, allow_nan=False)


def test_test_perturbation_does_not_change_features_or_fitted_model(dataset_store):
    from motionsense_app.models.training import train_model

    first = train_model(dataset_store, "reduced")
    X, y, _ = dataset_store.arrays("test")
    X[:] = -17
    y[:] = 7 - y
    second = train_model(dataset_store, "reduced")
    assert first["manifest"]["feature_ids"] == second["manifest"]["feature_ids"]
    for a, b in zip(first["estimator"].estimators_, second["estimator"].estimators_):
        np.testing.assert_array_equal(a.tree_.threshold, b.tree_.threshold)
        np.testing.assert_array_equal(a.tree_.value, b.tree_.value)
    assert first["manifest"]["model_id"] != second["manifest"]["model_id"]


@pytest.mark.parametrize("trees", [25, 50, 100, 150])
def test_supported_tree_counts(dataset_store, trees):
    from motionsense_app.models.training import train_model

    assert train_model(dataset_store, "reduced", trees)["estimator"].n_estimators == trees


@pytest.mark.parametrize("profile,trees", [("other", 50), ("full561", 50), ("full", 0),
                                          ("reduced", 51), ("full", 50.0), ("full", True)])
def test_invalid_config_is_domain_error(dataset_store, profile, trees):
    from motionsense_app.models.training import train_model

    with pytest.raises(DomainError):
        train_model(dataset_store, profile, trees)


@pytest.mark.parametrize("mutation", ["overlap", "missing_class", "nan"])
def test_training_revalidates_split_and_all_six_classes(dataset_store, mutation):
    from motionsense_app.models.training import train_model

    X, y, subjects = dataset_store.arrays("train")
    if mutation == "overlap":
        subjects[:] = 2
    elif mutation == "missing_class":
        y[y == 6] = 5
    else:
        X[0, 0] = np.nan
    with pytest.raises(DomainError):
        train_model(dataset_store, "full")


@pytest.mark.parametrize("vote", [0.0, float("nan")])
def test_incomplete_oob_is_null_with_localized_warning(dataset_store, monkeypatch, vote):
    from motionsense_app.models.training import train_model

    original_fit = RandomForestClassifier.fit

    def incomplete_fit(self, *args, **kwargs):
        result = original_fit(self, *args, **kwargs)
        if self.oob_score:
            self.oob_decision_function_[0] = vote
        return result

    monkeypatch.setattr(RandomForestClassifier, "fit", incomplete_fit)
    report = train_model(dataset_store, "full")["report"]
    assert report["oob_score"] is None
    assert "phiếu" in report["oob_warning"]
    json.dumps(report, allow_nan=False)


def test_missing_dataset_is_localized_unavailable(tmp_path):
    from motionsense_app.data.store import DatasetStore
    from motionsense_app.models.training import train_model

    with pytest.raises(DomainError) as error:
        train_model(DatasetStore(tmp_path), "full")
    assert error.value.status == 503


@pytest.mark.parametrize("mutation", ["width", "nan", "unknown_id", "duplicate_id", "labels"])
def test_selection_rejects_invalid_training_inputs(dataset_store, mutation):
    from motionsense_app.models.training import select_reduced_features

    X, y, _ = dataset_store.arrays("train")
    ids = [f["feature_id"] for f in dataset_store.features()]
    if mutation == "width":
        X = X[:, :5]
    elif mutation == "nan":
        X[0, 0] = np.nan
    elif mutation == "unknown_id":
        ids[0] = "subject_id"
    elif mutation == "duplicate_id":
        ids[0] = ids[1]
    else:
        y[0] = 99
    with pytest.raises(DomainError):
        select_reduced_features(X, y, ids)


def test_report_evaluation_uses_serial_inference_but_retains_parallel_fit(dataset_store, monkeypatch):
    from motionsense_app.models.training import train_model

    dataset_store.arrays("train")[0][:] = 0
    dataset_store.arrays("test")[0][:] = 0
    original_predict_proba = RandomForestClassifier.predict_proba
    workers = []

    def real_predict_proba(self, X):
        workers.append(self.n_jobs)
        return original_predict_proba(self, X)

    monkeypatch.setattr(RandomForestClassifier, "predict_proba", real_predict_proba)
    result = train_model(dataset_store, "full")
    assert workers == [1, 1]
    assert result["estimator"].n_jobs == result["manifest"]["config"]["n_jobs"] == 2
    serial = copy.copy(result["estimator"])
    serial.n_jobs = 1
    X, y, _ = dataset_store.arrays("test")
    assert result["report"]["accuracy"] == accuracy_score(
        y, serial.predict(pd.DataFrame(X, columns=result["manifest"]["feature_ids"]))
    )
