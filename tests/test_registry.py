import copy
import json
from pathlib import Path

import joblib
import pytest

from motionsense_app.errors import DomainError


def test_save_load_all_predictions_report_and_immutable_version(training_result, tmp_path):
    from motionsense_app.models.prediction import predict_rows
    from motionsense_app.models.registry import ModelRegistry

    registry = ModelRegistry(tmp_path / "mô hình")
    before = predict_rows(training_result, training_result["verification_rows"])
    info = registry.publish(training_result)
    assert info["status"] == "ready"
    assert predict_rows(registry.load(info["model_id"]), training_result["verification_rows"]) == before
    assert registry.report(info["model_id"]) == training_result["report"]
    assert registry.list() == [info]
    with pytest.raises(DomainError):
        registry.publish(training_result)
    assert registry.list() == [info]


@pytest.mark.parametrize("library", ["sklearn", "numpy", "scipy"])
def test_runtime_mismatch_unavailable_before_deserializing(training_result, tmp_path,
                                                         monkeypatch, library):
    from motionsense_app.models.registry import ModelRegistry

    model_registry = ModelRegistry(tmp_path / "models")
    trained_model_id = model_registry.publish(training_result)["model_id"]
    path = model_registry.model_dir / trained_model_id / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["versions"][library] = "0.0.0"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    def forbidden_load(*args, **kwargs):
        pytest.fail("Mismatched runtime must be rejected before deserialization")

    monkeypatch.setattr(joblib, "load", forbidden_load)
    with pytest.raises(DomainError) as error:
        model_registry.load(trained_model_id)
    assert "huấn luyện lại" in error.value.message
    items = [item for item in model_registry.list() if item["model_id"] == trained_model_id]
    assert items[0]["status"] == "unavailable"


@pytest.mark.parametrize("file,damage", [("model.joblib", "missing"), ("model.joblib", "corrupt"),
                                        ("report.json", "missing"), ("report.json", "corrupt"),
                                        ("manifest.json", "corrupt")])
def test_missing_or_corrupt_files_unavailable(model_registry, trained_model_id, file, damage):
    path = model_registry.model_dir / trained_model_id / file
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"not a valid artifact")
    with pytest.raises(DomainError):
        model_registry.load(trained_model_id)
    if file != "manifest.json":
        item = next(i for i in model_registry.list() if i["model_id"] == trained_model_id)
        assert item["status"] == "unavailable"


@pytest.mark.parametrize("field,value", [("feature_ids", ["f001"] * 5), ("trees", 100),
                                       ("source_sha256", "b" * 64), ("schema_version", 99)])
def test_manifest_estimator_inconsistency_unavailable(model_registry, trained_model_id, field, value):
    path = model_registry.model_dir / trained_model_id / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DomainError):
        model_registry.load(trained_model_id)


@pytest.mark.parametrize("failure", ["dump", "json", "reload", "verification", "rename"])
def test_failed_publish_never_changes_ready_versions(model_registry, dataset_store,
                                                    monkeypatch, failure):
    from pathlib import Path

    from motionsense_app.models.training import train_model

    result = train_model(dataset_store, "reduced")
    before = model_registry.list()

    def storage_failure(*args, **kwargs):
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        if failure == "dump":
            patch.setattr(joblib, "dump", storage_failure)
        elif failure == "json":
            patch.setattr(Path, "write_text", storage_failure)
        elif failure == "rename":
            patch.setattr(Path, "rename", storage_failure)
        else:
            original_load = joblib.load

            def damaged_load(*args, **kwargs):
                loaded = original_load(*args, **kwargs)
                if failure == "reload":
                    raise ValueError("invalid serialized model")
                loaded["estimator"].classes_ = loaded["estimator"].classes_[::-1]
                return loaded

            patch.setattr(joblib, "load", damaged_load)
        with pytest.raises(DomainError):
            model_registry.publish(result)
    assert model_registry.list() == before
    assert not (model_registry.model_dir / result["manifest"]["model_id"]).exists()
    assert not list(model_registry.model_dir.glob(".staging-*"))


@pytest.mark.parametrize("mutation", ["no_rows", "report_id", "nonfinite", "wrong_support"])
def test_publish_rejects_invalid_result(training_result, tmp_path, mutation):
    from motionsense_app.models.registry import ModelRegistry

    result = copy.deepcopy(training_result)
    if mutation == "no_rows":
        result["verification_rows"] = []
    elif mutation == "report_id":
        result["report"]["model_id"] = "model-" + "a" * 32
    elif mutation == "nonfinite":
        result["report"]["accuracy"] = float("nan")
    else:
        result["report"]["per_class"][0]["support"] += 1
    registry = ModelRegistry(tmp_path / "models")
    with pytest.raises(DomainError):
        registry.publish(result)
    assert registry.list() == []


@pytest.mark.parametrize("model_id", ["../outside", "", "model-" + "a" * 32])
def test_unknown_or_unsafe_id_is_domain_error(tmp_path, model_id):
    from motionsense_app.models.registry import ModelRegistry

    with pytest.raises(DomainError):
        ModelRegistry(tmp_path).load(model_id)


def test_has_ready_filters_before_hashing_and_stops_at_first_valid_model(
    training_result, tmp_path, monkeypatch
):
    import motionsense_app.models.registry as registry_module
    from motionsense_app.models.registry import ModelRegistry

    registry = ModelRegistry(tmp_path / "models")
    ready = registry.publish(training_result)
    ready_path = registry.model_dir / ready["model_id"]
    manifest = json.loads((ready_path / "manifest.json").read_text(encoding="utf-8"))
    candidates = []
    for marker in ("a", "b", "c"):
        path = registry.model_dir / ("model-" + marker * 32)
        path.mkdir()
        candidate = copy.deepcopy(manifest)
        candidate["model_id"] = path.name
        if marker == "a":
            candidate["dataset_id"] = candidate["source_sha256"] = "b" * 64
        (path / "manifest.json").write_text(json.dumps(candidate), encoding="utf-8")
        candidates.append(path)
    unrelated, corrupt, later = candidates
    original_iterdir = Path.iterdir
    original_checksum = registry_module._checksum
    original_load = joblib.load
    hashed = []
    deserialized = []

    def ordered_iterdir(self):
        if self == registry.model_dir:
            yield unrelated
            yield corrupt
            yield ready_path
            pytest.fail("Readiness must stop scanning after the first validated match")
        else:
            yield from original_iterdir(self)

    def checked_checksum(path):
        assert path.parent not in (unrelated, later)
        hashed.append(path.parent)
        return original_checksum(path)

    def checked_load(path, *args, **kwargs):
        assert path.parent == ready_path
        deserialized.append(path.parent)
        return original_load(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "iterdir", ordered_iterdir)
        patch.setattr(registry_module, "_checksum", checked_checksum)
        patch.setattr(joblib, "load", checked_load)
        assert registry.has_ready(ready["dataset_id"]) is True
    assert corrupt in hashed and ready_path in hashed
    assert deserialized == [ready_path]
    # Full listing still validates every historical artifact and retains unavailable entries.
    statuses = {item["model_id"]: item["status"] for item in registry.list()}
    assert statuses == {ready["model_id"]: "ready", unrelated.name: "unavailable",
                        corrupt.name: "unavailable", later.name: "unavailable"}


@pytest.mark.parametrize("damage", ["empty", "manifest", "missing_model", "model", "report", "runtime"])
def test_has_ready_requires_full_validation_of_compatible_candidate(training_result, tmp_path, damage):
    from motionsense_app.models.registry import ModelRegistry

    registry = ModelRegistry(tmp_path / "models")
    dataset_id = training_result["manifest"]["dataset_id"]
    if damage != "empty":
        info = registry.publish(training_result)
        folder = registry.model_dir / info["model_id"]
        if damage == "missing_model":
            (folder / "model.joblib").unlink()
        elif damage == "runtime":
            path = folder / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["versions"]["numpy"] = "0.0.0"
            path.write_text(json.dumps(manifest), encoding="utf-8")
        else:
            name = {"manifest": "manifest.json", "model": "model.joblib", "report": "report.json"}[damage]
            (folder / name).write_bytes(b"corrupt")
    assert registry.has_ready(dataset_id) is False
