import json
import shutil

from fastapi.testclient import TestClient

from motionsense_app.cli import main
from motionsense_app.main import create_app
from motionsense_app.settings import Settings


def test_cli_trains_both_and_predicts_actual_sample(dataset_store, monkeypatch, capsys, tmp_path):
    from motionsense_app.models.prediction import predict_rows
    from motionsense_app.models.registry import ModelRegistry

    settings = Settings(root=tmp_path)
    monkeypatch.setattr(Settings, "default", classmethod(lambda cls: settings))
    # Use the existing snapshot under isolated state, not production var.
    monkeypatch.setattr("motionsense_app.cli.DatasetStore", lambda path: dataset_store)
    assert main(["train", "--profile", "both", "--trees", "50"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert {item["profile"] for item in output["items"]} == {"full", "reduced"}
    registry = ModelRegistry(settings.model_dir)
    assert output["items"] == registry.list()
    model_id = next(i["model_id"] for i in output["items"] if i["profile"] == "reduced")
    assert main(["predict", "--model-id", model_id, "--sample-id", "test:000001"]) == 0
    actual = json.loads(capsys.readouterr().out)
    sample = dataset_store.sample("test:000001")
    expected = predict_rows(registry.load(model_id), [sample["features"]])[0]
    expected.update(sample_id=sample["sample_id"], actual_label=sample["actual_label"])
    assert actual == expected
    assert main(["predict", "--model-id", model_id, "--sample-id", "test:999999"]) == 1
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "SAMPLE_NOT_FOUND"


def test_health_requires_ready_model_for_current_dataset(dataset_store, model_registry, tmp_path):
    services = {"store": dataset_store, "registry": model_registry}
    with TestClient(create_app(Settings(root=tmp_path), services=services)) as client:
        assert client.get("/api/health").json()["models_ready"] is True

    class OtherStore:
        def info(self):
            return {"ready": True, "dataset_id": "b" * 64}

    services["store"] = OtherStore()
    with TestClient(create_app(Settings(root=tmp_path), services=services)) as client:
        assert client.get("/api/health").json()["models_ready"] is False
    services["store"] = dataset_store
    for item in model_registry.list():
        (model_registry.model_dir / item["model_id"] / "model.joblib").unlink()
    with TestClient(create_app(Settings(root=tmp_path), services=services)) as client:
        assert client.get("/api/health").json()["models_ready"] is False


def test_cli_missing_data_and_model_are_localized(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(Settings, "default", classmethod(lambda cls: Settings(root=tmp_path)))
    assert main(["train", "--profile", "full"]) == 1
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "DATA_UNAVAILABLE"
    assert main(["predict", "--model-id", "model-" + "a" * 32,
                 "--sample-id", "test:000001"]) == 1
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] in {"MODEL_NOT_FOUND", "MODEL_UNAVAILABLE"}


def test_default_registry_readiness_and_missing_dataset(dataset_store, model_registry, tmp_path):
    settings = Settings(root=tmp_path / "trạng thái sản phẩm")
    shutil.copytree(dataset_store.data_dir, settings.data_dir)
    shutil.copytree(model_registry.model_dir, settings.model_dir)
    with TestClient(create_app(settings)) as client:
        health = client.get("/api/health").json()
    assert health["data_ready"] is True and health["models_ready"] is True
    (settings.data_dir / "current.json").unlink()
    with TestClient(create_app(settings)) as client:
        health = client.get("/api/health").json()
    assert health["data_ready"] is False and health["models_ready"] is False


def test_falsey_registry_override_and_registry_failure(dataset_store, tmp_path):
    class FalseyRegistry:
        def __bool__(self):
            return False

        def recover_staging(self):
            # This health-only double owns no publication staging to recover.
            pass

        def has_ready(self, dataset_id):
            assert dataset_id == dataset_store.info()["dataset_id"]
            return True

        def list(self):
            raise AssertionError("Health must use lazy dataset-filtered readiness, not full listing")

    services = {"store": dataset_store, "registry": FalseyRegistry()}
    with TestClient(create_app(Settings(root=tmp_path), services=services)) as client:
        assert client.get("/api/health").json()["models_ready"] is True

    class BrokenRegistry:
        def recover_staging(self):
            # Startup succeeds; only the readiness query simulates storage failure.
            pass

        def has_ready(self, dataset_id):
            raise OSError("unreadable directory")

        def list(self):
            raise AssertionError("Health must use the injected readiness query")

    services["registry"] = BrokenRegistry()
    with TestClient(create_app(Settings(root=tmp_path), services=services)) as client:
        assert client.get("/api/health").json()["models_ready"] is False
