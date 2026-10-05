import pytest
from fastapi.testclient import TestClient

from motionsense_app.data.importer import load_uci, publish_dataset
from motionsense_app.errors import DomainError
from motionsense_app.main import create_app
from motionsense_app.settings import Settings


def test_health_without_setup(tmp_path):
    with TestClient(create_app(Settings(root=tmp_path))) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "app": "motionsense",
        "status": "ok",
        "data_ready": False,
        "models_ready": False,
    }


def test_health_uses_injected_dataset_store_readiness(tmp_path):
    class ReadyStore:
        def info(self):
            return {"ready": True}

    with TestClient(create_app(Settings(root=tmp_path), services={"store": ReadyStore()})) as client:
        response = client.get("/api/health")

    assert response.json()["data_ready"] is True


def test_unknown_api_route_returns_api_error_with_localized_root(tmp_path):
    root = tmp_path / "MotionSense dữ liệu người dùng"
    root.mkdir()

    with TestClient(create_app(Settings(root=root))) as client:
        response = client.get("/api/khong-ton-tai")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "NOT_FOUND",
            "message": "Không tìm thấy tài nguyên.",
            "details": [],
        }
    }


def test_validation_error_uses_vietnamese_api_error(tmp_path):
    app = create_app(Settings(root=tmp_path))

    @app.get("/api/validation-test/{item_id}")
    def validation_test(item_id: int):
        return {"item_id": item_id}

    with TestClient(app) as client:
        response = client.get("/api/validation-test/not-an-integer")

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert body["error"]["message"] == "Dữ liệu yêu cầu không hợp lệ."
    assert isinstance(body["error"]["details"], list)


def test_health_and_data_metadata_use_real_store(uci_root, tmp_path):
    settings = Settings(root=tmp_path)
    dataset_id = publish_dataset(load_uci(uci_root), settings.data_dir, "a" * 64)
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/health").json()["data_ready"] is True
        response = client.get("/api/data")
    assert response.status_code == 200
    info = response.json()
    assert info["dataset_id"] == dataset_id
    assert info["split_counts"] == {"train": 48, "test": 48}
    assert sum(activity["count"] for activity in info["activities"]) == 96
    assert "X" not in info


def test_corrupt_manifest_keeps_api_running_with_unready_metadata(uci_root, tmp_path):
    settings = Settings(root=tmp_path)
    dataset_id = publish_dataset(load_uci(uci_root), settings.data_dir, "a" * 64)
    (settings.data_dir / dataset_id / "manifest.json").write_text("[]", encoding="utf-8")
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/health").json()["data_ready"] is False
        response = client.get("/api/data")
    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert response.json()["split_counts"] == {"train": 0, "test": 0}


def test_falsey_service_override_is_preserved(tmp_path):
    class ReadyStore:
        def __bool__(self):
            return False

        def info(self):
            return {"ready": True}

    with TestClient(create_app(Settings(root=tmp_path), services={"store": ReadyStore()})) as client:
        assert client.get("/api/health").json()["data_ready"] is True


def test_unavailable_injected_store_keeps_health_available(tmp_path):
    class UnavailableStore:
        def info(self):
            raise DomainError("DATA_UNAVAILABLE", "Dữ liệu chưa sẵn sàng.", status=503)

    with TestClient(create_app(Settings(root=tmp_path), services={"store": UnavailableStore()})) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["data_ready"] is False


def test_lzma_corrupt_snapshot_keeps_startup_and_api_available(lzma_corrupt_store):
    settings = Settings(root=lzma_corrupt_store.parents[1])
    with TestClient(create_app(settings)) as client:
        health = client.get("/api/health")
        metadata = client.get("/api/data")
    assert health.status_code == metadata.status_code == 200
    assert health.json()["data_ready"] is False
    assert metadata.json()["ready"] is False


@pytest.mark.parametrize("state", ["missing", "corrupt"])
def test_unavailable_metadata_has_zero_feature_count(uci_root, tmp_path, state):
    settings = Settings(root=tmp_path)
    if state == "corrupt":
        dataset_id = publish_dataset(load_uci(uci_root), settings.data_dir, "a" * 64)
        (settings.data_dir / dataset_id / "manifest.json").write_text("{}", encoding="utf-8")
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/data")
    assert response.status_code == 200
    info = response.json()
    assert info["ready"] is False
    assert info["feature_count"] == 0
    assert info["split_counts"] == {"train": 0, "test": 0}
