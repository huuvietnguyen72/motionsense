import copy
import csv
import io

import pytest
from fastapi.testclient import TestClient

from motionsense_app.data.importer import load_uci, publish_dataset
from motionsense_app.data.store import DatasetStore
from motionsense_app.main import create_app
from motionsense_app.models.prediction import predict_rows
from motionsense_app.models.registry import ModelRegistry
from motionsense_app.settings import Settings


@pytest.fixture
def data_client(dataset_store, tmp_path):
    with TestClient(create_app(Settings(root=tmp_path), services={"store": dataset_store})) as client:
        yield client


def upload(client, model_id, content):
    return client.post("/api/recognition/csv", data={"model_id": model_id},
                       files={"file": ("sample.csv", content, "text/csv")})


def csv_bytes(header, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def assert_error(response, status, code):
    assert response.status_code == status
    error = response.json()["error"]
    assert error["code"] == code
    assert error["message"]
    assert isinstance(error["details"], list)
    assert all(set(item) == {"row", "column", "message"} for item in error["details"])
    return error


def test_features_and_paginated_summaries_do_not_expose_matrix(data_client):
    features = data_client.get("/api/data/features")
    assert features.status_code == 200
    items = features.json()["items"]
    assert len(items) == 561
    assert items[0] == {"feature_id": "f001", "name": "duplicate-name"}
    assert items[-1]["feature_id"] == "f561"
    response = data_client.get("/api/data/samples", params={
        "split": "test", "subject_id": 2, "offset": 1, "limit": 2,
    })
    assert response.status_code == 200
    assert response.json() == {"items": [
        {"sample_id": "test:000002", "subject_id": 2, "split": "test", "actual_label": 2},
        {"sample_id": "test:000003", "subject_id": 2, "split": "test", "actual_label": 3},
    ], "offset": 1, "limit": 2, "total": 48}
    default = data_client.get("/api/data/samples").json()
    assert default["offset"] == 0 and default["limit"] == 50 and default["total"] == 48
    train = data_client.get("/api/data/samples?split=train&limit=200").json()
    assert train["items"][0]["sample_id"] == "train:000001"
    assert len(train["items"]) == 48 and train["limit"] == 200
    assert data_client.get("/api/data/samples?subject_id=999").json()["total"] == 0
    beyond = data_client.get("/api/data/samples?offset=100").json()
    assert beyond["items"] == [] and beyond["total"] == 48


@pytest.mark.parametrize("query,column", [
    ("split=validation", "split"), ("offset=-1", "offset"), ("limit=0", "limit"),
    ("limit=201", "limit"), ("subject_id=0", "subject_id"),
    ("subject_id=1.5", "subject_id"),
])
def test_invalid_sample_filters_use_api_errors(data_client, query, column):
    error = assert_error(data_client.get(f"/api/data/samples?{query}"), 422, "VALIDATION_ERROR")
    assert error["details"][0]["column"] == column


def test_sample_detail_contains_real_identity_features_and_signal(data_client, dataset_store):
    response = data_client.get("/api/data/samples/test:000001")
    assert response.status_code == 200
    sample = response.json()
    assert sample == dataset_store.sample("test:000001")
    assert sample["actual_label"] == 1 and sample["subject_id"] == 2
    assert len(sample["features"]) == 561 and sample["signal"]["unit"] == "g"
    assert len(sample["signal"]["x"]) == 128 and sample["signal"]["time_seconds"][-1] == 2.54


@pytest.mark.parametrize("sample_id", ["validation:000001", "test:000000", "test:000049", "bad"])
def test_unknown_sample_is_404(data_client, sample_id):
    assert_error(data_client.get(f"/api/data/samples/{sample_id}"), 404, "SAMPLE_NOT_FOUND")


def test_sample_prediction_preserves_identity_and_real_probabilities(
    api_client, trained_model_id, dataset_store, model_registry,
):
    assert api_client.get("/api/health").json() == {
        "app": "motionsense", "status": "ok", "data_ready": True, "models_ready": True,
    }
    response = api_client.post("/api/recognition/sample", json={
        "model_id": trained_model_id, "sample_id": "test:000001",
    })
    assert response.status_code == 200
    expected = predict_rows(model_registry.load(trained_model_id),
                            [dataset_store.sample("test:000001")["features"]])[0]
    assert response.json() == dict(expected, sample_id="test:000001", actual_label=1)
    assert len(response.json()["probabilities"]) == 6
    assert_error(api_client.post("/api/recognition/sample", json={
        "model_id": trained_model_id, "sample_id": "test:999999",
    }), 404, "SAMPLE_NOT_FOUND")


def test_model_list_and_report_follow_master_contract(api_client, model_registry, trained_model_id):
    listing = api_client.get("/api/models")
    assert listing.status_code == 200
    assert listing.json() == {"items": model_registry.list()}
    assert {item["profile"] for item in listing.json()["items"]} == {"full", "reduced"}
    response = api_client.get(f"/api/models/{trained_model_id}/report")
    assert response.status_code == 200
    assert response.json() == model_registry.report(trained_model_id)
    assert response.json()["labels"] == [1, 2, 3, 4, 5, 6]
    assert response.json()["test_count"] == 48
    assert response.json()["split_strategy"] == "uci_subject_split"


def test_bom_template_roundtrip_and_full_template_accepted_by_reduced(api_client, model_registry):
    models = {item["profile"]: item["model_id"] for item in model_registry.list()}
    for profile, model_id in models.items():
        response = api_client.get("/api/recognition/template", params={"model_id": model_id})
        assert response.status_code == 200
        assert response.content.startswith(b"\xef\xbb\xbf")
        assert response.headers["content-type"].startswith("text/csv")
        assert "attachment" in response.headers["content-disposition"]
        records = list(csv.reader(io.StringIO(response.content.decode("utf-8-sig"))))
        assert len(records) == 2
        assert records[0] == model_registry.load(model_id)["manifest"]["feature_ids"] + ["activity"]
        expected = api_client.post("/api/recognition/sample", json={
            "model_id": model_id, "sample_id": "test:000001",
        }).json()
        actual = upload(api_client, model_id, response.content)
        assert actual.status_code == 200
        assert actual.json() == {"items": [dict(expected, sample_id=None)], "row_count": 1}
        if profile == "full":
            full_template = response.content
    reduced_sample = api_client.post("/api/recognition/sample", json={
        "model_id": models["reduced"], "sample_id": "test:000001",
    }).json()
    assert upload(api_client, models["reduced"], full_template).json() == {
        "items": [dict(reduced_sample, sample_id=None)], "row_count": 1,
    }


def test_csv_uses_manifest_names_and_never_inputs_metadata(api_client, model_registry, trained_model_id):
    artifact = model_registry.load(trained_model_id)
    ids = artifact["manifest"]["feature_ids"]
    row = dict(zip(ids, [0.1, 0.2, 0.3, 0.4, 0.5]))
    expected = predict_rows(artifact, [row])[0]
    header = list(reversed(ids)) + ["activity", "subject_id"]
    response = upload(api_client, trained_model_id, csv_bytes(header, [
        [row[fid] for fid in reversed(ids)] + [1, 2],
        [row[fid] for fid in reversed(ids)] + [6, 999],
    ]))
    assert response.status_code == 200
    assert response.json() == {"items": [dict(expected, actual_label=1),
                                         dict(expected, actual_label=6)], "row_count": 2}
    unlabeled = upload(api_client, trained_model_id, csv_bytes(ids, [[row[fid] for fid in ids]]))
    assert unlabeled.json() == {"items": [expected], "row_count": 1}


def test_csv_api_rejects_whole_file_with_precise_errors_and_limits(api_client, trained_model_id):
    template = api_client.get("/api/recognition/template", params={"model_id": trained_model_id})
    records = list(csv.reader(io.StringIO(template.content.decode("utf-8-sig"))))
    header, values = records
    broken = list(values)
    broken[0] = "1e300"
    error = assert_error(upload(api_client, trained_model_id,
                                csv_bytes(header, [values, broken])), 422, "CSV_INVALID")
    assert error["details"][0]["row"] == 3
    assert error["details"][0]["column"] == header[0]
    assert_error(upload(api_client, trained_model_id, b" " * (16 * 1024 * 1024 + 1)),
                 413, "CSV_TOO_LARGE")
    error = assert_error(upload(api_client, trained_model_id,
                                csv_bytes(header, [values] * 5001)), 413, "CSV_TOO_MANY_ROWS")
    assert error["details"][0]["row"] == 5002
    extra = next(f"f{i:03d}" for i in range(1, 562) if f"f{i:03d}" not in header)
    error = assert_error(upload(api_client, trained_model_id,
                                csv_bytes(header + [extra], [values + ["NaN"]])), 422, "CSV_INVALID")
    assert error["details"][0]["column"] == extra
    missing = api_client.post("/api/recognition/csv", data={"model_id": trained_model_id})
    assert assert_error(missing, 422, "VALIDATION_ERROR")["details"][0]["column"] == "file"


def test_no_setup_keeps_metadata_available_and_distinguishes_unknown_resources(tmp_path):
    with TestClient(create_app(Settings(root=tmp_path))) as client:
        assert client.get("/api/data").json()["ready"] is False
        assert client.get("/api/models").json() == {"items": []}
        for path in ("/api/data/features", "/api/data/samples", "/api/data/samples/test:000001"):
            assert_error(client.get(path), 503, "DATA_UNAVAILABLE")
        assert_error(client.get("/api/models/missing/report"), 404, "MODEL_NOT_FOUND")
        assert_error(client.get("/api/recognition/template?model_id=missing"), 404, "MODEL_NOT_FOUND")
        assert_error(client.post("/api/recognition/sample", json={
            "model_id": "missing", "sample_id": "test:000001",
        }), 404, "MODEL_NOT_FOUND")
        assert_error(upload(client, "missing", b"f001\n1\n"), 404, "MODEL_NOT_FOUND")


def test_external_csv_survives_missing_or_incompatible_dataset(
    model_registry, trained_model_id, uci_root, tmp_path,
):
    artifact = model_registry.load(trained_model_id)
    ids = artifact["manifest"]["feature_ids"]
    row = artifact["verification_rows"][0]
    content = csv_bytes(ids, [[row[fid] for fid in ids]])
    missing = DatasetStore(tmp_path / "missing-dataset")
    publish_dataset(load_uci(uci_root), tmp_path / "other-dataset", "b" * 64)
    incompatible = DatasetStore(tmp_path / "other-dataset")
    for store, code in [(missing, "DATA_UNAVAILABLE"), (incompatible, "DATASET_MISMATCH")]:
        with TestClient(create_app(Settings(root=tmp_path), services={
            "store": store, "registry": model_registry,
        })) as client:
            assert client.get("/api/health").json()["models_ready"] is False
            assert_error(client.post("/api/recognition/sample", json={
                "model_id": trained_model_id, "sample_id": "test:000001",
            }), 503, code)
            assert_error(client.get("/api/recognition/template", params={
                "model_id": trained_model_id,
            }), 503, code)
            response = upload(client, trained_model_id, content)
            assert response.status_code == 200
            assert response.json() == {"items": predict_rows(artifact, [row]), "row_count": 1}
            assert client.get(f"/api/models/{trained_model_id}/report").status_code == 200


def test_unavailable_stored_model_returns_503_not_unknown_404(
    dataset_store, model_registry, trained_model_id, tmp_path,
):
    registry = ModelRegistry(tmp_path / "broken-models")
    registry.publish(copy.deepcopy(model_registry.load(trained_model_id)))
    (registry.model_dir / trained_model_id / "report.json").unlink()
    with TestClient(create_app(Settings(root=tmp_path), services={
        "store": dataset_store, "registry": registry,
    })) as client:
        assert client.get("/api/models").json()["items"][0]["status"] == "unavailable"
        assert client.get("/api/health").json()["models_ready"] is False
        assert_error(client.get(f"/api/models/{trained_model_id}/report"), 503, "MODEL_UNAVAILABLE")
        assert_error(client.get("/api/recognition/template", params={"model_id": trained_model_id}),
                     503, "MODEL_UNAVAILABLE")
        assert_error(client.post("/api/recognition/sample", json={
            "model_id": trained_model_id, "sample_id": "test:000001",
        }), 503, "MODEL_UNAVAILABLE")
        assert_error(upload(client, trained_model_id, b"f001\n1\n"), 503, "MODEL_UNAVAILABLE")
