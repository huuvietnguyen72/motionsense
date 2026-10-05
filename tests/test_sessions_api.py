import csv
import io
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from motionsense_app.main import create_app
from motionsense_app.sessions.engine import SessionEngine
from motionsense_app.sessions.repository import SessionRepository
from motionsense_app.settings import Settings


def create_request(dataset_store, model_id):
    return {"dataset_id": dataset_store.info()["dataset_id"], "split": "test",
            "subject_id": 2, "model_id": model_id}


def test_create_persists_ordered_snapshot_and_exposes_session_page(
    api_client, dataset_store, trained_model_id,
):
    response = api_client.post("/api/sessions", json=create_request(dataset_store, trained_model_id))
    assert response.status_code == 201
    session = response.json()
    sid = session["session_id"]
    assert session == {
        "session_id": sid, **create_request(dataset_store, trained_model_id),
        "status": "ready", "cursor": 0, "total": 48, "speed": 1.0,
        "created_at": session["created_at"], "updated_at": session["updated_at"],
        "finish_reason": None, "last_prediction": None,
    }
    assert datetime.fromisoformat(session["created_at"]).utcoffset().total_seconds() == 0
    assert api_client.get(f"/api/sessions/{sid}").json() == session
    assert api_client.get("/api/sessions?offset=0&limit=1").json() == {
        "items": [session], "offset": 0, "limit": 1, "total": 1,
    }
    with closing(sqlite3.connect(api_client.app.state.settings.db_path)) as db, db:
        saved = db.execute("SELECT sample_ids_json, client_id FROM sessions").fetchone()
        assert json.loads(saved[0]) == [f"test:{i:06d}" for i in range(1, 49)]
        assert saved[1] is None
        assert db.execute("PRAGMA user_version").fetchone()[0] == 2
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


@pytest.mark.parametrize("changes,status,code", [
    ({"dataset_id": "b" * 64}, 503, "DATASET_MISMATCH"),
    ({"subject_id": 1}, 422, "SESSION_EMPTY"),
    ({"model_id": "missing"}, 404, "MODEL_NOT_FOUND"),
    ({"split": "other"}, 422, "VALIDATION_ERROR"),
    ({"subject_id": 0}, 422, "VALIDATION_ERROR"),
])
def test_create_rejects_unusable_snapshot(
    api_client, dataset_store, trained_model_id, changes, status, code,
):
    response = api_client.post("/api/sessions", json={
        **create_request(dataset_store, trained_model_id), **changes,
    })
    assert response.status_code == status
    assert response.json()["error"]["code"] == code


def test_unknown_session_is_localized_404(api_client):
    response = api_client.get("/api/sessions/missing")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "SESSION_NOT_FOUND"


def create_attached(api_client, dataset_store, model_id):
    sid = api_client.post("/api/sessions", json=create_request(dataset_store, model_id)).json()["session_id"]
    response = api_client.post(f"/api/sessions/{sid}/attach", json={"client_id": "a"})
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    return sid


def control(api_client, sid, action, **values):
    return api_client.post(f"/api/sessions/{sid}/control", json={
        "client_id": "a", "action": action, **values,
    })


def advance(api_client, sid, cursor, client_id="a"):
    return api_client.post(f"/api/sessions/{sid}/advance", json={
        "client_id": client_id, "expected_cursor": cursor,
    })


def test_playback_routes_return_committed_results_summary_and_export(
    api_client, session_engine, dataset_store, trained_model_id,
):
    _, repo, clock = session_engine
    sid = create_attached(api_client, dataset_store, trained_model_id)
    assert control(api_client, sid, "start").json()["status"] == "running"
    assert advance(api_client, sid, 0).json()["row"] is None
    clock[0] = 1
    first = advance(api_client, sid, 0)
    assert first.status_code == 200
    assert first.json()["session"]["cursor"] == 1
    assert advance(api_client, sid, 0).json() == first.json()
    assert control(api_client, sid, "pause").json()["status"] == "paused"
    clock[0] = 30
    assert advance(api_client, sid, 1).json()["row"] is None
    assert control(api_client, sid, "speed", speed=2).json()["speed"] == 2
    assert control(api_client, sid, "resume").json()["status"] == "running"
    clock[0] = 30.5
    assert advance(api_client, sid, 1).json()["session"]["cursor"] == 2
    assert control(api_client, sid, "finish").json()["finish_reason"] == "user"
    assert api_client.get(f"/api/sessions/{sid}/rows?offset=1&limit=1").json() == repo.rows(sid, 1, 1)
    assert api_client.get(f"/api/sessions/{sid}/summary").json() == repo.summary(sid)
    assert repo.summary(sid)["processed"] == 2
    response = api_client.get(f"/api/sessions/{sid}/export")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    assert response.content.startswith(b"\xef\xbb\xbf")
    assert len(list(csv.DictReader(io.StringIO(response.content.decode("utf-8-sig"))))) == 2
    assert api_client.get(f"/api/sessions/{sid}").json() == repo.get(sid)


def test_api_threaded_retry_and_future_cursor_conflict(
    api_client, session_engine, dataset_store, trained_model_id,
):
    _, repo, clock = session_engine
    sid = create_attached(api_client, dataset_store, trained_model_id)
    control(api_client, sid, "start")
    clock[0] = 1
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: advance(api_client, sid, 0), range(2)))
    assert all(response.status_code == 200 for response in responses)
    assert responses[0].json() == responses[1].json()
    response = advance(api_client, sid, 2)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "CURSOR_CONFLICT"
    assert repo.rows(sid, 0, 50)["total"] == 1


def test_api_new_tab_pauses_and_invalidates_old_requests(
    api_client, dataset_store, trained_model_id,
):
    sid = create_attached(api_client, dataset_store, trained_model_id)
    control(api_client, sid, "start")
    response = api_client.post(f"/api/sessions/{sid}/attach", json={"client_id": "b"})
    assert response.status_code == 200
    assert response.json()["status"] == "paused"
    for response in (control(api_client, sid, "resume"), advance(api_client, sid, 0)):
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "CLIENT_CONFLICT"


@pytest.mark.parametrize("path,body", [
    ("attach", {"client_id": ""}),
    ("attach", {"client_id": " "}),
    ("control", {"client_id": "a", "action": "speed", "speed": 0}),
    ("control", {"client_id": "a", "action": "speed", "speed": True}),
    ("control", {"client_id": "a", "action": "speed"}),
    ("advance", {"client_id": "a", "expected_cursor": -1}),
    ("advance", {"client_id": "a", "expected_cursor": True}),
    ("advance", {"client_id": "a", "expected_cursor": "0"}),
])
def test_api_invalid_playback_input_returns_422(
    api_client, dataset_store, trained_model_id, path, body,
):
    sid = create_attached(api_client, dataset_store, trained_model_id)
    response = api_client.post(f"/api/sessions/{sid}/{path}", json=body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.parametrize("suffix", ["", "/rows", "/summary", "/export"])
def test_unknown_session_read_endpoints_return_domain_404(api_client, suffix):
    response = api_client.get(f"/api/sessions/missing{suffix}")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "SESSION_NOT_FOUND"


@pytest.mark.parametrize("endpoint", ["/api/sessions", "/api/sessions/missing/rows"])
@pytest.mark.parametrize("query", ["offset=-1", "limit=0", "limit=201"])
def test_pagination_limits_are_validated(api_client, endpoint, query):
    response = api_client.get(f"{endpoint}?{query}")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_storage_write_error_is_localized_and_does_not_advance(
    api_client, session_engine, dataset_store, trained_model_id,
):
    _, repo, clock = session_engine
    sid = create_attached(api_client, dataset_store, trained_model_id)
    control(api_client, sid, "start")
    with closing(sqlite3.connect(repo.db_path)) as db, db:
        db.execute("""CREATE TRIGGER fail_cursor BEFORE UPDATE OF cursor ON sessions
                      BEGIN SELECT RAISE(ABORT,'private disk diagnostic'); END""")
    clock[0] = 1
    response = advance(api_client, sid, 0)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "STORAGE_ERROR"
    assert "private disk diagnostic" not in response.text
    assert repo.get(sid)["cursor"] == 0
    assert repo.get(sid)["status"] == "paused"
    assert repo.rows(sid, 0, 50)["total"] == 0


def test_missing_dataset_keeps_health_and_session_history_available(tmp_path):
    with TestClient(create_app(Settings(root=tmp_path))) as client:
        response = client.post("/api/sessions", json={
            "dataset_id": "a" * 64, "split": "test", "subject_id": 2, "model_id": "missing",
        })
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "DATA_UNAVAILABLE"
        assert client.get("/api/sessions").json()["total"] == 0
        assert client.get("/api/health").status_code == 200


def test_startup_recovers_durable_running_session_and_keeps_injected_engine(
    session_engine, dataset_store, trained_model_id, model_registry, tmp_path,
):
    engine, repo, clock = session_engine
    sid = engine.create(create_request(dataset_store, trained_model_id))["session_id"]
    engine.attach(sid, "old")
    engine.control(sid, "old", "start")
    clock[0] = 1
    saved_row = engine.advance(sid, "old", 0)["row"]
    app = create_app(Settings(root=tmp_path), services={
        "store": dataset_store, "registry": model_registry, "session_engine": engine,
    })
    with TestClient(app) as client:
        assert app.state.session_engine is engine
        session = client.get(f"/api/sessions/{sid}").json()
        assert session["status"] == "paused"
        assert session["cursor"] == 1
        assert advance(client, sid, 1, "old").status_code == 409
        assert client.get(f"/api/sessions/{sid}/rows").json()["items"] == [saved_row]
        client.post(f"/api/sessions/{sid}/attach", json={"client_id": "new"})
        response = client.post(f"/api/sessions/{sid}/control", json={"client_id": "new", "action": "resume"})
        assert response.json()["status"] == "running"
        # Recovery happens once, not on subsequent reads.
        assert client.get(f"/api/sessions/{sid}").json()["status"] == "running"
    assert repo.get(sid)["status"] == "paused"
    assert repo.get(sid)["cursor"] == 1


def test_default_engine_startup_recovers_existing_db_without_dependencies(
    session_engine, dataset_store, trained_model_id, tmp_path,
):
    engine, repo, _ = session_engine
    sid = engine.create(create_request(dataset_store, trained_model_id))["session_id"]
    engine.attach(sid, "a")
    engine.control(sid, "a", "start")
    with TestClient(create_app(Settings(root=tmp_path))) as client:
        assert client.get(f"/api/sessions/{sid}").json()["status"] == "paused"
        assert client.get(f"/api/sessions/{sid}/summary").json()["processed"] == 0
        assert client.get(f"/api/sessions/{sid}/export").status_code == 200
        client.post(f"/api/sessions/{sid}/attach", json={"client_id": "a"})
        assert control(client, sid, "resume").status_code == 503
    assert SessionRepository(repo.db_path).get(sid)["status"] == "paused"


def test_lifespan_watchdog_expires_injected_clock_without_browser_request(
    api_client, session_engine, dataset_store, trained_model_id,
):
    _, repo, clock = session_engine
    sid = create_attached(api_client, dataset_store, trained_model_id)
    control(api_client, sid, "start")
    clock[0] = 3
    deadline = time.monotonic() + 2
    while repo.get(sid)["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert repo.get(sid)["status"] == "paused"
    assert repo.rows(sid, 0, 50)["total"] == 0


def test_watchdog_waiting_on_engine_does_not_block_async_event_loop(
    session_engine, dataset_store, model_registry, trained_model_id, tmp_path,
):
    engine, _, _ = session_engine
    entered = threading.Event()
    release = threading.Event()

    class BlockingWatchdogEngine(SessionEngine):
        def expire_leases(self):
            entered.set()
            if not release.wait(timeout=5):
                raise AssertionError("watchdog did not leave the event loop responsive")
            super().expire_leases()

    blocking = BlockingWatchdogEngine(engine.repo, dataset_store, model_registry, engine.clock)
    app = create_app(Settings(root=tmp_path), services={
        "store": dataset_store, "registry": model_registry, "session_engine": blocking,
    })

    @app.get("/api/event-loop-probe")
    async def probe():
        return {"responsive": True}

    with TestClient(app) as client:
        assert entered.wait(timeout=2)
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(client.get, "/api/event-loop-probe")
                try:
                    response = future.result(timeout=1)
                finally:
                    release.set()
            assert response.status_code == 200
            assert response.json() == {"responsive": True}
        finally:
            release.set()


@pytest.mark.parametrize("action", ["resume", "advance"])
@pytest.mark.parametrize("kind", [
    "published", "missing_pointer", "invalid_json", "wrong_shape", "missing_id",
    "invalid_id", "wrong_type", "invalid_utf8", "duplicate_id",
    "missing:manifest.json", "missing:features.json", "missing:train.npz", "missing:test.npz",
    "corrupt:manifest.json", "corrupt:features.json", "corrupt:train.npz", "corrupt:test.npz",
])
def test_disk_source_change_blocks_playback_but_preserves_history(
    api_client, session_engine, dataset_store, trained_model_id, change_session_source,
    kind, action,
):
    engine, repo, clock = session_engine
    sid = create_attached(api_client, dataset_store, trained_model_id)
    assert control(api_client, sid, "start").status_code == 200
    clock[0] = 1
    first = advance(api_client, sid, 0).json()["row"]
    if action == "resume":
        assert control(api_client, sid, "pause").status_code == 200
    summary = repo.summary(sid)
    exported = api_client.get(f"/api/sessions/{sid}/export").content
    pinned_id = repo.get(sid)["dataset_id"]
    change_session_source(dataset_store, kind)
    assert engine.store is dataset_store
    assert dataset_store.info()["dataset_id"] == pinned_id  # cached info masks the disk change
    clock[0] = 2
    response = (control(api_client, sid, "resume") if action == "resume"
                else advance(api_client, sid, 1))
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == ("DATASET_MISMATCH" if kind == "published" else "DATA_UNAVAILABLE")
    assert error["message"] and isinstance(error["details"], list)
    assert repo.get(sid)["status"] == "paused"
    assert repo.get(sid)["cursor"] == 1
    assert api_client.get(f"/api/sessions/{sid}").json()["last_prediction"] == first["prediction"]
    assert api_client.get(f"/api/sessions/{sid}/rows").json()["items"] == [first]
    assert api_client.get(f"/api/sessions/{sid}/summary").json() == summary
    assert api_client.get(f"/api/sessions/{sid}/export").content == exported
    assert advance(api_client, sid, 0).json()["row"] == first
    assert advance(api_client, sid, 1).json()["row"] is None


@pytest.mark.parametrize("case", [
    "start", "start-repeat", "resume", "resume-repeat", "pause", "pause-repeat",
    "finish", "finish-repeat",
])
def test_api_optional_speed_rejected_before_control_or_noop(
    api_client, session_engine, dataset_store, trained_model_id, case,
):
    _, repo, clock = session_engine
    sid = create_attached(api_client, dataset_store, trained_model_id)
    if case != "start":
        assert control(api_client, sid, "start").status_code == 200
    if case in {"resume", "pause-repeat"}:
        assert control(api_client, sid, "pause").status_code == 200
    if case == "finish-repeat":
        assert control(api_client, sid, "finish").status_code == 200
    before = repo.get(sid)
    clock[0] = 0.25
    action = case.split("-", 1)[0]
    for speed in (0, 3, 1.5, 1.0000000000000002, 0.49999999999999994,
                  True, False, "1", None):
        response = control(api_client, sid, action, speed=speed)
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["code"] == "VALIDATION_ERROR"
        assert error["message"] and isinstance(error["details"], list)
        assert repo.get(sid) == before
    clock[0] = 0.99
    assert advance(api_client, sid, 0).json()["row"] is None
    clock[0] = 1
    result = advance(api_client, sid, 0).json()
    if before["status"] == "running":
        assert result["session"]["cursor"] == 1
    else:
        assert result["row"] is None
        assert repo.get(sid) == before
    # Omission remains valid for non-speed actions, including repeated controls.
    response = control(api_client, sid, action)
    assert response.status_code == 200
    assert response.json()["status"] == {
        "start": "running", "resume": "running", "pause": "paused", "finish": "finished",
    }[action]


def test_api_supported_optional_speeds_and_invalid_actions_preserve_control_contract(
    api_client, dataset_store, trained_model_id,
):
    sid = create_attached(api_client, dataset_store, trained_model_id)
    for action, speed, status in (("start", 0.5, "running"), ("pause", 2, "paused"),
                                 ("resume", 1.0, "running"), ("finish", 2.0, "finished")):
        response = control(api_client, sid, action, speed=speed)
        assert response.status_code == 200
        assert response.json()["status"] == status
        assert response.json()["speed"] == 1.0  # only the speed action changes stored speed
    response = control(api_client, sid, "start", speed=1.0)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "SESSION_STATE_CONFLICT"
    response = control(api_client, sid, "unknown", speed=0.5)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "SESSION_STATE_CONFLICT"
