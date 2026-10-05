import json
import sqlite3
import subprocess
import sys
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pytest

from motionsense_app.errors import DomainError


def test_failed_training_preserves_ready_models(
    dataset_store, model_registry, tmp_path, caplog,
):
    from motionsense_app.models.jobs import TrainingJobs

    original = {item["model_id"] for item in model_registry.list()}
    entered = threading.Event()
    release = threading.Event()

    def fail(store, profile, trees, on_stage):
        assert store is dataset_store
        assert (profile, trees) == ("full", 50)
        on_stage("fitting")
        entered.set()
        assert release.wait(5)
        raise ValueError("private training diagnostic")

    jobs = TrainingJobs(tmp_path / "jobs.sqlite3", dataset_store, model_registry, fail)
    try:
        job = jobs.submit("full", 50)
        assert entered.wait(5)
        assert jobs.get(job["job_id"])["stage"] == "fitting"
    finally:
        release.set()
        jobs.close()
    saved = jobs.get(job["job_id"])
    assert saved["status"] == "failed"
    assert saved["model_id"] is None
    assert saved["error"]["error"]["code"] == "TRAINING_FAILED"
    assert saved["error"]["error"]["message"]
    assert saved["error"]["error"]["details"] == []
    assert "private training diagnostic" not in json.dumps(saved)
    assert "private training diagnostic" in caplog.text
    assert {item["model_id"] for item in model_registry.list()} == original
    reopened = TrainingJobs(tmp_path / "jobs.sqlite3", dataset_store, model_registry, fail)
    try:
        reopened.recover()
        assert reopened.get(job["job_id"]) == saved
    finally:
        reopened.close()


def test_real_stages_publish_before_success_and_persist_identity(
    dataset_store, model_registry, tmp_path, job_worker,
):
    from motionsense_app.models.jobs import TrainingJobs

    trainer, entered, release, artifact = job_worker
    publishing = threading.Event()
    publish_release = threading.Event()
    original_publish = model_registry.publish

    def publish(result):
        publishing.set()
        assert publish_release.wait(5)
        return original_publish(result)

    model_registry.publish = publish
    jobs = TrainingJobs(tmp_path / "jobs.sqlite3", dataset_store, model_registry, trainer)
    try:
        job = jobs.submit("reduced", 50)
        assert set(job) == {"job_id", "status", "stage", "model_id", "error"}
        for stage in entered:
            assert entered[stage].wait(5)
            snapshot = jobs.get(job["job_id"])
            assert snapshot["status"] == "running"
            assert snapshot["stage"] == stage
            assert snapshot["model_id"] is None and snapshot["error"] is None
            release[stage].set()
        assert publishing.wait(5)
        assert jobs.get(job["job_id"])["status"] == "running"
        with pytest.raises(DomainError):
            model_registry.load(artifact["manifest"]["model_id"])
    finally:
        for event in release.values():
            event.set()
        publish_release.set()
        jobs.close()
    saved = jobs.get(job["job_id"])
    assert saved == {"job_id": job["job_id"], "status": "succeeded", "stage": "saving",
                     "model_id": artifact["manifest"]["model_id"], "error": None}
    assert model_registry.load(saved["model_id"])["manifest"]["status"] == "ready"
    with closing(sqlite3.connect(jobs.db_path)) as db, db:
        row = db.execute("SELECT dataset_id,profile,trees,created_at,updated_at FROM jobs").fetchone()
    assert row[:3] == (dataset_store.info()["dataset_id"], "reduced", 50)
    assert all(datetime.fromisoformat(value).utcoffset().total_seconds() == 0 for value in row[3:])
    assert row[4] >= row[3]


def test_concurrent_submits_accept_one_active_job_across_services(
    dataset_store, model_registry, tmp_path, job_worker,
):
    from motionsense_app.models.jobs import TrainingJobs

    trainer, entered, release, _ = job_worker
    db_path = tmp_path / "jobs.sqlite3"
    services = [TrainingJobs(db_path, dataset_store, model_registry, trainer) for _ in range(2)]
    barrier = threading.Barrier(3)

    def submit(service):
        barrier.wait(timeout=5)
        try:
            return service.submit("reduced", 50)
        except DomainError as error:
            return error

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(submit, service) for service in services]
            barrier.wait(timeout=5)
            results = [future.result(timeout=5) for future in futures]
        accepted = [result for result in results if isinstance(result, dict)]
        rejected = [result for result in results if isinstance(result, DomainError)]
        assert len(accepted) == len(rejected) == 1
        assert (rejected[0].status, rejected[0].code) == (409, "TRAINING_BUSY")
        assert entered["validating"].wait(5)
        with closing(sqlite3.connect(db_path)) as db, db:
            assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    finally:
        for event in release.values():
            event.set()
        for service in services:
            service.close()


@pytest.mark.parametrize("status", ["queued", "running"])
def test_recovery_interrupts_orphaned_jobs_without_retry(
    dataset_store, model_registry, tmp_path, status,
):
    from motionsense_app.models.jobs import TrainingJobs

    calls = []

    def forbidden(*args):
        calls.append(args)
        raise AssertionError("startup must not train")

    path = tmp_path / "jobs.sqlite3"
    jobs = TrainingJobs(path, dataset_store, model_registry, forbidden)
    jobs.close()
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("""INSERT INTO jobs(job_id,dataset_id,profile,trees,status,stage,
                      created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)""",
                   ("orphan", dataset_store.info()["dataset_id"], "full", 50, status,
                    "fitting" if status == "running" else "queued",
                    "2026-10-02T10:00:00+00:00", "2026-10-02T10:00:00+00:00"))
    reopened = TrainingJobs(path, dataset_store, model_registry, forbidden)
    try:
        reopened.recover()
        saved = reopened.get("orphan")
        assert saved["status"] == "interrupted"
        assert saved["model_id"] is None
        assert saved["error"]["error"]["code"] == "TRAINING_INTERRUPTED"
        assert "huấn luyện lại" in saved["error"]["error"]["message"]
        reopened.recover()
        assert reopened.get("orphan") == saved
        assert calls == []
    finally:
        reopened.close()


@pytest.mark.parametrize("profile,trees", [
    ("both", 50), (None, 50), ([], 50), ("full", 0), ("full", 51),
    ("full", True), ("full", 50.0), ("full", "50"),
])
def test_service_rejects_config_before_persisting(dataset_store, model_registry, tmp_path,
                                                profile, trees):
    from motionsense_app.models.jobs import TrainingJobs

    jobs = TrainingJobs(tmp_path / "jobs.sqlite3", dataset_store, model_registry)
    try:
        with pytest.raises(DomainError) as caught:
            jobs.submit(profile, trees)
        assert (caught.value.status, caught.value.code) == (422, "TRAINING_CONFIG")
        with closing(sqlite3.connect(jobs.db_path)) as db, db:
            assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    finally:
        jobs.close()


@pytest.mark.parametrize("kind", [
    "published", "missing_pointer", "invalid_json", "invalid_utf8", "duplicate_id",
    "missing:train.npz", "corrupt:test.npz",
])
def test_submit_rejects_stale_cached_dataset(dataset_store, model_registry, tmp_path,
                                           change_session_source, kind):
    from motionsense_app.models.jobs import TrainingJobs

    jobs = TrainingJobs(tmp_path / "jobs.sqlite3", dataset_store, model_registry)
    change_session_source(dataset_store, kind)
    try:
        with pytest.raises(DomainError) as caught:
            jobs.submit("full", 50)
        assert caught.value.status == 503
        with closing(sqlite3.connect(jobs.db_path)) as db, db:
            assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    finally:
        jobs.close()


def test_callback_storage_failure_fails_job_without_publishing(
    dataset_store, model_registry, tmp_path, job_worker, caplog,
):
    from motionsense_app.models.jobs import TrainingJobs

    trainer, entered, release, _ = job_worker
    original = {item["model_id"] for item in model_registry.list()}
    jobs = TrainingJobs(tmp_path / "jobs.sqlite3", dataset_store, model_registry, trainer)
    try:
        job = jobs.submit("reduced", 50)
        assert entered["validating"].wait(5)
        with closing(sqlite3.connect(jobs.db_path)) as db, db:
            db.execute("""CREATE TRIGGER fail_stage BEFORE UPDATE OF stage ON jobs
                          BEGIN SELECT RAISE(ABORT,'private callback diagnostic'); END""")
    finally:
        for event in release.values():
            event.set()
        jobs.close()
    saved = jobs.get(job["job_id"])
    assert saved["status"] == "failed" and saved["stage"] == "validating"
    assert saved["model_id"] is None
    assert "private callback diagnostic" not in json.dumps(saved)
    assert "private callback diagnostic" in caplog.text
    assert {item["model_id"] for item in model_registry.list()} == original


def test_publish_failure_keeps_previous_ready_versions(
    dataset_store, model_registry, tmp_path, training_result, caplog,
):
    from motionsense_app.models.jobs import TrainingJobs

    original = {item["model_id"] for item in model_registry.list()}
    # Real atomic publisher rejects inconsistent reports; do not stub away publication.
    training_result["report"]["model_id"] = "model-" + "0" * 32
    jobs = TrainingJobs(tmp_path / "jobs.sqlite3", dataset_store, model_registry,
                        lambda *args: training_result)
    job = jobs.submit("reduced", 50)
    try:
        jobs._future.result(timeout=5)
    finally:
        jobs.close()
    saved = jobs.get(job["job_id"])
    assert saved["status"] == "failed" and saved["model_id"] is None
    assert saved["error"]["error"]["code"] == "TRAINING_FAILED"
    assert {item["model_id"] for item in model_registry.list()} == original
    assert not list(model_registry.model_dir.glob(".staging-*"))
    assert "Training job" in caplog.text


def test_executor_rejection_persists_failure_and_logs_private_diagnostic(
    dataset_store, model_registry, tmp_path, caplog,
):
    from motionsense_app.models.jobs import TrainingJobs

    original = {item["model_id"] for item in model_registry.list()}
    jobs = TrainingJobs(tmp_path / "jobs.sqlite3", dataset_store, model_registry)
    # Simulate the real executor becoming unavailable before submission.
    jobs._executor.shutdown(wait=True)
    try:
        with pytest.raises(DomainError) as caught:
            jobs.submit("reduced", 50)
        assert (caught.value.status, caught.value.code) == (503, "TRAINING_UNAVAILABLE")
        assert "cannot schedule" not in caught.value.message
        with closing(sqlite3.connect(jobs.db_path)) as db, db:
            job_id = db.execute("SELECT job_id FROM jobs").fetchone()[0]
        saved = jobs.get(job_id)
        assert saved["status"] == "failed"
        assert saved["model_id"] is None
        assert saved["error"]["error"]["code"] == "TRAINING_FAILED"
        assert "cannot schedule" not in json.dumps(saved)
        assert {item["model_id"] for item in model_registry.list()} == original
        assert any(record.exc_info for record in caplog.records)
        assert "cannot schedule new futures after shutdown" in caplog.text
    finally:
        jobs.close()


def test_default_trainer_publishes_new_versions_and_releases_active_slot(
    dataset_store, model_registry, tmp_path,
):
    from motionsense_app.models.jobs import TrainingJobs

    jobs = TrainingJobs(tmp_path / "jobs.sqlite3", dataset_store, model_registry)
    completed = []
    try:
        for profile in ("full", "reduced"):
            job = jobs.submit(profile, 25)
            jobs._future.result(timeout=10)
            saved = jobs.get(job["job_id"])
            assert saved["status"] == "succeeded"
            assert saved["stage"] == "saving" and saved["error"] is None
            artifact = model_registry.load(saved["model_id"])
            assert artifact["manifest"]["dataset_id"] == dataset_store.info()["dataset_id"]
            assert artifact["manifest"]["profile"] == profile
            assert artifact["manifest"]["trees"] == 25
            assert artifact["report"]["test_count"] == 48
            completed.append(saved)
        assert completed[0]["model_id"] != completed[1]["model_id"]
        with closing(sqlite3.connect(jobs.db_path)) as db, db:
            assert db.execute("SELECT COUNT(*) FROM jobs WHERE status='succeeded'").fetchone()[0] == 2
    finally:
        jobs.close()


def test_queued_job_rechecks_live_dataset_before_calling_trainer(
    dataset_store, model_registry, tmp_path, change_session_source,
):
    from motionsense_app.models.jobs import TrainingJobs

    entered = threading.Event()
    release = threading.Event()
    calls = []
    original = {item["model_id"] for item in model_registry.list()}

    def occupy_worker():
        entered.set()
        assert release.wait(5)

    jobs = TrainingJobs(tmp_path / "jobs.sqlite3", dataset_store, model_registry,
                        lambda *args: calls.append(args))
    try:
        jobs._executor.submit(occupy_worker)
        assert entered.wait(5)
        job = jobs.submit("full", 50)
        assert job["status"] == "queued"
        change_session_source(dataset_store, "published")
        release.set()
        jobs._future.result(timeout=5)
        assert jobs.get(job["job_id"])["status"] == "failed"
        assert calls == []
        assert {item["model_id"] for item in model_registry.list()} == original
        with closing(sqlite3.connect(jobs.db_path)) as db, db:
            assert db.execute("SELECT dataset_id FROM jobs").fetchone()[0] == dataset_store.info()["dataset_id"]
    finally:
        release.set()
        jobs.close()


def test_jobs_migrate_v1_without_changing_pinned_session_or_rows(
    session_engine, dataset_store, model_registry, trained_model_id,
):
    from motionsense_app.models.jobs import TrainingJobs

    engine, repo, clock = session_engine
    sid = engine.create({
        "dataset_id": dataset_store.info()["dataset_id"], "split": "test",
        "subject_id": 2, "model_id": trained_model_id,
    })["session_id"]
    engine.attach(sid, "a")
    engine.control(sid, "a", "start")
    clock[0] = 1
    engine.advance(sid, "a", 0)
    snapshot = repo.get(sid)
    rows = repo.rows(sid)
    # Restore the persisted Task 2 schema shape: no jobs table, schema version 1.
    with closing(sqlite3.connect(repo.db_path)) as db, db:
        db.execute("DROP TABLE jobs")
        db.execute("PRAGMA user_version=1")
    jobs = TrainingJobs(repo.db_path, dataset_store, model_registry)
    try:
        assert repo.get(sid) == snapshot
        assert repo.rows(sid) == rows
        with closing(sqlite3.connect(repo.db_path)) as db, db:
            assert db.execute("PRAGMA user_version").fetchone()[0] == 2
            assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        job = jobs.submit("reduced", 25)
        jobs._future.result(timeout=10)
        assert jobs.get(job["job_id"])["status"] == "succeeded"
        assert repo.get(sid) == snapshot
        assert repo.rows(sid) == rows
    finally:
        jobs.close()


def test_close_waits_for_running_worker_and_rejects_new_submits(
    dataset_store, model_registry, tmp_path, job_worker,
):
    from motionsense_app.models.jobs import TrainingJobs

    trainer, entered, release, _ = job_worker
    jobs = TrainingJobs(tmp_path / "jobs.sqlite3", dataset_store, model_registry, trainer)
    job = jobs.submit("reduced", 50)
    assert entered["validating"].wait(5)
    close_started = threading.Event()
    closed = threading.Event()

    def close():
        close_started.set()
        jobs.close()
        closed.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(close)
        try:
            assert close_started.wait(5)
            assert not closed.is_set()
        finally:
            for event in release.values():
                event.set()
        future.result(timeout=5)
    assert closed.is_set()
    assert jobs.get(job["job_id"])["status"] == "succeeded"
    with pytest.raises(DomainError) as caught:
        jobs.submit("reduced", 50)
    assert caught.value.status == 503
    jobs.close()


def test_close_cancels_pending_job_and_does_not_run_trainer(
    dataset_store, model_registry, tmp_path,
):
    from motionsense_app.models.jobs import TrainingJobs

    calls = []
    entered = threading.Event()
    release = threading.Event()

    def occupy_worker():
        entered.set()
        assert release.wait(5)

    jobs = TrainingJobs(tmp_path / "jobs.sqlite3", dataset_store, model_registry,
                        lambda *args: calls.append(args))
    jobs._executor.submit(occupy_worker)
    assert entered.wait(5)
    job = jobs.submit("full", 50)
    assert jobs.get(job["job_id"])["status"] == "queued"
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(jobs.close)
        try:
            # close marks the durable pending job interrupted before waiting for workers.
            deadline = threading.Event()
            for _ in range(500):
                if jobs.get(job["job_id"])["status"] == "interrupted":
                    break
                deadline.wait(0.01)
            assert jobs.get(job["job_id"])["status"] == "interrupted"
        finally:
            release.set()
        future.result(timeout=5)
    assert calls == []


@pytest.mark.parametrize("payload", [
    {"profile": "both", "trees": 50}, {"profile": "full", "trees": 51},
    {"profile": "full", "trees": True}, {"profile": "full", "trees": 50.0},
    {"profile": "full", "trees": "50"}, {"trees": 50}, {"profile": "full"},
    {"profile": "full", "trees": 50, "extra": 1},
])
def test_train_api_rejects_invalid_input(api_client, payload):
    response = api_client.post("/api/models/train", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_unknown_job_is_localized_404(api_client):
    response = api_client.get("/api/jobs/missing")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "JOB_NOT_FOUND"


def test_missing_data_keeps_health_and_data_available(tmp_path):
    from fastapi.testclient import TestClient

    from motionsense_app.main import create_app
    from motionsense_app.settings import Settings

    with TestClient(create_app(Settings(root=tmp_path))) as client:
        response = client.post("/api/models/train", json={"profile": "full", "trees": 50})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "DATA_UNAVAILABLE"
        assert "Cai_dat.bat" in response.json()["error"]["message"]
        assert client.get("/api/health").json()["data_ready"] is False
        assert client.get("/api/data").json()["ready"] is False


def test_api_background_job_keeps_reads_responsive_and_session_pinned(
    dataset_store, model_registry, session_engine, trained_model_id, tmp_path, job_worker,
):
    from fastapi.testclient import TestClient

    from motionsense_app.main import create_app
    from motionsense_app.models.jobs import TrainingJobs
    from motionsense_app.settings import Settings

    trainer, entered, release, _ = job_worker
    engine, repo, clock = session_engine
    jobs = TrainingJobs(repo.db_path, dataset_store, model_registry, trainer)
    app = create_app(Settings(root=tmp_path), services={
        "store": dataset_store, "registry": model_registry, "session_engine": engine,
        "training_jobs": jobs,
    })

    @app.get("/api/event-loop-probe")
    async def probe():
        return {"responsive": True}

    try:
        with TestClient(app) as client:
            assert app.state.training_jobs is jobs
            session = client.post("/api/sessions", json={
                "dataset_id": dataset_store.info()["dataset_id"], "split": "test",
                "subject_id": 2, "model_id": trained_model_id,
            }).json()
            sid = session["session_id"]
            engine.attach(sid, "a")
            engine.control(sid, "a", "start")
            response = client.post("/api/models/train", json={"profile": "reduced", "trees": 50})
            assert response.status_code == 202
            jid = response.json()["job_id"]
            assert entered["validating"].wait(5)
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    assert pool.submit(client.get, "/api/event-loop-probe").result(2).json() == {
                        "responsive": True,
                    }
                    assert pool.submit(client.get, "/api/health").result(2).status_code == 200
                    current = pool.submit(client.get, f"/api/sessions/{sid}").result(2).json()
                    assert current["model_id"] == trained_model_id
                busy = client.post("/api/models/train", json={"profile": "full", "trees": 50})
                assert busy.status_code == 409
                assert busy.json()["error"]["code"] == "TRAINING_BUSY"
                clock[0] = 1
                first = engine.advance(sid, "a", 0)["row"]
                for event in release.values():
                    event.set()
                jobs._future.result(timeout=5)
                saved = client.get(f"/api/jobs/{jid}").json()
                assert saved["status"] == "succeeded"
                assert saved["model_id"] != trained_model_id
                clock[0] = 2
                second = engine.advance(sid, "a", 1)["row"]
                assert first["prediction"]["model_id"] == second["prediction"]["model_id"] == trained_model_id
                assert repo.get(sid)["model_id"] == trained_model_id
                assert client.get(f"/api/models/{saved['model_id']}/report").status_code == 200
                assert saved["model_id"] in {m["model_id"] for m in client.get("/api/models").json()["items"]}
            finally:
                for event in release.values():
                    event.set()
    finally:
        for event in release.values():
            event.set()
        jobs.close()


@pytest.mark.parametrize("status", ["queued", "running"])
def test_default_startup_recovers_jobs_without_data_or_training(
    session_engine, dataset_store, model_registry, trained_model_id, tmp_path, status,
):
    from fastapi.testclient import TestClient

    from motionsense_app.main import create_app
    from motionsense_app.models.jobs import TrainingJobs
    from motionsense_app.settings import Settings

    engine, repo, _ = session_engine
    sid = engine.create({
        "dataset_id": dataset_store.info()["dataset_id"], "split": "test",
        "subject_id": 2, "model_id": trained_model_id,
    })["session_id"]
    jobs = TrainingJobs(repo.db_path, dataset_store, model_registry)
    jobs.close()
    with closing(sqlite3.connect(repo.db_path)) as db, db:
        db.execute("""INSERT INTO jobs(job_id,dataset_id,profile,trees,status,stage,
                      created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)""",
                   ("orphan", dataset_store.info()["dataset_id"], "full", 50, status,
                    "fitting", "2026-10-02T10:00:00+00:00", "2026-10-02T10:00:00+00:00"))
    with TestClient(create_app(Settings(root=tmp_path))) as client:
        saved = client.get("/api/jobs/orphan").json()
        assert saved["status"] == "interrupted"
        assert saved["error"]["error"]["code"] == "TRAINING_INTERRUPTED"
        assert "huấn luyện lại" in saved["error"]["error"]["message"]
        assert client.get(f"/api/sessions/{sid}").json()["status"] == "paused"
        assert client.get("/api/health").json()["data_ready"] is False
        assert client.get("/api/data").json()["ready"] is False
        with closing(sqlite3.connect(repo.db_path)) as db, db:
            assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


def test_job_recovery_and_cleanup_still_run_when_session_recovery_fails(
    session_engine, dataset_store, model_registry, trained_model_id, tmp_path,
):
    from fastapi.testclient import TestClient

    from motionsense_app.main import create_app
    from motionsense_app.models.jobs import TrainingJobs
    from motionsense_app.settings import Settings

    engine, repo, _ = session_engine
    engine.create({
        "dataset_id": dataset_store.info()["dataset_id"], "split": "test",
        "subject_id": 2, "model_id": trained_model_id,
    })
    jobs = TrainingJobs(repo.db_path, dataset_store, model_registry)
    with closing(sqlite3.connect(repo.db_path)) as db, db:
        db.execute("""INSERT INTO jobs(job_id,dataset_id,profile,trees,status,stage,
                      created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)""",
                   ("orphan", dataset_store.info()["dataset_id"], "full", 50, "running",
                    "fitting", "2026-10-02T10:00:00+00:00", "2026-10-02T10:00:00+00:00"))
        db.execute("""CREATE TRIGGER fail_session_recovery BEFORE UPDATE OF client_id ON sessions
                      BEGIN SELECT RAISE(ABORT,'private recovery diagnostic'); END""")
    app = create_app(Settings(root=tmp_path), services={
        "store": dataset_store, "registry": model_registry, "session_engine": engine,
        "training_jobs": jobs,
    })
    try:
        with pytest.raises(DomainError) as caught, TestClient(app):
            pytest.fail("startup must surface the storage failure")
        assert caught.value.code == "STORAGE_ERROR"
        assert jobs.get("orphan")["status"] == "interrupted"
        with pytest.raises(DomainError) as caught:
            jobs.submit("full", 50)
        assert caught.value.code == "TRAINING_UNAVAILABLE"
    finally:
        jobs.close()


def test_lifespan_shutdown_pauses_sessions_and_waits_for_training_off_event_loop(
    session_engine, dataset_store, model_registry, trained_model_id, tmp_path, job_worker,
):
    from fastapi.testclient import TestClient

    from motionsense_app.main import create_app
    from motionsense_app.models.jobs import TrainingJobs
    from motionsense_app.settings import Settings

    trainer, entered, release, _ = job_worker
    closing = threading.Event()

    class ObservedJobs(TrainingJobs):
        def close(self):
            closing.set()
            super().close()

    engine, repo, _ = session_engine
    jobs = ObservedJobs(repo.db_path, dataset_store, model_registry, trainer)
    app = create_app(Settings(root=tmp_path), services={
        "store": dataset_store, "registry": model_registry, "session_engine": engine,
        "training_jobs": jobs,
    })

    @app.get("/api/event-loop-probe")
    async def probe():
        return {"responsive": True}

    client = TestClient(app)
    client.__enter__()
    shutdown = None
    try:
        sid = engine.create({
            "dataset_id": dataset_store.info()["dataset_id"], "split": "test",
            "subject_id": 2, "model_id": trained_model_id,
        })["session_id"]
        engine.attach(sid, "a")
        engine.control(sid, "a", "start")
        job = jobs.submit("reduced", 50)
        assert entered["validating"].wait(5)
        with ThreadPoolExecutor(max_workers=2) as pool:
            shutdown = pool.submit(client.__exit__, None, None, None)
            try:
                assert closing.wait(5)
                assert not shutdown.done()
                assert repo.get(sid)["status"] == "paused"
                assert jobs.get(job["job_id"])["status"] == "running"
                response = pool.submit(client.get, "/api/event-loop-probe").result(timeout=2)
                assert response.status_code == 200
                assert response.json() == {"responsive": True}
            finally:
                for event in release.values():
                    event.set()
            shutdown.result(timeout=5)
        assert jobs.get(job["job_id"])["status"] == "succeeded"
        with pytest.raises(DomainError) as caught:
            jobs.submit("full", 50)
        assert caught.value.status == 503
    finally:
        for event in release.values():
            event.set()
        if shutdown is None:
            client.__exit__(None, None, None)
        jobs.close()


def test_live_manager_recovery_cannot_release_another_workers_slot(
    dataset_store, model_registry, tmp_path, job_worker,
):
    from motionsense_app.models.jobs import TrainingJobs

    trainer, entered, release, _ = job_worker
    path = tmp_path / "jobs.sqlite3"
    first = TrainingJobs(path, dataset_store, model_registry, trainer)
    second = TrainingJobs(path, dataset_store, model_registry)
    try:
        first.recover()
        job = first.submit("reduced", 50)
        assert entered["validating"].wait(5)
        before = first.get(job["job_id"])
        for manager in (first, second):
            with pytest.raises(DomainError) as caught:
                manager.recover()
            assert (caught.value.status, caught.value.code) == (409, "TRAINING_BUSY")
            assert manager.get(job["job_id"]) == before
        with pytest.raises(DomainError) as caught:
            second.submit("reduced", 25)
        assert caught.value.code == "TRAINING_BUSY"
        for event in release.values():
            event.set()
        first._future.result(timeout=5)
        completed = second.get(job["job_id"])
        assert completed["status"] == "succeeded" and completed["error"] is None
        # Lifetime ownership remains exclusive until close, even after worker completion.
        with pytest.raises(DomainError):
            second.recover()
        first.close()
        second.recover()
        assert second.get(job["job_id"]) == completed
        next_job = second.submit("reduced", 25)
        second._future.result(timeout=10)
        assert second.get(next_job["job_id"])["status"] == "succeeded"
        assert second.get(next_job["job_id"])["error"] is None
    finally:
        for event in release.values():
            event.set()
        first.close()
        second.close()


def test_success_clears_persisted_error(dataset_store, model_registry, tmp_path, job_worker):
    from motionsense_app.errors import api_error
    from motionsense_app.models.jobs import TrainingJobs

    trainer, entered, release, _ = job_worker
    jobs = TrainingJobs(tmp_path / "jobs.sqlite3", dataset_store, model_registry, trainer)
    try:
        job = jobs.submit("reduced", 50)
        assert entered["validating"].wait(5)
        with closing(sqlite3.connect(jobs.db_path)) as db, db:
            db.execute("UPDATE jobs SET error_json=? WHERE job_id=?",
                       (json.dumps(api_error("OLD_ERROR", "Lỗi cũ.")), job["job_id"]))
        for event in release.values():
            event.set()
        jobs._future.result(timeout=5)
        assert jobs.get(job["job_id"])["status"] == "succeeded"
        assert jobs.get(job["job_id"])["error"] is None
        with closing(sqlite3.connect(jobs.db_path)) as db, db:
            assert db.execute("SELECT error_json FROM jobs").fetchone()[0] is None
    finally:
        for event in release.values():
            event.set()
        jobs.close()


@pytest.mark.parametrize("status", ["interrupted", "failed"])
def test_terminal_job_is_not_revived_or_published_by_its_worker(
    dataset_store, model_registry, tmp_path, job_worker, status,
):
    from motionsense_app.errors import api_error
    from motionsense_app.models.jobs import TrainingJobs

    trainer, entered, release, _ = job_worker
    original = {item["model_id"] for item in model_registry.list()}
    jobs = TrainingJobs(tmp_path / "jobs.sqlite3", dataset_store, model_registry, trainer)
    try:
        job = jobs.submit("reduced", 50)
        assert entered["validating"].wait(5)
        error = api_error("TERMINAL_ERROR", "Tác vụ đã kết thúc.")
        with closing(sqlite3.connect(jobs.db_path)) as db, db:
            db.execute("UPDATE jobs SET status=?,error_json=? WHERE job_id=?",
                       (status, json.dumps(error), job["job_id"]))
        terminal = jobs.get(job["job_id"])
        for event in release.values():
            event.set()
        jobs._future.result(timeout=5)
        assert jobs.get(job["job_id"]) == terminal
        assert {item["model_id"] for item in model_registry.list()} == original
    finally:
        for event in release.values():
            event.set()
        jobs.close()


def test_second_lifespan_cannot_recover_or_close_live_services(
    dataset_store, model_registry, session_engine, trained_model_id, tmp_path, job_worker,
):
    from fastapi.testclient import TestClient

    from motionsense_app.main import create_app
    from motionsense_app.models.jobs import TrainingJobs
    from motionsense_app.settings import Settings

    trainer, entered, release, _ = job_worker
    engine, repo, clock = session_engine
    jobs = TrainingJobs(repo.db_path, dataset_store, model_registry, trainer)
    services = {"store": dataset_store, "registry": model_registry,
                "session_engine": engine, "training_jobs": jobs}
    try:
        with TestClient(create_app(Settings(root=tmp_path), services=services)):
            sid = engine.create({
                "dataset_id": dataset_store.info()["dataset_id"], "split": "test",
                "subject_id": 2, "model_id": trained_model_id,
            })["session_id"]
            engine.attach(sid, "original")
            engine.control(sid, "original", "start")
            job = jobs.submit("reduced", 50)
            assert entered["validating"].wait(5)
            try:
                # Check both a distinct manager and a borrowed, already-live injected service.
                for overrides in ({"store": dataset_store, "registry": model_registry}, services):
                    with pytest.raises(DomainError) as caught, TestClient(
                        create_app(Settings(root=tmp_path), services=overrides)
                    ):
                        pytest.fail("a second lifespan must reject live ownership")
                    assert caught.value.code == "TRAINING_BUSY"
                    assert repo.get(sid)["status"] == "running"
                    assert jobs.get(job["job_id"])["status"] == "running"
                clock[0] = 1
                assert engine.advance(sid, "original", 0)["session"]["cursor"] == 1
            finally:
                for event in release.values():
                    event.set()
            jobs._future.result(timeout=5)
            assert jobs.get(job["job_id"])["status"] == "succeeded"
            assert jobs.get(job["job_id"])["error"] is None
    finally:
        for event in release.values():
            event.set()
        jobs.close()


def test_dead_publisher_recovery_reclaims_only_orphan_staging(
    dataset_store, model_registry, tmp_path,
):
    from motionsense_app.models.jobs import TrainingJobs

    original = {item["model_id"] for item in model_registry.list()}
    path = tmp_path / "child.sqlite3"
    # A real process holds ownership and pauses after writing a real staged artifact.
    source = textwrap.dedent("""
        import json
        import sys
        import threading
        from pathlib import Path
        import joblib
        from motionsense_app.data.store import DatasetStore
        from motionsense_app.models.jobs import TrainingJobs
        from motionsense_app.models.registry import ModelRegistry
        entered = threading.Event()
        release = threading.Event()
        original_dump = joblib.dump
        def dump(artifact, filename, *args, **kwargs):
            result = original_dump(artifact, filename, *args, **kwargs)
            entered.set()
            if not release.wait(60):
                raise RuntimeError("test publisher not released")
            return result
        joblib.dump = dump
        store = DatasetStore(Path(sys.argv[2]))
        registry = ModelRegistry(Path(sys.argv[3]))
        jobs = TrainingJobs(Path(sys.argv[1]), store, registry)
        jobs.recover()
        job = jobs.submit("reduced", 25)
        assert entered.wait(20)
        staging = next(registry.model_dir.glob(".staging-*"))
        print(json.dumps({"job_id": job["job_id"], "staging": str(staging)}), flush=True)
        sys.stdin.read()
        release.set()
        jobs.close()
    """)
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", source, str(path), str(dataset_store.data_dir),
         str(model_registry.model_dir)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    jobs = TrainingJobs(path, dataset_store, model_registry)
    other_db = TrainingJobs(tmp_path / "other.sqlite3", dataset_store, model_registry)
    reader = ThreadPoolExecutor(max_workers=1)
    try:
        line = reader.submit(process.stdout.readline).result(timeout=25)
        assert line, "child failed before reaching publication"
        saved = json.loads(line)
        staging = Path(saved["staging"])
        assert (staging / "model.joblib").is_file()
        with pytest.raises(DomainError) as caught:
            jobs.recover()
        assert caught.value.code == "TRAINING_BUSY"
        assert jobs.get(saved["job_id"])["status"] == "running"
        with pytest.raises(DomainError):
            jobs.submit("full", 25)
        # Even an independently owned DB must not clean another live registry publisher.
        other_db.recover()
        assert (staging / "model.joblib").is_file()
        other_db.close()
        unrelated = model_registry.model_dir / ".staging-user-work"
        unrelated.mkdir()
        (unrelated / "notes.txt").write_text("preserve", encoding="utf-8")
        lookalike_file = model_registry.model_dir / (".staging-" + "f" * 32)
        lookalike_file.write_text("not a directory", encoding="utf-8")
        process.kill()
        process.communicate(timeout=10)
        jobs.recover()
        terminal = jobs.get(saved["job_id"])
        assert terminal["status"] == "interrupted"
        assert terminal["error"]["error"]["code"] == "TRAINING_INTERRUPTED"
        assert not staging.exists()
        assert (unrelated / "notes.txt").read_text(encoding="utf-8") == "preserve"
        assert lookalike_file.read_text(encoding="utf-8") == "not a directory"
        assert {item["model_id"] for item in model_registry.list()} == original
        next_job = jobs.submit("reduced", 25)
        jobs._future.result(timeout=10)
        assert jobs.get(next_job["job_id"])["status"] == "succeeded"
        assert jobs.get(next_job["job_id"])["error"] is None
        assert jobs.get(saved["job_id"]) == terminal
    finally:
        if process.poll() is None:
            process.kill()
        reader.shutdown(wait=True)
        process.communicate(timeout=10)
        jobs.close()
        other_db.close()


def test_session_recovery_is_attempted_when_owned_job_recovery_fails(
    dataset_store, model_registry, session_engine, trained_model_id, tmp_path,
):
    from fastapi.testclient import TestClient

    from motionsense_app.main import create_app
    from motionsense_app.models.jobs import TrainingJobs
    from motionsense_app.settings import Settings

    engine, repo, _ = session_engine
    sid = engine.create({
        "dataset_id": dataset_store.info()["dataset_id"], "split": "test",
        "subject_id": 2, "model_id": trained_model_id,
    })["session_id"]
    engine.attach(sid, "old")
    engine.control(sid, "old", "start")
    jobs = TrainingJobs(repo.db_path, dataset_store, model_registry)
    with closing(sqlite3.connect(repo.db_path)) as db, db:
        db.execute("""INSERT INTO jobs(job_id,dataset_id,profile,trees,status,stage,
                      created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)""",
                   ("orphan", dataset_store.info()["dataset_id"], "full", 50, "running",
                    "fitting", "2026-10-02T10:00:00+00:00", "2026-10-02T10:00:00+00:00"))
        db.execute("""CREATE TRIGGER fail_job_recovery BEFORE UPDATE OF status ON jobs
                      BEGIN SELECT RAISE(ABORT,'private job recovery diagnostic'); END""")
    app = create_app(Settings(root=tmp_path), services={
        "store": dataset_store, "registry": model_registry, "session_engine": engine,
        "training_jobs": jobs,
    })
    try:
        with pytest.raises(DomainError) as caught, TestClient(app):
            pytest.fail("startup must surface job recovery's storage failure")
        assert caught.value.code == "STORAGE_ERROR"
        assert repo.get(sid)["status"] == "paused"
        with pytest.raises(DomainError) as caught:
            engine.control(sid, "old", "resume")
        assert caught.value.code == "CLIENT_CONFLICT"
        with pytest.raises(DomainError) as caught:
            jobs.submit("full", 50)
        assert caught.value.code == "TRAINING_UNAVAILABLE"
        assert jobs.get("orphan")["status"] == "running"  # failed write rolled back
    finally:
        jobs.close()
