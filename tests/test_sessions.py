import csv
import io
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime

import pytest

from motionsense_app.errors import DomainError
from motionsense_app.sessions.engine import SessionEngine
from motionsense_app.sessions.repository import SessionRepository
from motionsense_app.storage import transaction


def create_session(engine, dataset_store, model_id, client_id="a"):
    session = engine.create({
        "dataset_id": dataset_store.info()["dataset_id"], "split": "test",
        "subject_id": 2, "model_id": model_id,
    })
    engine.attach(session["session_id"], client_id)
    return session["session_id"]


def assert_conflict(call, code="SESSION_STATE_CONFLICT"):
    with pytest.raises(DomainError) as caught:
        call()
    assert caught.value.status == 409
    assert caught.value.code == code


def test_control_transitions_are_idempotent_and_finished_is_terminal(
    session_engine, dataset_store, trained_model_id,
):
    engine, repo, clock = session_engine
    sid = create_session(engine, dataset_store, trained_model_id)
    assert_conflict(lambda: engine.control(sid, "a", "resume"))
    assert_conflict(lambda: engine.control(sid, "a", "pause"))
    running = engine.control(sid, "a", "start")
    assert running["status"] == "running"
    clock[0] = 0.25
    assert engine.control(sid, "a", "start") == running
    assert engine.control(sid, "a", "resume") == running
    assert engine.attach(sid, "a") == running
    paused = engine.control(sid, "a", "pause")
    assert paused["status"] == "paused"
    assert engine.control(sid, "a", "pause") == paused
    assert_conflict(lambda: engine.control(sid, "a", "start"))
    assert engine.control(sid, "a", "resume")["status"] == "running"
    finished = engine.control(sid, "a", "finish")
    assert finished["status"] == "finished"
    assert finished["finish_reason"] == "user"
    assert engine.control(sid, "a", "finish") == finished
    for action in ("start", "resume", "pause", "speed", "invalid"):
        assert_conflict(lambda action=action: engine.control(sid, "a", action, 2.0))
    assert repo.get(sid)["cursor"] == 0


@pytest.mark.parametrize("action", ["start", "resume", "pause", "finish", "speed"])
def test_new_tab_pauses_running_and_rejects_old_tab_controls(
    session_engine, dataset_store, trained_model_id, action,
):
    engine, repo, _ = session_engine
    sid = create_session(engine, dataset_store, trained_model_id)
    engine.control(sid, "a", "start")
    assert engine.attach(sid, "b")["status"] == "paused"
    assert_conflict(lambda: engine.control(sid, "a", action, 2.0), "CLIENT_CONFLICT")
    assert repo.get(sid)["cursor"] == 0
    assert engine.control(sid, "b", "resume")["status"] == "running"


def test_unattached_browser_cannot_control(session_engine, dataset_store, trained_model_id):
    engine, _, _ = session_engine
    session = engine.create({
        "dataset_id": dataset_store.info()["dataset_id"], "split": "test",
        "subject_id": 2, "model_id": trained_model_id,
    })
    assert_conflict(lambda: engine.control(session["session_id"], "a", "start"), "CLIENT_CONFLICT")


def test_switching_sessions_pauses_previous_even_when_resuming_old_session(
    session_engine, dataset_store, trained_model_id,
):
    engine, repo, _ = session_engine
    first = create_session(engine, dataset_store, trained_model_id)
    engine.control(first, "a", "start")
    second = create_session(engine, dataset_store, trained_model_id, "b")
    assert repo.get(first)["status"] == "paused"
    engine.control(second, "b", "start")
    engine.control(first, "a", "resume")
    assert repo.get(second)["status"] == "paused"
    assert repo.get(first)["status"] == "running"


def test_invalid_create_does_not_pause_current_session(
    session_engine, dataset_store, trained_model_id,
):
    engine, repo, _ = session_engine
    sid = create_session(engine, dataset_store, trained_model_id)
    engine.control(sid, "a", "start")
    with pytest.raises(DomainError):
        engine.create({"dataset_id": dataset_store.info()["dataset_id"], "split": "test",
                       "subject_id": 1, "model_id": trained_model_id})
    assert repo.get(sid)["status"] == "running"


@pytest.mark.parametrize("speed", [0, -1, 3, True, float("nan"), None])
def test_invalid_speed_does_not_mutate_session(
    session_engine, dataset_store, trained_model_id, speed,
):
    engine, repo, _ = session_engine
    sid = create_session(engine, dataset_store, trained_model_id)
    before = repo.get(sid)
    with pytest.raises(DomainError) as caught:
        engine.control(sid, "a", "speed", speed)
    assert caught.value.status == 422
    assert repo.get(sid) == before


def test_recover_clears_browser_and_preserves_finished(
    session_engine, dataset_store, trained_model_id,
):
    engine, repo, _ = session_engine
    ready = create_session(engine, dataset_store, trained_model_id)
    finished = create_session(engine, dataset_store, trained_model_id)
    engine.control(finished, "a", "finish")
    paused = create_session(engine, dataset_store, trained_model_id)
    engine.control(paused, "a", "start")
    engine.control(paused, "a", "pause")
    running = create_session(engine, dataset_store, trained_model_id)
    engine.control(running, "a", "start")
    finished_before = repo.get(finished)
    repo.recover()
    for sid in (ready, paused, running):
        assert repo.get(sid)["status"] == "paused"
        assert_conflict(lambda sid=sid: engine.control(sid, "a", "resume"), "CLIENT_CONFLICT")
    assert repo.get(finished) == finished_before
    with closing(sqlite3.connect(repo.db_path)) as db, db:
        assert db.execute("SELECT COUNT(*) FROM sessions WHERE client_id IS NOT NULL").fetchone()[0] == 1


def start_session(session_engine, dataset_store, model_id):
    engine, _, _ = session_engine
    sid = create_session(engine, dataset_store, model_id)
    engine.control(sid, "a", "start")
    return sid


def test_repeated_advance_returns_exact_stored_row_once(
    session_engine, dataset_store, trained_model_id,
):
    engine, repo, clock = session_engine
    sid = start_session(session_engine, dataset_store, trained_model_id)
    clock[0] = 1.0
    first = engine.advance(sid, "a", 0)
    again = engine.advance(sid, "a", 0)
    assert first == again
    assert first["session"]["cursor"] == 1
    assert first["row"]["ordinal"] == 0
    assert first["row"]["sample_id"] == "test:000001"
    assert first["row"]["prediction"]["actual_label"] == 1
    assert first["row"]["prediction"]["model_id"] == trained_model_id
    assert datetime.fromisoformat(first["row"]["processed_at"]).utcoffset().total_seconds() == 0
    assert len(first["row"]["prediction"]["probabilities"]) == 6
    assert first["session"]["last_prediction"] == first["row"]["prediction"]
    assert repo.rows(sid, 0, 50) == {
        "items": [first["row"]], "offset": 0, "limit": 50, "total": 1,
    }


def test_threaded_advance_commits_only_one_row(
    session_engine, dataset_store, trained_model_id,
):
    engine, repo, clock = session_engine
    sid = start_session(session_engine, dataset_store, trained_model_id)
    clock[0] = 1
    barrier = threading.Barrier(2)

    def advance():
        barrier.wait(timeout=5)
        return engine.advance(sid, "a", 0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(advance) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]
    assert results[0] == results[1]
    assert repo.get(sid)["cursor"] == repo.rows(sid, 0, 50)["total"] == 1


def test_due_gating_and_no_catchup_at_slow_browser(
    session_engine, dataset_store, trained_model_id, monkeypatch,
):
    engine, repo, clock = session_engine
    sid = start_session(session_engine, dataset_store, trained_model_id)
    sample = dataset_store.sample

    def forbidden_sample(_id):
        pytest.fail("sample inference before server due time")

    monkeypatch.setattr(dataset_store, "sample", forbidden_sample)
    for now in (0, 0.25, 0.99):
        clock[0] = now
        assert engine.advance(sid, "a", 0)["row"] is None
    monkeypatch.setattr(dataset_store, "sample", sample)
    clock[0] = 2.9
    assert engine.advance(sid, "a", 0)["session"]["cursor"] == 1
    assert engine.advance(sid, "a", 1)["row"] is None
    clock[0] = 3.89
    assert engine.advance(sid, "a", 1)["row"] is None
    clock[0] = 3.9
    assert engine.advance(sid, "a", 1)["session"]["cursor"] == 2
    assert repo.rows(sid, 0, 50)["total"] == 2


def test_future_cursor_conflict_does_not_keep_dead_browser_alive(
    session_engine, dataset_store, trained_model_id,
):
    engine, repo, clock = session_engine
    sid = start_session(session_engine, dataset_store, trained_model_id)
    clock[0] = 2.9
    assert_conflict(lambda: engine.advance(sid, "a", 1), "CURSOR_CONFLICT")
    clock[0] = 3
    result = engine.advance(sid, "a", 0)
    assert result["session"]["status"] == "paused"
    assert result["row"] is None
    assert repo.rows(sid, 0, 50)["total"] == 0


@pytest.mark.parametrize("speed,delay", [(0.5, 2.0), (1.0, 1.0), (2.0, 0.5)])
def test_speed_reschedules_from_now_and_repeat_does_not_push_due(
    session_engine, dataset_store, trained_model_id, speed, delay,
):
    engine, _, clock = session_engine
    sid = create_session(engine, dataset_store, trained_model_id)
    engine.control(sid, "a", "speed", 2.0 if speed == 1 else 1.0)
    engine.control(sid, "a", "start")
    clock[0] = 0.25
    engine.control(sid, "a", "speed", speed)
    clock[0] = 0.25 + delay - 0.01
    assert engine.advance(sid, "a", 0)["row"] is None
    engine.control(sid, "a", "speed", speed)
    clock[0] = 0.25 + delay
    assert engine.advance(sid, "a", 0)["session"]["cursor"] == 1


def test_pause_resume_and_user_finish_preserve_cursor_and_stored_retry(
    session_engine, dataset_store, trained_model_id,
):
    engine, repo, clock = session_engine
    sid = start_session(session_engine, dataset_store, trained_model_id)
    clock[0] = 1
    first = engine.advance(sid, "a", 0)["row"]
    engine.control(sid, "a", "pause")
    clock[0] = 50
    assert engine.advance(sid, "a", 1)["row"] is None
    assert engine.advance(sid, "a", 0)["row"] == first
    engine.control(sid, "a", "resume")
    clock[0] = 50.99
    assert engine.advance(sid, "a", 1)["row"] is None
    clock[0] = 51
    assert engine.advance(sid, "a", 1)["session"]["cursor"] == 2
    engine.control(sid, "a", "finish")
    assert engine.advance(sid, "a", 2)["row"] is None
    assert engine.advance(sid, "a", 0)["row"] == first
    assert repo.rows(sid, 0, 50)["total"] == 2


def test_completion_processes_all_48_exactly_once_and_keeps_pinned_model(
    session_engine, dataset_store, trained_model_id, monkeypatch,
):
    engine, repo, clock = session_engine
    sid = start_session(session_engine, dataset_store, trained_model_id)
    # Selection is pinned once: subsequent selection changes cannot reinterpret it.
    monkeypatch.setattr(dataset_store, "sample_ids", lambda *_: ["test:000048"])
    for cursor in range(48):
        clock[0] = cursor + 1.0
        result = engine.advance(sid, "a", cursor)
        assert result["row"]["sample_id"] == f"test:{cursor + 1:06d}"
        assert result["row"]["prediction"]["model_id"] == trained_model_id
    assert result["session"]["status"] == "finished"
    assert result["session"]["finish_reason"] == "complete"
    assert result["session"]["cursor"] == result["session"]["total"] == 48
    assert engine.advance(sid, "a", 48)["row"] is None
    assert repo.rows(sid, 0, 200)["total"] == 48
    assert repo.rows(sid, 47, 1)["items"] == [result["row"]]


def test_expired_lease_pauses_before_even_stored_retry(
    session_engine, dataset_store, trained_model_id,
):
    engine, repo, clock = session_engine
    sid = start_session(session_engine, dataset_store, trained_model_id)
    clock[0] = 1
    engine.advance(sid, "a", 0)
    clock[0] = 4
    result = engine.advance(sid, "a", 0)
    assert result["session"]["status"] == "paused"
    assert result["row"] is None
    assert repo.get(sid)["cursor"] == 1


def test_watchdog_pauses_expired_session_without_advancing(
    session_engine, dataset_store, trained_model_id,
):
    engine, repo, clock = session_engine
    sid = start_session(session_engine, dataset_store, trained_model_id)
    clock[0] = 2.99
    engine.expire_leases()
    assert repo.get(sid)["status"] == "running"
    clock[0] = 3
    engine.expire_leases()
    assert repo.get(sid)["status"] == "paused"
    assert repo.get(sid)["cursor"] == 0


def test_stale_tab_cannot_advance_or_retry_after_reattach(
    session_engine, dataset_store, trained_model_id,
):
    engine, repo, clock = session_engine
    sid = start_session(session_engine, dataset_store, trained_model_id)
    clock[0] = 1
    engine.advance(sid, "a", 0)
    engine.attach(sid, "b")
    for cursor in (0, 1, 2):
        assert_conflict(lambda cursor=cursor: engine.advance(sid, "a", cursor), "CLIENT_CONFLICT")
    assert repo.get(sid)["cursor"] == 1


def test_reopen_storage_and_restart_engine_preserve_rows(
    session_engine, dataset_store, trained_model_id, model_registry,
):
    engine, repo, clock = session_engine
    sid = start_session(session_engine, dataset_store, trained_model_id)
    clock[0] = 1
    first = engine.advance(sid, "a", 0)["row"]
    reopened = SessionRepository(repo.db_path)
    reopened.recover()
    fresh = SessionEngine(reopened, dataset_store, model_registry, lambda: clock[0])
    assert reopened.get(sid)["cursor"] == 1
    assert reopened.get(sid)["status"] == "paused"
    assert reopened.get(sid)["last_prediction"] == first["prediction"]
    assert_conflict(lambda: fresh.advance(sid, "a", 1), "CLIENT_CONFLICT")
    fresh.attach(sid, "new")
    fresh.control(sid, "new", "resume")
    assert fresh.advance(sid, "new", 1)["row"] is None
    clock[0] = 2
    assert fresh.advance(sid, "new", 1)["row"]["ordinal"] == 1
    assert reopened.rows(sid, 0, 50)["total"] == 2


@pytest.mark.parametrize("unavailable", ["dataset", "missing_model", "corrupt_model"])
def test_unavailable_dependencies_block_resume_and_pause_advance_but_keep_history(
    session_engine, dataset_store, trained_model_id, unavailable, tmp_path,
):
    from motionsense_app.data.importer import load_uci, publish_dataset
    from motionsense_app.sessions.export import export_session

    engine, repo, clock = session_engine
    sid = start_session(session_engine, dataset_store, trained_model_id)
    clock[0] = 1
    first = engine.advance(sid, "a", 0)["row"]
    if unavailable == "dataset":
        # A real replacement snapshot, not a fabricated info() response.
        publish_dataset(load_uci(tmp_path / "UCI HAR Dataset"), dataset_store.data_dir, "b" * 64)
        assert engine.store is dataset_store
        code = "DATASET_MISMATCH"
    else:
        artifact_path = engine.registry.model_dir / trained_model_id
        if unavailable == "missing_model":
            artifact_path.rename(artifact_path.with_name("removed-model"))
        else:
            (artifact_path / "model.joblib").write_bytes(b"corrupt")
        code = "MODEL_UNAVAILABLE"
    clock[0] = 2
    with pytest.raises(DomainError) as caught:
        engine.advance(sid, "a", 1)
    assert caught.value.status == 503
    assert caught.value.code == code
    assert repo.get(sid)["status"] == "paused"
    with pytest.raises(DomainError) as caught:
        engine.control(sid, "a", "resume")
    assert caught.value.status == 503
    assert repo.rows(sid, 0, 50)["items"] == [first]
    assert repo.summary(sid)["processed"] == 1
    assert len(list(csv.DictReader(io.StringIO(export_session(repo, sid).decode("utf-8-sig"))))) == 1


def test_failed_cursor_update_rolls_back_insert_and_pauses(
    session_engine, dataset_store, trained_model_id,
):
    engine, repo, clock = session_engine
    sid = start_session(session_engine, dataset_store, trained_model_id)
    with closing(sqlite3.connect(repo.db_path)) as db, db:
        db.execute("""CREATE TRIGGER fail_cursor BEFORE UPDATE OF cursor ON sessions
                      WHEN NEW.cursor > OLD.cursor BEGIN SELECT RAISE(ABORT,'disk write failure'); END""")
    clock[0] = 1
    with pytest.raises(DomainError) as caught:
        engine.advance(sid, "a", 0)
    assert caught.value.code == "STORAGE_ERROR"
    assert caught.value.status == 500
    saved = repo.get(sid)
    assert saved["cursor"] == 0
    assert saved["last_prediction"] is None
    assert saved["status"] == "paused"
    assert repo.rows(sid, 0, 50)["total"] == 0
    with closing(sqlite3.connect(repo.db_path)) as db, db:
        db.execute("DROP TRIGGER fail_cursor")
    engine.control(sid, "a", "resume")
    clock[0] = 2
    assert engine.advance(sid, "a", 0)["session"]["cursor"] == 1


def test_database_rejects_duplicate_ordinals_samples_and_orphans(
    session_engine, dataset_store, trained_model_id,
):
    engine, repo, clock = session_engine
    sid = start_session(session_engine, dataset_store, trained_model_id)
    clock[0] = 1
    row = engine.advance(sid, "a", 0)["row"]
    for session_id, ordinal, sample in ((sid, 0, "test:000002"), (sid, 1, "test:000001"),
                                        ("missing", 0, "test:000001")):
        with pytest.raises(DomainError) as caught, transaction(repo.db_path, write=True) as db:
            assert db.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
            db.execute("INSERT INTO session_rows VALUES (?,?,?,?,?)",
                       (session_id, ordinal, sample, json.dumps(row["prediction"]), row["processed_at"]))
        assert caught.value.code == "STORAGE_ERROR"
    assert repo.rows(sid, 0, 50)["total"] == 1


def test_summary_and_bom_export_use_only_committed_rows(
    session_engine, dataset_store, trained_model_id,
):
    from motionsense_app.sessions.export import export_session

    engine, repo, clock = session_engine
    sid = start_session(session_engine, dataset_store, trained_model_id)
    assert repo.summary(sid) == {
        "processed": 0, "labeled": 0, "correct": 0, "session_accuracy": None,
        "counts": [{"label_id": i, "count": 0} for i in range(1, 7)],
    }
    for cursor in range(3):
        clock[0] = cursor + 1
        engine.advance(sid, "a", cursor)
    engine.control(sid, "a", "finish")
    rows = repo.rows(sid, 0, 50)["items"]
    # Historical rows may be unlabeled; summary must not divide by all rows.
    prediction = dict(rows[0]["prediction"], actual_label=None)
    with closing(sqlite3.connect(repo.db_path)) as db, db:
        db.execute("UPDATE session_rows SET prediction_json=? WHERE session_id=? AND ordinal=0",
                   (json.dumps(prediction), sid))
    rows = repo.rows(sid, 0, 50)["items"]
    summary = repo.summary(sid)
    assert summary["processed"] == 3
    assert summary["labeled"] == 2
    assert summary["correct"] == sum(
        row["prediction"]["actual_label"] == row["prediction"]["predicted_label"] for row in rows[1:]
    )
    assert summary["session_accuracy"] == summary["correct"] / 2
    assert sum(item["count"] for item in summary["counts"]) == 3
    for item in summary["counts"]:
        assert item["count"] == sum(row["prediction"]["predicted_label"] == item["label_id"] for row in rows)
    exported = export_session(repo, sid)
    assert exported.startswith(b"\xef\xbb\xbf")
    reader = csv.DictReader(io.StringIO(exported.decode("utf-8-sig")))
    assert reader.fieldnames == ["session_id", "model_id", "dataset_id", "split", "subject_id",
                                 "ordinal", "sample_id", "predicted_label", "actual_label",
                                 *[f"probability_{i}" for i in range(1, 7)], "processed_at"]
    csv_rows = list(reader)
    assert len(csv_rows) == 3
    assert csv_rows[0]["actual_label"] == ""
    for saved, exported in zip(rows, csv_rows):
        assert exported["session_id"] == sid
        assert exported["model_id"] == trained_model_id
        assert exported["dataset_id"] == dataset_store.info()["dataset_id"]
        assert exported["split"] == "test"
        assert exported["subject_id"] == "2"
        assert int(exported["ordinal"]) == saved["ordinal"]
        assert exported["sample_id"] == saved["sample_id"]
        assert int(exported["predicted_label"]) == saved["prediction"]["predicted_label"]
        assert exported["processed_at"] == saved["processed_at"]
        for p in saved["prediction"]["probabilities"]:
            assert float(exported[f"probability_{p['label_id']}"]) == p["value"]
    with closing(sqlite3.connect(repo.db_path)) as db, db:
        for row in rows:
            prediction = dict(row["prediction"], actual_label=None)
            db.execute("UPDATE session_rows SET prediction_json=? WHERE session_id=? AND ordinal=?",
                       (json.dumps(prediction), sid, row["ordinal"]))
        # Summary deliberately ignores inconsistent metadata, never inventing processed samples.
        db.execute("UPDATE sessions SET cursor=4 WHERE session_id=?", (sid,))
    assert repo.summary(sid)["processed"] == 3
    assert repo.summary(sid)["labeled"] == repo.summary(sid)["correct"] == 0
    assert repo.summary(sid)["session_accuracy"] is None


def test_watchdog_retries_failed_pause_after_storage_recovers(
    session_engine, dataset_store, trained_model_id,
):
    engine, repo, clock = session_engine
    sid = start_session(session_engine, dataset_store, trained_model_id)
    with closing(sqlite3.connect(repo.db_path)) as db, db:
        db.execute("""CREATE TRIGGER fail_cursor BEFORE UPDATE OF cursor ON sessions
                      BEGIN SELECT RAISE(ABORT,'cursor unavailable'); END""")
        db.execute("""CREATE TRIGGER fail_pause BEFORE UPDATE OF status ON sessions
                      WHEN NEW.status='paused' BEGIN SELECT RAISE(ABORT,'pause unavailable'); END""")
    clock[0] = 1
    with pytest.raises(DomainError) as caught:
        engine.advance(sid, "a", 0)
    assert caught.value.code == "STORAGE_ERROR"
    assert repo.get(sid)["status"] == "running"  # disk rejected the best-effort pause
    assert repo.get(sid)["cursor"] == repo.rows(sid, 0, 50)["total"] == 0
    with closing(sqlite3.connect(repo.db_path)) as db, db:
        db.execute("DROP TRIGGER fail_pause")
        db.execute("DROP TRIGGER fail_cursor")
    engine.expire_leases()
    assert repo.get(sid)["status"] == "paused"
    assert engine.advance(sid, "a", 0)["row"] is None


def test_new_session_insert_failure_rolls_back_previous_session_pause(
    session_engine, dataset_store, trained_model_id,
):
    engine, repo, clock = session_engine
    sid = start_session(session_engine, dataset_store, trained_model_id)
    with closing(sqlite3.connect(repo.db_path)) as db, db:
        db.execute("""CREATE TRIGGER fail_create BEFORE INSERT ON sessions
                      BEGIN SELECT RAISE(ABORT,'insert unavailable'); END""")
    with pytest.raises(DomainError) as caught:
        engine.create({"dataset_id": dataset_store.info()["dataset_id"], "split": "test",
                       "subject_id": 2, "model_id": trained_model_id})
    assert caught.value.code == "STORAGE_ERROR"
    assert repo.get(sid)["status"] == "running"
    assert repo.list(0, 50)["total"] == 1
    clock[0] = 1
    assert engine.advance(sid, "a", 0)["session"]["cursor"] == 1


def test_pause_waits_for_inflight_prediction_and_prevents_next_row(
    session_engine, dataset_store, trained_model_id, monkeypatch,
):
    engine, repo, clock = session_engine
    sid = start_session(session_engine, dataset_store, trained_model_id)
    clock[0] = 1
    entered = threading.Event()
    release = threading.Event()
    pause_requested = threading.Event()
    read_sample = dataset_store.sample

    def slow_sample(sample_id):
        entered.set()
        assert release.wait(timeout=5)
        return read_sample(sample_id)

    def pause():
        pause_requested.set()
        return engine.control(sid, "a", "pause")

    monkeypatch.setattr(dataset_store, "sample", slow_sample)
    with ThreadPoolExecutor(max_workers=2) as pool:
        processing = pool.submit(engine.advance, sid, "a", 0)
        try:
            assert entered.wait(timeout=3)
            pausing = pool.submit(pause)
            assert pause_requested.wait(timeout=2)
            assert repo.get(sid)["cursor"] == 0
        finally:
            release.set()
        assert processing.result(timeout=5)["session"]["cursor"] == 1
        assert pausing.result(timeout=5)["status"] == "paused"
    clock[0] = 2
    assert engine.advance(sid, "a", 1)["row"] is None
    assert repo.get(sid)["cursor"] == repo.rows(sid, 0, 50)["total"] == 1


def test_publishing_another_model_does_not_change_pinned_session(
    session_engine, dataset_store, trained_model_id, training_result,
):
    engine, repo, clock = session_engine
    sid = start_session(session_engine, dataset_store, trained_model_id)
    newer = engine.registry.publish(training_result)
    assert newer["model_id"] != trained_model_id
    assert newer["status"] == "ready"
    clock[0] = 1
    result = engine.advance(sid, "a", 0)
    assert repo.get(sid)["model_id"] == trained_model_id
    assert result["row"]["prediction"]["model_id"] == trained_model_id


def test_retries_do_not_require_model_but_do_not_extend_lease(
    session_engine, dataset_store, trained_model_id,
):
    engine, repo, clock = session_engine
    sid = start_session(session_engine, dataset_store, trained_model_id)
    clock[0] = 1
    first = engine.advance(sid, "a", 0)["row"]
    (engine.registry.model_dir / trained_model_id / "model.joblib").unlink()
    clock[0] = 3.9
    assert engine.advance(sid, "a", 0)["row"] == first
    clock[0] = 4
    engine.expire_leases()
    assert repo.get(sid)["status"] == "paused"
    assert repo.rows(sid, 0, 50)["total"] == 1


@pytest.mark.parametrize("kind", ["published", "missing_pointer", "corrupt:manifest.json"])
def test_start_rejects_changed_disk_source_without_replacing_cached_store(
    session_engine, dataset_store, trained_model_id, change_session_source, kind,
):
    engine, repo, _ = session_engine
    sid = create_session(engine, dataset_store, trained_model_id)
    before = repo.get(sid)
    change_session_source(dataset_store, kind)
    with pytest.raises(DomainError) as caught:
        engine.control(sid, "a", "start")
    assert caught.value.status == 503
    assert caught.value.code == ("DATASET_MISMATCH" if kind == "published" else "DATA_UNAVAILABLE")
    assert repo.get(sid) == before


def test_pointer_checks_keep_cached_arrays_without_reloading_snapshot(
    session_engine, dataset_store, trained_model_id, monkeypatch,
):
    engine, repo, clock = session_engine
    sid = create_session(engine, dataset_store, trained_model_id)

    def forbidden_reload(*_args, **_kwargs):
        pytest.fail("playback must not reopen or decompress the full dataset")

    monkeypatch.setattr("motionsense_app.data.store.read_snapshot", forbidden_reload)
    monkeypatch.setattr("motionsense_app.data.store.np.load", forbidden_reload)
    engine.control(sid, "a", "start")
    for now in (0, 0.25, 0.5, 0.75):
        clock[0] = now
        assert engine.advance(sid, "a", 0)["row"] is None
    clock[0] = 1
    assert engine.advance(sid, "a", 0)["row"]["sample_id"] == "test:000001"
    engine.control(sid, "a", "pause")
    engine.control(sid, "a", "resume")
    clock[0] = 2
    assert engine.advance(sid, "a", 1)["row"]["sample_id"] == "test:000002"
    assert repo.rows(sid, 0, 50)["total"] == 2


@pytest.mark.parametrize("case", [
    "start", "start-repeat", "resume", "resume-repeat", "pause", "pause-repeat",
    "finish", "finish-repeat",
])
def test_optional_speed_is_validated_before_any_engine_control_or_noop(
    session_engine, dataset_store, trained_model_id, case,
):
    engine, repo, clock = session_engine
    sid = create_session(engine, dataset_store, trained_model_id)
    if case != "start":
        engine.control(sid, "a", "start")
    if case in {"resume", "pause-repeat"}:
        engine.control(sid, "a", "pause")
    if case == "finish-repeat":
        engine.control(sid, "a", "finish")
    before = repo.get(sid)
    clock[0] = 0.25
    action = case.split("-", 1)[0]
    for speed in (0, 3, 1.5, 1.0000000000000002, True, False, "1", float("nan")):
        with pytest.raises(DomainError) as caught:
            engine.control(sid, "a", action, speed)
        assert caught.value.status == 422
        assert caught.value.code == "VALIDATION_ERROR"
        assert repo.get(sid) == before
    clock[0] = 0.99
    assert engine.advance(sid, "a", 0)["row"] is None
    clock[0] = 1
    result = engine.advance(sid, "a", 0)
    if before["status"] == "running":
        assert result["session"]["cursor"] == 1
    else:
        assert result["row"] is None
        assert repo.get(sid) == before


@pytest.mark.parametrize("actor,completion_time", [("attach", 2.5), ("watchdog", 4.25)])
def test_attach_or_watchdog_waits_behind_inflight_advance_and_keeps_one_committed_row(
    session_engine, dataset_store, trained_model_id, monkeypatch, actor, completion_time,
):
    engine, repo, clock = session_engine
    sid = start_session(session_engine, dataset_store, trained_model_id)
    entered = threading.Event()
    release = threading.Event()
    blocked = threading.Event()
    read_sample = dataset_store.sample
    real_lock = engine._lock

    class ObservedLock:
        """Observe actual contention without replacing the engine's real mutex."""
        def __enter__(self):
            if not real_lock.acquire(blocking=False):
                blocked.set()
                real_lock.acquire()
            return self

        def __exit__(self, *_exc):
            real_lock.release()

    def slow_sample(sample_id):
        entered.set()
        assert release.wait(timeout=5)
        return read_sample(sample_id)

    monkeypatch.setattr(engine, "_lock", ObservedLock())
    monkeypatch.setattr(dataset_store, "sample", slow_sample)
    clock[0] = 1
    with ThreadPoolExecutor(max_workers=2) as pool:
        processing = pool.submit(engine.advance, sid, "a", 0)
        try:
            assert entered.wait(timeout=3)
            clock[0] = completion_time
            pending = (pool.submit(engine.attach, sid, "b") if actor == "attach"
                       else pool.submit(engine.expire_leases))
            assert blocked.wait(timeout=2)  # contender actually failed to acquire the mutex
            assert not pending.done()
            assert repo.get(sid)["cursor"] == repo.rows(sid, 0, 50)["total"] == 0
        finally:
            release.set()
        committed = processing.result(timeout=5)
        after = pending.result(timeout=5)
    monkeypatch.setattr(dataset_store, "sample", read_sample)
    assert committed["session"]["status"] == "running"
    assert committed["session"]["cursor"] == 1
    assert committed["row"]["ordinal"] == 0
    assert committed["row"]["sample_id"] == "test:000001"
    assert repo.rows(sid, 0, 50)["items"] == [committed["row"]]
    assert repo.get(sid)["cursor"] == 1
    assert repo.get(sid)["status"] == "paused"
    assert repo.get(sid)["last_prediction"] == committed["row"]["prediction"]
    if actor == "attach":
        assert after["status"] == "paused" and after["cursor"] == 1
        for cursor in (0, 1, 2):
            assert_conflict(lambda cursor=cursor: engine.advance(sid, "a", cursor), "CLIENT_CONFLICT")
        assert_conflict(lambda: engine.control(sid, "a", "resume"), "CLIENT_CONFLICT")
    else:
        assert after is None
    owner = "b" if actor == "attach" else "a"
    assert engine.advance(sid, owner, 0)["row"] == committed["row"]
    assert engine.advance(sid, owner, 1)["row"] is None
    assert_conflict(lambda: engine.advance(sid, owner, 2), "CURSOR_CONFLICT")
    assert repo.rows(sid, 0, 50)["total"] == 1
    engine.control(sid, owner, "resume")
    clock[0] = completion_time + 0.99
    assert engine.advance(sid, owner, 1)["row"] is None
    clock[0] = completion_time + 1
    assert engine.advance(sid, owner, 1)["row"]["ordinal"] == 1
    assert repo.get(sid)["cursor"] == repo.rows(sid, 0, 50)["total"] == 2


def test_next_due_uses_clock_after_inflight_prediction_without_catchup(
    session_engine, dataset_store, trained_model_id, monkeypatch,
):
    engine, repo, clock = session_engine
    sid = start_session(session_engine, dataset_store, trained_model_id)
    entered = threading.Event()
    release = threading.Event()
    read_sample = dataset_store.sample

    def slow_sample(sample_id):
        entered.set()
        assert release.wait(timeout=5)
        return read_sample(sample_id)

    monkeypatch.setattr(dataset_store, "sample", slow_sample)
    clock[0] = 1
    with ThreadPoolExecutor(max_workers=1) as pool:
        processing = pool.submit(engine.advance, sid, "a", 0)
        try:
            assert entered.wait(timeout=3)
            clock[0] = 2.5  # due would be 2.0 if based on request-entry clock
        finally:
            release.set()
        assert processing.result(timeout=5)["session"]["cursor"] == 1
    monkeypatch.setattr(dataset_store, "sample", read_sample)
    assert engine.advance(sid, "a", 1)["row"] is None
    clock[0] = 3.49
    assert engine.advance(sid, "a", 1)["row"] is None
    clock[0] = 3.5
    assert engine.advance(sid, "a", 1)["row"]["ordinal"] == 1
    assert repo.get(sid)["cursor"] == repo.rows(sid, 0, 50)["total"] == 2
