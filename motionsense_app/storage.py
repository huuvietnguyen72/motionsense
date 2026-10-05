"""Versioned SQLite storage; connections never cross thread boundaries."""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from motionsense_app.errors import DomainError

logger = logging.getLogger(__name__)

SCHEMA_V1 = (
    """CREATE TABLE sessions (
        session_id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, split TEXT NOT NULL,
        subject_id INTEGER NOT NULL, model_id TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('ready','running','paused','finished')),
        cursor INTEGER NOT NULL CHECK(cursor >= 0), total INTEGER NOT NULL CHECK(total > 0),
        speed REAL NOT NULL CHECK(speed IN (0.5,1,2)),
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        finish_reason TEXT CHECK(finish_reason IN ('complete','user')),
        sample_ids_json TEXT NOT NULL, client_id TEXT, last_prediction_json TEXT,
        CHECK(cursor <= total)
    )""",
    """CREATE TABLE session_rows (
        session_id TEXT NOT NULL REFERENCES sessions(session_id),
        ordinal INTEGER NOT NULL CHECK(ordinal >= 0), sample_id TEXT NOT NULL,
        prediction_json TEXT NOT NULL, processed_at TEXT NOT NULL
    )""",
    "CREATE UNIQUE INDEX one_sample_per_session ON session_rows(session_id,sample_id)",
    "CREATE UNIQUE INDEX one_ordinal_per_session ON session_rows(session_id,ordinal)",
    "CREATE UNIQUE INDEX one_running_session ON sessions(status) WHERE status='running'",
)

SCHEMA_V2 = (
    """CREATE TABLE jobs (
        job_id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL,
        profile TEXT NOT NULL CHECK(profile IN ('full','reduced')),
        trees INTEGER NOT NULL CHECK(trees IN (25,50,100,150)),
        status TEXT NOT NULL CHECK(status IN
            ('queued','running','succeeded','failed','interrupted')),
        stage TEXT NOT NULL, model_id TEXT, error_json TEXT,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    )""",
    ("CREATE UNIQUE INDEX one_active_training_job ON jobs((1)) "
     "WHERE status IN ('queued','running')"),
)


@contextmanager
def transaction(db_path: Path, *, write: bool = False):
    db = None
    try:
        db = sqlite3.connect(db_path, timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=5000")
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        yield db
        db.commit()
    except (sqlite3.Error, OSError) as exc:
        logger.exception("Session storage operation failed")
        raise DomainError(
            "STORAGE_ERROR", "Không thể đọc hoặc ghi nhật ký phiên. Hãy kiểm tra ổ đĩa và thử lại.",
            500,
        ) from exc
    finally:
        if db is not None:
            # close rolls back any uncommitted transaction, including domain exceptions.
            db.close()


def initialize(db_path: Path) -> None:
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DomainError("STORAGE_ERROR", "Không thể tạo thư mục nhật ký phiên.", 500) from exc
    with transaction(db_path, write=True) as db:
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            for statement in SCHEMA_V1:
                db.execute(statement)
            db.execute("PRAGMA user_version=1")
            version = 1
        if version == 1:
            for statement in SCHEMA_V2:
                db.execute(statement)
            db.execute("PRAGMA user_version=2")
        elif version != 2:
            raise DomainError("STORAGE_VERSION", "Phiên bản nhật ký không được hỗ trợ.", 500)
