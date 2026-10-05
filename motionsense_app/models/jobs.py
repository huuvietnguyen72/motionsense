"""Durable, single-active training with thread-local SQLite transactions."""

import json
import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from motionsense_app.data.store import DatasetStore
from motionsense_app.errors import DomainError, api_error
from motionsense_app.models import Contract, ModelId
from motionsense_app.models.registry import ModelRegistry
from motionsense_app.models.training import train_model
from motionsense_app.ownership import ProcessLock
from motionsense_app.storage import initialize, transaction

logger = logging.getLogger(__name__)


class ErrorDetail(Contract):
    row: int | None
    column: str | None
    message: str


class ErrorBody(Contract):
    code: str
    message: str
    details: list[ErrorDetail]


class JobError(Contract):
    error: ErrorBody


class Job(Contract):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed", "interrupted"]
    stage: str
    model_id: ModelId | None
    error: JobError | None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _interrupted_error() -> dict:
    return api_error(
        "TRAINING_INTERRUPTED",
        "Tác vụ bị gián đoạn khi ứng dụng dừng. Bạn có thể huấn luyện lại.",
    )


class TrainingJobs:
    def __init__(
        self, db_path: Path, store: DatasetStore, registry: ModelRegistry,
        trainer: Callable = train_model,
    ):
        self.db_path = Path(db_path)
        self.store = store
        self.registry = registry
        self.trainer = trainer
        initialize(self.db_path)
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="motionsense-training")
        self._closed = False
        self._future: Future | None = None
        self._job_id: str | None = None
        self._ownership = ProcessLock(self.db_path.with_name(self.db_path.name + ".training.lock"))
        self._lifespan_active = False

    def _claim_ownership(self) -> None:
        if self._closed:
            raise DomainError(
                "TRAINING_UNAVAILABLE", "Dịch vụ huấn luyện đang dừng. Hãy mở lại ứng dụng.", 503,
            )
        if not self._ownership.acquire():
            raise DomainError(
                "TRAINING_BUSY", "Ứng dụng đang hoạt động với thư mục trạng thái này.", 409,
            )

    def _require_worker(self, db, job_id: str, status: str) -> None:
        row = db.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if not self._ownership.held or row is None or row["status"] != status:
            raise RuntimeError("training job no longer owns an active state")

    def submit(self, profile: str, trees: int) -> dict:
        if profile not in ("full", "reduced") or type(trees) is not int or trees not in (25, 50, 100, 150):
            raise DomainError(
                "TRAINING_CONFIG", "Chọn cấu hình đầy đủ/rút gọn và 25, 50, 100 hoặc 150 cây.",
            )
        with self._lock:
            if self._closed:
                raise DomainError(
                    "TRAINING_UNAVAILABLE", "Dịch vụ huấn luyện đang dừng. Hãy mở lại ứng dụng.", 503,
                )
            info = self.store.info()
            if not info.get("ready"):
                raise DomainError(
                    "DATA_UNAVAILABLE", "Dữ liệu chưa sẵn sàng. Hãy chạy Cai_dat.bat.", 503,
                )
            dataset_id = info["dataset_id"]
            self.store.require_current_snapshot(dataset_id)
            self._claim_ownership()
            job_id = f"job-{uuid4().hex}"
            timestamp = _now()
            with transaction(self.db_path, write=True) as db:
                if db.execute("SELECT 1 FROM jobs WHERE status IN ('queued','running')").fetchone():
                    raise DomainError(
                        "TRAINING_BUSY", "Đang có tác vụ huấn luyện. Hãy chờ tác vụ hoàn tất.", 409,
                    )
                db.execute(
                    """INSERT INTO jobs(job_id,dataset_id,profile,trees,status,stage,
                       created_at,updated_at) VALUES (?,?,?,?,'queued','queued',?,?)""",
                    (job_id, dataset_id, profile, trees, timestamp, timestamp),
                )
            try:
                self._future = self._executor.submit(self._run, job_id, dataset_id, profile, trees)
                self._job_id = job_id
            except Exception as exc:
                logger.exception("Could not schedule training job %s", job_id)
                self._fail(job_id)
                raise DomainError(
                    "TRAINING_UNAVAILABLE", "Không thể bắt đầu huấn luyện. Hãy thử lại.", 503,
                ) from exc
            return self.get(job_id)

    def get(self, job_id: str) -> dict:
        with transaction(self.db_path) as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise DomainError("JOB_NOT_FOUND", "Không tìm thấy tác vụ huấn luyện.", 404)
        return Job.model_validate({
            "job_id": row["job_id"], "status": row["status"], "stage": row["stage"],
            "model_id": row["model_id"],
            "error": json.loads(row["error_json"]) if row["error_json"] else None,
        }).model_dump()

    def _run(self, job_id: str, dataset_id: str, profile: str, trees: int) -> None:
        try:
            # submit/close cannot race the queued -> running transition.
            with self._lock, transaction(self.db_path, write=True) as db:
                self._require_worker(db, job_id, "queued")
                db.execute("UPDATE jobs SET status='running',updated_at=? WHERE job_id=? AND status='queued'",
                           (_now(), job_id))
            self.store.require_current_snapshot(dataset_id)
            result = self.trainer(
                self.store, profile, trees, lambda stage: self._update_job_stage(job_id, stage),
            )
            if (result["manifest"]["dataset_id"] != dataset_id
                    or result["manifest"]["profile"] != profile
                    or result["manifest"]["trees"] != trees):
                raise ValueError("trainer result differs from submitted snapshot/config")
            self._update_job_stage(job_id, "saving")
            # Lifetime ownership fences recovery; the local lock fences close and
            # callbacks without holding a SQLite write transaction during publication.
            with self._lock:
                with transaction(self.db_path) as db:
                    self._require_worker(db, job_id, "running")
                model = self.registry.publish(result)
                if model["status"] != "ready":
                    raise ValueError("publisher did not return a ready model")
                with transaction(self.db_path, write=True) as db:
                    self._require_worker(db, job_id, "running")
                    db.execute("""UPDATE jobs SET status='succeeded',model_id=?,error_json=NULL,updated_at=?
                                  WHERE job_id=? AND status='running'""",
                               (model["model_id"], _now(), job_id))
        except Exception:
            logger.exception("Training job %s failed", job_id)
            self._fail(job_id)

    def _update_job_stage(self, job_id: str, stage: str) -> None:
        if stage not in ("validating", "selecting_features", "fitting", "evaluating", "saving"):
            raise ValueError("unknown training stage")
        with self._lock, transaction(self.db_path, write=True) as db:
            self._require_worker(db, job_id, "running")
            db.execute("UPDATE jobs SET stage=?,updated_at=? WHERE job_id=? AND status='running'",
                       (stage, _now(), job_id))

    def _fail(self, job_id: str) -> None:
        error = api_error(
            "TRAINING_FAILED", "Không thể hoàn tất huấn luyện. Hãy kiểm tra dữ liệu và thử lại.",
        )
        try:
            # Do not update stage: a callback storage failure must still be recordable.
            with self._lock, transaction(self.db_path, write=True) as db:
                if not self._ownership.held:
                    return
                db.execute("""UPDATE jobs SET status='failed',model_id=NULL,error_json=?,updated_at=?
                              WHERE job_id=? AND status IN ('queued','running')""",
                           (json.dumps(error, ensure_ascii=False), _now(), job_id))
        except Exception:
            logger.exception("Could not persist failure of training job %s", job_id)

    def recover(self) -> None:
        with self._lock:
            self._claim_ownership()
            if self._future is not None and not self._future.done():
                raise DomainError("TRAINING_BUSY", "Tác vụ huấn luyện vẫn đang hoạt động.", 409)
            with transaction(self.db_path, write=True) as db:
                db.execute("""UPDATE jobs SET status='interrupted',model_id=NULL,error_json=?,updated_at=?
                              WHERE status IN ('queued','running')""",
                           (json.dumps(_interrupted_error(), ensure_ascii=False), _now()))
            self.registry.recover_staging()

    def start_lifecycle(self) -> None:
        """Claim exclusive app startup before session recovery or a watchdog exists."""
        with self._lock:
            if self._lifespan_active:
                raise DomainError("TRAINING_BUSY", "Ứng dụng vẫn đang hoạt động.", 409)
            self._claim_ownership()
            if self._future is not None and not self._future.done():
                raise DomainError("TRAINING_BUSY", "Tác vụ huấn luyện vẫn đang hoạt động.", 409)
            self._lifespan_active = True

    def close(self) -> None:
        try:
            with self._lock:
                self._closed = True
                if self._future is not None and self._future.cancel():
                    with transaction(self.db_path, write=True) as db:
                        db.execute("""UPDATE jobs SET status='interrupted',error_json=?,updated_at=?
                                      WHERE job_id=? AND status='queued'""",
                                   (json.dumps(_interrupted_error(), ensure_ascii=False),
                                    _now(), self._job_id))
        finally:
            # Running trainers finish (Python threads cannot be forcibly stopped).
            try:
                self._executor.shutdown(wait=True, cancel_futures=True)
            finally:
                with self._lock:
                    self._ownership.release()
                    self._lifespan_active = False
