import json
import logging
import threading
import time
from collections.abc import Callable
from uuid import uuid4

from motionsense_app.data.store import DatasetStore
from motionsense_app.errors import DomainError
from motionsense_app.models.prediction import predict_rows
from motionsense_app.models.registry import ModelRegistry
from motionsense_app.sessions import CreateSessionRequest
from motionsense_app.sessions.repository import SessionRepository, public_session, utc_now

logger = logging.getLogger(__name__)


class SessionEngine:
    def __init__(
        self, repo: SessionRepository, store: DatasetStore, registry: ModelRegistry,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.repo = repo
        self.store = store
        self.registry = registry
        self.clock = clock
        self._lock = threading.RLock()
        self._timing: dict[str, tuple[float, float]] = {}

    def _artifact(self, dataset_id: str, model_id: str, *, pinned: bool = True) -> dict:
        self.store.require_current_snapshot(dataset_id)
        try:
            artifact = self.registry.load(model_id)
        except DomainError as exc:
            if pinned and exc.code == "MODEL_NOT_FOUND":
                raise DomainError(
                    "MODEL_UNAVAILABLE", "Mô hình của phiên không còn sẵn sàng. Hãy chạy Cai_dat.bat.", 503,
                ) from exc
            raise
        if artifact["manifest"]["dataset_id"] != dataset_id:
            raise DomainError("DATASET_MISMATCH", "Mô hình không thuộc dữ liệu của phiên.", 503)
        return artifact

    def create(self, request: dict) -> dict:
        body = CreateSessionRequest.model_validate(request).model_dump()
        with self._lock:
            self._artifact(body["dataset_id"], body["model_id"], pinned=False)
            sample_ids = self.store.sample_ids(body["split"], body["subject_id"])
            if not sample_ids:
                raise DomainError("SESSION_EMPTY", "Người được chọn không có mẫu trong tập dữ liệu.", 422)
            session = self.repo.create(
                dict(body, session_id=f"session-{uuid4().hex}", created_at=utc_now()), sample_ids,
            )
            self._timing.clear()
            return session

    def _pause(self, id: str) -> dict:
        try:
            session = self.repo.change(id, status="paused")
        except DomainError:
            # Never infer again after a write failure; retain an expired lease so
            # the watchdog can retry the pause when the disk becomes writable.
            self._timing[id] = (float("inf"), float("-inf"))
            raise
        self._timing.pop(id, None)
        return session

    def advance(self, id: str, client_id: str, expected_cursor: int) -> dict:
        if (isinstance(expected_cursor, bool) or not isinstance(expected_cursor, int)
                or expected_cursor < 0):
            raise DomainError("VALIDATION_ERROR", "Vị trí mẫu phải là số nguyên không âm.", 422)
        with self._lock:
            saved = self.repo._snapshot(id)
            self._check_client(saved, client_id)
            now = self.clock()
            due, lease = self._timing.get(id, (0.0, 0.0))
            if saved["status"] == "running" and now >= lease:
                return {"session": self._pause(id), "row": None}
            if expected_cursor < saved["cursor"]:
                return {"session": public_session(saved),
                        "row": self.repo.row(id, expected_cursor)}
            if expected_cursor > saved["cursor"]:
                raise DomainError("CURSOR_CONFLICT", "Vị trí mẫu không khớp. Hãy đồng bộ lại phiên.", 409)
            if saved["status"] != "running":
                return {"session": public_session(saved), "row": None}
            self._timing[id] = (due, now + 3)
            if now < due:
                return {"session": public_session(saved), "row": None}
            try:
                artifact = self._artifact(saved["dataset_id"], saved["model_id"])
                sample_id = json.loads(saved["sample_ids_json"])[saved["cursor"]]
                sample = self.store.sample(sample_id)
                if sample["dataset_id"] != saved["dataset_id"]:
                    raise DomainError("DATASET_MISMATCH", "Mẫu không thuộc dữ liệu của phiên.", 503)
                prediction = predict_rows(artifact, [sample["features"]])[0]
                prediction.update(sample_id=sample_id, actual_label=sample["actual_label"])
                result = self.repo.commit_row(id, client_id, saved["cursor"], prediction)
            except (DomainError, OSError) as exc:
                logger.exception("Playback advance failed for %s", id)
                try:
                    self._pause(id)
                except DomainError:
                    # Disk may still be unavailable; only retrying the pause is allowed.
                    logger.exception("Could not persist pause for %s", id)
                if isinstance(exc, OSError):
                    raise DomainError(
                        "DATA_UNAVAILABLE", "Không thể đọc dữ liệu phiên. Hãy kiểm tra ổ đĩa.", 503,
                    ) from exc
                raise
            if result["session"]["status"] == "finished":
                self._timing.pop(id, None)
            else:
                self._timing[id] = (self.clock() + 1 / saved["speed"], now + 3)
            return result

    def expire_leases(self) -> None:
        with self._lock:
            now = self.clock()
            for id, (_, lease) in list(self._timing.items()):
                saved = self.repo._snapshot(id)
                if saved["status"] != "running":
                    self._timing.pop(id, None)
                elif now >= lease:
                    self._pause(id)

    def close(self) -> None:
        """Serialize shutdown with any last inference, then persist the stopped state."""
        with self._lock:
            self._timing.clear()
            self.repo.pause_running()

    def _check_client(self, saved: dict, client_id: str) -> None:
        if not client_id or saved["client_id"] != client_id:
            raise DomainError(
                "CLIENT_CONFLICT", "Phiên đã được mở ở cửa sổ khác. Hãy kết nối lại phiên.", 409,
            )

    def attach(self, id: str, client_id: str) -> dict:
        if not isinstance(client_id, str) or not client_id.strip():
            raise DomainError("VALIDATION_ERROR", "Mã cửa sổ không hợp lệ.", 422)
        with self._lock:
            session = self.repo.attach(id, client_id)
            if session["status"] != "running":
                self._timing.pop(id, None)
            return session

    def control(
        self, id: str, client_id: str, action: str, speed: float | None = None,
    ) -> dict:
        if (speed is not None or action == "speed") and (
            isinstance(speed, bool) or speed not in (0.5, 1, 2)
        ):
            raise DomainError("VALIDATION_ERROR", "Tốc độ phải là 0,5×, 1× hoặc 2×.", 422)
        with self._lock:
            saved = self.repo._snapshot(id)
            self._check_client(saved, client_id)
            status = saved["status"]
            if action == "finish" and status == "finished":
                return public_session(saved)
            actions = {"start", "resume", "pause", "finish", "speed"}
            if status == "finished" or action not in actions:
                raise DomainError(
                    "SESSION_STATE_CONFLICT", "Thao tác không phù hợp với trạng thái phiên.", 409,
                )
            if action == "speed":
                if speed == saved["speed"]:
                    return public_session(saved)
                session = self.repo.change(id, speed=speed)
                if status == "running":
                    _, lease = self._timing.get(id, (0.0, 0.0))
                    self._timing[id] = (self.clock() + 1 / speed, lease)
                return session
            target = {"start": "running", "resume": "running", "pause": "paused",
                      "finish": "finished"}[action]
            if status == target:
                return public_session(saved)
            allowed = {"start": {"ready"}, "resume": {"paused"},
                       "pause": {"running"}, "finish": {"ready", "running", "paused"}}
            if status not in allowed[action]:
                raise DomainError(
                    "SESSION_STATE_CONFLICT", "Thao tác không phù hợp với trạng thái phiên.", 409,
                )
            if target == "running":
                self._artifact(saved["dataset_id"], saved["model_id"])
            session = self.repo.change(id, status=target,
                                       finish_reason="user" if action == "finish" else None)
            if target == "running":
                now = self.clock()
                self._timing.clear()
                self._timing[id] = (now + 1 / saved["speed"], now + 3)
            else:
                self._timing.pop(id, None)
            return session
