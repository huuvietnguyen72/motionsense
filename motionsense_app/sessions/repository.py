from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from motionsense_app.errors import DomainError
from motionsense_app.storage import initialize, transaction


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def public_session(saved) -> dict:
    result = dict(saved)
    prediction = result.pop("last_prediction_json")
    result["last_prediction"] = json.loads(prediction) if prediction is not None else None
    result.pop("sample_ids_json")
    result.pop("client_id")
    return result


def public_row(saved) -> dict:
    return {"ordinal": saved["ordinal"], "sample_id": saved["sample_id"],
            "prediction": json.loads(saved["prediction_json"]), "processed_at": saved["processed_at"]}


class SessionRepository:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        initialize(self.db_path)

    def _get(self, db, session_id: str):
        saved = db.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if saved is None:
            raise DomainError("SESSION_NOT_FOUND", "Không tìm thấy phiên phát lại.", 404)
        return saved

    def get(self, id: str) -> dict:
        with transaction(self.db_path) as db:
            return public_session(self._get(db, id))

    def list(self, offset: int = 0, limit: int = 50) -> dict:
        with transaction(self.db_path) as db:
            total = db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            items = db.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC,session_id LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return {"items": [public_session(row) for row in items],
                    "offset": offset, "limit": limit, "total": total}

    def create(self, session: dict, sample_ids: list[str]) -> dict:
        with transaction(self.db_path, write=True) as db:
            db.execute("UPDATE sessions SET status='paused',updated_at=? WHERE status='running'",
                       (session["created_at"],))
            db.execute(
                """INSERT INTO sessions (session_id,dataset_id,split,subject_id,model_id,
                   status,cursor,total,speed,created_at,updated_at,finish_reason,
                   sample_ids_json,client_id,last_prediction_json)
                   VALUES (?,?,?,?,?,'ready',0,?,1,?,?,NULL,?,NULL,NULL)""",
                (session["session_id"], session["dataset_id"], session["split"],
                 session["subject_id"], session["model_id"], len(sample_ids),
                 session["created_at"], session["created_at"], json.dumps(sample_ids)),
            )
            return public_session(self._get(db, session["session_id"]))

    def _snapshot(self, id: str) -> dict:
        with transaction(self.db_path) as db:
            return dict(self._get(db, id))

    def attach(self, id: str, client_id: str) -> dict:
        with transaction(self.db_path, write=True) as db:
            saved = self._get(db, id)
            if saved["client_id"] != client_id:
                db.execute(
                    """UPDATE sessions SET client_id=?,updated_at=?,
                       status=CASE WHEN status='running' THEN 'paused' ELSE status END
                       WHERE session_id=?""", (client_id, utc_now(), id),
                )
            return public_session(self._get(db, id))

    def change(
        self, id: str, *, status: str | None = None, speed: float | None = None,
        finish_reason: str | None = None,
    ) -> dict:
        with transaction(self.db_path, write=True) as db:
            saved = self._get(db, id)
            now = utc_now()
            if status == "running":
                db.execute(
                    "UPDATE sessions SET status='paused',updated_at=? WHERE status='running' AND session_id<>?",
                    (now, id),
                )
            db.execute(
                """UPDATE sessions SET status=?,speed=?,finish_reason=?,updated_at=?
                   WHERE session_id=?""",
                (status or saved["status"], speed if speed is not None else saved["speed"],
                 finish_reason, now, id),
            )
            return public_session(self._get(db, id))

    def recover(self) -> None:
        with transaction(self.db_path, write=True) as db:
            db.execute(
                """UPDATE sessions SET status='paused',client_id=NULL,updated_at=?
                   WHERE status IN ('ready','running','paused')""", (utc_now(),),
            )

    def pause_running(self) -> None:
        with transaction(self.db_path, write=True) as db:
            db.execute("UPDATE sessions SET status='paused',updated_at=? WHERE status='running'",
                       (utc_now(),))

    def rows(self, id: str, offset: int = 0, limit: int = 50) -> dict:
        with transaction(self.db_path) as db:
            self._get(db, id)
            total = db.execute("SELECT COUNT(*) FROM session_rows WHERE session_id=?", (id,)).fetchone()[0]
            rows = db.execute(
                "SELECT * FROM session_rows WHERE session_id=? ORDER BY ordinal LIMIT ? OFFSET ?",
                (id, limit, offset),
            ).fetchall()
            return {"items": [public_row(row) for row in rows], "offset": offset,
                    "limit": limit, "total": total}

    def row(self, id: str, ordinal: int) -> dict:
        with transaction(self.db_path) as db:
            self._get(db, id)
            row = db.execute("SELECT * FROM session_rows WHERE session_id=? AND ordinal=?",
                             (id, ordinal)).fetchone()
            if row is None:
                raise DomainError("STORAGE_ERROR", "Nhật ký phiên thiếu mẫu đã xử lý.", 500)
            return public_row(row)

    def commit_row(self, id: str, client_id: str, cursor: int, prediction: dict) -> dict:
        """Insert and advance as one unit; a failed update cannot leave an orphan result."""
        with transaction(self.db_path, write=True) as db:
            saved = self._get(db, id)
            if saved["client_id"] != client_id or saved["cursor"] != cursor or saved["status"] != "running":
                raise DomainError("CURSOR_CONFLICT", "Vị trí phiên đã thay đổi. Hãy đồng bộ lại.", 409)
            sample_id = json.loads(saved["sample_ids_json"])[cursor]
            if prediction["sample_id"] != sample_id or prediction["model_id"] != saved["model_id"]:
                raise DomainError("STORAGE_ERROR", "Kết quả không thuộc mẫu hoặc mô hình của phiên.", 500)
            processed_at = utc_now()
            payload = json.dumps(prediction, ensure_ascii=False, allow_nan=False)
            db.execute("INSERT INTO session_rows VALUES (?,?,?,?,?)",
                       (id, cursor, sample_id, payload, processed_at))
            complete = cursor + 1 == saved["total"]
            updated = db.execute(
                """UPDATE sessions SET cursor=?,last_prediction_json=?,updated_at=?,status=?,finish_reason=?
                   WHERE session_id=? AND cursor=? AND status='running' AND client_id=?""",
                (cursor + 1, payload, processed_at, "finished" if complete else "running",
                 "complete" if complete else None, id, cursor, client_id),
            )
            if updated.rowcount != 1:
                raise DomainError("STORAGE_ERROR", "Không thể lưu vị trí phiên. Hãy thử lại.", 500)
            return {"session": public_session(self._get(db, id)),
                    "row": {"ordinal": cursor, "sample_id": sample_id,
                            "prediction": prediction, "processed_at": processed_at}}

    def export_snapshot(self, id: str) -> tuple[dict, list[dict]]:
        with transaction(self.db_path) as db:
            session = public_session(self._get(db, id))
            rows = db.execute("SELECT * FROM session_rows WHERE session_id=? ORDER BY ordinal",
                              (id,)).fetchall()
            return session, [public_row(row) for row in rows]

    def summary(self, id: str) -> dict:
        _, rows = self.export_snapshot(id)
        counts = {label_id: 0 for label_id in range(1, 7)}
        labeled = correct = 0
        for row in rows:
            prediction = row["prediction"]
            counts[prediction["predicted_label"]] += 1
            if prediction["actual_label"] is not None:
                labeled += 1
                correct += prediction["actual_label"] == prediction["predicted_label"]
        return {"processed": len(rows), "labeled": labeled, "correct": correct,
                "counts": [{"label_id": label_id, "count": count} for label_id, count in counts.items()],
                "session_accuracy": correct / labeled if labeled else None}
