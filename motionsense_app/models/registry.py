import copy
import hashlib
import json
import re
import shutil
import warnings
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning

from motionsense_app.errors import DomainError
from motionsense_app.models import ModelInfo, ModelManifest, ModelReport, runtime_versions
from motionsense_app.models.prediction import predict_rows, validate_artifact
from motionsense_app.ownership import ProcessLock


def _checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate_report(artifact: dict) -> None:
    report = ModelReport.model_validate(artifact["report"])
    manifest = validate_artifact(artifact)
    if (report.model_id != manifest.model_id
            or [f.feature_id for f in report.feature_importance] != manifest.feature_ids
            or not np.array_equal([f.value for f in report.feature_importance],
                                  artifact["estimator"].feature_importances_)):
        raise ValueError("report differs from estimator")


class ModelRegistry:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)

    def _path(self, model_id: str) -> Path:
        if not isinstance(model_id, str) or not re.fullmatch(r"model-[0-9a-f]{32}", model_id):
            raise DomainError("MODEL_NOT_FOUND", "Không tìm thấy phiên bản mô hình.", 404)
        path = self.model_dir / model_id
        if not path.is_dir() or path.is_symlink():
            raise DomainError("MODEL_NOT_FOUND", "Không tìm thấy phiên bản mô hình.", 404)
        return path

    def _read(self, path: Path, model_id: str) -> dict:
        try:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            parsed = ModelManifest.model_validate(manifest)
            if parsed.model_id != model_id or parsed.status != "ready":
                raise ValueError("invalid model identity")
            runtime = runtime_versions()
            if any(parsed.versions[k] != runtime[k] for k in ("sklearn", "numpy", "scipy")):
                raise DomainError("MODEL_UNAVAILABLE", "Phiên bản thư viện đã thay đổi; hãy huấn luyện lại.", 503)
            if set(parsed.file_sha256) != {"model.joblib", "report.json"}:
                raise ValueError("missing artifact checksums")
            for name, digest in parsed.file_sha256.items():
                if (path / name).is_symlink() or _checksum(path / name) != digest:
                    raise ValueError("corrupt model file")
            report = json.loads((path / "report.json").read_text(encoding="utf-8"))
            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                artifact = joblib.load(path / "model.joblib")
            bound_manifest = dict(manifest, file_sha256={})
            if artifact["manifest"] != bound_manifest or artifact["report"] != report:
                raise ValueError("serialized metadata differs from JSON")
            artifact["manifest"] = manifest
            _validate_report(artifact)
            rows = artifact["verification_rows"]
            if not rows or predict_rows(artifact, rows) != artifact["verification_predictions"]:
                raise ValueError("saved verification predictions differ")
            return artifact
        except DomainError:
            raise
        except Exception as exc:
            # Deserialization can raise many exception types for corrupt internal artifacts.
            raise DomainError("MODEL_UNAVAILABLE", "Tệp mô hình thiếu hoặc hỏng; hãy huấn luyện lại.", 503) from exc

    def publish(self, result: dict) -> dict:
        # Each call has its own OS handle; separate threads/registries/processes
        # cannot mistake another publisher's ownership for their own.
        ownership = ProcessLock(self.model_dir / ".publication.lock")
        if not ownership.acquire():
            raise DomainError(
                "MODEL_SAVE_BUSY", "Đang lưu một phiên bản mô hình. Hãy thử lại sau.", 503,
            )
        try:
            return self._publish(result)
        finally:
            ownership.release()

    def recover_staging(self) -> None:
        """Reclaim publisher-owned staging only when no live publisher holds the lock.

        Called after database lifetime ownership is acquired at safe startup.
        The registry-wide lock also protects publishers using another DB or the CLI.
        No age/PID heuristic and no deletion of ready models, links, or unknown names.
        """
        if not self.model_dir.is_dir():
            return
        ownership = ProcessLock(self.model_dir / ".publication.lock")
        if not ownership.acquire():
            return
        try:
            for path in self.model_dir.iterdir():
                if (re.fullmatch(r"\.staging-[0-9a-f]{32}", path.name)
                        and path.is_dir() and not path.is_symlink() and not path.is_junction()):
                    shutil.rmtree(path)
        except OSError as exc:
            raise DomainError(
                "STORAGE_ERROR", "Không thể dọn tệp mô hình bị gián đoạn. Hãy kiểm tra ổ đĩa.", 500,
            ) from exc
        finally:
            ownership.release()

    def _publish(self, result: dict) -> dict:
        staging = None
        try:
            artifact = copy.deepcopy(result)
            manifest = validate_artifact(artifact)
            _validate_report(artifact)
            rows = artifact["verification_rows"]
            if not isinstance(rows, list) or not rows:
                raise ValueError("missing verification rows")
            artifact["verification_predictions"] = predict_rows(artifact, rows)
            artifact["manifest"] = manifest.model_dump()
            artifact["manifest"]["file_sha256"] = {}
            self.model_dir.mkdir(parents=True, exist_ok=True)
            destination = self.model_dir / manifest.model_id
            if destination.exists():
                raise DomainError("MODEL_EXISTS", "Phiên bản mô hình đã tồn tại; hãy huấn luyện phiên bản mới.", 409)
            staging = self.model_dir / f".staging-{uuid4().hex}"
            staging.mkdir()
            joblib.dump(artifact, staging / "model.joblib", compress=3, protocol=5)
            (staging / "report.json").write_text(
                json.dumps(artifact["report"], ensure_ascii=False, allow_nan=False, indent=2),
                encoding="utf-8",
            )
            saved_manifest = dict(artifact["manifest"], file_sha256={
                name: _checksum(staging / name) for name in ("model.joblib", "report.json")
            })
            (staging / "manifest.json").write_text(
                json.dumps(saved_manifest, ensure_ascii=False, allow_nan=False, indent=2),
                encoding="utf-8",
            )
            loaded = self._read(staging, manifest.model_id)
            if predict_rows(loaded, rows) != artifact["verification_predictions"]:
                raise ValueError("reload verification failed")
            # Rename inside the same filesystem is atomic and cannot overwrite a ready directory.
            staging.rename(destination)
            return ModelInfo.model_validate({k: saved_manifest[k] for k in ModelInfo.model_fields}).model_dump()
        except DomainError:
            raise
        except Exception as exc:
            raise DomainError("MODEL_SAVE_FAILED", "Không thể lưu và kiểm chứng mô hình. Hãy thử huấn luyện lại.", 500) from exc
        finally:
            if staging is not None and staging.exists():
                shutil.rmtree(staging)

    def has_ready(self, dataset_id: str) -> bool:
        """Validate compatible candidates lazily; unrelated history needs only metadata reads."""
        if (not isinstance(dataset_id, str) or not re.fullmatch(r"[0-9a-f]{64}", dataset_id)
                or not self.model_dir.is_dir()):
            return False
        for path in self.model_dir.iterdir():
            if (not path.is_dir() or path.is_symlink()
                    or not re.fullmatch(r"model-[0-9a-f]{32}", path.name)):
                continue
            try:
                manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
                if not isinstance(manifest, dict) or manifest.get("dataset_id") != dataset_id:
                    continue
                parsed = ModelManifest.model_validate(manifest)
                if parsed.model_id != path.name or parsed.status != "ready":
                    continue
                # Full load retains hashes, runtime/config/report binding and exact inference checks.
                # Recheck compatibility after loading in case files changed during the metadata read.
                if self.load(path.name)["manifest"]["dataset_id"] == dataset_id:
                    return True
            except (OSError, ValueError, TypeError, KeyError, DomainError):
                continue
        return False

    def list(self) -> list[dict]:
        if not self.model_dir.is_dir():
            return []
        items = []
        for path in self.model_dir.iterdir():
            if not path.is_dir() or not re.fullmatch(r"model-[0-9a-f]{32}", path.name):
                continue
            try:
                manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
                info = ModelInfo.model_validate({k: manifest[k] for k in ModelInfo.model_fields})
                if info.model_id != path.name:
                    continue
            except (OSError, ValueError, TypeError, KeyError):
                continue
            try:
                self.load(path.name)
                info.status = "ready"
            except DomainError:
                info.status = "unavailable"
            items.append(info.model_dump())
        return sorted(items, key=lambda item: (item["created_at"], item["model_id"]))

    def load(self, model_id: str) -> dict:
        return self._read(self._path(model_id), model_id)

    def report(self, model_id: str) -> dict:
        return self.load(model_id)["report"]
