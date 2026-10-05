import json
import lzma
import re
import stat
import zipfile
import zlib
from pathlib import Path

import numpy as np

from motionsense_app.data.schema import (
    ACTIVITY_NAMES,
    DOI,
    LICENSE,
    SCHEMA_VERSION,
    SOURCE_URL,
    validate_bundle,
)
from motionsense_app.errors import DomainError

SNAPSHOT_FILES = ("manifest.json", "features.json", "train.npz", "test.npz")


def _pointer_object(pairs: list[tuple]) -> dict:
    if len(pairs) != 1 or pairs[0][0] != "dataset_id":
        raise ValueError("invalid or duplicate current dataset keys")
    return dict(pairs)


def _snapshot_signature(dataset_dir: Path) -> tuple:
    signature = []
    for name in SNAPSHOT_FILES:
        saved = (dataset_dir / name).stat()
        if not stat.S_ISREG(saved.st_mode):
            raise ValueError("snapshot entry is not a file")
        signature.append((saved.st_dev, saved.st_ino, saved.st_size,
                          saved.st_mtime_ns, saved.st_ctime_ns))
    return tuple(signature)


def read_snapshot(dataset_dir: Path, dataset_id: str) -> tuple[dict, list[dict], dict]:
    """Reopen and validate every saved file before accepting a snapshot."""
    try:
        manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
        features = json.loads((dataset_dir / "features.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or (
            manifest.get("dataset_id") != dataset_id
            or manifest.get("source_sha256") != dataset_id
            or manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("feature_count") != 561
            or manifest.get("source_url") != SOURCE_URL
            or manifest.get("doi") != DOI
            or manifest.get("license") != LICENSE
        ):
            raise ValueError("invalid manifest")
        labels = manifest["labels"]
        if not isinstance(labels, dict) or set(labels) != {str(i) for i in range(1, 7)}:
            raise ValueError("invalid label mapping")
        loaded = {}
        for split in ("train", "test"):
            with np.load(dataset_dir / f"{split}.npz", allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in ("X", "y", "subjects", "total_acc")}
            if (
                arrays["X"].dtype != np.float64
                or arrays["total_acc"].dtype != np.float32
                or arrays["y"].dtype != np.int64
                or arrays["subjects"].dtype != np.int64
            ):
                raise ValueError("invalid array dtype")
            loaded[split] = arrays
        bundle = {"features": features, "labels": {int(k): v for k, v in labels.items()}, **loaded}
        counts = validate_bundle(bundle)
        subject_count = len(set(loaded["train"]["subjects"]) | set(loaded["test"]["subjects"]))
        if manifest.get("split_counts") != counts or manifest.get("subject_count") != subject_count:
            raise ValueError("invalid manifest counts")
        return manifest, features, loaded
    except (
        ValueError, KeyError, TypeError, EOFError, RuntimeError,
        zipfile.BadZipFile, zlib.error, lzma.LZMAError,
    ) as exc:
        raise DomainError("DATA_INVALID", "Tệp dữ liệu đã lưu không hợp lệ.") from exc


class DatasetStore:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self._dataset_dir: Path | None = None
        self._manifest: dict | None = None
        self._features: list[dict] | None = None
        self._cache: dict[str, dict[str, np.ndarray]] = {}
        self._loaded_signature: tuple | None = None
        self._load()

    def _read_current_id(self) -> str:
        try:
            pointer = json.loads((self.data_dir / "current.json").read_text(encoding="utf-8"),
                                 object_pairs_hook=_pointer_object)
            if not isinstance(pointer, dict):
                raise TypeError("invalid current dataset pointer")
            dataset_id = pointer["dataset_id"]
            if not isinstance(dataset_id, str) or not re.fullmatch(r"[0-9a-f]{64}", dataset_id):
                raise ValueError("invalid current dataset identity")
            return dataset_id
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise DomainError(
                "DATA_UNAVAILABLE", "Không thể đọc nguồn dữ liệu hiện tại. Hãy chạy Cai_dat.bat.", 503,
            ) from exc

    def _load(self) -> None:
        try:
            dataset_id = self._read_current_id()
            dataset_dir = self.data_dir / dataset_id
            signature = _snapshot_signature(dataset_dir)
            manifest, features, loaded = read_snapshot(dataset_dir, dataset_id)
            if _snapshot_signature(dataset_dir) != signature:
                raise ValueError("snapshot changed during validation")
            self._dataset_dir = dataset_dir
            self._manifest = manifest
            self._features = features
            self._cache = loaded
            self._loaded_signature = signature
        except (OSError, ValueError, KeyError, TypeError, DomainError):
            return

    def require_current_snapshot(self, dataset_id: str) -> None:
        """Bind cached arrays to the live pointer and immutable, validated disk files.

        Only the small pointer and file metadata are read; no array decompression or
        dataset reload occurs on playback requests. Changed snapshots require setup
        and a newly validated store rather than silently reinterpreting cached data.
        """
        if self._manifest is None or self._dataset_dir is None:
            raise DomainError("DATA_UNAVAILABLE", "Dữ liệu chưa sẵn sàng. Hãy chạy Cai_dat.bat.", 503)
        current_id = self._read_current_id()
        if current_id != dataset_id or self._manifest["dataset_id"] != dataset_id:
            raise DomainError(
                "DATASET_MISMATCH", "Phiên không thuộc dữ liệu hiện tại; hãy chọn lại dữ liệu.", 503,
            )
        try:
            if _snapshot_signature(self._dataset_dir) != self._loaded_signature:
                raise ValueError("validated snapshot files changed")
        except (OSError, ValueError) as exc:
            raise DomainError(
                "DATA_UNAVAILABLE", "Dữ liệu của phiên thiếu hoặc đã thay đổi. Hãy chạy Cai_dat.bat.",
                503,
            ) from exc

    def info(self) -> dict:
        if self._manifest is None:
            return {
                "dataset_id": None,
                "ready": False,
                "source_url": SOURCE_URL,
                "license": LICENSE,
                "feature_count": 0,
                "split_counts": {"train": 0, "test": 0},
                "subjects": [],
                "activities": [],
            }
        assert self._manifest is not None
        subjects = []
        activity_counts = {label_id: 0 for label_id in range(1, 7)}
        for split in ("train", "test"):
            arrays = self._cache[split]
            subject_ids, counts = np.unique(arrays["subjects"], return_counts=True)
            subjects.extend(
                {"subject_id": int(subject_id), "split": split, "sample_count": int(count)}
                for subject_id, count in zip(subject_ids, counts)
            )
            values, label_counts = np.unique(arrays["y"], return_counts=True)
            for label_id, count in zip(values, label_counts):
                activity_counts[int(label_id)] += int(count)
        return {
            "dataset_id": self._manifest["dataset_id"],
            "ready": True,
            "source_url": self._manifest["source_url"],
            "license": self._manifest["license"],
            "feature_count": self._manifest["feature_count"],
            "split_counts": self._manifest["split_counts"],
            "subjects": subjects,
            "activities": [
                {"label_id": label_id, "name_vi": ACTIVITY_NAMES[label_id], "count": activity_counts[label_id]}
                for label_id in range(1, 7)
            ],
        }

    def features(self) -> list[dict]:
        if self._features is None:
            return []
        return [dict(feature) for feature in self._features]

    def arrays(self, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if split not in {"train", "test"}:
            raise DomainError("INVALID_SPLIT", "Tập dữ liệu phải là train hoặc test.")
        if split not in self._cache:
            raise DomainError("DATA_UNAVAILABLE", "Dữ liệu chưa sẵn sàng.", status=503)
        data = self._cache[split]
        return data["X"], data["y"], data["subjects"]

    def sample(self, sample_id: str) -> dict:
        match = re.fullmatch(r"(train|test):([0-9]{6})", sample_id)
        if match is None:
            raise DomainError("SAMPLE_NOT_FOUND", "Không tìm thấy mẫu dữ liệu.", status=404)
        split, ordinal = match.group(1), int(match.group(2)) - 1
        if split not in self._cache:
            raise DomainError("DATA_UNAVAILABLE", "Dữ liệu chưa sẵn sàng.", status=503)
        if ordinal < 0 or ordinal >= len(self._cache[split]["y"]):
            raise DomainError("SAMPLE_NOT_FOUND", "Không tìm thấy mẫu dữ liệu.", status=404)
        data = self._cache[split]
        signal = data["total_acc"][ordinal]
        return {
            "sample_id": sample_id,
            "dataset_id": self._manifest["dataset_id"],
            "split": split,
            "subject_id": int(data["subjects"][ordinal]),
            "features": {
                feature["feature_id"]: float(value)
                for feature, value in zip(self._features, data["X"][ordinal])
            },
            "actual_label": int(data["y"][ordinal]),
            "signal": {
                "time_seconds": (np.arange(128, dtype=np.float64) / 50.0).tolist(),
                "x": signal[:, 0].astype(float).tolist(),
                "y": signal[:, 1].astype(float).tolist(),
                "z": signal[:, 2].astype(float).tolist(),
                "unit": "g",
            },
        }

    def sample_ids(self, split: str, subject_id: int) -> list[str]:
        if split not in {"train", "test"}:
            raise DomainError("INVALID_SPLIT", "Tập dữ liệu phải là train hoặc test.")
        if split not in self._cache:
            raise DomainError("DATA_UNAVAILABLE", "Dữ liệu chưa sẵn sàng.", status=503)
        indexes = np.flatnonzero(self._cache[split]["subjects"] == subject_id)
        return [f"{split}:{index + 1:06d}" for index in indexes]
