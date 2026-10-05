import hashlib
import http.client
import json
import lzma
import os
import shutil
import ssl
import tempfile
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
from uuid import uuid4

import numpy as np

from motionsense_app.data.schema import DOI, LICENSE, SCHEMA_VERSION, SOURCE_URL
from motionsense_app.data.schema import validate_bundle as _validate_bundle
from motionsense_app.data.store import read_snapshot
from motionsense_app.errors import DomainError


def _invalid(message: str) -> DomainError:
    return DomainError("DATA_INVALID", f"Dữ liệu UCI HAR không hợp lệ: {message}")


def _read_matrix(path: Path, *, dtype: type = np.float64) -> np.ndarray:
    try:
        values = np.loadtxt(path, dtype=dtype, ndmin=2)
    except (OSError, ValueError) as exc:
        raise _invalid(f"không đọc được {path.name}.") from exc
    if not np.isfinite(values).all():
        raise _invalid(f"{path.name} phải chứa giá trị hữu hạn.")
    return values


def _read_vector(path: Path, *, integer: bool = False) -> np.ndarray:
    try:
        values = np.loadtxt(path, dtype=np.float64, ndmin=1)
    except (OSError, ValueError) as exc:
        raise _invalid(f"không đọc được {path.name}.") from exc
    if values.ndim != 1 or not np.isfinite(values).all():
        raise _invalid(f"{path.name} phải là vector hữu hạn.")
    if integer and not np.equal(values, np.floor(values)).all():
        subject_word = "người tham gia" if path.name.startswith("subject_") else "nhãn"
        raise _invalid(f"{subject_word} phải là số nguyên.")
    return values.astype(np.int64) if integer else values


def load_uci(root: Path) -> dict:
    root = Path(root)
    if root.name != "UCI HAR Dataset":
        candidate = root / "UCI HAR Dataset"
        if candidate.is_dir():
            root = candidate
    try:
        feature_lines = (root / "features.txt").read_text(encoding="utf-8").splitlines()
        label_lines = (root / "activity_labels.txt").read_text(encoding="utf-8").splitlines()
        features_source = [line.split(maxsplit=1) for line in feature_lines]
        labels_source = [line.split(maxsplit=1) for line in label_lines]
        if any(len(row) != 2 for row in features_source + labels_source):
            raise ValueError("malformed metadata")
        if [int(row[0]) for row in features_source] != list(range(1, 562)):
            raise ValueError("feature indices do not match original columns")
        if len(labels_source) != 6:
            raise ValueError("duplicate or missing labels")
        features = [{"feature_id": f"f{i:03d}", "name": row[1]} for i, row in enumerate(features_source, 1)]
        labels = {int(row[0]): row[1] for row in labels_source}
    except (OSError, UnicodeError, ValueError) as exc:
        raise _invalid("thiếu hoặc sai định dạng tệp đặc trưng/nhãn.") from exc
    if len(features) != 561:
        raise _invalid("phải có đúng 561 đặc trưng.")
    if set(labels) != set(range(1, 7)):
        raise _invalid("activity_labels.txt phải định nghĩa đủ nhãn 1–6.")

    result = {"features": features, "labels": labels}
    all_subjects = {}
    for split in ("train", "test"):
        folder = root / split
        X = _read_matrix(folder / f"X_{split}.txt")
        y = _read_vector(folder / f"y_{split}.txt", integer=True)
        subjects = _read_vector(folder / f"subject_{split}.txt", integer=True)
        if X.shape[1] != 561:
            raise _invalid(f"{split}: mỗi dòng phải có 561 đặc trưng.")
        if np.any((y < 1) | (y > 6)):
            raise _invalid(f"{split}: nhãn phải thuộc khoảng 1–6.")
        if np.any(subjects <= 0):
            raise _invalid(f"{split}: mã người tham gia phải là số nguyên dương.")
        signal_axes = []
        for axis in "xyz":
            signal = _read_matrix(
                folder / "Inertial Signals" / f"total_acc_{axis}_{split}.txt",
                dtype=np.float32,
            )
            if signal.shape[1] != 128:
                raise _invalid(f"{split}: mỗi trục tín hiệu phải có đúng 128 điểm.")
            signal_axes.append(signal)
        rows = {len(X), len(y), len(subjects), *(len(signal) for signal in signal_axes)}
        if len(rows) != 1:
            raise _invalid(f"{split}: số dòng của đặc trưng, nhãn, người và tín hiệu phải bằng nhau.")
        result[split] = {
            "X": X.astype(np.float64, copy=False),
            "y": y,
            "subjects": subjects,
            "total_acc": np.stack(signal_axes, axis=2).astype(np.float32, copy=False),
        }
        all_subjects[split] = set(subjects.tolist())
    overlap = all_subjects["train"] & all_subjects["test"]
    if overlap:
        raise DomainError(
            "SUBJECT_OVERLAP",
            "Hai tập dữ liệu có người tham gia trùng nhau.",
        )
    _validate_bundle(result)
    return result


def publish_dataset(bundle: dict, target: Path, source_sha256: str) -> str:
    counts = _validate_bundle(bundle)
    if len(source_sha256) != 64 or any(char not in "0123456789abcdef" for char in source_sha256.lower()):
        raise _invalid("SHA-256 nguồn không hợp lệ.")
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    dataset_id = source_sha256.lower()
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=target))
    pointer_tmp = target / f".current-{uuid4().hex}.tmp"
    try:
        for split in ("train", "test"):
            data = bundle[split]
            np.savez_compressed(
                staging / f"{split}.npz",
                X=np.asarray(data["X"], dtype=np.float64),
                y=np.asarray(data["y"], dtype=np.int64),
                subjects=np.asarray(data["subjects"], dtype=np.int64),
                total_acc=np.asarray(data["total_acc"], dtype=np.float32),
            )
        features_path = staging / "features.json"
        features_path.write_text(json.dumps(bundle["features"], ensure_ascii=False), encoding="utf-8")
        manifest = {
            "dataset_id": dataset_id,
            "source_url": SOURCE_URL,
            "doi": DOI,
            "license": LICENSE,
            "source_sha256": dataset_id,
            "schema_version": SCHEMA_VERSION,
            "feature_count": 561,
            "split_counts": counts,
            "subject_count": len(set(bundle["train"]["subjects"].tolist()) | set(bundle["test"]["subjects"].tolist())),
            "labels": {str(key): value for key, value in sorted(bundle["labels"].items())},
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        read_snapshot(staging, dataset_id)
        final_dir = target / dataset_id
        if final_dir.exists():
            # Reusing a hash must never promote a corrupt, previously saved snapshot.
            read_snapshot(final_dir, dataset_id)
        else:
            os.replace(staging, final_dir)
        pointer_tmp.write_text(json.dumps({"dataset_id": dataset_id}), encoding="utf-8")
        os.replace(pointer_tmp, target / "current.json")
        return dataset_id
    finally:
        pointer_tmp.unlink(missing_ok=True)
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _safe_member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name.replace("\\", "/"))
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"..", ""} or ":" in part or part.endswith((".", " ")) for part in path.parts)
    ):
        raise _invalid("đường dẫn trong ZIP thoát khỏi thư mục staging.")
    return path


def _extract_archive(archive: Path, staging: Path) -> Path:
    try:
        with zipfile.ZipFile(archive) as outer:
            files = outer.infolist()
            for member in files:
                _safe_member_path(member.filename)
            nested = [member for member in files if not member.is_dir() and member.filename.lower().endswith(".zip")]
            if nested:
                for member in nested:
                    content = outer.read(member)
                    inner_path = staging / Path(*_safe_member_path(member.filename).parts)
                    inner_path.parent.mkdir(parents=True, exist_ok=True)
                    inner_path.write_bytes(content)
                    with zipfile.ZipFile(inner_path) as inner:
                        for inner_member in inner.infolist():
                            rel = _safe_member_path(inner_member.filename)
                            destination = staging / Path(*rel.parts)
                            if inner_member.is_dir():
                                destination.mkdir(parents=True, exist_ok=True)
                                continue
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            with inner.open(inner_member) as source, destination.open("wb") as output:
                                shutil.copyfileobj(source, output)
                inner_path.unlink(missing_ok=True)
            else:
                outer.extractall(staging)
    except (OSError, zipfile.BadZipFile, zlib.error, lzma.LZMAError, RuntimeError) as exc:
        raise _invalid("archive ZIP bị hỏng hoặc không thể giải nén.") from exc
    roots = [path.parent for path in staging.rglob("features.txt")]
    if len(roots) != 1:
        raise _invalid("không tìm thấy duy nhất thư mục UCI HAR Dataset.")
    return roots[0]


def prepare_dataset(archive: Path, target: Path) -> str:
    archive = Path(archive)
    source_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-extract-", dir=target))
    try:
        root = _extract_archive(archive, staging)
        bundle = load_uci(root)
        counts = _validate_bundle(bundle)
        if sum(counts.values()) != 10299:
            raise _invalid(f"số mẫu production phải là 10.299 (hiện có {sum(counts.values())}).")
        subjects = set(bundle["train"]["subjects"].tolist()) | set(bundle["test"]["subjects"].tolist())
        observed_classes = set(bundle["train"]["y"].tolist()) | set(bundle["test"]["y"].tolist())
        if len(subjects) != 30 or observed_classes != set(range(1, 7)):
            raise _invalid("production phải có đúng 30 người và sáu lớp hoạt động.")
        return publish_dataset(bundle, target, source_hash)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


class _HTTPSOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # urllib resolves Location against the current URL before calling this hook.
        if urlsplit(newurl).scheme.lower() != "https":
            fp.close()
            raise DomainError(
                "HTTPS_REQUIRED",
                "Chuyển hướng tải dữ liệu phải sử dụng HTTPS.",
                status=503,
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_archive(destination: Path, *, attempts: int = 2) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    opener = urllib.request.build_opener(
        _HTTPSOnlyRedirectHandler(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )
    last_error = None
    for _ in range(attempts):
        try:
            request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "MotionSense/1.0"})
            with opener.open(request, timeout=60) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output)
            if not zipfile.is_zipfile(partial):
                raise _invalid("tải về không phải tệp ZIP hợp lệ.")
            os.replace(partial, destination)
            return destination
        except (OSError, urllib.error.URLError, http.client.HTTPException, DomainError) as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
    raise DomainError(
        "DOWNLOAD_FAILED",
        "Không thể tải dữ liệu UCI HAR qua HTTPS. Vui lòng kiểm tra kết nối và thử lại.",
        status=503,
    ) from last_error
