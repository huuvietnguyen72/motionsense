import http.client
import io
import json
import lzma
import os
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
import urllib.response
import zipfile
from email.message import Message
from urllib.parse import urlsplit

import numpy as np
import pytest

from motionsense_app.data.importer import (
    download_archive,
    load_uci,
    prepare_dataset,
    publish_dataset,
)
from motionsense_app.data.store import DatasetStore, read_snapshot
from motionsense_app.errors import DomainError


def _http_response(content, url, *, code=200, location=None):
    headers = Message()
    if location is not None:
        headers["Location"] = location
    stream = content if hasattr(content, "read") else io.BytesIO(content)
    response = urllib.response.addinfourl(stream, headers, url, code)
    response.msg = "OK" if code == 200 else "Redirect"
    return response


def test_duplicate_descriptions_keep_unique_feature_ids_and_signals(uci_root):
    bundle = load_uci(uci_root)
    ids = [feature["feature_id"] for feature in bundle["features"]]
    assert len(set(ids)) == 561
    assert ids[0] == "f001" and ids[-1] == "f561"
    assert bundle["test"]["total_acc"].shape == (48, 128, 3)
    assert bundle["test"]["total_acc"][0, 0, 0] == pytest.approx(0.35)
    assert bundle["test"]["X"].dtype == np.float64
    assert bundle["test"]["total_acc"].dtype == np.float32


def test_overlapping_subjects_are_rejected(uci_root):
    np.savetxt(uci_root / "test" / "subject_test.txt", np.ones(48), fmt="%d")
    with pytest.raises(DomainError, match="người"):
        load_uci(uci_root)


@pytest.mark.parametrize(
    ("relative_path", "contents", "message"),
    [
        ("train/Inertial Signals/total_acc_z_train.txt", None, "total_acc_z_train"),
        ("train/Inertial Signals/total_acc_x_train.txt", np.zeros((48, 127)), "128"),
        ("train/subject_train.txt", np.ones(47), "dòng"),
        ("train/X_train.txt", np.full((48, 561), np.nan), "hữu hạn"),
        ("test/y_test.txt", np.full(48, 7), "nhãn"),
        ("train/subject_train.txt", np.full(48, 0.5), "người"),
        ("train/subject_train.txt", np.zeros(48), "người"),
        ("test/y_test.txt", np.full(48, 1.5), "nhãn"),
        ("train/X_train.txt", np.zeros((48, 560)), "561"),
        ("test/Inertial Signals/total_acc_y_test.txt", np.full((48, 128), np.inf), "hữu hạn"),
    ],
)
def test_loader_rejects_invalid_source_values(uci_root, relative_path, contents, message):
    path = uci_root / relative_path
    if contents is None:
        path.unlink()
    else:
        np.savetxt(path, contents)
    with pytest.raises(DomainError, match=message):
        load_uci(uci_root)


def test_loader_rejects_infinite_feature_values(uci_root):
    values = np.loadtxt(uci_root / "train" / "X_train.txt")
    values[0, 0] = np.inf
    np.savetxt(uci_root / "train" / "X_train.txt", values)
    with pytest.raises(DomainError, match="hữu hạn"):
        load_uci(uci_root)


def test_loader_rejects_unequal_split_row_counts(uci_root):
    path = uci_root / "test" / "subject_test.txt"
    np.savetxt(path, np.array([2] * 47), fmt="%d")
    with pytest.raises(DomainError, match="dòng"):
        load_uci(uci_root)


def test_publish_store_keeps_sample_fields_aligned(uci_root, tmp_path):
    bundle = load_uci(uci_root)
    target = tmp_path / "datasets"
    dataset_id = publish_dataset(bundle, target, "a" * 64)
    store = DatasetStore(target)

    sample = store.sample("test:000001")
    X, y, subjects = store.arrays("test")
    assert sample["features"]["f001"] == X[0, 0]
    assert sample["actual_label"] == y[0] == 1
    assert sample["subject_id"] == subjects[0] == 2
    assert sample["signal"]["x"][0] == pytest.approx(0.35)
    assert sample["signal"]["time_seconds"][-1] == pytest.approx(127 / 50)
    assert sample["signal"]["unit"] == "g"
    assert store.sample_ids("test", 2) == [f"test:{i:06d}" for i in range(1, 49)]
    assert store.info()["dataset_id"] == dataset_id
    assert len(store.features()) == 561
    with pytest.raises(DomainError) as error:
        store.sample("test:000049")
    assert (error.value.code, error.value.status) == ("SAMPLE_NOT_FOUND", 404)
    with pytest.raises(DomainError) as error:
        store.sample("other:000001")
    assert (error.value.code, error.value.status) == ("SAMPLE_NOT_FOUND", 404)


def test_publish_failure_preserves_current_pointer(uci_root, tmp_path):
    bundle = load_uci(uci_root)
    target = tmp_path / "datasets"
    publish_dataset(bundle, target, "a" * 64)
    pointer = (target / "current.json").read_bytes()

    # A real filesystem collision prevents this snapshot from being published.
    (target / ("b" * 64)).write_text("not a dataset directory", encoding="utf-8")
    with pytest.raises(OSError):
        publish_dataset(bundle, target, "b" * 64)
    assert (target / "current.json").read_bytes() == pointer
    assert list(target.glob(".staging-*")) == []


def test_store_reports_corrupt_manifest_not_ready(uci_root, tmp_path):
    target = tmp_path / "datasets"
    dataset_id = publish_dataset(load_uci(uci_root), target, "a" * 64)
    manifest_path = target / dataset_id / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    store = DatasetStore(target)
    assert store.info()["ready"] is False
    with pytest.raises(DomainError):
        store.sample("test:000001")


@pytest.mark.parametrize("member", ["../escape.txt", "UCI HAR Dataset/../../escape.txt"])
def test_prepare_rejects_zip_path_traversal_and_cleans_staging(uci_root, tmp_path, member):
    archive = tmp_path / "malicious.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(member, "bad")
    target = tmp_path / "datasets"
    with pytest.raises(DomainError, match="đường dẫn"):
        prepare_dataset(archive, target)
    assert not (tmp_path / "escape.txt").exists()
    assert list(target.glob(".staging-*")) == [] if target.exists() else True


def test_prepare_supports_nested_zip_archive(uci_root, tmp_path):
    inner = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner, "w") as zf:
        for path in uci_root.rglob("*"):
            if path.is_file():
                zf.write(path, f"UCI HAR Dataset/{path.relative_to(uci_root)}")
    archive = tmp_path / "outer.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(inner, "human+activity+recognition+using+smartphones/UCI HAR Dataset.zip")
    with pytest.raises(DomainError, match="production.*10.299"):
        prepare_dataset(archive, tmp_path / "datasets")
    assert list((tmp_path / "datasets").glob(".staging-*")) == []


def test_store_invalid_pointer_is_not_ready(tmp_path):
    data_dir = tmp_path / "datasets"
    data_dir.mkdir()
    (data_dir / "current.json").write_text(json.dumps({"dataset_id": "../bad"}), encoding="utf-8")
    assert DatasetStore(data_dir).info()["ready"] is False


def test_download_retries_and_atomically_renames_verified_zip(tmp_path, monkeypatch):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("ok.txt", "ok")
    content = payload.getvalue()
    calls = 0

    def open_url(_handler, request):
        nonlocal calls
        calls += 1
        assert request.timeout == 60
        if calls == 1:
            raise urllib.error.URLError("temporary network failure")
        return _http_response(content, request.full_url)

    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", open_url)
    destination = tmp_path / "source" / "uci.zip"
    assert download_archive(destination, attempts=2) == destination
    assert calls == 2
    assert destination.read_bytes() == content
    assert not destination.with_name("uci.zip.part").exists()


def test_failed_download_keeps_existing_archive_and_removes_partial(tmp_path, monkeypatch):
    destination = tmp_path / "uci.zip"
    destination.write_bytes(b"previous archive")

    def fail(_handler, _request):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", fail)
    with pytest.raises(DomainError, match="HTTPS") as error:
        download_archive(destination, attempts=1)
    assert "offline" not in error.value.message
    assert isinstance(error.value.__cause__, urllib.error.URLError)
    assert destination.read_bytes() == b"previous archive"
    assert not destination.with_name("uci.zip.part").exists()


@pytest.mark.parametrize("metadata", ["features", "labels"])
def test_loader_rejects_ambiguous_metadata(uci_root, metadata):
    path = uci_root / ("features.txt" if metadata == "features" else "activity_labels.txt")
    lines = path.read_text(encoding="utf-8").splitlines()
    if metadata == "features":
        lines[0] = "2 wrong-column"
    else:
        lines.append("1 conflicting-label")
    path.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(DomainError):
        load_uci(uci_root)


@pytest.mark.parametrize(
    "corruption",
    [
        "manifest-list", "missing-source", "wrong-hash", "wrong-subject-count",
        "features-list", "duplicate-feature", "bad-label", "fractional-subject",
        "overlap", "wrong-dtype", "wrong-shape", "missing-file", "broken-npz", "bad-compression",
        "encrypted-npz",
    ],
)
def test_store_rejects_corrupt_snapshot_without_crashing(uci_root, tmp_path, corruption):
    target = tmp_path / "datasets"
    dataset_id = publish_dataset(load_uci(uci_root), target, "a" * 64)
    folder = target / dataset_id
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if corruption == "manifest-list":
        manifest = []
    elif corruption == "missing-source":
        del manifest["source_url"]
    elif corruption == "wrong-hash":
        manifest["source_sha256"] = "b" * 64
    elif corruption == "wrong-subject-count":
        manifest["subject_count"] = 30
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    if corruption in {"features-list", "duplicate-feature"}:
        features_path = folder / "features.json"
        features = json.loads(features_path.read_text(encoding="utf-8"))
        if corruption == "features-list":
            features = [None] * 561
        else:
            features[1]["feature_id"] = "f001"
        features_path.write_text(json.dumps(features), encoding="utf-8")
    array_path = folder / "test.npz"
    if corruption == "missing-file":
        array_path.unlink()
    elif corruption == "broken-npz":
        array_path.write_bytes(b"PK\x03\x04broken")
    elif corruption == "encrypted-npz":
        content = bytearray(array_path.read_bytes())
        central_header = content.index(b"PK\x01\x02")
        # A real ZIP reader now requires a password; saved dataset files cannot be encrypted.
        content[central_header + 8] |= 1
        content[6] |= 1
        array_path.write_bytes(content)
    elif corruption == "bad-compression":
        with zipfile.ZipFile(array_path) as archive:
            member = archive.getinfo("X.npy")
        content = bytearray(array_path.read_bytes())
        # Damage the real DEFLATE body while retaining the ZIP directory/header.
        extra_size = int.from_bytes(content[member.header_offset + 28:member.header_offset + 30], "little")
        start = member.header_offset + 30 + len(member.filename.encode()) + extra_size
        content[start:start + member.compress_size] = b"\xff" * member.compress_size
        array_path.write_bytes(content)
    elif corruption in {"bad-label", "fractional-subject", "overlap", "wrong-dtype", "wrong-shape"}:
        with np.load(array_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if corruption == "bad-label":
            arrays["y"][0] = 7
        elif corruption == "fractional-subject":
            arrays["subjects"] = np.full(48, 2.5)
        elif corruption == "overlap":
            arrays["subjects"][:] = 1
        elif corruption == "wrong-dtype":
            arrays["X"] = arrays["X"].astype(np.float32)
        else:
            arrays["total_acc"] = arrays["total_acc"][:, :127]
        np.savez_compressed(array_path, **arrays)
    assert DatasetStore(target).info()["ready"] is False


def test_publish_does_not_point_to_corrupt_existing_snapshot(uci_root, tmp_path):
    bundle = load_uci(uci_root)
    target = tmp_path / "datasets"
    corrupt_id = publish_dataset(bundle, target, "a" * 64)
    publish_dataset(bundle, target, "b" * 64)
    pointer = (target / "current.json").read_bytes()
    (target / corrupt_id / "test.npz").write_bytes(b"broken")
    with pytest.raises(DomainError):
        publish_dataset(bundle, target, "a" * 64)
    assert (target / "current.json").read_bytes() == pointer
    assert list(target.glob(".staging-*")) == []
    assert list(target.glob(".current-*")) == []


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("member", ["../escape.txt", "C:/escape.txt", "..\\escape.txt"])
def test_prepare_rejects_outer_and_inner_unsafe_paths(tmp_path, nested, member):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(member, "unsafe")
    source = tmp_path / "source.zip"
    if nested:
        with zipfile.ZipFile(source, "w") as outer:
            outer.writestr("dataset.zip", payload.getvalue())
    else:
        source.write_bytes(payload.getvalue())
    target = tmp_path / "datasets"
    with pytest.raises(DomainError, match="đường dẫn"):
        prepare_dataset(source, target)
    assert not (tmp_path / "escape.txt").exists()
    assert list(target.glob(".staging-*")) == []


@pytest.mark.parametrize("nested", [False, True])
def test_prepare_rejects_malformed_zip_and_cleans_staging(tmp_path, nested):
    source = tmp_path / "source.zip"
    if nested:
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("dataset.zip", b"broken zip")
    else:
        source.write_bytes(b"broken zip")
    target = tmp_path / "datasets"
    with pytest.raises(DomainError, match="ZIP"):
        prepare_dataset(source, target)
    assert list(target.glob(".staging-*")) == []


def test_store_sample_alignment_uses_source_row_not_sorted_labels(uci_root, tmp_path):
    path = uci_root / "test" / "X_test.txt"
    values = np.loadtxt(path)
    values[47, 560] = 9.125
    np.savetxt(path, values)
    signal_path = uci_root / "test" / "Inertial Signals" / "total_acc_z_test.txt"
    signal = np.loadtxt(signal_path)
    signal[47, 127] = 1.125
    np.savetxt(signal_path, signal)
    target = tmp_path / "datasets"
    publish_dataset(load_uci(uci_root), target, "a" * 64)
    store = DatasetStore(target)
    sample = store.sample("test:000048")
    assert sample["features"]["f561"] == 9.125
    assert sample["actual_label"] == 6
    assert sample["subject_id"] == 2
    assert sample["signal"]["z"][127] == 1.125
    assert store.sample_ids("test", 1) == []
    for sample_id in ["test:000000", "test:999999", "test:1", "train:000049", "test:٠٠٠٠٠١"]:
        with pytest.raises(DomainError):
            store.sample(sample_id)
    with pytest.raises(DomainError):
        store.arrays("unknown")
    with pytest.raises(DomainError):
        store.sample_ids("unknown", 1)


def test_download_interrupted_body_retries_and_cleans_partial(tmp_path, monkeypatch):
    destination = tmp_path / "uci.zip"
    destination.write_bytes(b"previous archive")

    class InterruptedResponse(io.BytesIO):
        def read(self, size=-1):
            if self.tell() == 0:
                return super().read(8)
            raise http.client.IncompleteRead(b"partial", 100)

    monkeypatch.setattr(
        urllib.request.HTTPSHandler,
        "https_open",
        lambda _handler, request: _http_response(
            InterruptedResponse(b"not a complete archive"), request.full_url,
        ),
    )
    with pytest.raises(DomainError, match="HTTPS"):
        download_archive(destination, attempts=2)
    assert destination.read_bytes() == b"previous archive"
    assert not destination.with_name("uci.zip.part").exists()


def test_cli_invalid_archive_returns_localized_error_without_traceback(tmp_path):
    archive = tmp_path / "broken.zip"
    archive.write_bytes(b"broken")
    result = subprocess.run(
        [sys.executable, "-m", "motionsense_app.cli", "prepare-data", "--archive", str(archive)],
        env={**os.environ, "MOTIONSENSE_STATE_ROOT": str(tmp_path), "PYTHONUTF8": "1"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert json.loads(result.stderr)["error"]["code"] == "DATA_INVALID"


def test_data_dtos_validate_signal_lengths_and_sample_contract(uci_root, tmp_path):
    from pydantic import ValidationError

    from motionsense_app.data import schema

    target = tmp_path / "datasets"
    publish_dataset(load_uci(uci_root), target, "a" * 64)
    store = DatasetStore(target)
    sample = store.sample("test:000048")
    assert schema.Sample.model_validate(sample).model_dump() == sample
    assert schema.DatasetInfo.model_validate(store.info()).ready is True
    sample["signal"]["z"] = sample["signal"]["z"][:127]
    with pytest.raises(ValidationError):
        schema.Sample.model_validate(sample)


def _production_archive(tmp_path, *, all_classes=True):
    """Synthetic source for production invariants; never written to product var/."""
    source = tmp_path / "synthetic-production.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        prefix = "UCI HAR Dataset/"
        archive.writestr(prefix + "features.txt", "\n".join(f"{i} name" for i in range(1, 562)))
        archive.writestr(prefix + "activity_labels.txt", "\n".join(f"{i} label-{i}" for i in range(1, 7)))
        for split, rows, subjects in [("train", 6000, range(1, 22)), ("test", 4299, range(22, 31))]:
            archive.writestr(prefix + f"{split}/X_{split}.txt", (" ".join(["0"] * 561) + "\n") * rows)
            labels = [(i % 6 + 1) if all_classes else 1 for i in range(rows)]
            archive.writestr(prefix + f"{split}/y_{split}.txt", "\n".join(map(str, labels)))
            subject_list = list(subjects)
            archive.writestr(
                prefix + f"{split}/subject_{split}.txt",
                "\n".join(str(subject_list[i % len(subject_list)]) for i in range(rows)),
            )
            for axis in "xyz":
                archive.writestr(
                    prefix + f"{split}/Inertial Signals/total_acc_{axis}_{split}.txt",
                    (" ".join(["0.1"] * 128) + "\n") * rows,
                )
    return source


def test_prepare_derives_production_split_counts_from_source(tmp_path):
    target = tmp_path / "datasets"
    dataset_id = prepare_dataset(_production_archive(tmp_path), target)
    store = DatasetStore(target)
    assert store.info()["dataset_id"] == dataset_id
    assert store.info()["split_counts"] == {"train": 6000, "test": 4299}
    assert len(store.info()["subjects"]) == 30


def test_production_rejects_missing_observed_activity_classes(tmp_path):
    target = tmp_path / "datasets"
    with pytest.raises(DomainError, match="sáu lớp"):
        prepare_dataset(_production_archive(tmp_path, all_classes=False), target)
    assert not (target / "current.json").exists()
    assert list(target.glob(".staging-*")) == []


@pytest.mark.parametrize("member", [".. /.. /escape.txt", "folder/file:stream", "folder./escape.txt"])
def test_prepare_rejects_windows_ambiguous_paths(tmp_path, member):
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(member, "unsafe")
    target = tmp_path / "datasets"
    with pytest.raises(DomainError, match="đường dẫn"):
        prepare_dataset(source, target)
    assert list(target.glob(".staging-*")) == []
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file-sharing semantics")
def test_pointer_replace_failure_preserves_previous_pointer_and_cleans_temp(uci_root, tmp_path):
    bundle = load_uci(uci_root)
    target = tmp_path / "datasets"
    publish_dataset(bundle, target, "a" * 64)
    pointer_path = target / "current.json"
    before = pointer_path.read_bytes()
    # CPython on Windows opens without FILE_SHARE_DELETE, so rename genuinely fails.
    with pointer_path.open("rb"), pytest.raises(OSError):
        publish_dataset(bundle, target, "b" * 64)
    assert pointer_path.read_bytes() == before
    assert list(target.glob(".staging-*")) == []
    assert list(target.glob(".current-*")) == []


def test_prepare_handles_corrupt_deflate_body_as_domain_error(tmp_path):
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("features.txt", "1 feature" * 100)
    with zipfile.ZipFile(source) as archive:
        member = archive.getinfo("features.txt")
    content = bytearray(source.read_bytes())
    start = member.header_offset + 30 + len(member.filename.encode())
    content[start:start + member.compress_size] = b"\xff" * member.compress_size
    source.write_bytes(content)
    target = tmp_path / "datasets"
    with pytest.raises(DomainError, match="ZIP"):
        prepare_dataset(source, target)
    assert list(target.glob(".staging-*")) == []


def test_lzma_corrupt_npz_is_normalized_and_store_is_unready(lzma_corrupt_store):
    target = lzma_corrupt_store
    with pytest.raises(DomainError) as error:
        read_snapshot(target / ("a" * 64), "a" * 64)
    assert error.value.code == "DATA_INVALID"
    assert isinstance(error.value.__cause__, lzma.LZMAError)
    assert DatasetStore(target).info()["ready"] is False


@pytest.mark.parametrize("location", ["outer-extract", "outer-nested-read", "inner-extract"])
def test_lzma_corrupt_zip_is_normalized_and_cleans_staging(
    uci_root, tmp_path, corrupt_lzma_zip, location,
):
    target = tmp_path / "datasets"
    publish_dataset(load_uci(uci_root), target, "a" * 64)
    pointer = (target / "current.json").read_bytes()
    source = tmp_path / "lzma.zip"
    if location == "outer-nested-read":
        source.write_bytes(corrupt_lzma_zip({"dataset.zip": b"damaged inner archive body"}))
    else:
        damaged = corrupt_lzma_zip({"UCI HAR Dataset/features.txt": b"1 feature"})
        if location == "inner-extract":
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("dataset.zip", damaged)
        else:
            source.write_bytes(damaged)
    with pytest.raises(DomainError) as error:
        prepare_dataset(source, target)
    assert error.value.code == "DATA_INVALID"
    assert isinstance(error.value.__cause__, lzma.LZMAError)
    assert (target / "current.json").read_bytes() == pointer
    assert list(target.glob(".staging-*")) == []


@pytest.mark.parametrize(
    "redirects",
    [
        ["http://insecure.example/data.zip"],
        ["ftp://insecure.example/data.zip"],
        ["https://secure.example/next", "http://insecure.example/data.zip"],
        ["https://secure.example/next", "ftp://insecure.example/data.zip"],
    ],
)
def test_download_rejects_non_https_redirect_before_following_hop(tmp_path, monkeypatch, redirects):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("ok.txt", "ok")
    requests = []

    def transport(_handler, request):
        requests.append(request.full_url)
        assert request.timeout == 60
        if request.full_url.endswith("human+activity+recognition+using+smartphones.zip"):
            return _http_response(b"", request.full_url, code=302, location=redirects[0])
        if len(redirects) == 2 and request.full_url == redirects[0]:
            return _http_response(b"", request.full_url, code=302, location=redirects[1])
        return _http_response(payload.getvalue(), request.full_url)

    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", transport)
    monkeypatch.setattr(urllib.request.HTTPHandler, "http_open", transport)
    monkeypatch.setattr(urllib.request.FTPHandler, "ftp_open", transport)
    destination = tmp_path / "source.zip"
    destination.write_bytes(b"previous archive")
    partial = destination.with_name(destination.name + ".part")
    partial.write_bytes(b"previous partial")
    with pytest.raises(DomainError) as error:
        download_archive(destination, attempts=2)
    assert (error.value.code, error.value.status) == ("DOWNLOAD_FAILED", 503)
    assert all(urlsplit(url).scheme == "https" for url in requests)
    assert len(requests) == len(redirects) * 2
    assert destination.read_bytes() == b"previous archive"
    assert not partial.exists()


def test_download_follows_secure_relative_redirects_with_verified_tls(tmp_path, monkeypatch):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("ok.txt", "ok")
    destination = tmp_path / "source.zip"
    destination.write_bytes(b"previous archive")
    requests = []

    def transport(handler, request):
        requests.append(request.full_url)
        assert request.timeout == 60
        context = handler._context or ssl.create_default_context()
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True
        assert destination.read_bytes() == b"previous archive"
        if len(requests) == 1:
            return _http_response(b"", request.full_url, code=302, location="/next")
        if len(requests) == 2:
            return _http_response(b"", request.full_url, code=307, location="//secure.example/data.zip")
        return _http_response(payload.getvalue(), request.full_url)

    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", transport)
    assert download_archive(destination, attempts=1) == destination
    assert requests[1:] == ["https://archive.ics.uci.edu/next", "https://secure.example/data.zip"]
    assert destination.read_bytes() == payload.getvalue()
    assert not destination.with_name(destination.name + ".part").exists()


@pytest.mark.parametrize("state", ["missing", "corrupt"])
@pytest.mark.parametrize("sample_id", ["train:000001", "test:000001"])
def test_canonical_sample_on_unavailable_store_returns_503(uci_root, tmp_path, state, sample_id):
    target = tmp_path / "datasets"
    if state == "corrupt":
        dataset_id = publish_dataset(load_uci(uci_root), target, "a" * 64)
        (target / dataset_id / "manifest.json").write_text("{}", encoding="utf-8")
    store = DatasetStore(target)
    with pytest.raises(DomainError) as error:
        store.sample(sample_id)
    assert (error.value.code, error.value.status) == ("DATA_UNAVAILABLE", 503)
    for malformed_id in ["other:000001", "test:1"]:
        with pytest.raises(DomainError) as error:
            store.sample(malformed_id)
        assert (error.value.code, error.value.status) == ("SAMPLE_NOT_FOUND", 404)


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["prepare-data"],
        ["prepare-data", "--download", "--archive", "some.zip"],
        ["unknown-command"],
        ["prepare-data", "--archive"],
    ],
)
def test_cli_argument_errors_have_localized_headline_and_usage(tmp_path, args):
    result = subprocess.run(
        [sys.executable, "-m", "motionsense_app.cli", *args],
        env={**os.environ, "MOTIONSENSE_STATE_ROOT": str(tmp_path), "PYTHONUTF8": "1"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 2
    assert result.stderr.startswith("Cách dùng:")
    assert "lỗi: Tham số không hợp lệ." in result.stderr
    assert "--help" in result.stderr
    assert "Traceback" not in result.stderr
