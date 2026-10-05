import hashlib
import io
import lzma
import threading
import zipfile
from pathlib import Path

import numpy as np
import pytest

from motionsense_app.data.importer import load_uci, publish_dataset
from motionsense_app.data.store import DatasetStore


@pytest.fixture(scope='session')
def real_store():
    store = DatasetStore(Path(__file__).resolve().parents[1] / 'var/datasets')
    assert store.info()['ready'], 'Dữ liệu thật thiếu/hỏng: chạy Cai_dat.bat.'
    return store


@pytest.fixture(scope='session')
def real_source():
    source = Path(__file__).resolve().parents[1] / 'var/source/uci-har.zip'
    assert source.is_file(), 'Thiếu archive nguồn thật: chạy Cai_dat.bat.'
    return source


@pytest.fixture(scope='session')
def real_registry(real_store):
    from motionsense_app.models.registry import ModelRegistry
    registry = ModelRegistry(Path(__file__).resolve().parents[1] / 'var/models')
    assert registry.has_ready(real_store.info()['dataset_id']), 'Thiếu model thật: chạy Cai_dat.bat.'
    return registry


@pytest.fixture
def uci_root(tmp_path):
    root = tmp_path / "UCI HAR Dataset"
    root.mkdir()
    (root / "features.txt").write_text(
        "\n".join(f"{i} duplicate-name" for i in range(1, 562)), encoding="utf-8"
    )
    names = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING",
    ]
    (root / "activity_labels.txt").write_text(
        "\n".join(f"{i} {name}" for i, name in enumerate(names, 1)), encoding="utf-8"
    )
    for split, subject, offset in [("train", 1, 0.0), ("test", 2, 0.25)]:
        folder = root / split
        signals = folder / "Inertial Signals"
        signals.mkdir(parents=True)
        y = np.tile(np.arange(1, 7), 8)
        x = np.repeat((y / 6.0)[:, None], 561, axis=1) + offset
        np.savetxt(folder / f"X_{split}.txt", x)
        np.savetxt(folder / f"y_{split}.txt", y, fmt="%d")
        np.savetxt(folder / f"subject_{split}.txt", np.full(48, subject), fmt="%d")
        for axis, value in [("x", 0.1), ("y", 0.2), ("z", 0.3)]:
            np.savetxt(signals / f"total_acc_{axis}_{split}.txt", np.full((48, 128), value + offset))
    return root


@pytest.fixture
def dataset_store(uci_root, tmp_path):
    digest = hashlib.sha256()
    for path in sorted(uci_root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(uci_root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    target = tmp_path / "datasets"
    publish_dataset(load_uci(uci_root), target, digest.hexdigest())
    return DatasetStore(target)


@pytest.fixture
def training_result(dataset_store):
    from motionsense_app.models.training import train_model

    return train_model(dataset_store, "reduced", 50, lambda stage: None)


@pytest.fixture
def job_worker(training_result, dataset_store):
    """Pause at actual trainer stages without timing-dependent sleeps."""
    stages = ["validating", "selecting_features", "fitting", "evaluating", "saving"]
    entered = {stage: threading.Event() for stage in stages}
    release = {stage: threading.Event() for stage in stages}

    def trainer(store, profile, trees, on_stage):
        assert store is dataset_store
        assert (profile, trees) == ("reduced", 50)
        for stage in stages:
            on_stage(stage)
            entered[stage].set()
            assert release[stage].wait(5), f"worker stuck at {stage}"
        return training_result

    yield trainer, entered, release, training_result
    for event in release.values():
        event.set()


@pytest.fixture
def model_registry(dataset_store, tmp_path):
    from motionsense_app.models.registry import ModelRegistry
    from motionsense_app.models.training import train_model

    registry = ModelRegistry(tmp_path / "models")
    for profile in ("full", "reduced"):
        registry.publish(train_model(dataset_store, profile, 50, lambda stage: None))
    return registry


@pytest.fixture
def trained_model_id(model_registry):
    return next(item["model_id"] for item in model_registry.list() if item["profile"] == "reduced")


@pytest.fixture
def session_engine(dataset_store, model_registry, tmp_path):
    from motionsense_app.sessions.engine import SessionEngine
    from motionsense_app.sessions.repository import SessionRepository

    clock = [0.0]
    repo = SessionRepository(tmp_path / "var" / "motionsense.sqlite3")
    return SessionEngine(repo, dataset_store, model_registry, lambda: clock[0]), repo, clock


@pytest.fixture
def change_session_source(uci_root):
    def change(store, kind):
        pointer = store.data_dir / "current.json"
        snapshot = store.data_dir / store.info()["dataset_id"]
        if kind == "published":
            publish_dataset(load_uci(uci_root), store.data_dir, "b" * 64)
        elif kind == "missing_pointer":
            pointer.unlink()
        elif kind.startswith("missing:"):
            (snapshot / kind.split(":", 1)[1]).unlink()
        elif kind.startswith("corrupt:"):
            (snapshot / kind.split(":", 1)[1]).write_bytes(b"corrupt snapshot")
        else:
            content = {
                "invalid_json": b"{", "wrong_shape": b"[]", "missing_id": b"{}",
                "invalid_id": b'{"dataset_id":"../../other"}',
                "wrong_type": b'{"dataset_id":123}', "invalid_utf8": b"\xff",
                "duplicate_id": ('{"dataset_id":"' + "b" * 64
                                 + '","dataset_id":"' + store.info()["dataset_id"] + '"}').encode(),
            }[kind]
            pointer.write_bytes(content)
    return change


@pytest.fixture
def api_client(dataset_store, model_registry, session_engine, tmp_path):
    from fastapi.testclient import TestClient

    from motionsense_app.main import create_app
    from motionsense_app.settings import Settings

    app = create_app(Settings(root=tmp_path), services={
        "store": dataset_store, "registry": model_registry, "session_engine": session_engine[0],
    })
    with TestClient(app) as client:
        yield client


@pytest.fixture
def corrupt_lzma_zip():
    def build(members):
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_LZMA) as archive:
            for name, content in members.items():
                archive.writestr(name, content)
        with zipfile.ZipFile(payload) as archive:
            member = archive.getinfo(next(iter(members)))
        content = bytearray(payload.getvalue())
        offset = member.header_offset
        extra_size = int.from_bytes(content[offset + 28:offset + 30], "little")
        body = offset + 30 + len(member.filename.encode()) + extra_size
        # ZIP_LZMA has a four-byte prefix followed by its filter properties.
        content[body + 4] = 0xff
        result = bytes(content)
        with zipfile.ZipFile(io.BytesIO(result)) as archive, pytest.raises(lzma.LZMAError):
            archive.read(member.filename)
        return result

    return build


@pytest.fixture
def lzma_corrupt_store(uci_root, tmp_path, corrupt_lzma_zip):
    target = tmp_path / "var" / "datasets"
    dataset_id = publish_dataset(load_uci(uci_root), target, "a" * 64)
    npz_path = target / dataset_id / "test.npz"
    with zipfile.ZipFile(npz_path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    npz_path.write_bytes(corrupt_lzma_zip(members))
    return target
