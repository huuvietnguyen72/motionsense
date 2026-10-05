from pathlib import Path

import pytest


@pytest.mark.real_data
def test_acceptance_copy_excludes_live_locks_and_user_database(tmp_path):
    from motionsense_app.data.store import DatasetStore
    from motionsense_app.models.registry import ModelRegistry
    from scripts.acceptance import copy_ready_state, fingerprint
    root = Path(__file__).resolve().parents[1]
    before = fingerprint(root / 'var')
    target = tmp_path / 'nghiệm thu riêng'
    copy_ready_state(root, target)
    assert not (target / 'var/motionsense.sqlite3').exists()
    assert not list(target.rglob('*.lock'))
    data = DatasetStore(target / 'var/datasets').info()
    assert data['split_counts'] == {'train': 7352, 'test': 2947}
    assert ModelRegistry(target / 'var/models').has_ready(data['dataset_id'])
    assert fingerprint(root / 'var') == before


def test_acceptance_missing_real_source_fails_with_setup_instruction(tmp_path):
    from scripts.acceptance import copy_ready_state
    with pytest.raises(RuntimeError, match='Cai_dat.bat'):
        copy_ready_state(tmp_path, tmp_path / 'isolated')


def test_acceptance_unknown_port_is_not_killed(tmp_path):
    import socket

    from scripts.acceptance import Acceptance
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        runner = Acceptance(listener.getsockname()[1], tmp_path)
        try:
            with pytest.raises(RuntimeError, match='không dừng tiến trình lạ'):
                runner.start(tmp_path)
            assert runner.child is None
            assert listener.getsockname()[1] == runner.port
        finally:
            runner.client.close()
