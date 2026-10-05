import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from test_install_recovery import OwnedProcessHandles

ROOT = Path(__file__).resolve().parents[1]


def test_windows_boot_identity_is_stable_across_controllers():
    from scripts.install import windows_boot_identity

    identity = windows_boot_identity()
    assert len(identity) == 32 and int(identity, 16) != 0
    result = subprocess.run(
        [sys.executable, '-c',
         'from scripts.install import windows_boot_identity; print(windows_boot_identity())'],
        cwd=ROOT, capture_output=True, encoding='utf-8', timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == identity


@pytest.mark.parametrize('stored', [None, '', 'invalid', 'a' * 32])
def test_guard_without_verified_changed_boot_blocks_tools(tmp_path, monkeypatch, stored):
    from scripts import install as installer

    root = tmp_path / 'Guard boot validation'
    guard = root / 'var/logs/.setup.lock.pending'
    guard.parent.mkdir(parents=True)
    guard.write_text(json.dumps({'schema_version': 1, 'boot_id': stored}), encoding='utf-8')
    monkeypatch.setattr(installer, 'windows_boot_identity', lambda: 'a' * 32)
    assert installer.install(root) == 2
    assert not (root / '.venv').exists()
    assert json.loads(guard.read_text('utf-8'))['boot_id'] == stored


@pytest.mark.parametrize('location', ['pending', 'lock', 'both'])
@pytest.mark.parametrize('stored', ['0' * 32, None, '', 'invalid', 'a' * 31,
                                  'a' * 33, 'g' * 32, 0])
def test_invalid_stored_boot_preserves_guards_and_blocks_tools(
        tmp_path, monkeypatch, capsys, location, stored):
    from scripts import install as installer

    root = tmp_path / 'Guard boot có dấu'
    lock = root / 'var/logs/.setup.lock'
    guard = lock.with_name(lock.name + '.pending')
    lock.parent.mkdir(parents=True)
    record = json.dumps({'schema_version': 1, 'boot_id': stored}).encode('utf-8')
    lock_bytes = b'\0' + (record if location in ('lock', 'both') else b'')
    lock.write_bytes(lock_bytes)
    if location in ('pending', 'both'):
        guard.write_bytes(record)
    tools = []

    def unexpected_tool(command, *args, **kwargs):
        tools.append(command)
        raise RuntimeError('Unexpected tool execution')

    monkeypatch.setattr(installer, 'windows_boot_identity', lambda: 'a' * 32)
    monkeypatch.setattr(installer, 'run_step', unexpected_tool)
    for _ in range(2):
        assert installer.install(root) == 2
        assert tools == [], 'Invalid stored boot authorized tool execution'
        error = capsys.readouterr().err
        assert 'Guard .pending thiếu/hỏng định danh boot' in error
        assert 'Không tự xóa guard' in error
        assert 'Traceback' not in error
        assert not (root / '.venv').exists()
        assert lock.read_bytes() == lock_bytes
        if location in ('pending', 'both'):
            assert guard.read_bytes() == record
        else:
            assert not guard.exists()


def test_unavailable_boot_identity_blocks_install_before_tools(tmp_path, monkeypatch):
    from scripts import install as installer

    def unavailable():
        raise OSError('Boot query unavailable')

    monkeypatch.setattr(installer, 'windows_boot_identity', unavailable)
    assert installer.install(tmp_path) == 2
    assert not (tmp_path / '.venv').exists()


@pytest.mark.real_data
@pytest.mark.parametrize('fault', ['cleanup-error', 'accounting-timeout'])
def test_unconfirmed_cleanup_keeps_guards_after_bounded_installer_exit(tmp_path, fault):
    from scripts import install as installer

    state = tmp_path / 'MotionSense guard có dấu'
    for folder in ('datasets', 'models'):
        shutil.copytree(ROOT / 'var' / folder, state / 'var' / folder)
    tree = tmp_path / 'Cây cleanup chưa xác nhận'
    tree.mkdir()
    batch = tree / 'npm-hang.cmd'
    batch.write_bytes((f'@echo off\r\nchcp 65001 >nul\r\n"{sys.executable}" -X utf8 '
                       f'"{ROOT / "tests/windows_step_tree.py"}" parent "{tree}"\r\n'
                       'exit /b %ERRORLEVEL%\r\n').encode())
    handles = OwnedProcessHandles(tree)
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
    paths = [ROOT / 'var/logs/.setup.lock', state / 'var/logs/.install.lock']
    try:
        started = time.monotonic()
        controller = subprocess.Popen(
            [str(Path(sys.base_prefix) / 'python.exe'), '-I', '-S', '-X', 'utf8',
             str(ROOT / 'tests/windows_cleanup_fault_driver.py'), str(ROOT), str(state),
             str(tree), fault], cwd=tmp_path, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, encoding='utf-8')
        deadline = time.monotonic() + 10
        while not (tree / 'ready').exists():
            assert controller.poll() is None, 'Controller exited before tree startup'
            assert time.monotonic() < deadline, 'Owned tree did not start'
            time.sleep(0.02)
        handles.capture()
        stdout, stderr = controller.communicate(timeout=20)
        elapsed = time.monotonic() - started
        assert controller.returncode == 2, stderr + stdout
        assert elapsed < 15
        # OS locks die with the controller; durable guards must still exclude retry.
        for path in paths:
            assert path.with_name(path.name + '.pending').exists()
        evidence = json.loads((tree / 'controller-evidence.json').read_text('utf-8'))
        assert evidence['ordinary_guard_release_calls'] == []
        assert unrelated.poll() is None
        blocked = subprocess.run(
            [sys.executable, '-X', 'utf8', str(ROOT / 'scripts/install.py')], cwd=tmp_path,
            env=dict(os.environ, MOTIONSENSE_STATE_ROOT=str(state)), capture_output=True,
            encoding='utf-8', timeout=10, check=False)
        assert blocked.returncode == 2
        assert '.pending' in blocked.stderr
        # Known handles can be signaled without proving inventory completeness.
        for _, handle in handles.handles.values():
            assert handles.kernel.WaitForSingleObject(handle, 10000) == 0
        assert handles.all_stopped()
        for path in paths:
            path.with_name(path.name + '.pending').unlink()
            guard = installer.ProcessLock(path)
            try:
                assert guard.acquire(), 'Guard was not closed after confirmed cleanup'
            finally:
                guard.release()
        deleted = subprocess.run(
            [sys.executable, '-X', 'utf8', str(ROOT / 'scripts/install.py')], cwd=tmp_path,
            env=dict(os.environ, MOTIONSENSE_STATE_ROOT=str(state)), capture_output=True,
            encoding='utf-8', timeout=10, check=False)
        assert deleted.returncode == 2, 'Manual pending deletion was accepted as completion'
        # Read-only boot API seam: inject another boot, never reboot the live machine.
        retry = subprocess.run(
            [sys.executable, '-X', 'utf8', '-c',
             ('import sys; sys.path.insert(0, sys.argv[1]); '
              'from scripts import install as i; i.windows_boot_identity=lambda: "f"*32; '
              'raise SystemExit(i.install())'), str(ROOT)], cwd=tmp_path,
            env=dict(os.environ, MOTIONSENSE_STATE_ROOT=str(state)), capture_output=True,
            encoding='utf-8', timeout=60, check=False)
        assert retry.returncode == 0, retry.stderr + retry.stdout
        assert unrelated.poll() is None
        (tree / 'round3-evidence.json').write_text(json.dumps({
            'fault': fault, 'installer_exit': controller.returncode,
            'installer_elapsed_seconds': elapsed, 'ordinary_release_calls': [],
            'guards_blocked_after_controller_exit': True, 'retry_while_pending_exit': 2,
            'manual_pending_deletion_retry_exit': deleted.returncode,
            'changed_boot_identity_injected': True,
            'owned_pids': {name: pid for name, (pid, _) in handles.handles.items()},
            'owned_tree_signaled_before_retry': handles.all_stopped(), 'retry_exit': 0,
            'unrelated_pid': unrelated.pid, 'unrelated_alive': unrelated.poll() is None,
        }, indent=2), encoding='utf-8')
    finally:
        if 'controller' in locals() and controller.poll() is None:
            controller.terminate()
            controller.wait(timeout=5)
        handles.close()
        for path in paths:
            path.with_name(path.name + '.pending').unlink(missing_ok=True)
            # Test-owned fault records only; exact helper handles were cleaned above.
            with path.open('r+b') as stream:
                stream.truncate(1)
        unrelated.terminate()
        unrelated.wait(timeout=5)
