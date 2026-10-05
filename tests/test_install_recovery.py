import ctypes
import json
import shutil
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def small_build(tmp_path):
    root = tmp_path / 'MotionSense bản dựng có dấu'
    frontend = root / 'frontend'
    (frontend / 'src').mkdir(parents=True)
    (frontend / 'dist/assets').mkdir(parents=True)
    for name in ('package.json', 'package-lock.json', 'index.html', 'src/main.tsx',
                 'dist/index.html', 'dist/assets/app.js'):
        (frontend / name).write_text('content', encoding='utf-8')
    from scripts.preflight import write_build_manifest

    write_build_manifest(root)
    shutil.copytree(ROOT / 'scripts', root / 'scripts', ignore=shutil.ignore_patterns('__pycache__'))
    return root


MANIFEST_CASES = [
    'root-null', 'root-list', 'root-string', 'root-number', 'root-empty',
    'schema-missing', 'schema-bool', 'schema-list', 'schema-float', 'extra-field',
    'inputs-null', 'inputs-list', 'inputs-string', 'inputs-number', 'inputs-missing',
    'outputs-null', 'outputs-list', 'outputs-string', 'outputs-number', 'outputs-missing',
    'input-hash-null', 'input-hash-list', 'output-hash-object', 'output-hash-invalid',
    'input-traversal', 'output-absolute', 'duplicate-inputs', 'invalid-json', 'invalid-utf8',
    'too-deep',
]


def malformed_manifest(original, case):
    manifest = json.loads(json.dumps(original))
    root_values = {'root-null': None, 'root-list': [], 'root-string': 'manifest',
                   'root-number': 42, 'root-empty': {}}
    if case in root_values:
        manifest = root_values[case]
    elif case == 'schema-missing':
        del manifest['schema_version']
    elif case == 'schema-bool':
        manifest['schema_version'] = True
    elif case == 'schema-list':
        manifest['schema_version'] = [1]
    elif case == 'schema-float':
        manifest['schema_version'] = 1.0
    elif case == 'extra-field':
        manifest['unexpected'] = {'schema_version': 1}
    elif case.startswith(('inputs-', 'outputs-')):
        field, kind = case.split('-')
        if kind == 'missing':
            del manifest[field]
        else:
            manifest[field] = {'null': None, 'list': [], 'string': 'index.html',
                               'number': 42}[kind]
    elif case.startswith('input-hash-'):
        manifest['inputs']['package.json'] = None if case.endswith('null') else []
    elif case == 'output-hash-object':
        manifest['outputs']['index.html'] = {'hash': 'a' * 64}
    elif case == 'output-hash-invalid':
        manifest['outputs']['index.html'] = 'not-sha256'
    elif case == 'input-traversal':
        manifest['inputs']['../../outside'] = 'a' * 64
    elif case == 'output-absolute':
        manifest['outputs']['C:\\outside.js'] = 'a' * 64
    elif case == 'duplicate-inputs':
        valid = json.dumps(manifest)
        return ('{"inputs": [],' + valid[1:]).encode('utf-8')
    elif case == 'invalid-json':
        return b'{broken'
    elif case == 'invalid-utf8':
        return b'\xff'
    elif case == 'too-deep':
        return b'[' * 20000 + b'0' + b']' * 20000
    return json.dumps(manifest).encode('utf-8')


@pytest.mark.parametrize('case', MANIFEST_CASES)
def test_malformed_manifest_is_notready_json_without_traceback(small_build, tmp_path, case):
    from scripts.preflight import frontend_ready

    path = small_build / 'frontend/dist/build-manifest.json'
    original = json.loads(path.read_text('utf-8'))
    path.write_bytes(malformed_manifest(original, case))
    assert frontend_ready(small_build) is False
    result = subprocess.run(
        [sys.executable, '-X', 'utf8', str(small_build / 'scripts/launch.py'), '--check'],
        cwd=tmp_path, capture_output=True, encoding='utf-8', timeout=10, check=False)
    assert result.returncode == 2, result.stderr + result.stdout
    body = json.loads(result.stdout)
    assert body['ready'] is False and 'frontend' in body['missing']
    assert 'Traceback' not in result.stderr
    assert 'AttributeError' not in result.stderr


def test_installer_rebuilds_malformed_manifest_and_can_reuse_without_node(tmp_path, monkeypatch):
    from scripts.install import ensure_frontend
    from scripts.preflight import frontend_ready

    root = tmp_path / 'MotionSense rebuild có dấu'
    shutil.copytree(ROOT / 'frontend', root / 'frontend', ignore=shutil.ignore_patterns(
        'node_modules', 'test-results', 'playwright-report'))
    (root / 'frontend/dist/build-manifest.json').write_text(
        '{"schema_version":1,"inputs":null,"outputs":{}}', encoding='utf-8')
    log_path = root / 'rebuild.log'
    with log_path.open('w', encoding='utf-8') as log:
        ensure_frontend(root, log)
    assert frontend_ready(root)
    text = log_path.read_text('utf-8')
    assert 'ci --no-audit --no-fund' in text and 'run build' in text
    manifest = (root / 'frontend/dist/build-manifest.json').read_bytes()
    monkeypatch.setenv('PATH', '')
    with log_path.open('a', encoding='utf-8') as log:
        ensure_frontend(root, log)
    assert (root / 'frontend/dist/build-manifest.json').read_bytes() == manifest


class OwnedProcessHandles:
    """Test-only handles to the exact processes recorded by our own tree helper."""

    def __init__(self, directory):
        self.directory = directory
        self.handles = {}
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel.TerminateProcess.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL

    def capture(self):
        for name in ('parent', 'grandchild'):
            pid = int((self.directory / f'{name}.pid').read_text('utf-8'))
            handle = self.kernel.OpenProcess(0x100000 | 0x1, False, pid)
            assert handle, ctypes.WinError(ctypes.get_last_error())
            self.handles[name] = (pid, handle)

    def all_stopped(self):
        return len(self.handles) == 2 and all(
            self.kernel.WaitForSingleObject(handle, 0) == 0
            for _, handle in self.handles.values())

    def close(self):
        for _, handle in self.handles.values():
            if self.kernel.WaitForSingleObject(handle, 0) != 0:
                self.kernel.TerminateProcess(handle, 99)
                self.kernel.WaitForSingleObject(handle, 5000)
            self.kernel.CloseHandle(handle)


@pytest.mark.real_data
@pytest.mark.parametrize('cleanup_failure', ['none', 'query-once', 'terminate-once',
                                            'accounting-timeout', 'query-persistent',
                                            'terminate-persistent'])
def test_installer_timeout_stops_owned_parent_grandchild_before_unlock_and_retry(
        tmp_path, monkeypatch, cleanup_failure):
    import threading

    from motionsense_app.ownership import ProcessLock
    from scripts import install as installer
    from scripts.process_tree import _WindowsJob

    monkeypatch.setattr(installer, 'windows_boot_identity', lambda: 'a' * 32, raising=False)
    state = tmp_path / 'MotionSense npm timeout có dấu'
    for folder in ('datasets', 'models'):
        shutil.copytree(ROOT / 'var' / folder, state / 'var' / folder)
    monkeypatch.setenv('MOTIONSENSE_STATE_ROOT', str(state))
    tree = tmp_path / 'Cây process thuộc test'
    tree.mkdir()
    batch = tree / 'npm-hang.cmd'
    batch.write_bytes((f'@echo off\r\nchcp 65001 >nul\r\n"{sys.executable}" -X utf8 '
                       f'"{ROOT / "tests/windows_step_tree.py"}" parent "{tree}"\r\n'
                       'exit /b %ERRORLEVEL%\r\n').encode())
    handles = OwnedProcessHandles(tree)
    ready = threading.Event()
    errors = []

    def capture_tree():
        try:
            deadline = time.monotonic() + 10
            while not (tree / 'ready').exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError('Owned process tree did not start')
                time.sleep(0.02)
            handles.capture()
            ready.set()
        except (OSError, ValueError, RuntimeError, AssertionError) as exc:
            errors.append(exc)

    observer = threading.Thread(target=capture_tree)
    observer.start()
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
    original_frontend = installer.ensure_frontend
    original_release = ProcessLock.release
    releases = []

    def timeout_frontend(root, log):
        installer.run_step(['cmd.exe', '/d', '/c', str(batch)], tree, log, timeout=4)

    def observed_release(lock):
        if lock.held and lock.path.name in {'.setup.lock', '.install.lock'}:
            releases.append((lock.path.name, handles.all_stopped()))
        original_release(lock)

    monkeypatch.setattr(installer, 'ensure_frontend', timeout_frontend)
    monkeypatch.setattr(ProcessLock, 'release', observed_release)
    original_count = _WindowsJob.active_processes
    original_stop = _WindowsJob.stop_and_wait
    injected = False

    def count(job):
        nonlocal injected
        if (ready.is_set() and cleanup_failure in {'query-once', 'query-persistent'}
                and (not injected or cleanup_failure == 'query-persistent')):
            injected = True
            raise OSError('Injected accounting failure')
        return original_count(job)

    def stop(job, timeout=5):
        nonlocal injected
        if ready.is_set() and (not injected or cleanup_failure == 'terminate-persistent'):
            if cleanup_failure in {'terminate-once', 'terminate-persistent'}:
                injected = True
                # Real Win32 failure: NULL is not a valid job handle.
                assert not job.kernel.TerminateJobObject(None, 2)
                raise ctypes.WinError(ctypes.get_last_error())
            if cleanup_failure == 'accounting-timeout':
                injected = True
                # Exercise the actual accounting deadline while the real tree is alive.
                monkeypatch.setattr(_WindowsJob, 'active_processes', lambda job: 1)
                try:
                    return original_stop(job, timeout=0.05)
                finally:
                    monkeypatch.setattr(_WindowsJob, 'active_processes', count)
        return original_stop(job, timeout)

    monkeypatch.setattr(_WindowsJob, 'active_processes', count)
    monkeypatch.setattr(_WindowsJob, 'stop_and_wait', stop)
    try:
        started = time.monotonic()
        assert installer.install(ROOT) == 2
        elapsed = time.monotonic() - started
        assert elapsed < 30
        observer.join(timeout=10)
        assert not errors, errors
        assert ready.is_set()
        assert releases == [], 'Forced cleanup released installer locks on the same boot'
        assert (ROOT / 'var/logs/.setup.lock.pending').exists()
        assert installer.install(ROOT) == 2
        assert releases == []
        if not cleanup_failure.endswith('persistent'):
            assert handles.all_stopped(), 'Known timed-out descendants survived'
        assert unrelated.poll() is None
        log_path = state / 'var/logs/install.log'
        text = log_path.read_text('utf-8')
        assert 'Thiết lập thất bại' in text and 'timed out' in text
        assert 'MotionSense sẵn sàng.' not in text
        # Inject a changed boot identity; do not reboot or change the live OS.
        monkeypatch.setattr(_WindowsJob, 'active_processes', original_count)
        monkeypatch.setattr(_WindowsJob, 'stop_and_wait', original_stop)
        monkeypatch.setattr(installer, 'windows_boot_identity', lambda: 'b' * 32)
        monkeypatch.setattr(installer, 'ensure_frontend', original_frontend)
        assert installer.install(ROOT) == 0
        assert handles.all_stopped()
        assert releases[:2] == [('.install.lock', True), ('.setup.lock', True)]
        assert unrelated.poll() is None
        assert 'MotionSense sẵn sàng.' in log_path.read_text('utf-8')
        (tmp_path / 'fixround3-evidence.json').write_text(json.dumps({
            'owned_pids': {name: pid for name, (pid, _) in handles.handles.items()},
            'owned_process_handles_signaled': handles.all_stopped(),
            'lock_release_observations': releases,
            'unrelated_pid': unrelated.pid, 'unrelated_alive_after_retry': unrelated.poll() is None,
            'timeout_install_exit': 2, 'retry_install_exit': 0,
            'timeout_install_elapsed_seconds': elapsed,
            'same_boot_retry_exit': 2, 'changed_boot_identity_injected': True,
        }, indent=2), encoding='utf-8')
    finally:
        monkeypatch.setattr(_WindowsJob, 'active_processes', original_count)
        monkeypatch.setattr(_WindowsJob, 'stop_and_wait', original_stop)
        monkeypatch.setattr(installer, 'windows_boot_identity', lambda: 'b' * 32)
        if hasattr(installer, 'retry_pending_cleanup'):
            installer.retry_pending_cleanup()
        observer.join(timeout=12)
        handles.close()
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_owned_step_preserves_nonzero_exit_and_unicode_output(tmp_path):
    from scripts.install import run_step

    log_path = tmp_path / 'bước thất bại.log'
    with (log_path.open('w', encoding='utf-8') as log,
          pytest.raises(RuntimeError, match='exit 7') as caught):
        run_step([sys.executable, '-X', 'utf8', '-c',
                  'print("Lỗi có dấu"); raise SystemExit(7)'], tmp_path, log, timeout=5)
    text = log_path.read_text('utf-8')
    assert 'Lỗi có dấu' in text
    assert 'Windows step tree stopped (active=0).' in text
    if hasattr(caught.value, 'retry'):
        caught.value.retry()


def test_owned_step_fails_closed_if_job_assignment_fails(tmp_path, monkeypatch):
    from scripts.install import run_step
    from scripts.process_tree import _WindowsJob

    children = []

    def denied(job, child):
        children.append(child)
        raise OSError('Job assignment denied')

    monkeypatch.setattr(_WindowsJob, 'assign', denied)
    with ((tmp_path / 'assignment.log').open('w', encoding='utf-8') as log,
          pytest.raises(OSError, match='assignment denied')):
        run_step([sys.executable, '-c',
                  'from pathlib import Path; Path("must-not-run").touch()'],
                 tmp_path, log, timeout=5)
    assert not (tmp_path / 'must-not-run').exists()
    assert len(children) == 1 and children[0].poll() is not None


def test_stale_cleanup_guard_blocks_new_installer_process(tmp_path):
    root = tmp_path / 'Guarded installer'
    guard = root / 'var/logs/.setup.lock.pending'
    guard.parent.mkdir(parents=True)
    guard.write_text('Unconfirmed previous tree', encoding='utf-8')
    result = subprocess.run(
        [sys.executable, '-X', 'utf8', '-c',
         ('from pathlib import Path; from scripts.install import install; import sys; '
          'raise SystemExit(install(Path(sys.argv[1])))'), str(root)],
        cwd=ROOT, capture_output=True, encoding='utf-8', timeout=10, check=False)
    assert result.returncode == 2
    assert '.pending' in result.stderr and 'Khởi động lại Windows' in result.stderr
    assert guard.read_text('utf-8') == 'Unconfirmed previous tree'
    assert not (root / '.venv').exists()


def test_cleanup_retry_preserves_tool_exit_seven(tmp_path, monkeypatch):
    from scripts.install import run_step
    from scripts.process_tree import _WindowsJob

    original = _WindowsJob.active_processes
    failed = False

    def count(job):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError('Transient accounting error')
        return original(job)

    monkeypatch.setattr(_WindowsJob, 'active_processes', count)
    with ((tmp_path / 'exit.log').open('w', encoding='utf-8') as log,
          pytest.raises(RuntimeError, match='exit 7') as caught):
        run_step([sys.executable, '-c', 'raise SystemExit(7)'], tmp_path, log, timeout=5)
    if hasattr(caught.value, 'retry'):
        caught.value.retry()


def test_capture_resize_is_bounded_by_attempts(tmp_path, monkeypatch):
    from scripts.process_tree import _WindowsJob

    job = _WindowsJob()
    calls = 0

    def growing(*args):
        nonlocal calls
        calls += 1
        if calls > 8:
            raise AssertionError('Unbounded inventory resize')
        ctypes.set_last_error(234)
        return False

    monkeypatch.setattr(job.kernel, 'QueryInformationJobObject', growing)
    try:
        with pytest.raises(RuntimeError, match='inventory'):
            job.capture_processes()
        assert calls <= 8
    finally:
        job.close()


def test_capture_checks_deadline_before_retry(monkeypatch):
    from scripts.process_tree import _WindowsJob

    job = _WindowsJob()
    clock = iter([0.0, 0.0, 6.0, 6.0])
    calls = 0

    def growing(*args):
        nonlocal calls
        calls += 1
        assert calls == 1, 'Inventory retried after deadline'
        ctypes.set_last_error(234)
        return False

    monkeypatch.setattr(time, 'monotonic', lambda: next(clock, 6.0))
    monkeypatch.setattr(job.kernel, 'QueryInformationJobObject', growing)
    try:
        with pytest.raises(RuntimeError, match='inventory'):
            job.capture_processes()
    finally:
        job.close()


def test_unconfirmed_cleanup_preserves_tool_exit_evidence(tmp_path, monkeypatch):
    from scripts.install import run_step
    from scripts.process_tree import CleanupUnconfirmed, _WindowsJob

    original = _WindowsJob.stop_and_wait

    def denied(job, timeout=5):
        raise OSError('Persistent cleanup denied')

    monkeypatch.setattr(_WindowsJob, 'stop_and_wait', denied)
    try:
        with ((tmp_path / 'unconfirmed.log').open('w', encoding='utf-8') as log,
              pytest.raises(CleanupUnconfirmed) as caught):
            run_step([sys.executable, '-c', 'raise SystemExit(7)'], tmp_path, log, timeout=5)
        assert 'exit 7' in str(caught.value)
        assert 'Persistent cleanup denied' in str(caught.value)
    finally:
        monkeypatch.setattr(_WindowsJob, 'stop_and_wait', original)
        caught.value.retry()
