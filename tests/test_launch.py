import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from motionsense_app.main import create_app
from motionsense_app.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


def run_script(name, *args, cwd=None, env=None):
    return subprocess.run(
        [sys.executable, '-X', 'utf8', str(ROOT / 'scripts' / name), *args],
        cwd=cwd or ROOT, env=env, capture_output=True, encoding='utf-8', timeout=60, check=False,
    )


@pytest.mark.real_data
def test_launch_check_from_unrelated_directory(tmp_path):
    result = run_script('launch.py', '--check', cwd=tmp_path)
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(result.stdout) == {'ready': True, 'missing': []}


def test_check_missing_state_is_json_without_traceback(tmp_path):
    env = dict(os.environ, MOTIONSENSE_STATE_ROOT=str(tmp_path / 'Trạng thái có dấu'))
    result = run_script('launch.py', '--check', cwd=tmp_path, env=env)
    assert result.returncode == 2, result.stderr + result.stdout
    body = json.loads(result.stdout)
    assert body['ready'] is False
    assert {'dataset', 'models'}.issubset(body['missing'])
    assert 'Traceback' not in result.stderr


def test_health_marker_distinguishes_state_and_normalizes_root(tmp_path):
    root = tmp_path / 'MotionSense có dấu'
    root.mkdir()
    markers = []
    for settings in [Settings(root), Settings(root / '..' / root.name),
                     Settings(root, tmp_path / 'other')]:
        with TestClient(create_app(settings)) as client:
            response = client.get('/api/health')
            assert set(response.json()) == {'app', 'status', 'data_ready', 'models_ready'}
            marker = response.headers.get('X-MotionSense-Instance')
            assert marker is not None and len(marker) == 64
            markers.append(marker)
    assert markers[0] == markers[1]
    assert markers[0] != markers[2]


def test_preflight_empty_root_reports_all_missing(tmp_path):
    from scripts.launch import check_installation

    result = check_installation(tmp_path / 'MotionSense chưa cài')
    assert result['ready'] is False
    assert {'environment', 'frontend', 'dataset', 'models'}.issubset(result['missing'])


def test_build_manifest_rejects_changed_assets_source_and_lock(tmp_path):
    from scripts.preflight import frontend_ready, write_build_manifest

    frontend = tmp_path / 'frontend'
    (frontend / 'src').mkdir(parents=True)
    (frontend / 'dist' / 'assets').mkdir(parents=True)
    files = {'package-lock.json': '{}', 'package.json': '{}', 'src/main.tsx': 'hello',
             'index.html': '<div/>', 'dist/index.html': '<div/>', 'dist/assets/app.js': 'hello'}
    for name, value in files.items():
        (frontend / name).write_text(value, encoding='utf-8')
    assert not frontend_ready(tmp_path)
    write_build_manifest(tmp_path)
    assert frontend_ready(tmp_path)
    for name in ('dist/assets/app.js', 'src/main.tsx', 'package-lock.json'):
        original = (frontend / name).read_bytes()
        (frontend / name).write_bytes(b'changed')
        assert not frontend_ready(tmp_path), name
        (frontend / name).write_bytes(original)
    assert frontend_ready(tmp_path)


def test_environment_checks_real_lock_versions_and_interpreter(tmp_path):
    from scripts.preflight import environment_ready

    assert environment_ready(ROOT)
    (tmp_path / '.venv' / 'Scripts').mkdir(parents=True)
    import shutil
    shutil.copy2(ROOT / '.venv' / 'Scripts' / 'python.exe',
                 tmp_path / '.venv' / 'Scripts' / 'python.exe')
    (tmp_path / 'requirements.lock.txt').write_text('fastapi==0.0.1\n', encoding='utf-8')
    assert not environment_ready(tmp_path)


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def test_readiness_detects_dead_child_and_timeout_cleans_own_child():
    from scripts.lifecycle import stop_child, wait_for_health

    url = f'http://127.0.0.1:{free_port()}'
    child = subprocess.Popen([sys.executable, '-c', 'raise SystemExit(7)'])
    try:
        with pytest.raises(RuntimeError, match='7'):
            wait_for_health(child, url, 'missing', timeout=5)
    finally:
        child.wait(timeout=5)
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                             creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    try:
        with pytest.raises(TimeoutError):
            wait_for_health(child, url, 'missing', timeout=0.3)
    finally:
        stop_child(child, timeout=1)
    assert child.poll() is not None


@pytest.mark.real_data
def test_other_occupied_process_is_untouched(tmp_path):
    port = free_port()
    child = subprocess.Popen([sys.executable, '-m', 'http.server', str(port),
                              '--bind', '127.0.0.1'], cwd=tmp_path,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}', timeout=0.2):
                    break
            except OSError:
                time.sleep(0.05)
        result = run_script('launch.py', '--port', str(port), '--no-browser', cwd=tmp_path)
        assert result.returncode == 2, result.stderr + result.stdout
        assert 'cổng' in result.stderr.lower()
        assert child.poll() is None
        with urllib.request.urlopen(f'http://127.0.0.1:{port}', timeout=2) as response:
            assert response.status == 200
    finally:
        child.terminate()
        child.wait(timeout=5)


@pytest.fixture
def installed_state(tmp_path):
    import shutil

    state = tmp_path / 'MotionSense trạng thái có dấu'
    for folder in ('datasets', 'models'):
        shutil.copytree(ROOT / 'var' / folder, state / 'var' / folder)
    return state


@pytest.mark.real_data
def test_windows_batch_start_reuse_cli_and_graceful_shutdown(installed_state, tmp_path):
    from scripts.lifecycle import probe_health, stop_child

    port = free_port()
    env = dict(os.environ, MOTIONSENSE_STATE_ROOT=os.path.relpath(installed_state, ROOT),
               PYTHONUTF8='1')
    command = ['cmd.exe', '/d', '/c', str(ROOT / 'Khoi_dong.bat'),
               '--port', str(port), '--no-browser']
    log = tmp_path / 'launcher.log'
    with log.open('w', encoding='utf-8') as stream:
        launcher = subprocess.Popen(command, cwd=tmp_path, env=env, stdout=stream,
                                    stderr=stream,
                                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        try:
            marker = Settings(ROOT, installed_state).instance_marker
            url = f'http://127.0.0.1:{port}'
            deadline = time.monotonic() + 45
            while not probe_health(url, marker):
                assert launcher.poll() is None, log.read_text('utf-8')
                assert time.monotonic() < deadline, log.read_text('utf-8')
                time.sleep(0.2)
            repeated = run_script('launch.py', '--port', str(port), '--no-browser',
                                  cwd=tmp_path, env=env)
            assert repeated.returncode == 0, repeated.stderr + repeated.stdout
            assert 'đang chạy' in repeated.stdout
            cli = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'motionsense_app.cli',
                                  'serve', '--port', str(port), '--no-browser'],
                                 cwd=ROOT, env=env, capture_output=True,
                                 encoding='utf-8', timeout=45, check=False)
            assert cli.returncode == 0, cli.stderr + cli.stdout
            assert 'đang chạy' in cli.stdout
            other = run_script('launch.py', '--port', str(port), '--no-browser')
            assert other.returncode == 2
            assert probe_health(url, marker)
            launcher.send_signal(signal.CTRL_BREAK_EVENT)
            launcher.wait(timeout=25)
            assert not probe_health(url, marker)
        finally:
            stop_child(launcher, timeout=5)
    text = (installed_state / 'var/logs/app.log').read_text('utf-8')
    assert 'Application shutdown complete' in text
    assert 'MotionSense' in log.read_text('utf-8')


def test_installer_preserves_incomplete_existing_environment(tmp_path):
    from scripts.install import install

    root = tmp_path / 'MotionSense môi trường chưa hoàn tất'
    root.mkdir()
    (root / '.venv').mkdir()
    sentinel = root / '.venv/user.txt'
    sentinel.write_text('giữ nguyên', encoding='utf-8')
    assert install(root) == 2
    assert sentinel.read_text('utf-8') == 'giữ nguyên'
    assert not (root / '.venv/Scripts/python.exe').exists()
    assert 'môi trường' in (root / 'var/logs/install.log').read_text('utf-8').lower()


def test_installer_lock_excludes_second_process(tmp_path):
    from motionsense_app.ownership import ProcessLock

    state = tmp_path / 'Cài đặt có dấu'
    lock = ProcessLock(state / 'var/logs/.install.lock')
    assert lock.acquire()
    try:
        result = run_script('install.py', env=dict(os.environ,
                            MOTIONSENSE_STATE_ROOT=str(state)), cwd=tmp_path)
        assert result.returncode == 2, result.stderr + result.stdout
        assert 'đang' in result.stderr.lower()
    finally:
        lock.release()


@pytest.mark.real_data
def test_installer_reuses_actual_environment_build_dataset_models(installed_state, tmp_path):
    from scripts.preflight import digest

    paths = [p for folder in ('datasets', 'models')
             for p in (installed_state / 'var' / folder).rglob('*') if p.is_file()]
    paths.extend([ROOT / '.venv/pyvenv.cfg', ROOT / 'frontend/dist/build-manifest.json'])
    before = {p: (digest(p), p.stat().st_mtime_ns) for p in paths}
    # Deliberately exclude Node/npm. An already validated build is usable without them.
    env = dict(os.environ, MOTIONSENSE_STATE_ROOT=str(installed_state), PATH='')
    result = run_script('install.py', cwd=tmp_path, env=env)
    assert result.returncode == 0, result.stderr + result.stdout
    assert 'sẵn sàng' in result.stdout
    assert before == {p: (digest(p), p.stat().st_mtime_ns) for p in paths}
    log = (installed_state / 'var/logs/install.log').read_text('utf-8')
    assert 'pip check' in log
    assert 'Huấn luyện' not in log
    assert 'npm ci' not in log


def test_batch_check_from_unrelated_directory(tmp_path):
    result = subprocess.run(['cmd.exe', '/d', '/c', str(ROOT / 'Khoi_dong.bat'), '--check'],
                            cwd=tmp_path, capture_output=True, encoding='utf-8', timeout=40,
                            check=False)
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(result.stdout) == {'ready': True, 'missing': []}


def test_check_script_emits_utf8_even_with_legacy_stdio(tmp_path):
    env = dict(os.environ, MOTIONSENSE_STATE_ROOT=str(tmp_path),
               PYTHONUTF8='0', PYTHONIOENCODING='ascii')
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/launch.py'), '--port', '0'],
                            cwd=tmp_path, env=env, capture_output=True,
                            encoding='utf-8', timeout=10, check=False)
    assert result.returncode == 2
    assert 'Cổng' in result.stderr
    assert 'UnicodeEncodeError' not in result.stderr


def test_runtime_lock_rejects_invalid_hash(tmp_path):
    from scripts.preflight import locked_versions

    (tmp_path / 'requirements.lock.txt').write_text(
        'fastapi==0.142.2 --hash=sha256:not-a-digest\n', encoding='utf-8')
    with pytest.raises(ValueError):
        locked_versions(tmp_path)


def test_manifest_can_recover_interrupted_manifest_write(tmp_path):
    from scripts.preflight import frontend_ready, write_build_manifest

    frontend = tmp_path / 'frontend'
    (frontend / 'dist/assets').mkdir(parents=True)
    for name in ('package.json', 'package-lock.json', 'index.html', 'dist/index.html',
                 'dist/assets/app.js', 'dist/.build-manifest.tmp'):
        (frontend / name).write_text('content', encoding='utf-8')
    write_build_manifest(tmp_path)
    assert frontend_ready(tmp_path)


@pytest.mark.real_data
def test_preflight_distinguishes_missing_models_from_current_dataset(installed_state, tmp_path):
    import shutil

    shutil.rmtree(installed_state / 'var/models')
    env = dict(os.environ, MOTIONSENSE_STATE_ROOT=str(installed_state))
    result = run_script('launch.py', '--check', cwd=tmp_path, env=env)
    assert result.returncode == 2
    assert json.loads(result.stdout) == {'ready': False, 'missing': ['models']}
    pointer = installed_state / 'var/datasets/current.json'
    pointer.write_text('{"dataset_id":"broken"}', encoding='utf-8')
    result = run_script('launch.py', '--check', cwd=tmp_path, env=env)
    assert result.returncode == 2
    assert json.loads(result.stdout) == {'ready': False, 'missing': ['dataset', 'models']}


@pytest.mark.real_data
def test_installer_batch_reuses_from_unrelated_cwd(installed_state, tmp_path):
    result = subprocess.run(['cmd.exe', '/d', '/c', str(ROOT / 'Cai_dat.bat')],
                            cwd=tmp_path, env=dict(os.environ,
                                MOTIONSENSE_STATE_ROOT=str(installed_state)),
                            capture_output=True, encoding='utf-8', timeout=90, check=False)
    assert result.returncode == 0, result.stderr + result.stdout
    assert 'sẵn sàng' in result.stdout


def test_windows_batch_packaging_is_utf8_crlf():
    # Artifact encoding is a distribution contract, in addition to real cmd.exe tests.
    for name in ('Cai_dat.bat', 'Khoi_dong.bat'):
        content = (ROOT / name).read_bytes()
        content.decode('utf-8')
        assert b'\r\n' in content
        assert b'\n' not in content.replace(b'\r\n', b'')


@pytest.mark.real_data
def test_actual_ctrl_c_in_private_windows_console(installed_state, tmp_path):
    # A private console permits a real Ctrl+C broadcast without touching the harness
    # or any unrelated process. The driver owns every process in that console.
    result = subprocess.run(
        [sys.executable, '-X', 'utf8', str(ROOT / 'tests/windows_console_driver.py'),
         str(ROOT), str(free_port())], cwd=tmp_path,
        env=dict(os.environ, MOTIONSENSE_STATE_ROOT=str(installed_state), PYTHONUTF8='1'),
        creationflags=subprocess.CREATE_NEW_CONSOLE, capture_output=True,
        encoding='utf-8', timeout=65, check=False)
    assert result.returncode == 0, result.stderr + result.stdout
    assert 'Ctrl+C: shutdown verified' in result.stdout
    text = (installed_state / 'var/logs/app.log').read_text('utf-8')
    assert 'Application shutdown complete' in text


def test_health_probe_does_not_follow_another_services_redirect():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from scripts.lifecycle import probe_health

    visits = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            visits.append(self.path)
            if self.path == '/api/health':
                self.send_response(302)
                self.send_header('Location', '/foreign-target')
                self.end_headers()
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{}')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        assert not probe_health(f'http://127.0.0.1:{server.server_port}', 'marker')
        assert visits == ['/api/health']
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_relative_state_environment_is_anchored_from_unrelated_cwd(tmp_path):
    env = dict(os.environ, MOTIONSENSE_STATE_ROOT='trạng thái riêng', PYTHONPATH=str(ROOT),
               PYTHONUTF8='1')
    result = subprocess.run(
        [sys.executable, '-c', ('import json; from motionsense_app.settings import Settings; '
                               'print(json.dumps(str(Settings.default().var_root)))')],
        cwd=tmp_path, env=env, capture_output=True, encoding='utf-8', timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == str(ROOT / 'trạng thái riêng' / 'var')
