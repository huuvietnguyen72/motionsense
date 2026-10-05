"""Single loopback service; only the child we created belongs to this launcher."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

from motionsense_app.errors import DomainError
from motionsense_app.ownership import ProcessLock
from motionsense_app.settings import Settings
from scripts.preflight import utf8_env, venv_python


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def probe_health(url: str, marker: str, timeout: float = 1) -> bool:
    try:
        # Loopback must not go through a system HTTP proxy or follow foreign redirects.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        with opener.open(url + '/api/health', timeout=timeout) as response:
            body = json.loads(response.read(8192))
            return (response.status == 200 and response.geturl() == url + '/api/health'
                    and response.headers.get('X-MotionSense-Instance') == marker
                    and body.get('app') == 'motionsense' and body.get('status') == 'ok'
                    and body.get('data_ready') is True and body.get('models_ready') is True)
    except (OSError, ValueError, TypeError, AttributeError, urllib.error.URLError):
        return False


def port_occupied(port: int) -> bool:
    with socket.socket() as sock:
        if os.name == 'nt':
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            sock.bind(('127.0.0.1', port))
        except OSError:
            return True
    return False


def wait_for_health(child: subprocess.Popen, url: str, marker: str,
                    timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code = child.poll()
        if code is not None:
            raise RuntimeError(f'Dịch vụ dừng sớm (exit {code}); xem var/logs/app.log.')
        if probe_health(url, marker, timeout=min(1, max(0.01, deadline - time.monotonic()))):
            if child.poll() is not None:
                raise RuntimeError('Dịch vụ đã dừng trong lúc kiểm tra; xem var/logs/app.log.')
            return
        time.sleep(0.1)
    raise TimeoutError('Dịch vụ chưa sẵn sàng sau thời gian chờ; xem var/logs/app.log.')


def stop_child(child: subprocess.Popen, timeout: float = 15) -> None:
    if child.poll() is not None:
        return
    try:
        child.send_signal(signal.CTRL_BREAK_EVENT if os.name == 'nt' else signal.SIGINT)
        child.wait(timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)


def _interrupt(_signum, _frame):
    raise KeyboardInterrupt


def serve(root: Path, port: int = 8765, *, no_browser: bool = False) -> int:
    root = root.resolve()
    settings = Settings.default(root)
    url = f'http://127.0.0.1:{port}'
    marker = settings.instance_marker
    child = None
    lock = ProcessLock(settings.var_root / 'logs' / f'.launch-{port}.lock')
    previous = {}
    try:
        for sig in (signal.SIGINT, signal.SIGBREAK) if os.name == 'nt' else (signal.SIGINT,):
            previous[sig] = signal.signal(sig, _interrupt)
        deadline = time.monotonic() + 30
        while not lock.acquire():
            if time.monotonic() >= deadline:
                raise RuntimeError('Đang có trình khởi động khác. Hãy thử lại sau.')
            time.sleep(0.2)
        if probe_health(url, marker):
            print(f'MotionSense đang chạy tại {url}', flush=True)
            if not no_browser:
                webbrowser.open(url)
            return 0
        if port_occupied(port):
            print(f'Cổng {port} đang bận hoặc thuộc phiên bản khác. '
                  'Dùng Khoi_dong.bat --port 8766 (hoặc cổng trống khác).', file=sys.stderr)
            return 2
        logs = settings.var_root / 'logs'
        logs.mkdir(parents=True, exist_ok=True)
        command = [str(venv_python(root)), '-X', 'utf8', '-m', 'uvicorn',
                   'motionsense_app.main:create_app', '--factory', '--host', '127.0.0.1',
                   '--port', str(port), '--workers', '1', '--no-use-colors']
        with (logs / 'app.log').open('a', encoding='utf-8') as stream:
            child = subprocess.Popen(command, cwd=root, env=utf8_env(), stdout=stream,
                                     stderr=stream, creationflags=(
                                         subprocess.CREATE_NEW_PROCESS_GROUP
                                         if os.name == 'nt' else 0))
            wait_for_health(child, url, marker)
            lock.release()
            print(f'MotionSense sẵn sàng tại {url}. Nhấn Ctrl+C để dừng.', flush=True)
            if not no_browser:
                webbrowser.open(url)
            # Windows' infinite WaitForSingleObject delays Python SIGBREAK handlers.
            # Bounded child.wait keeps console shutdown responsive without busy waiting.
            while True:
                try:
                    return 0 if child.wait(timeout=0.2) == 0 else 1
                except subprocess.TimeoutExpired:
                    pass
    except KeyboardInterrupt:
        print('Đang dừng MotionSense và lưu trạng thái...', flush=True)
        return 0
    except (OSError, RuntimeError, TimeoutError, DomainError) as exc:
        print(f'Không thể khởi động: {exc}. Hãy kiểm tra nhật ký và chạy Cai_dat.bat.',
              file=sys.stderr)
        return 2
    finally:
        # Ignore repeated console interrupts while the owned child's lifespan closes.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if child is not None:
                stop_child(child)
        finally:
            lock.release()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
