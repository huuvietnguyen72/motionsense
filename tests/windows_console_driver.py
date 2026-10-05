"""Test utility: broadcast Ctrl+C only inside our newly created Windows console."""

import ctypes
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    root, port = Path(sys.argv[1]), int(sys.argv[2])
    sys.path.insert(0, str(root))
    from motionsense_app.settings import Settings
    from scripts.lifecycle import probe_health, stop_child

    state = Path(os.environ['MOTIONSENSE_STATE_ROOT'])
    marker = Settings(root, state).instance_marker
    url = f'http://127.0.0.1:{port}'
    # Child inherits normal Ctrl+C handling; ignore it in the driver only afterward.
    with (state / 'console-launch.log').open('w', encoding='utf-8') as log:
        launcher = subprocess.Popen(
            [sys.executable, '-X', 'utf8', str(root / 'scripts/launch.py'),
             '--port', str(port), '--no-browser'], stdout=log, stderr=log)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            deadline = time.monotonic() + 40
            while not probe_health(url, marker):
                if launcher.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError('Private-console launcher did not become ready')
                time.sleep(0.1)
            # This driver runs with CREATE_NEW_CONSOLE, never the user's console.
            if not ctypes.windll.kernel32.GenerateConsoleCtrlEvent(0, 0):
                raise ctypes.WinError()
            launcher.wait(timeout=20)
            if launcher.returncode != 0 or probe_health(url, marker):
                raise RuntimeError('Ctrl+C did not cleanly stop the owned server')
            print('Ctrl+C: shutdown verified')
        finally:
            stop_child(launcher, timeout=3)


if __name__ == '__main__':
    main()
