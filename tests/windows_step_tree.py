"""Test-only cmd/npm-shaped tree with a detached-group descendant still writing."""

import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    role, directory = sys.argv[1], Path(sys.argv[2])
    (directory / f'{role}.pid').write_text(str(os.getpid()), encoding='utf-8')
    if role == 'parent':
        subprocess.Popen([sys.executable, '-X', 'utf8', __file__, 'grandchild', str(directory)],
                         creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        deadline = time.monotonic() + 10
        while not (directory / 'grandchild.pid').exists():
            if time.monotonic() > deadline:
                raise RuntimeError('Grandchild did not start')
            time.sleep(0.02)
        (directory / 'ready').write_text('ready', encoding='utf-8')
    deadline = time.monotonic() + 60
    with (directory / f'{role}.heartbeat').open('a', encoding='utf-8') as stream:
        while time.monotonic() < deadline:
            stream.write('writing\n')
            stream.flush()
            time.sleep(0.05)


if __name__ == '__main__':
    main()
