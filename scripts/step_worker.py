"""Stdlib-only installer worker. No child creation before the Job Object gate."""

import json
import subprocess
import sys


def main() -> int:
    if sys.stdin.buffer.read(1) != b'1':
        return 2
    command = json.loads(sys.argv[1])
    # Inherit the controlling job, absolute cwd/environment and log handles.
    # Tools cannot inherit the start-gate pipe or request job breakaway here.
    with subprocess.Popen(command, stdin=subprocess.DEVNULL) as child:
        return child.wait()


if __name__ == '__main__':
    raise SystemExit(main())
