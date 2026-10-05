"""Test-only independent monitor: postpone confirmation until the test permits it."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import cleanup_monitor


def main():
    directory = Path(os.environ['MOTIONSENSE_TEST_CLEANUP_GATE'])
    (directory / 'monitor.pid').write_text(str(os.getpid()), encoding='utf-8')
    original = cleanup_monitor.confirm_cleanup

    def pending(job, worker):
        if not (directory / 'permit-cleanup').exists():
            (directory / 'monitor-pending').touch()
            raise OSError('Injected independent cleanup/accounting unavailable')
        return original(job, worker)

    cleanup_monitor.confirm_cleanup = pending
    return cleanup_monitor.main()


if __name__ == '__main__':
    raise SystemExit(main())
