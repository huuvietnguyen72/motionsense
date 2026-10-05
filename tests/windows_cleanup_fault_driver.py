"""Test-only installer faults; keeps native job open to model unconfirmed close."""

import json
import os
import sys
from pathlib import Path


def main():
    root, state, tree, mode = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    sys.path.insert(0, str(root))
    from scripts import install as installer
    from scripts import process_tree

    os.environ['MOTIONSENSE_STATE_ROOT'] = str(state)
    original_stop = process_tree._WindowsJob.stop_and_wait
    original_close = process_tree._WindowsJob.close
    original_release = installer.ProcessLock.release
    retained = []
    releases = []
    faulty = [False]

    def failed_stop(job, timeout=5):
        if not faulty[0]:
            return original_stop(job, timeout)
        if mode == 'cleanup-error':
            raise OSError('Injected cleanup failure')
        # The real bounded accounting-wait loop times out on stale accounting;
        # the injected terminate request reports success but has not completed.
        job.active_processes = lambda: 1
        job.kernel.TerminateJobObject = lambda handle, code: True
        return original_stop(job, timeout=0.05)

    def unconfirmed_close(job):
        if faulty[0]:
            retained.append(job)
        else:
            original_close(job)

    def observed_release(lock):
        if lock.held:
            releases.append(lock.path.name)
        original_release(lock)

    def frontend(root, log):
        faulty[0] = True
        installer.run_step(['cmd.exe', '/d', '/c', str(tree / 'npm-hang.cmd')],
                           tree, log, timeout=2)

    process_tree._WindowsJob.stop_and_wait = failed_stop
    process_tree._WindowsJob.close = unconfirmed_close
    installer.ProcessLock.release = observed_release
    installer.ensure_frontend = frontend
    result = installer.install(root)
    (tree / 'controller-evidence.json').write_text(json.dumps({
        'exit': result, 'ordinary_guard_release_calls': releases,
    }), encoding='utf-8')
    return result


if __name__ == '__main__':
    raise SystemExit(main())
