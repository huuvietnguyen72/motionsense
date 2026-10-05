"""Windows installer steps contained in an unnamed, non-breakaway Job Object."""

import ctypes
import json
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path


class CleanupUnconfirmed(RuntimeError):
    """Forced/uncertain termination requires reboot-gated installer recovery."""

    def __init__(self, job, child, assigned, cause, original):
        super().__init__(f'{original or "Bước thiết lập kết thúc"}; cleanup: {cause}; '
                         'Cây thiết lập bị dừng bất thường; cần xác nhận reboot trước retry.')
        self.job, self.child, self.assigned = job, child, assigned

    def retry(self):
        if self.assigned:
            self.job.stop_and_wait()
        elif self.child is not None and self.child.poll() is None:
            self.child.terminate()
        if self.child is not None:
            self.child.wait(timeout=5)
        self.job.close()
        if self in _unconfirmed:
            _unconfirmed.remove(self)


# Also retain ownership for callers other than install(); never rely on exception
# traceback lifetime or Popen's destructor to protect an unconfirmed tree.
_unconfirmed = []


class _BasicLimits(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong),
                ('PerJobUserTimeLimit', ctypes.c_longlong), ('LimitFlags', wintypes.DWORD),
                ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t),
                ('ActiveProcessLimit', wintypes.DWORD), ('Affinity', ctypes.c_size_t),
                ('PriorityClass', wintypes.DWORD), ('SchedulingClass', wintypes.DWORD)]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        'ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
        'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [('BasicLimitInformation', _BasicLimits), ('IoInfo', _IoCounters),
                ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]


class _Accounting(ctypes.Structure):
    _fields_ = [(name, ctypes.c_longlong) for name in (
        'TotalUserTime', 'TotalKernelTime', 'ThisPeriodTotalUserTime', 'ThisPeriodTotalKernelTime')
    ] + [(name, wintypes.DWORD) for name in (
        'TotalPageFaultCount', 'TotalProcesses', 'ActiveProcesses', 'TotalTerminatedProcesses')]


class _WindowsJob:
    def __init__(self):
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                   ctypes.c_void_p, wintypes.DWORD]
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.TerminateJobObject.restype = wintypes.BOOL
        kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                     ctypes.c_void_p, wintypes.DWORD,
                                                     ctypes.c_void_p]
        kernel.QueryInformationJobObject.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE,
                                         ctypes.POINTER(wintypes.BOOL)]
        kernel.IsProcessInJob.restype = wintypes.BOOL
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.process_handles = []
        self.kernel = kernel
        self.handle = kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _ExtendedLimits()
        # Neither BREAKAWAY_OK nor SILENT_BREAKAWAY_OK is allowed. The handle is
        # non-inheritable; only this controller can close/terminate its owned job.
        limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits),
                                              ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign(self, child: subprocess.Popen) -> None:
        if not self.kernel.AssignProcessToJobObject(self.handle, int(child._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def active_processes(self) -> int:
        accounting = _Accounting()
        if not self.kernel.QueryInformationJobObject(
                self.handle, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return accounting.ActiveProcesses

    def stop_and_wait(self, timeout: float = 5) -> None:
        deadline = time.monotonic() + timeout
        self.capture_processes(deadline)
        # Termination is attempted even when accounting is unavailable. A failed
        # request is never proof of completion; keep the job open for a retry.
        if not self.kernel.TerminateJobObject(self.handle, 2):
            raise ctypes.WinError(ctypes.get_last_error())
        self.capture_processes(deadline)
        while self.active_processes():
            if time.monotonic() >= deadline:
                raise RuntimeError('Không xác nhận được cây tiến trình thiết lập đã dừng.')
            time.sleep(0.02)
        # Windows accounting can reach zero before process handles are signaled.
        # Wait on the known handles as best-effort cleanup, never as proof that
        # the instantaneous inventory covered every descendant. Guards stay.
        for handle in self.process_handles:
            remaining = max(0, int((deadline - time.monotonic()) * 1000))
            result = self.kernel.WaitForSingleObject(handle, remaining)
            if result == 0xFFFFFFFF:
                raise ctypes.WinError(ctypes.get_last_error())
            if result != 0:
                raise RuntimeError('Owned process termination wait timed out.')

    def capture_processes(self, deadline: float | None = None) -> None:
        if deadline is None:
            deadline = time.monotonic() + 5
        capacity = 64
        for _ in range(8):
            if time.monotonic() >= deadline:
                raise RuntimeError('Job inventory capture deadline expired.')
            class ProcessIds(ctypes.Structure):
                _fields_ = [('assigned', wintypes.DWORD), ('count', wintypes.DWORD),
                            ('ids', ctypes.c_size_t * capacity)]

            ids = ProcessIds()
            if self.kernel.QueryInformationJobObject(
                    self.handle, 3, ctypes.byref(ids), ctypes.sizeof(ids), None):
                break
            error = ctypes.get_last_error()
            if error != 234:  # ERROR_MORE_DATA
                raise ctypes.WinError(error)
            capacity = max(capacity * 2, ids.assigned)
            if capacity > 65536:
                raise RuntimeError('Job inventory capacity limit exceeded.')
        else:
            raise RuntimeError('Job inventory resize retry limit exceeded.')
        for pid in ids.ids[:ids.count]:
            if time.monotonic() >= deadline:
                raise RuntimeError('Job inventory handle capture deadline expired.')
            handle = self.kernel.OpenProcess(0x100000 | 0x1000, False, pid)
            if not handle:
                error = ctypes.get_last_error()
                if error == 87:  # Process already disappeared.
                    continue
                raise ctypes.WinError(error)
            owned = wintypes.BOOL()
            if not self.kernel.IsProcessInJob(handle, self.handle, ctypes.byref(owned)):
                error = ctypes.get_last_error()
                self.kernel.CloseHandle(handle)
                raise ctypes.WinError(error)
            if owned.value:
                self.process_handles.append(handle)
            else:
                # PID reuse: never terminate or wait on a foreign process.
                self.kernel.CloseHandle(handle)

    def close(self) -> None:
        for handle in self.process_handles:
            self.kernel.CloseHandle(handle)
        self.process_handles.clear()
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def run_owned_step(command: list[str], cwd: Path, env: dict, log, timeout: float,
                   *, output=None) -> int:
    job = _WindowsJob()
    child = None
    assigned = False
    confirmed = False
    exit_code = None
    try:
        # Use the real base interpreter, not .venv's redirector: it can spawn a
        # Python child before AssignProcessToJobObject. This stdlib-only worker
        # cannot spawn anything until our stdin gate is released after assignment.
        interpreter = Path(sys.base_prefix) / 'python.exe'
        worker = Path(__file__).with_name('step_worker.py')
        child = subprocess.Popen(
            [str(interpreter), '-I', '-S', '-X', 'utf8', str(worker), json.dumps(command)],
            cwd=cwd, env=env, stdin=subprocess.PIPE,
            stdout=output if output is not None else log, stderr=log,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        job.assign(child)
        assigned = True
        child.stdin.write(b'1')
        child.stdin.flush()
        child.stdin.close()
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                exit_code = child.wait(timeout=min(0.2, remaining))
                return exit_code
            except subprocess.TimeoutExpired:
                pass
    finally:
        original = sys.exception()
        if original is None and exit_code is not None:
            original = f'{subprocess.list2cmdline(command)} (exit {exit_code})'
        try:
            if assigned:
                # Successful tools with no active processes follow normal setup.
                # Every other path is abnormal: snapshot waits cannot authorize
                # same-boot installer retry, even when cleanup appears successful.
                if exit_code == 0 and sys.exception() is None and job.active_processes() == 0:
                    confirmed = True
                    log.write('Windows step completed normally (active=0).\n')
                    log.flush()
                else:
                    try:
                        job.stop_and_wait()
                        child.wait(timeout=5)
                    except (OSError, RuntimeError, subprocess.SubprocessError):
                        # Bounded fallback with the same owned job, not kill-by-PID.
                        job.stop_and_wait()
                        child.wait(timeout=5)
                    log.write('Windows step tree stopped (active=0).\n')
                    log.write('Forced cleanup: inventory is incomplete evidence; reboot required.\n')
                    log.flush()
                    raise RuntimeError('Forced cleanup requires reboot-confirmed recovery.')
            elif child is not None:
                # Failed containment: gate was never released, so there are no
                # descendants. Terminate only this Popen-owned waiting worker.
                child.terminate()
                child.wait(timeout=5)
                confirmed = True
            else:
                confirmed = True
        except (OSError, RuntimeError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
            if not confirmed:
                pending = CleanupUnconfirmed(job, child, assigned, exc, original)
                _unconfirmed.append(pending)
                raise pending from exc
            raise
        finally:
            if child is not None and child.stdin is not None:
                child.stdin.close()
            if confirmed:
                job.close()
