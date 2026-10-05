"""Local OS locks: live owners exclude peers and process death releases ownership.

Lock files stay in place so peers always lock the same filesystem object.
No PID heuristics, global system configuration, or SQLite lease timeouts.
"""

import errno
import os
from pathlib import Path

from motionsense_app.errors import DomainError

if os.name == "nt":
    import msvcrt
else:
    import fcntl


class ProcessLock:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._stream = None

    @property
    def held(self) -> bool:
        return self._stream is not None

    def acquire(self) -> bool:
        if self.held:
            return True
        stream = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            stream = self.path.open("a+b")
            # Windows byte-range locking needs a stable byte, even in a new lock file.
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            try:
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    return False
                raise
            self._stream = stream
            stream = None
            return True
        except OSError as exc:
            raise DomainError(
                "STORAGE_ERROR", "Không thể khóa thư mục trạng thái. Hãy kiểm tra ổ đĩa.", 500,
            ) from exc
        finally:
            if stream is not None:
                stream.close()

    def release(self) -> None:
        stream = self._stream
        if stream is None:
            return
        try:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()
            self._stream = None
