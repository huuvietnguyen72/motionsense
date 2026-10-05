"""Retryable Windows setup. Runtime imports run only in the project environment."""

import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motionsense_app.errors import DomainError
from motionsense_app.ownership import ProcessLock
from motionsense_app.settings import Settings
from scripts.preflight import (
    check_installation,
    environment_ready,
    frontend_ready,
    inspect_environment,
    locked_versions,
    utf8_env,
    utf8_stdio,
    valid_interpreter,
    venv_python,
    write_build_manifest,
)
from scripts.process_tree import CleanupUnconfirmed, run_owned_step

_pending_cleanup = []


def windows_boot_identity() -> str:
    """Read Windows' boot GUID, independent of clocks, PIDs and sleep/resume."""
    query = ctypes.WinDLL('ntdll').NtQuerySystemInformation
    query.argtypes = [ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong,
                      ctypes.POINTER(ctypes.c_ulong)]
    query.restype = ctypes.c_long
    # SYSTEM_BOOT_ENVIRONMENT_INFORMATION: GUID, firmware enum, padding, flags.
    buffer = ctypes.create_string_buffer(32)
    length = ctypes.c_ulong()
    status = query(90, buffer, ctypes.sizeof(buffer), ctypes.byref(length))
    if status < 0 or length.value < 16 or buffer.raw[:16] == bytes(16):
        raise RuntimeError('Không đọc được định danh boot Windows; chưa cho phép thiết lập.')
    return buffer.raw[:16].hex()


def guard_record(boot_id: str) -> bytes:
    if (not isinstance(boot_id, str) or len(boot_id) != 32
            or boot_id == '0' * 32
            or any(char not in '0123456789abcdef' for char in boot_id)):
        raise RuntimeError('Định danh boot Windows không hợp lệ.')
    return json.dumps({'schema_version': 1, 'boot_id': boot_id}).encode('utf-8')


def write_lock_record(lock, data: bytes) -> None:
    # Keep byte zero and the filesystem object used by ProcessLock intact.
    # The redundant record prevents deleting .pending from authorizing retry.
    with lock.path.open('r+b') as stream:
        stream.seek(1)
        stream.write(data)
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())


def check_guard_recovery(lock, path, boot_id: str) -> None:
    with lock.path.open('rb') as stream:
        stream.seek(1)
        stored = stream.read(4097)
    records = [stored] if stored else []
    if path.exists():
        with path.open('rb') as stream:
            records.append(stream.read(4097))
    for data in records:
        try:
            record = json.loads(data)
            valid = (isinstance(record, dict) and set(record) == {'schema_version', 'boot_id'}
                     and type(record['schema_version']) is int and record['schema_version'] == 1
                     and guard_record(record['boot_id']))
        except (ValueError, KeyError, RuntimeError, UnicodeError, RecursionError):
            valid = False
        if not valid:
            raise RuntimeError(f'Guard .pending thiếu/hỏng định danh boot: {path}. '
                               'Không tự xóa guard; cần khôi phục bản ghi boot hợp lệ. '
                               'Khởi động lại Windows không thay thế bản ghi bị mất.')
        if record['boot_id'] == boot_id:
            raise RuntimeError(f'Guard .pending còn thuộc boot hiện tại: {path}. '
                               'Khởi động lại Windows (Restart) rồi chạy lại Cai_dat.bat; '
                               'không xóa guard thủ công.')


def clear_guards(locks, guards) -> None:
    for lock, guard in zip(locks, guards, strict=True):
        guard.unlink(missing_ok=True)
        write_lock_record(lock, b'')


def retry_pending_cleanup(boot_id: str | None = None) -> None:
    """Only a changed boot authorizes recovery, never an inventory snapshot."""
    boot_id = boot_id or windows_boot_identity()
    for pending, locks, guards, previous_boot in list(_pending_cleanup):
        if previous_boot == boot_id:
            raise RuntimeError('Guard .pending yêu cầu Khởi động lại Windows (Restart); '
                               'cleanup cùng boot không cho phép retry.')
        pending.retry()
        clear_guards(locks, guards)
        for lock in reversed(locks):
            lock.release()
        _pending_cleanup.remove((pending, locks, guards, previous_boot))


def run_step(command: list[str], cwd: Path, log, timeout: float = 900, *, output=None) -> None:
    message = subprocess.list2cmdline(command)
    print(message, flush=True)
    log.write(message + '\n')
    log.flush()
    code = run_owned_step(command, cwd, utf8_env(), log, timeout, output=output)
    if code:
        raise RuntimeError(f'Bước thiết lập thất bại (exit {code}): {message}')


def tool_version(command: list[str], cwd: Path, log) -> str:
    with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as output:
        run_step(command, cwd, log, timeout=30, output=output)
        output.seek(0)
        return output.read().strip()


def bootstrap(root: Path) -> None:
    from motionsense_app.data.importer import download_archive, prepare_dataset
    from motionsense_app.data.store import DatasetStore
    from motionsense_app.models.registry import ModelRegistry
    from motionsense_app.models.training import train_model
    settings = Settings.default(root)
    store = DatasetStore(settings.data_dir)
    if not store.info()['ready']:
        # A corrupt or user-selected pointer must not be silently overwritten.
        if (settings.data_dir / 'current.json').exists():
            raise RuntimeError('Dữ liệu hiện tại hỏng. Hãy kiểm tra/khôi phục bản sao var/datasets; '
                               'không tự ghi đè snapshot hiện có.')
        archive = settings.var_root / 'source/uci-har.zip'
        if not archive.is_file():
            print('Tải dữ liệu UCI HAR từ nguồn chính thức.', flush=True)
            archive = download_archive(archive)
        prepare_dataset(archive, settings.data_dir)
        store = DatasetStore(settings.data_dir)
    dataset_id = store.info()['dataset_id']
    registry = ModelRegistry(settings.model_dir)
    ready = {item['profile'] for item in registry.list()
             if item['status'] == 'ready' and item['dataset_id'] == dataset_id
             and item['trees'] == 50 and item['seed'] == 42}
    for profile in ('full', 'reduced'):
        if profile not in ready:
            print(f'Huấn luyện {profile}, 50 cây, seed 42 (phiên bản mới).', flush=True)
            registry.publish(train_model(store, profile, 50))
        else:
            print(f'Dùng lại mô hình {profile}/50 tương thích.', flush=True)


def ensure_frontend(root: Path, log) -> None:
    if frontend_ready(root):
        print('Dùng lại giao diện đã kiểm chứng; không cần Node.', flush=True)
        return
    node, npm = shutil.which('node.exe'), shutil.which('npm.cmd')
    if not node or not npm:
        raise RuntimeError('Cần Node.js >=22.12 và npm >=10 để build giao diện; '
                           'cài Node rồi mở lại cửa sổ Cai_dat.bat.')
    node_version = tool_version([node, '--version'], root, log)
    npm_version = tool_version([npm, '--version'], root, log)
    if tuple(map(int, node_version.lstrip('v').split('.'))) < (22, 12, 0):
        raise RuntimeError(f'Node {node_version} quá cũ; cần >=22.12.')
    if int(npm_version.split('.')[0]) < 10:
        raise RuntimeError(f'npm {npm_version} quá cũ; cần >=10.')
    run_step([npm, 'ci', '--no-audit', '--no-fund'], root / 'frontend', log)
    run_step([npm, 'run', 'build'], root / 'frontend', log)
    write_build_manifest(root)
    if not frontend_ready(root):
        raise RuntimeError('Giao diện build không hợp lệ; xem install.log.')


def install(root: Path = ROOT) -> int:
    utf8_stdio()
    root = root.resolve()
    settings = Settings.default(root)
    state_root = settings.state_root or root
    log_path = state_root / 'var/logs/install.log'
    # Project lock protects shared .venv/dist even when state roots differ.
    locks = [ProcessLock(root / 'var/logs/.setup.lock'),
              ProcessLock(state_root / 'var/logs/.install.lock')]
    guards = []
    retain = False
    try:
        boot_id = windows_boot_identity()
        data = guard_record(boot_id)
        retry_pending_cleanup(boot_id)
        for lock in locks:
            if not lock.acquire():
                raise RuntimeError('Đang có bộ cài đặt khác. Chờ hoàn tất rồi thử lại.')
        # Persist BEFORE any owned tool starts: process exit releases OS locks,
        # but KILL_ON_JOB_CLOSE alone does not confirm completion. A stale guard
        # therefore blocks new installers even after this controller exits.
        paths = [lock.path.with_name(lock.path.name + '.pending') for lock in locks]
        for lock, path in zip(locks, paths, strict=True):
            check_guard_recovery(lock, path, boot_id)
        for lock, path in zip(locks, paths, strict=True):
            write_lock_record(lock, data)
            with path.open('wb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            guards.append(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('a', encoding='utf-8') as log:
            log.write('\nThiết lập MotionSense\n')
            log.flush()
            if (platform.python_implementation() != 'CPython' or sys.version_info[:2] != (3, 13)
                    or sysconfig.get_platform() != 'win-amd64'):
                raise RuntimeError('Cần CPython Windows 3.13 64-bit; chạy bằng Cai_dat.bat.')
            environment = root / '.venv'
            if environment.exists():
                if not venv_python(root).is_file():
                    raise RuntimeError('Môi trường .venv chưa hoàn tất. Hãy sao lưu và kiểm tra '
                                       '.venv; bộ cài không xóa môi trường hiện có.')
                if not valid_interpreter(inspect_environment(root), root):
                    raise RuntimeError('Môi trường .venv không phải CPython Windows 3.13 64-bit.')
            else:
                run_step([sys.executable, '-X', 'utf8', '-m', 'venv', str(environment)], root, log)
            locked_versions(root)
            if not environment_ready(root):
                run_step([str(venv_python(root)), '-m', 'pip', 'install', '--require-hashes',
                          '--timeout', '20', '--retries', '2', '-r',
                          str(root / 'requirements.lock.txt')], root, log)
            else:
                print('Dùng lại môi trường đúng runtime lock.', flush=True)
            run_step([str(venv_python(root)), '-m', 'pip', 'check'], root, log, timeout=60)
            ensure_frontend(root, log)
            run_step([str(venv_python(root)), '-X', 'utf8', '-c',
                      ('from pathlib import Path; from scripts.install import bootstrap; '
                       'import sys; bootstrap(Path(sys.argv[1]))'), str(root)], root, log,
                     timeout=1800)
            result = check_installation(root)
            if not result['ready']:
                raise RuntimeError('Thiết lập chưa hoàn tất: ' + ', '.join(result['missing']))
            log.write('MotionSense sẵn sàng.\n')
        print('MotionSense sẵn sàng. Chạy Khoi_dong.bat.', flush=True)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError,
            DomainError) as exc:
        if isinstance(exc, CleanupUnconfirmed):
            retain = True
            _pending_cleanup.append((exc, locks, guards, boot_id))
        message = (f'Thiết lập thất bại: {exc}\n'
                   f'Xem {log_path}; sửa nguyên nhân rồi chạy lại Cai_dat.bat.')
        if retain:
            message += ('\nGiữ khóa thiết lập vì cleanup chưa xác nhận; retry bị chặn. '
                        'Khởi động lại Windows (Restart) rồi chạy lại Cai_dat.bat; '
                        'bộ cài tự khôi phục khi định danh boot đã đổi. Không xóa guard: '
                        + ', '.join(map(str, guards)))
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open('a', encoding='utf-8') as log:
                log.write(message + '\n')
        except OSError:
            message += '\nKhông ghi được nhật ký; hãy kiểm tra quyền ghi và dung lượng ổ đĩa.'
        print(message, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('Đã ngừng thiết lập. Chạy lại Cai_dat.bat để tiếp tục.', file=sys.stderr)
        return 2
    finally:
        if not retain:
            clear_guards(locks[:len(guards)], guards)
            for lock in reversed(locks):
                lock.release()


if __name__ == '__main__':
    raise SystemExit(install())
