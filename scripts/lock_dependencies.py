"""Developer-only lock regeneration; never called by the product installer."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preflight import (
    inspect_environment,
    locked_versions,
    utf8_env,
    utf8_stdio,
    valid_interpreter,
    venv_python,
)


def main() -> int:
    utf8_stdio()
    try:
        if not valid_interpreter(inspect_environment(ROOT), ROOT):
            raise ValueError('Cần môi trường CPython Windows 3.13 64-bit của dự án.')
        for manifest, lock in [('requirements.in', 'requirements.lock.txt'),
                               ('requirements-dev.in', 'requirements-dev.lock.txt')]:
            command = [str(venv_python(ROOT)), '-m', 'piptools', 'compile',
                       '--generate-hashes', '--resolver=backtracking',
                       '--pip-args=--timeout 20 --retries 2', '--output-file',
                       str(ROOT / lock), str(ROOT / manifest)]
            subprocess.run(command, cwd=ROOT, env=utf8_env(), timeout=900, check=True)
            locked_versions(ROOT, lock)
        subprocess.run([str(venv_python(ROOT)), '-m', 'pip', 'check'],
                       cwd=ROOT, env=utf8_env(), timeout=60, check=True)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f'Không thể khóa phụ thuộc: {exc}. Kiểm tra mạng, pip-tools và manifest.',
              file=sys.stderr)
        return 2
    print('Đã tạo lock có hash. Cài các lock và chạy kiểm thử trước khi phát hành.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
