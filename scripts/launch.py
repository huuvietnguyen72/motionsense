"""Launch the local application from any working directory."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preflight import check_installation, utf8_stdio


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    parser = argparse.ArgumentParser(description='Khởi động MotionSense tại máy.')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('Cổng phải trong khoảng 1–65535.')
    result = check_installation(ROOT)
    if args.check:
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result['ready'] else 2
    if not result['ready']:
        print('Chưa sẵn sàng: ' + ', '.join(result['missing']) +
              '. Hãy chạy Cai_dat.bat; xem var/logs/install.log.', file=sys.stderr)
        return 2
    from scripts.lifecycle import serve
    return serve(ROOT, args.port, no_browser=args.no_browser)


if __name__ == '__main__':
    raise SystemExit(main())
