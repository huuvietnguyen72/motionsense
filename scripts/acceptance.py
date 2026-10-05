"""Actual acceptance on an owned loopback server and disposable Vietnamese state."""

import argparse
import csv
import hashlib
import io
import json
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx

from motionsense_app.data.store import DatasetStore
from motionsense_app.models.registry import ModelRegistry
from motionsense_app.settings import Settings
from scripts.lifecycle import port_occupied, stop_child, wait_for_health
from scripts.preflight import frontend_ready, utf8_env, utf8_stdio, write_build_manifest

TEMP = Path(r'C:\Users\Admin\AppData\Local\Temp\opencode')


def fingerprint(var):
    """Protected user artifacts; operational logs are deliberately separate."""
    result = {}
    if var.exists():
        for path in sorted(var.rglob('*')):
            if path.is_file() and path.relative_to(var).parts[0] != 'logs':
                with path.open('rb') as stream:
                    result[path.relative_to(var).as_posix()] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return result


def copy_ready_state(root, state):
    store = DatasetStore(root / 'var/datasets')
    data = store.info()
    if not data['ready'] or data['split_counts'] != {'train': 7352, 'test': 2947}:
        raise RuntimeError('Thiếu dữ liệu thật. Chạy Cai_dat.bat.')
    registry = ModelRegistry(root / 'var/models')
    models = [m for m in registry.list() if m['status'] == 'ready' and m['dataset_id'] == data['dataset_id']]
    if not {'full', 'reduced'} <= {m['profile'] for m in models if m['trees'] == 50 and m['seed'] == 42}:
        raise RuntimeError('Thiếu model mặc định thật. Chạy Cai_dat.bat.')
    datasets = state / 'var/datasets'
    datasets.mkdir(parents=True)
    shutil.copy2(root / 'var/datasets/current.json', datasets / 'current.json')
    shutil.copytree(root / 'var/datasets' / data['dataset_id'], datasets / data['dataset_id'])
    for model in models:
        shutil.copytree(root / 'var/models' / model['model_id'], state / 'var/models' / model['model_id'])
    return data, models


class Acceptance:
    def __init__(self, port, evidence):
        self.port, self.evidence = port, evidence
        self.child = None
        self.commands, self.checks = [], []
        self.env = utf8_env()
        self.url = f'http://127.0.0.1:{port}'
        self.client = httpx.Client(base_url=self.url, timeout=30, trust_env=False)

    def command(self, args, cwd=ROOT, timeout=1800):
        index = len(self.commands) + 1
        started = time.monotonic()
        log = self.evidence / f'command-{index:02}.log'
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run(args, cwd=cwd, env=self.env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=timeout, check=False)
        entry = {'command': subprocess.list2cmdline([str(a) for a in args]), 'cwd': str(cwd),
                 'exit_code': result.returncode, 'seconds': round(time.monotonic() - started, 3), 'log': log.name}
        self.commands.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        if result.returncode:
            raise RuntimeError(f'Command failed: {entry}')

    def start(self, state, *, restart=False, ready=True):
        if port_occupied(self.port):
            if not restart:
                raise RuntimeError(f'Cổng {self.port} bận; không dừng tiến trình lạ.')
            previous_port = self.port
            with socket.socket() as reservation:
                reservation.bind(('127.0.0.1', 0))
                self.port = reservation.getsockname()[1]
            self.client.close()
            self.url = f'http://127.0.0.1:{self.port}'
            self.client = httpx.Client(base_url=self.url, timeout=30, trust_env=False)
            self.checks.append({'gate': 'restart_port_selection', 'previous': previous_port,
                               'selected': self.port, 'reason': 'Windows bind remains busy after owned exit; no foreign process killed'})
        self.env['MOTIONSENSE_STATE_ROOT'] = str(state)
        command = [sys.executable, '-X', 'utf8', '-m', 'uvicorn',
                   'motionsense_app.main:create_app', '--factory', '--host', '127.0.0.1',
                   '--port', str(self.port), '--workers', '1', '--no-use-colors']
        with (self.evidence / 'server.log').open('a', encoding='utf-8') as stream:
            self.child = subprocess.Popen(command, cwd=ROOT, env=self.env, stdout=stream,
                                          stderr=stream, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        marker = Settings(ROOT, state).instance_marker
        if ready:
            wait_for_health(self.child, self.url, marker)
        else:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if self.child.poll() is not None:
                    raise RuntimeError('Unready server exited before health.')
                try:
                    response = self.client.get('/api/health', timeout=1)
                    if response.status_code == 200 and response.headers.get('X-MotionSense-Instance') == marker:
                        assert response.json()['data_ready'] is False
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.1)
            else:
                raise TimeoutError('Unready server health timeout.')
        self.checks.append({'gate': 'owned_start', 'pid': self.child.pid, 'command': subprocess.list2cmdline(command), 'state': str(state)})

    def stop(self, forced=False):
        if self.child is not None:
            pid = self.child.pid
            if forced and self.child.poll() is None:
                self.child.kill()
                self.child.wait(timeout=10)
            else:
                stop_child(self.child)
            assert self.child.poll() is not None
            self.checks.append({'gate': 'owned_stop', 'pid': pid, 'forced': forced, 'exit_code': self.child.returncode})
            self.child = None

    def api(self, method, path, body=None):
        response = self.client.request(method, path, json=body)
        response.raise_for_status()
        return response.json()

    def session_gate(self, state, data, models):
        reduced = next(m for m in models if m['profile'] == 'reduced' and m['trees'] == 50)
        person = next(s for s in data['subjects'] if s['split'] == 'test')
        session = self.api('POST', '/api/sessions', {'dataset_id': data['dataset_id'], 'split': 'test',
                           'subject_id': person['subject_id'], 'model_id': reduced['model_id']})
        sid = session['session_id']
        base = '/api/sessions/' + sid
        owner = 'acceptance-owner'
        self.api('POST', base + '/attach', {'client_id': owner})

        def control(action, **kwargs):
            return self.api('POST', base + '/control', dict(client_id=owner, action=action, **kwargs))

        def advance(target):
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                saved = self.api('GET', base)
                if saved['cursor'] >= target:
                    return saved
                self.api('POST', base + '/advance', {'client_id': owner, 'expected_cursor': saved['cursor']})
                time.sleep(0.15)
            raise TimeoutError('Không advance đủ mẫu.')

        control('start')
        saved = advance(3)
        saved = control('pause')
        for forced in (False, True):
            if forced:
                control('resume')
                saved = advance(saved['cursor'] + 1)
            self.stop(forced=forced)
            self.start(state, restart=True)
            recovered = self.api('GET', base)
            assert recovered['status'] == 'paused'
            assert saved['cursor'] <= recovered['cursor'] <= saved['cursor'] + int(forced)
            self.api('POST', base + '/attach', {'client_id': owner})
            control('resume')
            saved = advance(recovered['cursor'] + 1)
            saved = control('pause')
            self.checks.append({'gate': 'forced_recovery' if forced else 'restart', 'session': saved})
        control('resume')
        time.sleep(3.8)
        assert self.api('GET', base)['status'] == 'paused'
        self.checks.append({'gate': 'lease_over_3_5s', 'status': 'passed'})
        control('speed', speed=2)
        control('resume')
        job = self.api('POST', '/api/models/train', {'profile': 'full', 'trees': 25})
        started = time.monotonic()
        timings, cursors, stages = [], [], []
        while time.monotonic() - started < 300:
            tick = time.monotonic()
            health = self.api('GET', '/api/health')
            timings.append(time.monotonic() - tick)
            assert health['status'] == 'ok'
            saved = self.api('GET', base)
            assert saved['model_id'] == reduced['model_id']
            self.api('POST', base + '/advance', {'client_id': owner, 'expected_cursor': saved['cursor']})
            cursors.append(saved['cursor'])
            job = self.api('GET', '/api/jobs/' + job['job_id'])
            stages.append(job['status'] + ':' + job['stage'])
            if job['status'] in ('succeeded', 'failed', 'interrupted'):
                break
            time.sleep(0.25)
        assert job['status'] == 'succeeded', job
        assert max(cursors) > min(cursors), 'Playback must progress during the real training job'
        ready = self.api('GET', '/api/models')['items']
        assert any(m['model_id'] == job['model_id'] and m['status'] == 'ready' and m['trees'] == 25 for m in ready)
        saved = control('pause')
        rows = self.api('GET', base + '/rows?limit=200')
        summary = self.api('GET', base + '/summary')
        exported = list(csv.DictReader(io.StringIO(self.client.get(base + '/export').text.lstrip('\ufeff'))))
        assert len(exported) == rows['total'] == saved['cursor'] == summary['processed']
        assert len({r['sample_id'] for r in exported}) == saved['cursor']
        assert [r['ordinal'] for r in rows['items']] == list(range(len(rows['items'])))
        self.checks.append({'gate': 'real_full25_background_reduced50_playback', 'job': job,
                           'seconds': time.monotonic() - started, 'health_seconds': timings,
                           'stages': stages, 'cursors': cursors, 'pinned_model': reduced['model_id'], 'summary': summary})
        control('finish')


def main():
    utf8_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    stamp = time.strftime('%Y%m%d-%H%M%S')
    evidence = ROOT / 'reports' / ('run-' + stamp)
    evidence.mkdir(parents=True)
    before = fingerprint(ROOT / 'var')
    runner = Acceptance(args.port, evidence)
    status, error = 'FAIL', None
    started = time.monotonic()
    python = sys.executable
    npm = shutil.which('npm.cmd')
    state = None
    try:
        if not npm or not TEMP.is_dir():
            raise RuntimeError('Cần npm.cmd và thư mục temp đã phê duyệt.')
        if not frontend_ready(ROOT):
            runner.command([npm, 'run', 'build'], ROOT / 'frontend')
            write_build_manifest(ROOT)
            assert frontend_ready(ROOT)
        runner.command([python, '-m', 'pytest', '-m', 'not real_data', '-q', '-ra'])
        runner.command([python, '-m', 'ruff', 'check', 'motionsense_app', 'scripts', 'tests'])
        runner.command([python, '-m', 'pip', 'check'])
        runner.command([npm, 'run', 'typecheck'], ROOT / 'frontend')
        runner.command([npm, 'run', 'build'], ROOT / 'frontend')
        write_build_manifest(ROOT)
        assert frontend_ready(ROOT), 'Manifest installer phải khớp dist mới.'
        runner.checks.append({'gate': 'build_manifest', 'status': 'passed'})
        runner.command([python, '-m', 'pytest', '-m', 'real_data', '-q', '-ra'])
        runner.command([python, '--version'])
        runner.command(['node.exe', '--version'])
        runner.command([npm, '--version'])
        runner.command([python, '-m', 'pip', 'freeze'])
        with tempfile.TemporaryDirectory(prefix='MotionSense nghiệm thu ', dir=TEMP) as temporary:
            state = Path(temporary)
            data, models = copy_ready_state(ROOT, state)
            runner.checks.append({'gate': 'source', 'data': data, 'models': models})
            registry = ModelRegistry(state / 'var/models')
            runner.checks.append({'gate': 'real_reports', 'reports': [registry.report(m['model_id']) for m in models]})
            try:
                runner.start(state)
                runner.session_gate(state, data, models)
                runner.env.update(MOTIONSENSE_BASE_URL=runner.url, MOTIONSENSE_TEST_MODE='real',
                                  MOTIONSENSE_BROWSER_CHANNEL='chrome',
                                  MOTIONSENSE_EVIDENCE_DIR=str(evidence / 'browser'),
                                  PLAYWRIGHT_JSON_OUTPUT_NAME=str(evidence / 'playwright.json'))
                unready_title = 'chưa thiết lập hoặc snapshot hỏng không hiện số liệu sẵn sàng'
                runner.command([npm, 'run', 'test:e2e', '--', '--reporter=list,json',
                                '--grep-invert', unready_title], ROOT / 'frontend', timeout=1800)
                browser = json.loads((evidence / 'playwright.json').read_text(encoding='utf-8'))
                runner.checks.append({'gate': 'browser_stats', 'stats': browser['stats']})
                assert browser['stats']['skipped'] == 0, 'Required browser acceptance must not skip'
                runner.stop()
                for mode in ('empty', 'corrupt'):
                    unready_state = state / mode
                    unready_state.mkdir()
                    if mode == 'corrupt':
                        copy_ready_state(ROOT, unready_state)
                        (unready_state / 'var/datasets/current.json').write_text('{invalid', encoding='utf-8')
                    runner.start(unready_state, restart=True, ready=False)
                    runner.env.update(MOTIONSENSE_BASE_URL=runner.url, MOTIONSENSE_TEST_MODE=mode,
                                      MOTIONSENSE_EVIDENCE_DIR=str(evidence / ('browser-' + mode)),
                                      PLAYWRIGHT_JSON_OUTPUT_NAME=str(evidence / ('playwright-' + mode + '.json')))
                    runner.command([npm, 'run', 'test:e2e', '--', '--reporter=list,json', '--grep',
                                    unready_title + '|no ready models retains history'], ROOT / 'frontend')
                    stats = json.loads((evidence / ('playwright-' + mode + '.json')).read_text(encoding='utf-8'))['stats']
                    assert stats['skipped'] == 0 and stats['expected'] == 2, stats
                    runner.checks.append({'gate': 'browser_' + mode, 'stats': stats})
                    runner.stop()
                status = 'PASS'
            finally:
                runner.stop()
                if (state / 'var/logs').exists():
                    shutil.copytree(state / 'var/logs', evidence / 'isolated-logs')
    except Exception:  # noqa: BLE001 -- persist every acceptance failure before cleanup/report
        error = traceback.format_exc()
        print(error, flush=True)
    finally:
        runner.stop()
        runner.client.close()
        after = fingerprint(ROOT / 'var')
        unchanged = before == after
        if not unchanged:
            status = 'FAIL'
        result = {'status': status, 'error': error, 'seconds': round(time.monotonic() - started, 3),
                  'environment': {'os': platform.platform(), 'python': sys.version, 'executable': python},
                  'commands': runner.commands, 'checks': runner.checks, 'user_state_unchanged': unchanged,
                  'before': before, 'after': after, 'owned_child_stopped': runner.child is None,
                  'temporary_state_removed': state is None or not state.exists()}
        (evidence / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        lines = ['# Nghiệm thu MotionSense', '', f'Kết quả thực tế: **{status}**; {result["seconds"]} giây.',
                 f'Artifacts: [{evidence.name}/result.json]({evidence.name}/result.json).',
                 f'User DB/data/models/source không đổi: {unchanged}; owned child đã dừng: {runner.child is None}.',
                 '', '## Lệnh thực tế', '', '| Lệnh | Exit | Giây | Log |', '|---|---:|---:|---|']
        for c in runner.commands:
            lines.append(f'| `{c["command"]}` | {c["exit_code"]} | {c["seconds"]} | [{c["log"]}]({evidence.name}/{c["log"]}) |')
        lines += ['', '## Phạm vi và giới hạn', '',
                  'Chrome channel đã cài; không tải lại bundled Chromium. Browser version được đính kèm offline test.',
                  'Chặn request ngoài bằng page.route; APIRequestContext không nằm trong route. Không ngắt mạng vật lý.',
                  'Bộ browser bao gồm fixture lỗi có kiểm soát; các chỉ số chất lượng lấy từ mô hình/dataset thật.',
                  'Các bước sau lệnh fail: **Chưa kiểm chứng**, không được tính PASS.',
                  'Chi tiết counts/skips/timings/model IDs/matrices/fingerprints trong logs và result.json.',
                  '[Hướng dẫn](../README.md): `Khoi_dong.bat`.', '', '## Lỗi', '', '```', error or 'Không có lỗi trong các gate đã chạy.', '```']
        (ROOT / 'reports/acceptance.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
