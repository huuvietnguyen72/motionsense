import { test, expect } from '@playwright/test';
import type { APIRequestContext, Page as BrowserPage } from '@playwright/test';
import type { DatasetInfo, ModelInfo, Session, SessionSummary, Sample } from '../src/api/types';
import { execFileSync } from 'node:child_process';
import path from 'node:path';

// Catches local cursor clocks, auto-resume, stale signal pairing, lost ownership,
// fabricated summaries and exports, and history accidentally requiring artifacts.
test.setTimeout(45_000);
const mode = process.env.MOTIONSENSE_TEST_MODE ?? 'real';
async function setup(page: BrowserPage, request: APIRequestContext) {
  const data: DatasetInfo = await (await request.get('/api/data')).json();
  expect(data.ready).toBe(true);
  expect(data.split_counts).toEqual({ train: 7352, test: 2947 });
  const { items }: { items: ModelInfo[] } = await (await request.get('/api/models')).json();
  const model = items.find(m => m.status === 'ready' && m.profile === 'full' && m.dataset_id === data.dataset_id)!;
  expect(model).toBeTruthy();
  const subject = data.subjects.find(s => s.split === 'test')!;
  await page.goto('/#theo-doi');
  await expect(page.getByLabel('Người tham gia')).toHaveValue(`test:${subject.subject_id}`);
  await expect(page.getByLabel('Mô hình')).toHaveValue(model.model_id);
  const before = (await (await request.get('/api/sessions')).json()).total;
  await page.waitForTimeout(350);
  expect((await (await request.get('/api/sessions')).json()).total).toBe(before);
  await page.getByRole('button', { name: 'Tạo phiên', exact: true }).click();
  await expect(page.getByTestId('session-status')).toHaveText('Sẵn sàng');
  const id = await page.getByTestId('session-id').innerText();
  return { data, model, subject, id };
}
async function snapshot(request: APIRequestContext, id: string): Promise<Session> {
  return (await request.get(`/api/sessions/${id}`)).json();
}
async function run(page: BrowserPage) {
  await page.getByRole('button', { name: 'Bắt đầu', exact: true }).click();
  await expect.poll(async () => Number(await page.getByTestId('processed-count').innerText())).toBeGreaterThan(0);
}
async function finish(page: BrowserPage) {
  await page.getByRole('button', { name: 'Kết thúc', exact: true }).click();
  await expect(page.getByTestId('session-status')).toHaveText('Đã kết thúc');
}

test('pause/reload preserves persisted cursor, prediction, signal and separate comparison', async ({ page, request }, info) => {
  test.skip(mode !== 'real');
  const { id, data, model } = await setup(page, request);
  await expect(page.getByTestId('prediction-card')).toContainText('Chưa có kết quả');
  await run(page);
  await page.getByRole('button', { name: 'Tạm dừng', exact: true }).click();
  await expect(page.getByTestId('session-status')).toHaveText('Tạm dừng');
  const saved = await snapshot(request, id);
  await expect(page.getByTestId('processed-count')).toHaveText(String(saved.cursor));
  const sample: Sample = await (await request.get(`/api/data/samples/${saved.last_prediction!.sample_id}`)).json();
  await expect(page.getByTestId('signal-chart')).toHaveAttribute('data-sample-id', sample.sample_id);
  const traces = page.getByTestId('signal-chart').locator('polyline');
  await expect(traces).toHaveCount(3);
  for (const [index, axis] of ['x', 'y', 'z'].entries()) {
    const values = await traces.nth(index).getAttribute('data-values');
    const raw = sample.signal[axis as 'x' | 'y' | 'z'];
    expect(JSON.parse(values!)).toEqual(raw);
    const points = (await traces.nth(index).getAttribute('points'))!.split(' ').map(p => p.split(',').map(Number));
    expect(points).toHaveLength(128);
    expect(points[0][0]).toBe(48);
    expect(points[127][0]).toBe(672);
    // Independently verify actual SVG geometry, not just its evidence attributes.
    // Normalize each trace's range: higher g must plot higher on the chart.
    const plotted = points.map(p => p[1]);
    const top = Math.min(...plotted), bottom = Math.max(...plotted);
    const low = Math.min(...raw), high = Math.max(...raw);
    for (let i = 0; i < 128; i++) {
      expect((points[i][0] - 48) / 624).toBeCloseTo(sample.signal.time_seconds[i] / 2.54, 8);
      expect((bottom - plotted[i]) / (bottom - top)).toBeCloseTo((raw[i] - low) / (high - low), 8);
    }
  }
  const predicted = data.activities.find(a => a.label_id === saved.last_prediction!.predicted_label)!.name_vi;
  await expect(page.getByTestId('prediction-label')).toHaveText(predicted);
  await expect(page.getByTestId('prediction-card')).not.toContainText('Nhãn thực tế');
  await expect(page.getByTestId('probabilities').locator('meter')).toHaveCount(6);
  for (const p of saved.last_prediction!.probabilities) {
    await expect(page.getByTestId(`probability-${p.label_id}`)).toHaveAttribute('value', String(p.value));
  }
  await page.reload();
  await expect(page.getByTestId('session-status')).toHaveText('Tạm dừng');
  await expect(page.getByTestId('processed-count')).toHaveText(String(saved.cursor));
  await page.waitForTimeout(1400);
  expect((await snapshot(request, id)).cursor).toBe(saved.cursor);
  await expect(page.getByTestId('pinned-model')).toHaveText(model.model_id);
  await page.getByText('Đối chiếu nhãn thực tế', { exact: true }).click();
  await expect(page.getByRole('table', { name: 'Đối chiếu nhãn thực tế' })).toContainText(data.activities.find(a => a.label_id === sample.actual_label)!.name_vi);
  await page.screenshot({ path: info.outputPath('monitor-comparison.png'), fullPage: true });
  await finish(page);
});

test('running reload attaches a new owner and pauses before explicit resume; speed/finish/export', async ({ page, request }, info) => {
  test.skip(mode !== 'real');
  const { id, data } = await setup(page, request);
  await run(page);
  const before = await snapshot(request, id);
  await page.reload();
  await expect(page.getByTestId('session-status')).toHaveText('Tạm dừng');
  const recovered = await snapshot(request, id);
  expect(recovered.cursor).toBeGreaterThanOrEqual(before.cursor);
  expect(recovered.cursor).toBeLessThanOrEqual(before.cursor + 1);
  await page.waitForTimeout(1400);
  expect((await snapshot(request, id)).cursor).toBe(recovered.cursor);
  for (const speed of [0.5, 1, 2]) {
    await page.getByLabel('Tốc độ phát lại').selectOption(String(speed));
    await expect.poll(async () => (await snapshot(request, id)).speed).toBe(speed);
  }
  await page.getByRole('button', { name: 'Tiếp tục', exact: true }).click();
  await expect.poll(async () => (await snapshot(request, id)).cursor).toBeGreaterThan(recovered.cursor);
  await finish(page);
  const summary: SessionSummary = await (await request.get(`/api/sessions/${id}/summary`)).json();
  await expect(page.getByTestId('summary-processed')).toHaveText(String(summary.processed));
  for (const count of summary.counts) {
    const row = page.getByTestId(`distribution-${count.label_id}`);
    await expect(row.locator('strong')).toHaveText(`${count.count} mẫu`);
    await expect(row.locator('span').nth(0)).toHaveText(data.activities.find(a => a.label_id === count.label_id)!.name_vi);
    await expect(row.locator('span').nth(1)).toHaveText(new Intl.NumberFormat('vi-VN', { style: 'percent', maximumFractionDigits: 1 }).format(count.count / summary.processed));
  }
  const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('link', { name: 'Xuất CSV', exact: true }).click()]);
  expect(await download.failure()).toBeNull();
  const csv = await (await request.get(`/api/sessions/${id}/export`)).text();
  const lines = csv.replace(/^\uFEFF/, '').trim().split(/\r?\n/);
  expect(lines.length - 1).toBe(summary.processed);
  expect(lines[0]).toContain('actual_label');
  expect(new Set(lines.slice(1).map(line => line.split(',')[5])).size).toBe(summary.processed);
  expect((await snapshot(request, id)).finish_reason).toBe('user');
  await page.screenshot({ path: info.outputPath('monitor-finished.png'), fullPage: true });
  await expect(page.getByRole('button', { name: 'Tạo phiên mới', exact: true })).toBeVisible();
});

test('connection loss stops polling; reconnect pauses and synchronizes saved rows before resume', async ({ page, request }) => {
  test.skip(mode !== 'real');
  const { id } = await setup(page, request);
  await run(page);
  await page.route('**/advance', route => route.abort('failed'));
  await expect(page.getByRole('alert')).toContainText('Mất kết nối');
  await page.unroute('**/advance');
  await page.getByRole('button', { name: 'Kết nối lại', exact: true }).click();
  await expect(page.getByTestId('session-status')).toHaveText('Tạm dừng');
  const saved = await snapshot(request, id);
  await expect(page.getByTestId('processed-count')).toHaveText(String(saved.cursor));
  await expect(page.getByRole('table', { name: 'Lịch sử nhận diện' }).locator('tbody tr')).toHaveCount(saved.cursor);
  await page.waitForTimeout(1100);
  expect((await snapshot(request, id)).cursor).toBe(saved.cursor);
  await finish(page);
});

test('new tab ownership blocks old tab; navigation pauses without auto restart and history reopens', async ({ page, context, request }) => {
  test.skip(mode !== 'real');
  const { id } = await setup(page, request);
  await run(page);
  const other = await context.newPage();
  await other.goto('/#theo-doi');
  await expect(other.getByTestId('session-status')).toHaveText('Tạm dừng');
  await expect(page.getByRole('alert')).toContainText('cửa sổ khác');
  await other.getByRole('button', { name: 'Tiếp tục', exact: true }).click();
  await expect(other.getByTestId('session-status')).toHaveText('Đang chạy');
  await other.getByRole('link', { name: 'Dữ liệu', exact: true }).click();
  await expect.poll(async () => (await snapshot(request, id)).status).toBe('paused');
  await other.getByRole('link', { name: 'Theo dõi', exact: true }).click();
  await expect(other.getByTestId('session-status')).toHaveText('Tạm dừng');
  await other.getByRole('button', { name: `Mở phiên ${id}`, exact: true }).click();
  await expect(other.getByTestId('session-id')).toHaveText(id);
  await finish(other);
  await other.close();
});

test('slow real signal clears old chart and ignores stale response while prediction advances', async ({ page, request }) => {
  test.skip(mode !== 'real');
  await setup(page, request);
  await run(page);
  await expect(page.getByTestId('signal-chart')).toBeVisible();
  const old = await page.getByTestId('signal-chart').getAttribute('data-sample-id');
  let release!: () => void;
  const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/api/data/samples/*', async route => {
    const response = await route.fetch();
    await gate;
    await route.fulfill({ response });
  });
  await expect(page.getByTestId('prediction-sample')).not.toHaveText(old!);
  await expect(page.getByTestId('signal-chart')).toHaveCount(0);
  await page.getByRole('button', { name: 'Tạm dừng', exact: true }).click();
  const current = await page.getByTestId('prediction-sample').innerText();
  release();
  await expect(page.getByTestId('signal-chart')).toHaveAttribute('data-sample-id', current);
  await page.unroute('**/api/data/samples/*');
  await finish(page);
});

test('no ready models retains history and disables creation; empty setup stays honest', async ({ page, request }, info) => {
  if (mode === 'real') {
    const { id } = await setup(page, request);
    await run(page);
    await finish(page);
    await page.route('**/api/models', route => route.fulfill({ json: { items: [] } }));
    await page.reload();
    await expect(page.getByRole('status').filter({ hasText: 'Chưa có mô hình' })).toBeVisible();
    await page.getByRole('button', { name: `Mở phiên ${id}`, exact: true }).click();
    await expect(page.getByTestId('session-status')).toHaveText('Đã kết thúc');
    await expect(page.getByRole('link', { name: 'Xuất CSV', exact: true })).toBeVisible();
  } else {
    await page.goto('/#theo-doi');
    await expect(page.getByRole('status').filter({ hasText: 'Cai_dat.bat' }).first()).toBeVisible();
    await expect(page.getByTestId('processed-count')).toHaveCount(0);
  }
  await expect(page.getByRole('button', { name: /Tạo phiên/ })).toBeDisabled();
  await page.screenshot({ path: info.outputPath(`monitor-${mode}-setup.png`), fullPage: true });
});

test('monitor layout, keyboard focus, local assets and no console errors', async ({ page, request }, info) => {
  test.skip(mode !== 'real');
  const errors: string[] = [], external: string[] = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('request', r => { if (!r.url().startsWith(info.project.use.baseURL!)) external.push(r.url()); });
  await setup(page, request);
  await run(page);
  await page.getByRole('button', { name: 'Tạm dừng', exact: true }).click();
  for (const width of [1366, 1024]) {
    await page.setViewportSize({ width, height: 768 });
    await expect(page.getByTestId('signal-chart')).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: info.outputPath(`monitor-${width}x768.png`), fullPage: true });
    await page.screenshot({ path: info.outputPath(`monitor-${width}x768-viewport.png`) });
  }
  await page.getByLabel('Tốc độ phát lại').focus();
  await page.keyboard.press('Tab');
  const focused = page.locator(':focus');
  expect(await focused.evaluate(el => getComputedStyle(el).outlineStyle)).not.toBe('none');
  expect(errors).toEqual([]);
  expect(external).toEqual([]);
  await finish(page);
});

test('real persisted near-end session completes with paginated rows; missing model/source history remains exportable', async ({ page, request }, info) => {
  test.skip(mode !== 'real');
  // Only the owned runner's disposable state may be seeded/moved. Official
  // samples and the published estimator generate every persisted prediction.
  const root = path.resolve('..');
  expect(process.env.MOTIONSENSE_STATE_ROOT).toContain('opencode');
  execFileSync(path.join(root, '.venv/Scripts/python.exe'), ['-c', `
import json, os
from pathlib import Path
from motionsense_app.settings import Settings
from motionsense_app.data.store import DatasetStore
from motionsense_app.models.registry import ModelRegistry
from motionsense_app.models.prediction import predict_rows
from motionsense_app.sessions.engine import SessionEngine
from motionsense_app.sessions.repository import SessionRepository
s = Settings.default()
assert s.state_root and s.state_root.resolve() != s.root.resolve()
store = DatasetStore(s.data_dir)
registry = ModelRegistry(s.model_dir)
repo = SessionRepository(s.db_path)
data = store.info()
subject = next(v for v in data['subjects'] if v['split'] == 'test')
model = next(v for v in registry.list() if v['status'] == 'ready' and v['profile'] == 'full')
engine = SessionEngine(repo, store, registry)
session = engine.create(dict(dataset_id=data['dataset_id'], split='test', subject_id=subject['subject_id'], model_id=model['model_id']))
id = session['session_id']
engine.attach(id, 'seed')
repo.change(id, status='running', speed=2)
ids = store.sample_ids('test', subject['subject_id'])
samples = [store.sample(v) for v in ids[:-1]]
predictions = predict_rows(registry.load(model['model_id']), [v['features'] for v in samples])
for ordinal, (sample, prediction) in enumerate(zip(samples, predictions)):
    prediction.update(sample_id=sample['sample_id'], actual_label=sample['actual_label'])
    repo.commit_row(id, 'seed', ordinal, prediction)
repo.change(id, status='paused')
print(id)
`, ], { cwd: root, encoding: 'utf8' });
  const seeded: Session = (await (await request.get('/api/sessions?offset=0&limit=1')).json()).items[0];
  await page.goto('/#theo-doi');
  await page.getByRole('button', { name: `Mở phiên ${seeded.session_id}`, exact: true }).click();
  await expect(page.getByTestId('processed-count')).toHaveText(String(seeded.total - 1));
  await expect(page.getByRole('table', { name: 'Lịch sử nhận diện' }).locator('tbody tr')).toHaveCount(50);
  await page.getByRole('button', { name: 'Mẫu kế tiếp', exact: true }).click();
  await expect(page.getByRole('table', { name: 'Lịch sử nhận diện' }).locator('tbody tr').first()).toContainText('51');
  await page.getByRole('button', { name: 'Tiếp tục', exact: true }).click();
  await expect(page.getByTestId('session-status')).toHaveText('Đã kết thúc');
  expect((await snapshot(request, seeded.session_id)).finish_reason).toBe('complete');
  await expect(page.getByTestId('summary-processed')).toHaveText(String(seeded.total));
  await page.screenshot({ path: info.outputPath('monitor-complete.png'), fullPage: true });
  // Use a normally paused, unfinished API/UI session for unavailable artifacts.
  // Never turn a completed session into an impossible resumable fixture.
  await page.getByRole('button', { name: 'Tạo phiên mới', exact: true }).click();
  await expect(page.getByTestId('session-status')).toHaveText('Sẵn sàng');
  await run(page);
  await page.getByRole('button', { name: 'Tạm dừng', exact: true }).click();
  await expect(page.getByTestId('session-status')).toHaveText('Tạm dừng');
  const pausedId = await page.getByTestId('session-id').innerText();
  const paused = await snapshot(request, pausedId);
  expect(paused.cursor).toBeGreaterThan(0);
  expect(paused.cursor).toBeLessThan(paused.total);
  execFileSync(path.join(root, '.venv/Scripts/python.exe'), ['-c', `
from motionsense_app.settings import Settings
s = Settings.default()
assert s.state_root and s.state_root.resolve() != s.root.resolve()
p = s.model_dir / '${paused.model_id}'
p.rename(s.var_root / p.name)
`], { cwd: root, encoding: 'utf8' });
  try {
    await page.reload();
    await expect(page.getByTestId('session-status')).toHaveText('Tạm dừng');
    await page.getByRole('button', { name: 'Tiếp tục', exact: true }).click();
    await expect(page.getByRole('alert')).toContainText('Mô hình');
    await expect(page.getByRole('alert')).toContainText('Cai_dat.bat');
    const csv = await (await request.get(`/api/sessions/${pausedId}/export`)).text();
    expect(csv.trim().split(/\r?\n/)).toHaveLength(paused.cursor + 1);
    await page.screenshot({ path: info.outputPath('monitor-missing-model.png'), fullPage: true });
    execFileSync(path.join(root, '.venv/Scripts/python.exe'), ['-c', `
from motionsense_app.settings import Settings
s = Settings.default()
assert s.state_root and s.state_root.resolve() != s.root.resolve()
(s.var_root / '${paused.model_id}').rename(s.model_dir / '${paused.model_id}')
p = s.data_dir
p.rename(s.var_root / 'missing-datasets')
`], { cwd: root, encoding: 'utf8' });
    await page.reload();
    await expect(page.getByTestId('session-status')).toHaveText('Tạm dừng');
    await expect(page.getByRole('table', { name: 'Lịch sử nhận diện' }).locator('tbody tr')).toHaveCount(paused.cursor);
    await page.getByRole('button', { name: 'Tiếp tục', exact: true }).click();
    await expect(page.getByRole('alert').filter({ hasText: /dữ liệu/i }).first()).toBeVisible();
    await expect(page.getByRole('link', { name: 'Xuất CSV', exact: true })).toBeVisible();
    await page.screenshot({ path: info.outputPath('monitor-missing-source.png'), fullPage: true });
  } finally {
    execFileSync(path.join(root, '.venv/Scripts/python.exe'), ['-c', `
from motionsense_app.settings import Settings
s = Settings.default()
assert s.state_root and s.state_root.resolve() != s.root.resolve()
p = s.var_root / '${paused.model_id}'
if p.exists(): p.rename(s.model_dir / p.name)
p = s.var_root / 'missing-datasets'
if p.exists(): p.rename(s.data_dir)
`], { cwd: root, encoding: 'utf8' });
  }
});

test('slow advance is serialized with pause and new selection does not change pinned session', async ({ page, request }) => {
  test.skip(mode !== 'real');
  const { id, data, model } = await setup(page, request);
  let active = 0, max = 0;
  await page.route('**/api/sessions/**', async route => {
    if (!route.request().url().endsWith('/advance')) return route.continue();
    active++; max = Math.max(max, active);
    const result = await route.fetch();
    await new Promise(resolve => setTimeout(resolve, 550));
    await route.fulfill({ response: result });
    active--;
  });
  await run(page);
  await page.getByLabel('Người tham gia').selectOption(`train:${data.subjects.find(s => s.split === 'train')!.subject_id}`);
  await expect(page.getByTestId('pinned-model')).toHaveText(model.model_id);
  expect((await snapshot(request, id)).split).toBe('test');
  await page.getByRole('button', { name: 'Tạm dừng', exact: true }).click();
  await expect(page.getByTestId('session-status')).toHaveText('Tạm dừng');
  const saved = await snapshot(request, id);
  await page.waitForTimeout(1600);
  expect((await snapshot(request, id)).cursor).toBe(saved.cursor);
  expect(max).toBe(1);
  const rows = (await (await request.get(`/api/sessions/${id}/rows`)).json()).items;
  expect(new Set(rows.map((r: { ordinal: number }) => r.ordinal)).size).toBe(saved.cursor);
  await finish(page);
});

test('reopening the running session in the same document pauses instead of restarting polling', async ({ page, request }) => {
  test.skip(mode !== 'real');
  const { id } = await setup(page, request);
  await run(page);
  await page.getByRole('button', { name: `Mở phiên ${id}`, exact: true }).click();
  await expect(page.getByTestId('session-status')).toHaveText('Tạm dừng');
  const saved = await snapshot(request, id);
  await page.waitForTimeout(1300);
  expect((await snapshot(request, id)).cursor).toBe(saved.cursor);
  await finish(page);
});

test('explicit new person/model session pauses the old server session and pins the new selection', async ({ page, request }) => {
  test.skip(mode !== 'real');
  const { id, data } = await setup(page, request);
  await run(page);
  const { items }: { items: ModelInfo[] } = await (await request.get('/api/models')).json();
  const reduced = items.find(m => m.profile === 'reduced' && m.status === 'ready' && m.dataset_id === data.dataset_id)!;
  expect(reduced).toBeTruthy();
  const person = data.subjects.find(s => s.split === 'train')!;
  await page.getByLabel('Người tham gia').selectOption(`train:${person.subject_id}`);
  await page.getByLabel('Mô hình', { exact: true }).selectOption(reduced.model_id);
  expect((await snapshot(request, id)).split).toBe('test');
  await page.getByRole('button', { name: 'Tạo phiên', exact: true }).click();
  await expect(page.getByTestId('session-status')).toHaveText('Sẵn sàng');
  expect((await snapshot(request, id)).status).toBe('paused');
  const nextId = await page.getByTestId('session-id').innerText();
  expect(nextId).not.toBe(id);
  expect(await snapshot(request, nextId)).toMatchObject({ split: 'train', subject_id: person.subject_id, model_id: reduced.model_id, cursor: 0 });
  await page.waitForTimeout(1100);
  expect((await snapshot(request, nextId)).cursor).toBe(0);
  await finish(page);
});

test('development StrictMode restores once and disposal never automatically starts playback', async ({ page, request }, info) => {
  test.skip(mode !== 'real');
  const { id } = await setup(page, request);
  await run(page);
  const { createServer } = await import('vite');
  const server = await createServer({
    root: process.cwd(), mode: 'development',
    server: { host: '127.0.0.1', port: 0, proxy: { '/api': info.project.use.baseURL! } },
  });
  const requests: string[] = [], errors: string[] = [];
  page.on('request', r => { if (r.url().includes('/api/sessions')) requests.push(`${r.method()} ${new URL(r.url()).pathname} ${r.postData() ?? ''}`); });
  page.on('pageerror', e => errors.push(e.message));
  try {
    await server.listen();
    const address = server.httpServer!.address();
    expect(address && typeof address !== 'string').toBeTruthy();
    const url = `http://127.0.0.1:${(address as { port: number }).port}`;
    await page.addInitScript(id => localStorage.setItem('motionsense.session_id', id), id);
    await page.goto(`${url}/#theo-doi`);
    await expect(page.getByTestId('session-status')).toHaveText('Tạm dừng');
    await expect(page.getByRole('button', { name: 'Tiếp tục', exact: true })).toBeEnabled();
    const restored = await snapshot(request, id);
    await page.waitForTimeout(1100);
    expect((await snapshot(request, id)).cursor).toBe(restored.cursor);
    expect(requests.filter(r => r.includes('/attach'))).toHaveLength(1);
    expect(requests.filter(r => r.includes('"action":"start"') || r.includes('"action":"resume"'))).toHaveLength(0);
    await page.getByRole('button', { name: 'Tiếp tục', exact: true }).click();
    await expect(page.getByTestId('session-status')).toHaveText('Đang chạy');
    await page.getByRole('link', { name: 'Dữ liệu', exact: true }).click();
    await page.getByRole('link', { name: 'Theo dõi', exact: true }).click();
    await expect(page.getByTestId('session-status')).toHaveText('Tạm dừng');
    await page.screenshot({ path: info.outputPath('monitor-development-strictmode.png'), fullPage: true });
    expect(errors).toEqual([]);
    await finish(page);
  } finally {
    await page.goto('about:blank');
    await server.close();
  }
});

test('saved sessions paginate by 50 and opening the second page selects that existing session', async ({ page, request }) => {
  test.skip(mode !== 'real');
  const data: DatasetInfo = await (await request.get('/api/data')).json();
  const { items }: { items: ModelInfo[] } = await (await request.get('/api/models')).json();
  const model = items.find(m => m.status === 'ready' && m.profile === 'full' && m.dataset_id === data.dataset_id)!;
  const person = data.subjects.find(s => s.split === 'test')!;
  let oldest = '';
  for (let i = 0; i < 51; i++) {
    const response = await request.post('/api/sessions', { data: { dataset_id: data.dataset_id, model_id: model.model_id, split: 'test', subject_id: person.subject_id } });
    expect(response.status()).toBe(201);
    const session: Session = await response.json();
    if (i === 0) oldest = session.session_id;
  }
  await page.goto('/#theo-doi');
  await expect(page.getByRole('table', { name: 'Phiên đã lưu' }).locator('tbody tr')).toHaveCount(50);
  await expect(page.getByRole('button', { name: `Mở phiên ${oldest}`, exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Phiên kế tiếp', exact: true }).click();
  await page.getByRole('button', { name: `Mở phiên ${oldest}`, exact: true }).click();
  await expect(page.getByTestId('session-id')).toHaveText(oldest);
  await expect(page.getByTestId('processed-count')).toHaveText('0');
  await expect(page.getByTestId('session-status')).toHaveText('Sẵn sàng');
  await finish(page);
});

for (const failure of ['committed advance response', 'control ownership', 'rows request'] as const) {
  test(`R1: successful row pagination preserves ${failure} failure until genuine reconnect`, async ({ page, request }, info) => {
    test.skip(mode !== 'real');
    // Catches clearing the rendered failure while the hook is still blocked,
    // or unblocking on a row GET without fresh ownership/full synchronization.
    const root = path.resolve('..');
    expect(process.env.MOTIONSENSE_STATE_ROOT).toContain('opencode');
    const id = execFileSync(path.join(root, '.venv/Scripts/python.exe'), ['-c', `
from motionsense_app.settings import Settings
from motionsense_app.data.store import DatasetStore
from motionsense_app.models.registry import ModelRegistry
from motionsense_app.models.prediction import predict_rows
from motionsense_app.sessions.engine import SessionEngine
from motionsense_app.sessions.repository import SessionRepository
s = Settings.default()
assert s.state_root and s.state_root.resolve() != s.root.resolve()
store = DatasetStore(s.data_dir)
registry = ModelRegistry(s.model_dir)
repo = SessionRepository(s.db_path)
data = store.info()
assert data['ready'] and data['split_counts'] == dict(train=7352, test=2947)
subject = next(v for v in data['subjects'] if v['split'] == 'test')
model = next(v for v in registry.list() if v['status'] == 'ready' and v['profile'] == 'full')
engine = SessionEngine(repo, store, registry)
session = engine.create(dict(dataset_id=data['dataset_id'], split='test', subject_id=subject['subject_id'], model_id=model['model_id']))
id = session['session_id']
engine.attach(id, 'seed')
repo.change(id, status='running', speed=2)
samples = [store.sample(v) for v in store.sample_ids('test', subject['subject_id'])[:51]]
predictions = predict_rows(registry.load(model['model_id']), [v['features'] for v in samples])
for ordinal, (sample, prediction) in enumerate(zip(samples, predictions)):
    prediction.update(sample_id=sample['sample_id'], actual_label=sample['actual_label'])
    repo.commit_row(id, 'seed', ordinal, prediction)
repo.change(id, status='paused')
print(id)
`], { cwd: root, encoding: 'utf8' }).trim();
    const sent: { path: string; body: { client_id: string; expected_cursor?: number; action?: string } }[] = [];
    page.on('request', r => {
      if (r.method() === 'POST' && r.url().includes(`/api/sessions/${id}/`)) {
        sent.push({ path: new URL(r.url()).pathname, body: r.postDataJSON() });
      }
    });
    let releaseSummary = () => {};
    try {
      await page.goto('/#theo-doi');
      await page.getByRole('button', { name: `Mở phiên ${id}`, exact: true }).click();
      await expect(page.getByRole('button', { name: 'Tiếp tục', exact: true })).toBeEnabled();
      await expect(page.getByTestId('processed-count')).toHaveText('51');
      const initialClient = sent.find(r => r.path.endsWith('/attach'))!.body.client_id;
      expect(initialClient).toBeTruthy();
      const expectedError = failure === 'control ownership' ? 'cửa sổ khác' : 'Mất kết nối';
      if (failure === 'committed advance response') {
        await page.route(`**/api/sessions/${id}/advance`, async route => {
          const response = await route.fetch();
          const result: { session: Session; row: { ordinal: number } | null } = await response.json();
          if (result.row) {
            // Lose only the response AFTER the real server commits ordinal 51.
            expect(result.row.ordinal).toBe(51);
            expect(result.session.cursor).toBe(52);
            await route.abort('failed');
          } else await route.fulfill({ response });
        });
        await page.getByRole('button', { name: 'Tiếp tục', exact: true }).click();
      } else if (failure === 'control ownership') {
        const response = await request.post(`/api/sessions/${id}/attach`, { data: { client_id: 'another-owner' } });
        expect(response.ok()).toBe(true);
        const rejected = page.waitForResponse(r => r.url().endsWith(`/sessions/${id}/control`) && r.status() === 409);
        await page.getByRole('button', { name: 'Tiếp tục', exact: true }).click();
        expect((await (await rejected).json()).error.code).toBe('CLIENT_CONFLICT');
      } else {
        await page.route(`**/api/sessions/${id}/rows?**`, route => route.abort('failed'));
        await page.getByRole('button', { name: 'Mẫu kế tiếp', exact: true }).click();
      }
      await expect(page.getByRole('alert')).toContainText(expectedError);
      await expect(page.getByRole('button', { name: 'Kết nối lại', exact: true })).toBeEnabled();
      await page.unroute(`**/api/sessions/${id}/advance`);
      await page.unroute(`**/api/sessions/${id}/rows?**`);
      const saved = await snapshot(request, id);
      expect(saved.cursor).toBe(failure === 'committed advance response' ? 52 : 51);
      const advancesBeforeReads = sent.filter(r => r.path.endsWith('/advance')).length;
      await page.getByRole('button', { name: 'Mẫu kế tiếp', exact: true }).click();
      const rows = page.getByRole('table', { name: 'Lịch sử nhận diện' }).locator('tbody tr');
      await expect(rows.first().getByRole('rowheader')).toHaveText('51');
      await expect(rows).toHaveCount(saved.cursor - 50);
      // The failing pre-fix assertion: successful pagination used to hide this.
      await expect(page.getByRole('alert')).toContainText(expectedError);
      await expect(page.getByRole('button', { name: 'Kết nối lại', exact: true })).toBeEnabled();
      await expect(page.getByRole('button', { name: 'Kết thúc', exact: true })).toBeDisabled();
      await expect(page.getByLabel('Tốc độ phát lại')).toBeDisabled();
      for (const button of await page.locator('.session-controls button').all()) await expect(button).toBeDisabled();
      await page.getByRole('button', { name: 'Mẫu trước', exact: true }).click();
      await expect(rows.first().getByRole('rowheader')).toHaveText('1');
      await expect(rows).toHaveCount(50);
      await expect(page.getByRole('alert')).toContainText(expectedError);
      await page.waitForTimeout(600);
      expect(sent.filter(r => r.path.endsWith('/advance')).length).toBe(advancesBeforeReads);
      expect(sent.filter(r => r.path.endsWith('/attach'))).toHaveLength(1);
      expect((await snapshot(request, id)).cursor).toBe(saved.cursor);
      await page.screenshot({ path: info.outputPath('r1-blocked-after-pagination.png'), fullPage: true });

      // Even successful attach/snapshot/rows is insufficient until summary also
      // finishes: keep recovery actionable and every control disabled meanwhile.
      let summaryStarted!: () => void;
      const started = new Promise<void>(resolve => { summaryStarted = resolve; });
      const gate = new Promise<void>(resolve => { releaseSummary = resolve; });
      await page.route(`**/api/sessions/${id}/summary`, async route => {
        const response = await route.fetch();
        summaryStarted();
        await gate;
        await route.fulfill({ response });
      });
      await page.getByRole('button', { name: 'Kết nối lại', exact: true }).click();
      await started;
      await expect(page.getByTestId('session-status')).toHaveText('Tạm dừng');
      await expect(page.getByRole('alert')).toContainText(expectedError);
      await expect(page.getByRole('button', { name: 'Kết nối lại', exact: true })).toBeDisabled();
      await expect(page.getByRole('button', { name: 'Tiếp tục', exact: true })).toBeDisabled();
      releaseSummary();
      await expect(page.getByRole('button', { name: 'Tiếp tục', exact: true })).toBeEnabled();
      await expect(page.getByRole('alert')).toHaveCount(0);
      await expect(page.getByRole('button', { name: 'Kết nối lại', exact: true })).toHaveCount(0);
      await page.unroute(`**/api/sessions/${id}/summary`);
      const attaches = sent.filter(r => r.path.endsWith('/attach'));
      expect(attaches).toHaveLength(2);
      const currentClient = attaches[1].body.client_id;
      expect(currentClient).not.toBe(initialClient);
      expect(currentClient).not.toBe('another-owner');
      expect(await snapshot(request, id)).toMatchObject({ status: 'paused', cursor: saved.cursor });
      const summary: SessionSummary = await (await request.get(`/api/sessions/${id}/summary`)).json();
      expect(summary.processed).toBe(saved.cursor);
      await expect(page.getByTestId('processed-count')).toHaveText(String(saved.cursor));
      await expect(page.getByTestId('summary-processed')).toHaveText(String(summary.processed));
      const firstPage = await (await request.get(`/api/sessions/${id}/rows?offset=0&limit=50`)).json();
      expect(firstPage.total).toBe(saved.cursor);
      expect(await rows.getByRole('rowheader').allTextContents()).toEqual(firstPage.items.map((r: { ordinal: number }) => String(r.ordinal + 1)));
      await page.waitForTimeout(600);
      expect(sent.filter(r => r.path.endsWith('/advance')).length).toBe(advancesBeforeReads);

      const resumed = page.waitForResponse(r => r.url().endsWith(`/sessions/${id}/control`) && r.request().postDataJSON().action === 'resume');
      await page.getByRole('button', { name: 'Tiếp tục', exact: true }).click();
      expect((await resumed).status()).toBe(200);
      expect(sent.filter(r => r.body.action === 'resume').at(-1)!.body.client_id).toBe(currentClient);
      await expect.poll(async () => (await snapshot(request, id)).cursor).toBeGreaterThan(saved.cursor);
      const firstAdvance = sent.filter(r => r.path.endsWith('/advance')).slice(advancesBeforeReads)[0];
      expect(firstAdvance.body).toEqual({ client_id: currentClient, expected_cursor: saved.cursor });
      await page.getByRole('button', { name: 'Tạm dừng', exact: true }).click();
      await expect(page.getByTestId('session-status')).toHaveText('Tạm dừng');
      await page.getByRole('button', { name: 'Mẫu kế tiếp', exact: true }).click();
      const finalPage = await (await request.get(`/api/sessions/${id}/rows?offset=50&limit=50`)).json();
      const visibleOrdinals = await rows.getByRole('rowheader').allTextContents();
      expect(visibleOrdinals).toEqual(finalPage.items.map((r: { ordinal: number }) => String(r.ordinal + 1)));
      expect(new Set(visibleOrdinals).size).toBe(visibleOrdinals.length);
      await expect(page.getByRole('alert')).toHaveCount(0);
      await expect(page.getByRole('button', { name: 'Tiếp tục', exact: true })).toBeEnabled();
      await page.screenshot({ path: info.outputPath('r1-reconnected-working-controls.png'), fullPage: true });
      await finish(page);
    } finally {
      releaseSummary();
      // Also close test-created sessions if a regression assertion fails.
      await request.post(`/api/sessions/${id}/attach`, { data: { client_id: 'r1-test-cleanup' } });
      await request.post(`/api/sessions/${id}/control`, { data: { client_id: 'r1-test-cleanup', action: 'finish' } });
    }
  });
}
