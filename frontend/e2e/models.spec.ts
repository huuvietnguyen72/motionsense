import { test, expect } from '@playwright/test';
import type { DatasetInfo, ModelInfo, ModelReport, Job } from '../src/api/types';

// Catches fabricated metrics, transposed matrices, cross-snapshot comparison,
// stale report acceptance, duplicate training and automatic model/session switching.
test.setTimeout(120_000);
const percent = (n: number) => new Intl.NumberFormat('vi-VN', { style: 'percent', maximumFractionDigits: 2 }).format(n);
test('reports show exact real metrics, all matrix cells, per-class and feature values', async ({ page, request }, info) => {
  const { items }: { items: ModelInfo[] } = await (await request.get('/api/models')).json();
  const full = items.find(m => m.profile === 'full' && m.status === 'ready')!;
  const reduced = items.find(m => m.profile === 'reduced' && m.status === 'ready')!;
  const data: DatasetInfo = await (await request.get('/api/data')).json();
  await page.goto('/#mo-hinh');
  for (const model of [full, reduced]) {
    await page.getByLabel('Mô hình phân tích').selectOption(model.model_id);
    const report: ModelReport = await (await request.get(`/api/models/${model.model_id}/report`)).json();
    await info.attach(`${model.profile}-source-report`, { body: JSON.stringify({ model, report, heldout_subjects: data.subjects.filter(s => s.split === 'test') }, null, 2), contentType: 'application/json' });
    await expect(page.getByTestId('report-model')).toHaveText(model.model_id);
    await expect(page.getByTestId('model-accuracy')).toHaveText(percent(report.accuracy));
    await expect(page.getByTestId('model-f1')).toHaveText(percent(report.macro_f1));
    await expect(page.getByTestId('model-test-count')).toHaveText(new Intl.NumberFormat('vi-VN').format(report.test_count));
    await expect(page.getByTestId('model-trees')).toHaveText(String(model.trees));
    await expect(page.getByTestId('heldout-subjects')).toContainText(data.subjects.filter(s => s.split === 'test').map(s => s.subject_id).join(', '));
    const matrix = page.getByRole('table', { name: 'Ma trận nhầm lẫn' });
    await expect(matrix).toContainText('Hàng: thực tế · Cột: dự đoán');
    for (let r = 0; r < 6; r++) for (let c = 0; c < 6; c++) await expect(matrix.locator('tbody tr').nth(r).locator('td').nth(c)).toHaveText(String(report.confusion_matrix[r][c]));
    for (const [i, cls] of report.per_class.entries()) {
      const cells = page.getByRole('table', { name: 'Chỉ số theo lớp' }).locator('tbody tr').nth(i).locator('td');
      await expect(cells.nth(0)).toHaveText(percent(cls.precision));
      await expect(cells.nth(1)).toHaveText(percent(cls.recall));
      await expect(cells.nth(2)).toHaveText(percent(cls.f1));
      await expect(cells.nth(3)).toHaveText(String(cls.support));
    }
    const table = page.getByRole('table', { name: 'Độ quan trọng đặc trưng' });
    await expect(table.locator('tbody tr')).toHaveCount(model.profile === 'full' ? 15 : 5);
    const ranked = [...report.feature_importance].sort((a, b) => b.value - a.value || a.feature_id.localeCompare(b.feature_id));
    await expect(table.locator('tbody tr').first().getByRole('rowheader')).toHaveText(ranked[0].feature_id);
    for (const feature of ranked.slice(0, model.profile === 'full' ? 15 : 5)) {
      const row = table.getByRole('row').filter({ has: page.getByRole('rowheader', { name: feature.feature_id, exact: true }) });
      await expect(row).toContainText(feature.name);
      await expect(row.locator('meter')).toHaveAttribute('value', String(feature.value));
    }
    if (model.profile === 'full') { await page.getByRole('button', { name: 'Xem tất cả đặc trưng' }).click(); await expect(table.locator('tbody tr')).toHaveCount(561); }
    await expect(page.getByTestId('train-accuracy')).toHaveText(percent(report.train_accuracy));
    await expect(page.getByTestId('oob-score')).toHaveText(report.oob_score === null ? 'Không có ước lượng OOB' : percent(report.oob_score));
  }
});

test('comparison requires full/reduced on same snapshot and delayed report cannot replace selection', async ({ page, request }) => {
  const { items }: { items: ModelInfo[] } = await (await request.get('/api/models')).json();
  const full = items.find(m => m.profile === 'full' && m.status === 'ready')!;
  const reduced = items.find(m => m.profile === 'reduced' && m.status === 'ready')!;
  const other = { ...reduced, model_id: 'other-snapshot', dataset_id: 'different-dataset' };
  await page.route('**/api/models', route => route.fulfill({ json: { items: [...items, other] } }));
  await page.goto('/#mo-hinh');
  await expect(page.getByTestId('report-model')).toHaveText(full.model_id);
  await expect(page.getByRole('table', { name: 'So sánh mô hình' })).toContainText(reduced.model_id);
  await page.getByLabel('Mô hình rút gọn để so sánh').selectOption(other.model_id);
  await expect(page.getByRole('alert')).toContainText('cùng snapshot');
  await expect(page.getByRole('table', { name: 'So sánh mô hình' })).toHaveCount(0);
  await page.getByLabel('Mô hình rút gọn để so sánh').selectOption(reduced.model_id);
  await expect(page.getByRole('table', { name: 'So sánh mô hình' })).toContainText(full.model_id);
  let release!: () => void; const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route(`**/api/models/${reduced.model_id}/report`, async route => { const response = await route.fetch(); await gate; await route.fulfill({ response }); });
  const sent = page.waitForRequest(`**/api/models/${reduced.model_id}/report`);
  await page.getByLabel('Mô hình phân tích').selectOption(reduced.model_id); await sent;
  await expect(page.getByTestId('report-model')).toHaveCount(0);
  await page.getByLabel('Mô hình phân tích').selectOption(full.model_id);
  await expect(page.getByTestId('report-model')).toHaveText(full.model_id);
  release(); await page.waitForTimeout(300);
  await expect(page.getByTestId('report-model')).toHaveText(full.model_id);
});

test('failed/interrupted jobs restore without submitting and busy is actionable', async ({ page }) => {
  let submits = 0;
  const job: Job = { job_id: 'job-scoped-failure', status: 'running', stage: 'fitting', model_id: null, error: null };
  await page.route('**/api/models/train', route => { submits++; return route.fulfill({ status: 202, json: job }); });
  await page.route('**/api/jobs/job-scoped-failure', route => route.fulfill({ json: { ...job, status: 'failed', error: { error: { code: 'TRAINING_FAILED', message: 'Không thể hoàn tất huấn luyện. Hãy kiểm tra dữ liệu và thử lại.', details: [] } } } }));
  await page.goto('/#mo-hinh');
  await expect(page.getByLabel('Số cây')).toHaveValue('50');
  await page.getByRole('button', { name: 'Huấn luyện', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('Không thể hoàn tất huấn luyện');
  await page.reload();
  await expect(page.getByRole('alert')).toContainText('Không thể hoàn tất huấn luyện');
  expect(submits).toBe(1);
  await page.unroute('**/api/jobs/job-scoped-failure');
  await page.route('**/api/jobs/job-scoped-failure', route => route.fulfill({ json: { ...job, status: 'interrupted', error: { error: { code: 'TRAINING_INTERRUPTED', message: 'Tác vụ bị gián đoạn. Bạn có thể huấn luyện lại.', details: [] } } } }));
  await page.reload();
  await expect(page.getByRole('alert')).toContainText('gián đoạn');
  await page.unroute('**/api/models/train');
  await page.route('**/api/models/train', route => route.fulfill({ status: 409, json: { error: { code: 'TRAINING_BUSY', message: 'Đang có tác vụ huấn luyện. Hãy chờ tác vụ hoàn tất.', details: [] } } }));
  await page.getByRole('button', { name: 'Huấn luyện lại', exact: true }).click();
  await expect(page.getByRole('alert').filter({ hasText: 'Đang có tác vụ' })).toBeVisible();
});

test('real training survives reload, serializes polling, refreshes list and preserves pinned session', async ({ page, request }, info) => {
  const data: DatasetInfo = await (await request.get('/api/data')).json();
  const { items }: { items: ModelInfo[] } = await (await request.get('/api/models')).json();
  const model = items.find(m => m.status === 'ready' && m.profile === 'full')!;
  await page.goto('/#theo-doi');
  await page.getByRole('button', { name: 'Tạo phiên', exact: true }).click();
  await expect(page.getByTestId('session-status')).toHaveText('Sẵn sàng');
  const sessionId = await page.getByTestId('session-id').innerText();
  await page.getByRole('button', { name: 'Bắt đầu', exact: true }).click();
  await expect.poll(async () => Number(await page.getByTestId('processed-count').innerText())).toBeGreaterThan(0);
  await page.getByRole('link', { name: 'Phân tích mô hình', exact: true }).click();
  await expect(page.getByTestId('report-model')).toHaveText(model.model_id);
  let submits = 0; page.on('request', r => { if (r.url().endsWith('/api/models/train')) submits++; });
  let active = 0, maximum = 0;
  await page.route('**/api/jobs/*', async route => { active++; maximum = Math.max(maximum, active); const response = await route.fetch(); await new Promise(resolve => setTimeout(resolve, 1200)); active--; await route.fulfill({ response }); });
  await page.getByLabel('Cấu hình huấn luyện').selectOption('full');
  await page.getByRole('button', { name: 'Huấn luyện', exact: true }).click();
  await expect(page.getByTestId('job-id')).toContainText('job-');
  const id = await page.getByTestId('job-id').innerText();
  await expect(page.getByRole('button', { name: 'Huấn luyện', exact: true })).toBeDisabled();
  await page.waitForTimeout(1500);
  await page.unroute('**/api/jobs/*');
  await page.reload();
  await expect(page.getByTestId('job-id')).toHaveText(id);
  await expect(page.getByTestId('job-status')).toHaveText('Hoàn tất', { timeout: 100_000 });
  const job: Job = await (await request.get(`/api/jobs/${id}`)).json();
  expect(job.status).toBe('succeeded'); expect(job.model_id).toBeTruthy();
  await expect(page.getByTestId('report-model')).toHaveText(model.model_id);
  await expect(page.getByLabel('Mô hình phân tích').locator(`option[value="${job.model_id}"]`)).toHaveCount(1);
  expect(submits).toBe(1); expect(maximum).toBe(1);
  await expect(page.getByTestId('job-panel')).not.toContainText('%');
  await page.getByRole('button', { name: 'Chọn mô hình mới' }).click();
  await expect(page.getByTestId('report-model')).toHaveText(job.model_id!);
  await page.getByRole('link', { name: 'Theo dõi', exact: true }).click();
  await expect(page.getByTestId('session-id')).toHaveText(sessionId);
  await expect(page.getByTestId('pinned-model')).toHaveText(model.model_id);
  const session = await (await request.get(`/api/sessions/${sessionId}`)).json();
  expect(session.dataset_id).toBe(data.dataset_id); expect(session.model_id).toBe(model.model_id); expect(session.status).toBe('paused');
  await info.attach('real-training-and-pinning', { body: JSON.stringify({ job, selected_before_explicit_choose: model.model_id, pinned_session: session, submit_count: submits, maximum_concurrent_delayed_polls: maximum }, null, 2), contentType: 'application/json' });
  await page.getByRole('button', { name: 'Kết thúc', exact: true }).click();
});

test('recognition and model layouts use local assets, keyboard focus and fit responsive widths', async ({ page }, info) => {
  const errors: string[] = [], external: string[] = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('request', r => { if (!r.url().startsWith(info.project.use.baseURL!)) external.push(r.url()); });
  for (const route of ['nhan-dien', 'mo-hinh']) for (const [width, height] of [[1366, 768], [1024, 768], [390, 844]]) {
    await page.setViewportSize({ width, height }); await page.goto(`/#${route}`);
    if (route === 'nhan-dien') { await expect(page.getByRole('button', { name: 'Nhận diện', exact: true })).toBeEnabled(); await page.getByRole('button', { name: 'Nhận diện', exact: true }).click(); await expect(page.getByTestId('recognition-results')).toBeVisible(); }
    else await expect(page.getByTestId('report-model')).toBeVisible();
    await page.waitForLoadState('networkidle');
    await page.evaluate(() => window.scrollTo(0, 0));
    const overflow = await page.evaluate(() => ({ fits: document.documentElement.scrollWidth <= innerWidth, offenders: [...document.querySelectorAll('main *')].filter(el => el.getBoundingClientRect().right > innerWidth).map(el => ({ tag: el.tagName, cls: el.className, text: el.textContent?.slice(0, 80), right: el.getBoundingClientRect().right })) }));
    expect(overflow.fits, JSON.stringify({ route, width, offenders: overflow.offenders })).toBe(true);
    await page.screenshot({ path: info.outputPath(`${route}-${width}x${height}.png`), fullPage: true });
    await page.screenshot({ path: info.outputPath(`${route}-${width}x${height}-viewport.png`) });
  }
  await page.goto('/#nhan-dien'); await page.keyboard.press('Tab');
  await expect(page.getByRole('link', { name: 'Đến nội dung chính' })).toBeFocused();
  expect(external).toEqual([]); expect(errors).toEqual([]);
});

test('report errors recover and absent OOB shows backend warning separately from test metrics', async ({ page, request }) => {
  const { items }: { items: ModelInfo[] } = await (await request.get('/api/models')).json();
  const full = items.find(m => m.profile === 'full' && m.status === 'ready')!;
  await page.route(`**/api/models/${full.model_id}/report`, route => route.abort('failed'));
  await page.goto('/#mo-hinh');
  await expect(page.getByRole('alert').first()).toContainText('Mất kết nối');
  await expect(page.getByTestId('model-accuracy')).toHaveCount(0);
  await page.unroute(`**/api/models/${full.model_id}/report`);
  const report: ModelReport = await (await request.get(`/api/models/${full.model_id}/report`)).json();
  await page.route(`**/api/models/${full.model_id}/report`, route => route.fulfill({ json: { ...report, oob_score: null, oob_warning: 'Không có ước lượng OOB đáng tin cậy với số cây này.' } }));
  await page.getByRole('button', { name: 'Thử lại', exact: true }).first().click();
  await expect(page.getByTestId('model-accuracy')).toHaveText(percent(report.accuracy));
  await expect(page.getByTestId('oob-score')).toHaveText('Không có ước lượng OOB');
  await expect(page.getByRole('note')).toContainText('Không có ước lượng OOB đáng tin cậy');
});

test('polling connection failure preserves job and retries GET; a missing persisted job can be dismissed explicitly', async ({ page }) => {
  await page.goto('/#mo-hinh');
  await page.evaluate(() => localStorage.setItem('motionsense.training-job', 'job-scoped-reconnect'));
  let submits = 0; page.on('request', r => { if (r.url().endsWith('/api/models/train')) submits++; });
  await page.route('**/api/jobs/job-scoped-reconnect', route => route.abort('failed'));
  await page.reload();
  await expect(page.getByRole('alert')).toContainText('Mất kết nối');
  await expect(page.getByRole('button', { name: 'Huấn luyện', exact: true })).toBeDisabled();
  await page.unroute('**/api/jobs/job-scoped-reconnect');
  await page.route('**/api/jobs/job-scoped-reconnect', route => route.fulfill({ status: 404, json: { error: { code: 'JOB_NOT_FOUND', message: 'Không tìm thấy tác vụ huấn luyện.', details: [] } } }));
  await page.getByRole('button', { name: 'Thử lại', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('Không tìm thấy tác vụ');
  await page.getByRole('button', { name: 'Bỏ tác vụ không tồn tại' }).click();
  await expect(page.getByTestId('job-panel')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Huấn luyện', exact: true })).toBeEnabled();
  expect(submits).toBe(0);
});

test('development StrictMode restores saved job without resubmission or automatic model choice', async ({ page, request }, info) => {
  const { items }: { items: ModelInfo[] } = await (await request.get('/api/models')).json();
  const full = items.find(m => m.profile === 'full' && m.status === 'ready')!;
  const reduced = items.find(m => m.profile === 'reduced' && m.status === 'ready')!;
  const { createServer } = await import('vite');
  const server = await createServer({ root: process.cwd(), mode: 'development', server: { host: '127.0.0.1', port: 0, proxy: { '/api': info.project.use.baseURL! } } });
  let submits = 0; const errors: string[] = [];
  page.on('request', r => { if (r.url().endsWith('/api/models/train')) submits++; });
  page.on('pageerror', e => errors.push(e.message));
  await page.route('**/api/jobs/job-strict-restore', route => route.fulfill({ json: { job_id: 'job-strict-restore', status: 'succeeded', stage: 'saving', model_id: reduced.model_id, error: null } }));
  await page.addInitScript(id => { localStorage.setItem('motionsense.training-job', 'job-strict-restore'); localStorage.setItem('motionsense.analysis-model', id); }, full.model_id);
  try {
    await server.listen();
    const address = server.httpServer!.address() as { port: number };
    await page.goto(`http://127.0.0.1:${address.port}/#mo-hinh`);
    await expect(page.getByTestId('job-status')).toHaveText('Hoàn tất');
    await expect(page.getByTestId('report-model')).toHaveText(full.model_id);
    await page.getByRole('link', { name: 'Nhận diện', exact: true }).click();
    await expect(page.getByLabel('Mô hình')).toHaveValue(full.model_id);
    await page.getByRole('button', { name: 'Nhận diện', exact: true }).click();
    await expect(page.getByTestId('recognition-model')).toHaveText(full.model_id);
    await page.getByRole('link', { name: 'Phân tích mô hình', exact: true }).click();
    await expect(page.getByTestId('report-model')).toHaveText(full.model_id);
    expect(submits).toBe(0); expect(errors).toEqual([]);
  } finally { await page.goto('about:blank'); await server.close(); }
});

for (const previous of ['none', 'terminal'] as const) test(`fixround1 S1: pending real training acknowledgement survives route return with ${previous} previous job`, async ({ page, request }, info) => {
  const { items }: { items: ModelInfo[] } = await (await request.get('/api/models')).json();
  const selected = items.find(m => m.profile === 'reduced' && m.status === 'ready')!;
  await page.goto('/#mo-hinh');
  await page.getByLabel('Mô hình phân tích').selectOption(selected.model_id);
  if (previous === 'terminal') {
    await page.route('**/api/jobs/job-previous-terminal', route => route.fulfill({ json: { job_id: 'job-previous-terminal', status: 'succeeded', stage: 'saving', model_id: selected.model_id, error: null } }));
    await page.evaluate(() => localStorage.setItem('motionsense.training-job', 'job-previous-terminal'));
    await page.reload();
    await expect(page.getByTestId('job-status')).toHaveText('Hoàn tất');
  }
  let release!: () => void; const gate = new Promise<void>(resolve => { release = resolve; });
  let accepted!: (job: Job) => void; const serverAccepted = new Promise<Job>(resolve => { accepted = resolve; });
  let submissions = 0;
  await page.route('**/api/models/train', async route => {
    submissions++; const response = await route.fetch(); accepted(await response.json());
    await gate; await route.fulfill({ response });
  });
  try {
    await page.getByRole('button', { name: 'Huấn luyện', exact: true }).click();
    const actual = await serverAccepted;
    expect(actual.job_id).toContain('job-');
    await page.getByRole('link', { name: 'Nhận diện', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Nhận diện', exact: true })).toBeVisible();
    await page.getByRole('link', { name: 'Phân tích mô hình', exact: true }).click();
    await expect(page.getByTestId('report-model')).toHaveText(selected.model_id);
    await expect(page.getByRole('button', { name: 'Huấn luyện', exact: true })).toBeDisabled();
    const polled = page.waitForRequest(`**/api/jobs/${actual.job_id}`);
    release(); await polled;
    await expect(page.getByTestId('job-id')).toHaveText(actual.job_id);
    await expect(page.getByTestId('job-status')).toHaveText('Hoàn tất', { timeout: 90_000 });
    const saved: Job = await (await request.get(`/api/jobs/${actual.job_id}`)).json();
    expect(saved.status).toBe('succeeded');
    expect(await page.evaluate(() => localStorage.getItem('motionsense.training-job'))).toBe(actual.job_id);
    await expect(page.getByTestId('report-model')).toHaveText(selected.model_id);
    expect(submissions).toBe(1);
    await page.screenshot({ path: info.outputPath(`s1-${previous}-ack-restored.png`), fullPage: true });
    await page.getByRole('button', { name: 'Chọn mô hình mới' }).click();
    await expect(page.getByTestId('report-model')).toHaveText(saved.model_id!);
    await page.reload();
    await expect(page.getByTestId('job-id')).toHaveText(actual.job_id);
    await expect(page.getByTestId('report-model')).toHaveText(saved.model_id!);
    expect(submissions).toBe(1);
  } finally { release(); }
});

for (const side of ['full', 'reduced'] as const) test(`fixround1 Q1: ${side} comparison report pending has feedback, errors take precedence and selection cannot mix snapshots`, async ({ page, request }, info) => {
  const { items }: { items: ModelInfo[] } = await (await request.get('/api/models')).json();
  const full = items.find(m => m.profile === 'full' && m.status === 'ready')!;
  const reduced = items.find(m => m.profile === 'reduced' && m.status === 'ready')!;
  const delayed = side === 'full' ? full : reduced;
  const completed = side === 'full' ? reduced : full;
  const completedReport: ModelReport = await (await request.get(`/api/models/${completed.model_id}/report`)).json();
  // A second same-snapshot selectable version for one-side change; values stay
  // sourced from the real report, with only the controlled version ID changed.
  const alternate = { ...delayed, model_id: `comparison-${side}-alternate` };
  const delayedReport: ModelReport = await (await request.get(`/api/models/${delayed.model_id}/report`)).json();
  await page.route('**/api/models', route => route.fulfill({ json: { items: [...items, alternate] } }));
  let release!: () => void; let gate = new Promise<void>(resolve => { release = resolve; });
  await page.route(`**/api/models/${delayed.model_id}/report`, async route => { const response = await route.fetch(); await gate; await route.fulfill({ response }); });
  const completedResponse = page.waitForResponse(`**/api/models/${completed.model_id}/report`);
  await page.goto('/#mo-hinh'); await completedResponse;
  const comparison = page.locator('section').filter({ has: page.getByRole('heading', { name: 'So sánh đầy đủ / rút gọn', exact: true }) });
  try {
    await expect(comparison.getByRole('status')).toContainText('Đang tải báo cáo so sánh');
    await expect(comparison.getByRole('table')).toHaveCount(0);
    release();
    await expect(comparison.getByRole('table')).toContainText(delayed.model_id);
    gate = new Promise<void>(resolve => { release = resolve; });
    let responseMode: 'success' | 'error' = 'success';
    await page.route(`**/api/models/${alternate.model_id}/report`, async route => {
      const outcome = responseMode;
      await gate;
      await route.fulfill(outcome === 'error' ? { status: 503, json: { error: { code: 'REPORT_UNAVAILABLE', message: 'Không thể đọc báo cáo so sánh. Hãy thử lại.', details: [] } } } : { json: { ...delayedReport, model_id: alternate.model_id } });
    });
    const select = page.getByLabel(side === 'full' ? 'Mô hình đầy đủ để so sánh' : 'Mô hình rút gọn để so sánh');
    await select.selectOption(alternate.model_id);
    await expect(comparison.getByRole('status')).toContainText('Đang tải báo cáo so sánh');
    await expect(comparison.getByRole('table')).toHaveCount(0);
    // Change back before the old version arrives: neither old rows nor old errors
    // may contaminate the newly selected complete comparison.
    await select.selectOption(delayed.model_id);
    release();
    await expect(comparison.getByRole('table')).toContainText(delayed.model_id);
    await expect(comparison.getByRole('table')).not.toContainText(alternate.model_id);
    responseMode = 'error';
    gate = new Promise<void>(resolve => { release = resolve; });
    await select.selectOption(alternate.model_id);
    await expect(comparison.getByRole('status')).toContainText('Đang tải báo cáo so sánh');
    release();
    await expect(comparison.getByRole('alert')).toContainText('Không thể đọc báo cáo so sánh');
    await expect(comparison.getByRole('status')).toHaveCount(0);
    await expect(comparison.getByRole('table')).toHaveCount(0);
    responseMode = 'success';
    await comparison.getByRole('button', { name: 'Thử lại', exact: true }).click();
    await expect(comparison.getByRole('table')).toContainText(alternate.model_id);
    const completedRow = comparison.getByRole('row').filter({ hasText: completed.model_id });
    await expect(completedRow).toContainText(percent(completedReport.accuracy));
    const delayedRow = comparison.getByRole('row').filter({ hasText: alternate.model_id });
    await expect(delayedRow).toContainText(percent(delayedReport.accuracy));
    await page.screenshot({ path: info.outputPath(`q1-${side}-complete.png`), fullPage: true });
  } finally { release(); }
});
