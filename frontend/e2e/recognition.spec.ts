import { test, expect } from '@playwright/test';
import type { DatasetInfo, ModelInfo, Prediction, Sample } from '../src/api/types';

// Catches accepting invalid files, feature/metadata leakage, stale predictions,
// incorrect signal pairing and result pagination that silently loses rows.
test.setTimeout(40_000);
test('CSV errors include row/column and never retain a successful result', async ({ page, request }) => {
  const { items }: { items: ModelInfo[] } = await (await request.get('/api/models')).json();
  const full = items.find(m => m.status === 'ready' && m.profile === 'full')!;
  expect(full).toBeTruthy();
  await page.goto('/#nhan-dien');
  await expect(page.getByLabel('Mô hình')).toHaveValue(full.model_id);
  await page.getByRole('button', { name: 'Nhận diện', exact: true }).click();
  await expect(page.getByTestId('recognition-results')).toBeVisible();
  await page.getByRole('tab', { name: 'Tệp CSV', exact: true }).click();
  await expect(page.getByTestId('recognition-results')).toHaveCount(0);
  await page.getByLabel('Tệp đặc trưng').setInputFiles({ name: 'missing.csv', mimeType: 'text/csv', buffer: Buffer.from('f001\n0.2\n') });
  await page.getByRole('button', { name: 'Nhận diện', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText(/thiếu/i);
  await expect(page.getByRole('table', { name: 'Lỗi CSV' })).toContainText('f002');
  await expect(page.getByTestId('recognition-results')).toHaveCount(0);
  const csv = (await (await request.get(`/api/recognition/template?model_id=${full.model_id}`)).text()).replace(/^\uFEFF/, '');
  const [header, row] = csv.trim().split(/\r?\n/);
  const values = row.split(','); values[0] = 'NaN';
  await page.getByLabel('Tệp đặc trưng').setInputFiles({ name: 'nan.csv', mimeType: 'text/csv', buffer: Buffer.from(`${header}\n${values.join(',')}\n`) });
  await page.getByRole('button', { name: 'Nhận diện', exact: true }).click();
  await expect(page.getByRole('table', { name: 'Lỗi CSV' })).toContainText('2');
  await expect(page.getByRole('table', { name: 'Lỗi CSV' })).toContainText('f001');
  await expect(page.getByTestId('recognition-results')).toHaveCount(0);
});

test('real sample, full template CSV and reduced input preserve probabilities and separate metadata', async ({ page, request }) => {
  const data: DatasetInfo = await (await request.get('/api/data')).json();
  const { items }: { items: ModelInfo[] } = await (await request.get('/api/models')).json();
  const full = items.find(m => m.status === 'ready' && m.profile === 'full')!;
  const reduced = items.find(m => m.status === 'ready' && m.profile === 'reduced' && m.dataset_id === full.dataset_id)!;
  expect(reduced).toBeTruthy();
  await page.goto('/#nhan-dien');
  const subject = data.subjects.find(s => s.split === 'test')!;
  await expect(page.getByLabel('Người tham gia')).toHaveValue(`test:${subject.subject_id}`);
  const sampleId = await page.getByLabel('Mẫu nguồn').inputValue();
  const sample: Sample = await (await request.get(`/api/data/samples/${sampleId}`)).json();
  const prediction: Prediction = await (await request.post('/api/recognition/sample', { data: { model_id: full.model_id, sample_id: sampleId } })).json();
  await page.getByRole('button', { name: 'Nhận diện', exact: true }).click();
  await expect(page.getByTestId('recognition-model')).toHaveText(full.model_id);
  for (const p of prediction.probabilities) await expect(page.getByTestId(`probability-${p.label_id}`)).toHaveAttribute('value', String(p.value));
  await expect(page.getByTestId('signal-chart')).toHaveAttribute('data-sample-id', sampleId);
  expect(JSON.parse((await page.getByTestId('signal-chart').locator('polyline').first().getAttribute('data-values'))!)).toEqual(sample.signal.x);
  await expect(page.getByTestId('recognition-prediction')).not.toContainText('Nhãn thực tế');
  await page.getByText('Đối chiếu nhãn thực tế', { exact: true }).click();
  await expect(page.getByTestId('recognition-actual')).toContainText(data.activities.find(a => a.label_id === prediction.actual_label)!.name_vi);
  await page.getByRole('button', { name: 'Mẫu sau' }).click();
  await expect(page.getByTestId('recognition-results')).toHaveCount(0);
  await page.getByRole('button', { name: 'Mẫu trước' }).click();
  const ids = Object.keys(sample.features).reverse();
  const csv = `${ids.join(',')},activity,subject_id\n${ids.map(id => sample.features[id]).join(',')},${sample.actual_label},${sample.subject_id}\n`;
  await page.getByRole('tab', { name: 'Tệp CSV', exact: true }).click();
  for (const model of [full, reduced]) {
    await page.getByLabel('Mô hình').selectOption(model.model_id);
    await expect(page.getByRole('link', { name: 'Tải CSV mẫu' })).toHaveAttribute('href', `/api/recognition/template?model_id=${encodeURIComponent(model.model_id)}`);
    const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('link', { name: 'Tải CSV mẫu' }).click()]);
    expect(await download.failure()).toBeNull();
    const template = await (await request.get(`/api/recognition/template?model_id=${model.model_id}`)).text();
    await page.getByLabel('Tệp đặc trưng').setInputFiles({ name: 'template.csv', mimeType: 'text/csv', buffer: Buffer.from(template) });
    const expected = await (await request.post('/api/recognition/csv', { multipart: { model_id: model.model_id, file: { name: 'template.csv', mimeType: 'text/csv', buffer: Buffer.from(template) } } })).json();
    const templateSample: Prediction = await (await request.post('/api/recognition/sample', { data: { model_id: model.model_id, sample_id: 'test:000001' } })).json();
    expect(expected.items[0].predicted_label).toBe(templateSample.predicted_label);
    expect(expected.items[0].probabilities).toEqual(templateSample.probabilities);
    await page.getByRole('button', { name: 'Nhận diện', exact: true }).click();
    for (const p of expected.items[0].probabilities) await expect(page.getByTestId(`probability-${p.label_id}`)).toHaveAttribute('value', String(p.value));
    await page.getByLabel('Tệp đặc trưng').setInputFiles({ name: 'full-reordered.csv', mimeType: 'text/csv', buffer: Buffer.from(csv) });
    const expectedSample: Prediction = await (await request.post('/api/recognition/sample', { data: { model_id: model.model_id, sample_id: sampleId } })).json();
    await page.getByRole('button', { name: 'Nhận diện', exact: true }).click();
    for (const p of expectedSample.probabilities) await expect(page.getByTestId(`probability-${p.label_id}`)).toHaveAttribute('value', String(p.value));
    await expect(page.getByTestId('signal-chart')).toHaveCount(0);
  }
  const template = (await (await request.get(`/api/recognition/template?model_id=${reduced.model_id}`)).text()).replace(/^\uFEFF/, '').trim().split(/\r?\n/);
  await page.getByLabel('Tệp đặc trưng').setInputFiles({ name: 'many.csv', mimeType: 'text/csv', buffer: Buffer.from(`${template[0]}\n${Array(51).fill(template[1]).join('\n')}\n`) });
  await page.getByRole('button', { name: 'Nhận diện', exact: true }).click();
  await expect(page.getByRole('table', { name: 'Kết quả nhận diện' }).locator('tbody tr')).toHaveCount(50);
  await page.getByRole('button', { name: 'Kết quả sau' }).click();
  await expect(page.getByRole('table', { name: 'Kết quả nhận diện' }).locator('tbody tr')).toHaveCount(1);
  await page.getByRole('button', { name: 'Xem kết quả 51' }).click();
  await expect(page.getByRole('button', { name: 'Xem kết quả 51' })).toHaveAttribute('aria-pressed', 'true');
});

test('changing model or input cancels delayed real predictions and never pairs old signal', async ({ page, request }) => {
  const { items }: { items: ModelInfo[] } = await (await request.get('/api/models')).json();
  const full = items.find(m => m.profile === 'full' && m.status === 'ready')!;
  const reduced = items.find(m => m.profile === 'reduced' && m.status === 'ready')!;
  await page.goto('/#nhan-dien');
  await expect(page.getByRole('button', { name: 'Nhận diện', exact: true })).toBeEnabled();
  let release!: () => void;
  const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/api/recognition/sample', async route => { const response = await route.fetch(); await gate; await route.fulfill({ response }); });
  const sent = page.waitForRequest('**/api/recognition/sample');
  await page.getByRole('button', { name: 'Nhận diện', exact: true }).click(); await sent;
  await expect(page.getByRole('button', { name: 'Nhận diện', exact: true })).toBeDisabled();
  await page.getByLabel('Mô hình').selectOption(reduced.model_id);
  release();
  await page.waitForTimeout(250);
  await expect(page.getByTestId('recognition-results')).toHaveCount(0);
  await page.unroute('**/api/recognition/sample');
  await page.getByRole('button', { name: 'Nhận diện', exact: true }).click();
  await expect(page.getByTestId('recognition-model')).toHaveText(reduced.model_id);
  await page.getByLabel('Mô hình').selectOption(full.model_id);
  await expect(page.getByTestId('recognition-results')).toHaveCount(0);
  await page.route('**/api/recognition/sample', route => route.abort('failed'));
  await page.getByRole('button', { name: 'Nhận diện', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('Mất kết nối');
  await page.unroute('**/api/recognition/sample');
  await page.getByRole('button', { name: 'Nhận diện', exact: true }).click();
  await expect(page.getByTestId('recognition-model')).toHaveText(full.model_id);
});

test('CSV size and row limits reject without results; people filter uses real sample pages', async ({ page, request }) => {
  const data: DatasetInfo = await (await request.get('/api/data')).json();
  const { items }: { items: ModelInfo[] } = await (await request.get('/api/models')).json();
  const model = items.find(m => m.profile === 'reduced' && m.status === 'ready')!;
  await page.goto('/#nhan-dien');
  const subject = data.subjects.find(s => s.split === 'train')!;
  await page.getByLabel('Người tham gia').selectOption(`train:${subject.subject_id}`);
  const source = await (await request.get(`/api/data/samples?split=train&subject_id=${subject.subject_id}&offset=0&limit=50`)).json();
  await expect(page.getByLabel('Mẫu nguồn')).toHaveValue(source.items[0].sample_id);
  const ids = await page.getByLabel('Mẫu nguồn').locator('option').evaluateAll(nodes => nodes.map(n => (n as HTMLOptionElement).value));
  expect(ids).toEqual(source.items.map((s: Sample) => s.sample_id));
  await page.getByRole('button', { name: 'Mẫu sau' }).click();
  const next = await (await request.get(`/api/data/samples?split=train&subject_id=${subject.subject_id}&offset=50&limit=50`)).json();
  await expect(page.getByLabel('Mẫu nguồn')).toHaveValue(next.items[0].sample_id);
  await page.getByLabel('Mô hình').selectOption(model.model_id);
  await page.getByRole('tab', { name: 'Tệp CSV' }).click();
  await page.getByLabel('Tệp đặc trưng').setInputFiles({ name: 'large.csv', mimeType: 'text/csv', buffer: Buffer.alloc(16 * 1024 * 1024 + 1, 65) });
  let uploads = 0; page.on('request', r => { if (r.url().endsWith('/api/recognition/csv')) uploads++; });
  await page.getByRole('button', { name: 'Nhận diện', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('16 MiB');
  await expect(page.getByTestId('recognition-results')).toHaveCount(0); expect(uploads).toBe(0);
  const template = (await (await request.get(`/api/recognition/template?model_id=${model.model_id}`)).text()).replace(/^\uFEFF/, '').trim().split(/\r?\n/);
  await page.getByLabel('Tệp đặc trưng').setInputFiles({ name: 'too-many.csv', mimeType: 'text/csv', buffer: Buffer.from(`${template[0]}\n${Array(5001).fill(template[1]).join('\n')}\n`) });
  await page.getByRole('button', { name: 'Nhận diện', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('5000');
  await expect(page.getByTestId('recognition-results')).toHaveCount(0);
});

test('delayed real signal is hidden after changing source sample and setup is actionable', async ({ page, request }) => {
  await page.goto('/#nhan-dien');
  await expect(page.getByRole('button', { name: 'Nhận diện', exact: true })).toBeEnabled();
  const id = await page.getByLabel('Mẫu nguồn').inputValue();
  let release!: () => void; const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route(`**/api/data/samples/${encodeURIComponent(id)}`, async route => { const response = await route.fetch(); await gate; await route.fulfill({ response }); });
  const sent = page.waitForRequest(`**/api/data/samples/${encodeURIComponent(id)}`);
  await page.getByRole('button', { name: 'Nhận diện', exact: true }).click(); await sent;
  const options = await page.getByLabel('Mẫu nguồn').locator('option').evaluateAll(nodes => nodes.map(n => (n as HTMLOptionElement).value));
  await page.getByLabel('Mẫu nguồn').selectOption(options[1]); release();
  await expect(page.getByTestId('recognition-results')).toHaveCount(0);
  await page.getByRole('button', { name: 'Nhận diện', exact: true }).click();
  await expect(page.getByTestId('signal-chart')).toHaveAttribute('data-sample-id', options[1]);
  const expected: Sample = await (await request.get(`/api/data/samples/${options[1]}`)).json();
  expect(JSON.parse((await page.getByTestId('signal-chart').locator('polyline').first().getAttribute('data-values'))!)).toEqual(expected.signal.x);
  await page.route('**/api/models', route => route.fulfill({ json: { items: [] } }));
  await page.reload();
  await expect(page.getByRole('status')).toContainText('Cai_dat.bat');
  await expect(page.getByRole('button', { name: 'Nhận diện', exact: true })).toHaveCount(0);
});

test('returning to CSV cannot submit a file no longer displayed; tabs work from keyboard', async ({ page }) => {
  await page.goto('/#nhan-dien');
  const sampleTab = page.getByRole('tab', { name: 'Mẫu có sẵn', exact: true });
  await sampleTab.focus(); await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('tab', { name: 'Tệp CSV', exact: true })).toBeFocused();
  await page.getByLabel('Tệp đặc trưng').setInputFiles({ name: 'selected.csv', mimeType: 'text/csv', buffer: Buffer.from('f001\n0.2\n') });
  await expect(page.getByRole('button', { name: 'Nhận diện', exact: true })).toBeEnabled();
  await page.getByRole('tab', { name: 'Mẫu có sẵn', exact: true }).click();
  await page.getByRole('tab', { name: 'Tệp CSV', exact: true }).click();
  await expect(page.getByLabel('Tệp đặc trưng')).toHaveValue('');
  await expect(page.getByRole('button', { name: 'Nhận diện', exact: true })).toBeDisabled();
});

test('fixround1 S2: failed source signal has terminal error and retry; delayed retry never pairs an old sample', async ({ page, request }, info) => {
  await page.goto('/#nhan-dien');
  await expect(page.getByRole('button', { name: 'Nhận diện', exact: true })).toBeEnabled();
  const originalId = await page.getByLabel('Mẫu nguồn').inputValue();
  let predictions = 0; page.on('request', r => { if (r.url().endsWith('/api/recognition/sample')) predictions++; });
  let mode: 'fail' | 'delayed' | 'success' = 'fail';
  let release!: () => void; const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route(`**/api/data/samples/${encodeURIComponent(originalId)}`, async route => {
    if (mode === 'fail') return route.abort('failed');
    const response = await route.fetch(); if (mode === 'delayed') await gate;
    await route.fulfill({ response });
  });
  await page.getByRole('button', { name: 'Nhận diện', exact: true }).click();
  const signalCard = page.locator('section.card').filter({ has: page.getByRole('heading', { name: 'Tín hiệu mẫu nguồn', exact: true }) });
  await expect(signalCard.getByRole('alert')).toContainText('Mất kết nối');
  await expect(signalCard.getByText('Đang tải cửa sổ tín hiệu đúng mẫu…', { exact: true })).toHaveCount(0);
  await expect(signalCard.getByTestId('signal-chart')).toHaveCount(0);
  mode = 'success';
  await signalCard.getByRole('button', { name: 'Thử lại', exact: true }).click();
  await expect(signalCard.getByTestId('signal-chart')).toHaveAttribute('data-sample-id', originalId);
  const original: Sample = await (await request.get(`/api/data/samples/${originalId}`)).json();
  expect(JSON.parse((await signalCard.locator('polyline').first().getAttribute('data-values'))!)).toEqual(original.signal.x);
  expect(predictions).toBe(1);
  // Fail the same sample again, then switch input while its explicit retry waits.
  mode = 'fail';
  await page.getByRole('button', { name: 'Nhận diện', exact: true }).click();
  await expect(signalCard.getByRole('alert')).toBeVisible();
  mode = 'delayed';
  const sent = page.waitForRequest(`**/api/data/samples/${encodeURIComponent(originalId)}`);
  try {
    await signalCard.getByRole('button', { name: 'Thử lại', exact: true }).click(); await sent;
    await expect(signalCard.getByText('Đang tải cửa sổ tín hiệu đúng mẫu…', { exact: true })).toBeVisible();
    const ids = await page.getByLabel('Mẫu nguồn').locator('option').evaluateAll(nodes => nodes.map(n => (n as HTMLOptionElement).value));
    await page.getByLabel('Mẫu nguồn').selectOption(ids[1]);
    await expect(page.getByTestId('recognition-results')).toHaveCount(0);
    await page.getByRole('button', { name: 'Nhận diện', exact: true }).click();
    await expect(page.getByTestId('signal-chart')).toHaveAttribute('data-sample-id', ids[1]);
    release();
    const current: Sample = await (await request.get(`/api/data/samples/${ids[1]}`)).json();
    await expect(page.getByTestId('signal-chart')).toHaveAttribute('data-sample-id', ids[1]);
    expect(JSON.parse((await page.getByTestId('signal-chart').locator('polyline').first().getAttribute('data-values'))!)).toEqual(current.signal.x);
    await expect(page.getByRole('alert')).toHaveCount(0);
    await page.screenshot({ path: info.outputPath('s2-retry-current-sample.png'), fullPage: true });
  } finally { release(); }
});
