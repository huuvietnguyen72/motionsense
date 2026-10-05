import { test, expect } from '@playwright/test';
import type { DatasetInfo, FeatureInfo, ModelInfo } from '../src/api/types';

const format = (value: number) => new Intl.NumberFormat('vi-VN').format(value);
const mode = process.env.MOTIONSENSE_TEST_MODE ?? 'real';

test('trang dữ liệu hiện nguồn và số liệu từ API', async ({ page, request }) => {
  test.skip(mode !== 'real', 'Checked separately against an unready real server');
  const data: DatasetInfo = await (await request.get('/api/data')).json();
  expect(data.ready).toBe(true);
  expect(data.split_counts).toEqual({ train: 7352, test: 2947 });
  await page.goto('/#du-lieu');
  await expect(page.getByRole('heading', { name: 'Dữ liệu', exact: true })).toBeVisible();
  await expect(page.getByRole('link', { name: 'Nguồn UCI HAR' })).toHaveAttribute('href', data.source_url);
  await expect(page.getByRole('link', { name: '10.24432/C54S4K' })).toHaveAttribute('href', 'https://doi.org/10.24432/C54S4K');
  await expect(page.getByTestId('dataset-license')).toHaveText(data.license);
  await expect(page.getByTestId('dataset-total')).toHaveText(format(data.split_counts.train + data.split_counts.test));
  await expect(page.getByTestId('dataset-train')).toHaveText(format(data.split_counts.train));
  await expect(page.getByTestId('dataset-test')).toHaveText(format(data.split_counts.test));
  await expect(page.getByTestId('dataset-subjects')).toHaveText(format(new Set(data.subjects.map(s => s.subject_id)).size));
  await expect(page.getByTestId('dataset-features')).toHaveText(format(data.feature_count));
  for (const activity of data.activities) {
    const row = page.getByRole('table', { name: 'Hoạt động trong dữ liệu' }).getByRole('row').filter({ has: page.getByRole('rowheader', { name: activity.name_vi, exact: true }) });
    await expect(row).toContainText(format(activity.count));
  }
  await expect(page.getByText('50 Hz', { exact: true })).toBeVisible();
  await expect(page.getByText(/0–2,54 giây/)).toBeVisible();
  await expect(page.getByText(/không suy ra thời lượng hoạt động/)).toBeVisible();
});

test('tìm đặc trưng theo ID và tên giữ nguyên các tên trùng', async ({ page, request }) => {
  test.skip(mode !== 'real');
  const { items }: { items: FeatureInfo[] } = await (await request.get('/api/data/features')).json();
  const duplicates = items.filter(item => item.name === items.find(candidate => items.filter(f => f.name === candidate.name).length > 1)!.name);
  expect(duplicates.length).toBeGreaterThan(1);
  await page.goto('/#du-lieu');
  const search = page.getByRole('searchbox', { name: 'Tìm đặc trưng theo ID hoặc tên' });
  await search.fill('F001');
  await expect(page.getByRole('table', { name: 'Danh sách đặc trưng' }).getByRole('row')).toHaveCount(2);
  await expect(page.getByRole('cell', { name: 'f001', exact: true })).toBeVisible();
  await search.fill(duplicates[0].name);
  for (const feature of duplicates) await expect(page.getByRole('cell', { name: feature.feature_id, exact: true })).toBeVisible();
  await search.fill('không-có-đặc-trưng-này');
  await expect(page.getByRole('status')).toContainText('Không tìm thấy đặc trưng');
});

test('CSV tải trực tiếp đúng phiên bản mô hình, ưu tiên full ready', async ({ page, request }) => {
  test.skip(mode !== 'real');
  const data: DatasetInfo = await (await request.get('/api/data')).json();
  const { items }: { items: ModelInfo[] } = await (await request.get('/api/models')).json();
  const ready = items.filter(m => m.status === 'ready' && m.dataset_id === data.dataset_id);
  expect(ready.some(m => m.profile === 'full')).toBe(true);
  await page.goto('/#du-lieu');
  const select = page.getByLabel('Mô hình cho tệp CSV');
  await expect(select).toHaveValue(ready.find(m => m.profile === 'full')!.model_id);
  for (const model of ready) {
    await select.selectOption(model.model_id);
    const link = page.getByRole('link', { name: 'Tải CSV mẫu' });
    await expect(link).toHaveAttribute('href', `/api/recognition/template?model_id=${encodeURIComponent(model.model_id)}`);
    const [download] = await Promise.all([page.waitForEvent('download'), link.click()]);
    expect(await download.failure()).toBeNull();
    const response = await request.get(`/api/recognition/template?model_id=${model.model_id}`);
    expect(response.headers()['content-type']).toContain('text/csv');
    const header = (await response.text()).replace(/^\uFEFF/, '').split(/\r?\n/)[0].split(',');
    for (const id of model.feature_ids) expect(header).toContain(id);
  }
});

test('chưa thiết lập hoặc snapshot hỏng không hiện số liệu sẵn sàng', async ({ page, request }, testInfo) => {
  test.skip(mode === 'real');
  const health = await (await request.get('/api/health')).json();
  expect(health.data_ready).toBe(false);
  const data = await (await request.get('/api/data')).json();
  expect(data.ready).toBe(false);
  await page.goto('/#du-lieu');
  await expect(page.getByRole('status')).toContainText('Cai_dat.bat');
  await expect(page.getByTestId('dataset-total')).toHaveCount(0);
  await expect(page.getByRole('link', { name: 'Nguồn UCI HAR' })).toHaveAttribute('href', data.source_url);
  await page.screenshot({ path: testInfo.outputPath(`${mode}-setup.png`), fullPage: true });
});

test('tải chậm, lỗi mạng, phản hồi lỗi hỏng và thử lại', async ({ page }) => {
  test.skip(mode !== 'real');
  await page.route('**/api/data', async route => {
    await new Promise(resolve => setTimeout(resolve, 400));
    await route.abort('failed');
  });
  await page.goto('/#du-lieu');
  await expect(page.getByRole('status')).toContainText('Đang tải');
  await expect(page.getByRole('alert')).toContainText('Mất kết nối với dịch vụ tại máy');
  await page.unroute('**/api/data');
  await page.route('**/api/data', route => route.fulfill({ status: 503, contentType: 'text/html', body: '<html>broken</html>' }));
  await page.getByRole('button', { name: 'Thử lại' }).click();
  await expect(page.getByRole('alert')).toContainText('Phản hồi từ dịch vụ không hợp lệ');
  await page.unroute('**/api/data');
  await page.route('**/api/data', route => route.fulfill({ status: 500, json: {} }));
  await page.getByRole('button', { name: 'Thử lại' }).click();
  await expect(page.getByRole('alert')).toContainText('Phản hồi từ dịch vụ không hợp lệ');
  await page.unroute('**/api/data');
  await page.getByRole('button', { name: 'Thử lại' }).click();
  await expect(page.getByTestId('dataset-total')).toHaveText('10.299');
});

test('lỗi API tiếng Việt và lỗi từng phần không che nguồn dữ liệu', async ({ page }) => {
  test.skip(mode !== 'real');
  await page.route('**/api/data/features', route => route.fulfill({ status: 503, json: { error: { code: 'DATA_UNAVAILABLE', message: 'Dữ liệu đã thay đổi. Hãy chạy Cai_dat.bat.', details: [] } } }));
  await page.route('**/api/models', route => route.fulfill({ json: { items: [] } }));
  await page.goto('/#du-lieu');
  await expect(page.getByTestId('dataset-total')).toHaveText('10.299');
  await expect(page.getByRole('alert')).toContainText('Cai_dat.bat');
  await expect(page.getByRole('status')).toContainText('Chưa có mô hình');
  await expect(page.getByRole('link', { name: 'Tải CSV mẫu' })).toHaveCount(0);
});

test('đổi trang hủy phản hồi cũ, điều hướng lạ về Theo dõi', async ({ page }) => {
  test.skip(mode !== 'real');
  const started = page.waitForRequest(request => request.url().endsWith('/api/data'));
  await page.route('**/api/data', async route => {
    await new Promise(resolve => setTimeout(resolve, 300));
    await route.fulfill({ status: 500, json: { error: { code: 'FAILED', message: 'Không được hiện trên trang mới.', details: [] } } });
  });
  await page.goto('/#du-lieu');
  await started;
  await expect(page.getByRole('status')).toContainText('Đang tải');
  const [cancelled] = await Promise.all([
    page.waitForEvent('requestfailed', request => request.url().endsWith('/api/data')),
    page.getByRole('link', { name: 'Theo dõi', exact: true }).click(),
  ]);
  expect(cancelled.failure()?.errorText).toContain('ERR_ABORTED');
  await expect(page.getByRole('heading', { name: 'Theo dõi', exact: true })).toBeVisible();
  await expect(page.getByRole('alert')).toHaveCount(0);
  await page.goto('/#khong-ton-tai');
  await expect(page.getByRole('link', { name: 'Theo dõi', exact: true })).toHaveAttribute('aria-current', 'page');
});

test('API 404 vẫn JSON, assets cục bộ, bố cục và bàn phím', async ({ page, request }, testInfo) => {
  test.skip(mode !== 'real');
  const response = await request.get('/api/khong-ton-tai');
  expect(response.status()).toBe(404);
  expect((await response.json()).error.code).toBe('NOT_FOUND');
  const external: string[] = [];
  const errors: string[] = [];
  page.on('request', request => { if (!request.url().startsWith(testInfo.project.use.baseURL!)) external.push(request.url()); });
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  for (const [width, height] of [[1366, 768], [1024, 768], [390, 844]]) {
    await page.setViewportSize({ width, height });
    await page.goto('/#du-lieu');
    await expect(page.getByTestId('dataset-total')).toHaveText('10.299');
    await page.waitForLoadState('networkidle');
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: testInfo.outputPath(`data-${width}x${height}.png`), fullPage: true });
  }
  await page.goto('/');
  await page.keyboard.press('Tab');
  await expect(page.getByRole('link', { name: 'Đến nội dung chính' })).toBeFocused();
  const outline = await page.getByRole('link', { name: 'Đến nội dung chính' }).evaluate(el => getComputedStyle(el).outlineStyle);
  expect(outline).not.toBe('none');
  await page.emulateMedia({ reducedMotion: 'reduce' });
  expect(await page.getByRole('link', { name: 'Theo dõi', exact: true }).evaluate(el => getComputedStyle(el).transitionDuration)).toBe('0s');
  expect(external).toEqual([]);
  expect(errors).toEqual([]);
});
