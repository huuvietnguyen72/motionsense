import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';

test('four real screens, recognition, playback, history and CSV with external requests blocked', async ({ page, browser, request }, info) => {
  test.setTimeout(60_000);
  const external: string[] = [];
  await page.route('**/*', async route => {
    const url = new URL(route.request().url());
    if (['127.0.0.1', 'localhost'].includes(url.hostname)) await route.continue();
    else { external.push(url.href); await route.abort(); }
  });
  await page.goto('/#du-lieu');
  await expect(page.getByRole('heading', { name: 'Dữ liệu', exact: true })).toBeVisible();
  await page.screenshot({ path: info.outputPath('offline-data.png'), fullPage: true });
  await page.getByRole('link', { name: 'Nhận diện', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Nhận diện', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'Nhận diện', exact: true }).click();
  await expect(page.getByTestId('recognition-results')).toBeVisible();
  await page.screenshot({ path: info.outputPath('offline-recognition.png'), fullPage: true });
  await page.getByRole('link', { name: 'Phân tích mô hình', exact: true }).click();
  await expect(page.getByRole('table', { name: 'Ma trận nhầm lẫn' })).toBeVisible();
  await page.screenshot({ path: info.outputPath('offline-models.png'), fullPage: true });
  await page.getByRole('link', { name: 'Theo dõi', exact: true }).click();
  await page.getByRole('button', { name: 'Tạo phiên', exact: true }).click();
  await expect(page.getByTestId('session-status')).toHaveText('Sẵn sàng');
  const id = await page.getByTestId('session-id').innerText();
  await page.getByRole('button', { name: 'Bắt đầu', exact: true }).click();
  await expect.poll(async () => Number(await page.getByTestId('processed-count').innerText())).toBeGreaterThanOrEqual(3);
  await page.getByRole('button', { name: 'Tạm dừng', exact: true }).click();
  await expect(page.getByTestId('session-status')).toHaveText('Tạm dừng');
  await page.getByLabel('Tốc độ phát lại').selectOption('2');
  await page.getByRole('button', { name: 'Tiếp tục', exact: true }).click();
  await expect(page.getByTestId('session-status')).toHaveText('Đang chạy');
  await page.getByRole('button', { name: 'Kết thúc', exact: true }).click();
  await expect(page.getByTestId('session-status')).toHaveText('Đã kết thúc');
  await page.reload();
  await page.getByRole('button', { name: `Mở phiên ${id}`, exact: true }).click();
  const summary = await (await request.get(`/api/sessions/${id}/summary`)).json();
  await expect(page.getByTestId('summary-processed')).toHaveText(String(summary.processed));
  const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('link', { name: 'Xuất CSV', exact: true }).click()]);
  expect(await download.failure()).toBeNull();
  const target = info.outputPath('offline-session.csv');
  await download.saveAs(target);
  const csv = await readFile(target, 'utf8');
  expect(csv.replace(/^\uFEFF/, '').trim().split(/\r?\n/).length - 1).toBe(summary.processed);
  await page.screenshot({ path: info.outputPath('offline-history.png'), fullPage: true });
  expect(external).toEqual([]);
  await info.attach('offline-scope-and-browser', { body: JSON.stringify({ browser_version: browser.version(), channel: info.project.use.channel, external_attempts: external, scope: 'page.route only; APIRequestContext not intercepted; machine network unchanged', summary }, null, 2), contentType: 'application/json' });
});
