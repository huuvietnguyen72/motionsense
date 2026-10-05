import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  workers: 1,
  retries: 0,
  reporter: 'list',
  outputDir: process.env.MOTIONSENSE_EVIDENCE_DIR ?? './test-results',
  use: {
    baseURL: process.env.MOTIONSENSE_BASE_URL ?? process.env.MOTIONSENSE_TEST_URL ?? 'http://127.0.0.1:8765',
    channel: process.env.MOTIONSENSE_BROWSER_CHANNEL,
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
});
