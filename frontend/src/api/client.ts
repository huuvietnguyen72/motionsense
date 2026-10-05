import type { ErrorDetail } from './types';

export class ApiFailure extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public details: ErrorDetail[] = [],
  ) {
    super(message);
    this.name = 'ApiFailure';
  }
}

export function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError';
}

function invalidResponse(status: number): ApiFailure {
  return new ApiFailure(status, 'INVALID_RESPONSE',
    'Phản hồi từ dịch vụ không hợp lệ. Hãy thử lại hoặc khởi động lại MotionSense.');
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (typeof init.body === 'string' && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }
  let response: Response;
  try {
    response = await fetch(`/api${path}`, { ...init, headers });
  } catch (error) {
    if (isAbortError(error) || init.signal?.aborted) throw error;
    throw new ApiFailure(0, 'NETWORK_ERROR',
      'Mất kết nối với dịch vụ tại máy. Kiểm tra MotionSense đang chạy rồi thử lại.');
  }
  let body: unknown;
  try {
    body = await response.json();
  } catch (error) {
    if (isAbortError(error) || init.signal?.aborted) throw error;
    throw invalidResponse(response.status);
  }
  if (!response.ok) {
    if (typeof body !== 'object' || body === null || !('error' in body)) throw invalidResponse(response.status);
    const failure = body.error;
    if (typeof failure !== 'object' || failure === null || !('code' in failure)
      || !('message' in failure) || typeof failure.code !== 'string' || typeof failure.message !== 'string') {
      throw invalidResponse(response.status);
    }
    const details = 'details' in failure ? failure.details : [];
    if (!Array.isArray(details) || !details.every(d => d && typeof d.message === 'string'
      && (d.row === null || typeof d.row === 'number')
      && (d.column === null || typeof d.column === 'string'))) throw invalidResponse(response.status);
    throw new ApiFailure(response.status, failure.code, failure.message, details);
  }
  return body as T;
}
