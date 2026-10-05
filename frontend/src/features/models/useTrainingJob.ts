import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { api, ApiFailure, isAbortError } from '../../api/client';
import type { Job } from '../../api/types';

const key = 'motionsense.training-job';
// A submission belongs to this document, not a ModelsPage mount. Keep its lock
// and acknowledged identity alive while routes dispose/recreate their hooks.
let submission: { jobId: string | null; acknowledged: Job | null; pending: boolean; error: string | null } = {
  jobId: localStorage.getItem(key), acknowledged: null, pending: false, error: null,
};
const listeners = new Set<() => void>();
function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}
function getSubmission() { return submission; }
function publish(next: typeof submission) {
  submission = next;
  listeners.forEach(listener => listener());
}

export function useTrainingJob(onSuccess: () => void) {
  const shared = useSyncExternalStore(subscribe, getSubmission);
  const { jobId } = shared;
  const [polled, setPolled] = useState<Job | null>(null);
  const job = polled?.job_id === jobId ? polled : shared.acknowledged?.job_id === jobId ? shared.acknowledged : null;
  const [error, setError] = useState<string | null>(null);
  const [missing, setMissing] = useState(false);
  const [retry, setRetry] = useState(0);
  const notified = useRef<string | null>(null);
  useEffect(() => {
    if (!jobId) return;
    const abort = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    setError(null); setMissing(false);
    async function poll() {
      try {
        const current = await api<Job>(`/jobs/${encodeURIComponent(jobId!)}`, { signal: abort.signal });
        if (abort.signal.aborted) return;
        setPolled(current);
        if (current.status === 'queued' || current.status === 'running') timer = setTimeout(() => void poll(), 1000);
        else if (current.status === 'succeeded' && notified.current !== current.job_id) { notified.current = current.job_id; onSuccess(); }
      } catch (e) { if (!abort.signal.aborted && !isAbortError(e)) { setError(e instanceof Error ? e.message : 'Không thể đọc tác vụ. Hãy kết nối lại.'); setMissing(e instanceof ApiFailure && e.status === 404); } }
    }
    void poll();
    return () => { abort.abort(); clearTimeout(timer); };
  }, [jobId, retry, onSuccess]);
  async function submit(profile: 'full' | 'reduced', trees: number) {
    if (submission.pending || (jobId && (!job || job.status === 'queued' || job.status === 'running'))) return;
    setError(null);
    publish({ ...submission, pending: true, error: null });
    try {
      // Do not abort this POST on route disposal. Its acknowledgement updates
      // storage and every current subscriber, including a newly mounted page.
      const next = await api<Job>('/models/train', { method: 'POST', body: JSON.stringify({ profile, trees }) });
      localStorage.setItem(key, next.job_id);
      publish({ jobId: next.job_id, acknowledged: next, pending: false, error: null });
    } catch (e) { publish({ ...submission, pending: false, error: e instanceof Error ? e.message : 'Không thể bắt đầu huấn luyện. Hãy thử lại.' }); }
  }
  function dismissMissing() {
    if (!missing) return;
    localStorage.removeItem(key); publish({ jobId: null, acknowledged: null, pending: false, error: null }); setPolled(null); setError(null); setMissing(false);
  }
  return { job, jobId, error: error ?? shared.error, missing, dismissMissing, submitting: shared.pending, busy: shared.pending || Boolean(jobId && (!job || job.status === 'queued' || job.status === 'running')), submit, reconnect: () => { publish({ ...submission, error: null }); setRetry(n => n + 1); } };
}
