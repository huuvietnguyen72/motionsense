import { useEffect, useState } from 'react';
import { api, isAbortError } from '../../api/client';
import type { ModelReport } from '../../api/types';

export function useModelReport(modelId: string, retry: number) {
  const [snapshot, setSnapshot] = useState<{ id: string; report?: ModelReport; error?: string } | null>(null);
  useEffect(() => {
    if (!modelId) return;
    const abort = new AbortController();
    setSnapshot(null);
    void api<ModelReport>(`/models/${encodeURIComponent(modelId)}/report`, { signal: abort.signal })
      .then(report => { if (!abort.signal.aborted && report.model_id === modelId) setSnapshot({ id: modelId, report }); })
      .catch(e => { if (!abort.signal.aborted && !isAbortError(e)) setSnapshot({ id: modelId, error: e.message }); });
    return () => abort.abort();
  }, [modelId, retry]);
  return snapshot?.id === modelId ? snapshot : null;
}
