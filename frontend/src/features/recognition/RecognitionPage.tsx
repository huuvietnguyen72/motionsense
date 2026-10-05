import { useEffect, useRef, useState } from 'react';
import type React from 'react';
import { api, ApiFailure, isAbortError } from '../../api/client';
import type { DatasetInfo, ModelInfo, Page, Prediction, Sample } from '../../api/types';
import { StatusPanel } from '../../components/StatusPanel';
import { ProbabilityBars } from '../../components/ProbabilityBars';
import { SignalChart } from '../../components/SignalChart';

type SourceRow = Pick<Sample, 'sample_id' | 'subject_id' | 'split' | 'actual_label'>;

export function RecognitionPage(): React.JSX.Element {
  const [data, setData] = useState<DatasetInfo | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelId, setModelId] = useState('');
  const [subject, setSubject] = useState('');
  const [tab, setTab] = useState<'sample' | 'csv'>('sample');
  const [file, setFile] = useState<File | null>(null);
  const [offset, setOffset] = useState(0);
  const [samples, setSamples] = useState<Page<SourceRow> | null>(null);
  const [sampleId, setSampleId] = useState('');
  const [results, setResults] = useState<Prediction[]>([]);
  const [sourceResult, setSourceResult] = useState(false);
  const [active, setActive] = useState(0);
  const [resultOffset, setResultOffset] = useState(0);
  const [signal, setSignal] = useState<Sample | null>(null);
  const [error, setError] = useState<ApiFailure | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [signalError, setSignalError] = useState<string | null>(null);
  const [signalRetry, setSignalRetry] = useState(0);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [retry, setRetry] = useState(0);
  const operation = useRef<AbortController | null>(null);
  const sequence = useRef(0);
  const labels = Object.fromEntries(data?.activities.map(a => [a.label_id, a.name_vi]) ?? []);
  const prediction = results[active];
  const model = models.find(m => m.model_id === modelId);

  function clearResult() {
    sequence.current++;
    operation.current?.abort();
    setBusy(false); setResults([]); setError(null); setSignal(null); setSignalError(null);
    setActive(0); setResultOffset(0);
  }
  function changeTab(next: 'sample' | 'csv') {
    if (next === tab) return;
    clearResult(); setFile(null); setTab(next);
  }
  useEffect(() => {
    const abort = new AbortController();
    setLoading(true); setListError(null);
    void Promise.all([api<DatasetInfo>('/data', { signal: abort.signal }), api<{ items: ModelInfo[] }>('/models', { signal: abort.signal })])
      .then(([info, list]) => {
        if (abort.signal.aborted) return;
        setData(info);
        const ready = list.items.filter(m => m.status === 'ready' && m.dataset_id === info.dataset_id);
        setModels(ready);
        setModelId(current => ready.some(m => m.model_id === current) ? current : (ready.find(m => m.profile === 'full') ?? ready[0])?.model_id ?? '');
        const first = info.subjects.find(s => s.split === 'test') ?? info.subjects[0];
        setSubject(current => current || (first ? `${first.split}:${first.subject_id}` : ''));
      }).catch(e => { if (!abort.signal.aborted && !isAbortError(e)) setListError(e.message); })
      .finally(() => { if (!abort.signal.aborted) setLoading(false); });
    return () => abort.abort();
  }, [retry]);
  useEffect(() => {
    if (!subject || !data?.ready || tab !== 'sample') return;
    const abort = new AbortController();
    const [split, id] = subject.split(':');
    setSamples(null); setSampleId(''); setListError(null);
    void api<Page<SourceRow>>(`/data/samples?split=${split}&subject_id=${id}&offset=${offset}&limit=50`, { signal: abort.signal })
      .then(page => { if (!abort.signal.aborted) { setSamples(page); setSampleId(page.items[0]?.sample_id ?? ''); } })
      .catch(e => { if (!abort.signal.aborted && !isAbortError(e)) setListError(e.message); });
    return () => abort.abort();
  }, [subject, offset, data, tab, retry]);
  useEffect(() => () => { sequence.current++; operation.current?.abort(); }, []);
  useEffect(() => {
    setSignal(null); setSignalError(null);
    if (!sourceResult || !prediction?.sample_id) return;
    const abort = new AbortController();
    const id = prediction.sample_id;
    void api<Sample>(`/data/samples/${encodeURIComponent(id)}`, { signal: abort.signal })
      .then(sample => { if (!abort.signal.aborted && sample.sample_id === id) setSignal(sample); })
      .catch(e => { if (!abort.signal.aborted && !isAbortError(e)) setSignalError(e.message); });
    return () => abort.abort();
  }, [prediction, sourceResult, signalRetry]);

  async function recognize() {
    if (busy || !modelId || (tab === 'sample' ? !sampleId : !file)) return;
    clearResult();
    if (file && tab === 'csv' && file.size > 16 * 1024 * 1024) {
      setError(new ApiFailure(413, 'FILE_TOO_LARGE', 'Tệp vượt giới hạn 16 MiB. Hãy chọn tệp nhỏ hơn.')); return;
    }
    const abort = new AbortController(); operation.current = abort;
    const version = sequence.current;
    setBusy(true);
    try {
      let items: Prediction[];
      if (tab === 'sample') {
        items = [await api<Prediction>('/recognition/sample', { method: 'POST', signal: abort.signal, body: JSON.stringify({ model_id: modelId, sample_id: sampleId }) })];
      } else {
        const body = new FormData(); body.append('model_id', modelId); body.append('file', file!);
        items = (await api<{ items: Prediction[]; row_count: number }>('/recognition/csv', { method: 'POST', signal: abort.signal, body })).items;
      }
      if (!abort.signal.aborted && version === sequence.current) { setResults(items); setSourceResult(tab === 'sample'); }
    } catch (e) {
      if (!abort.signal.aborted && version === sequence.current && !isAbortError(e)) setError(e instanceof ApiFailure ? e : new ApiFailure(0, 'UNKNOWN', 'Không thể nhận diện. Hãy thử lại.'));
    } finally { if (!abort.signal.aborted && version === sequence.current) setBusy(false); }
  }

  return <>
    <div className="page-heading"><div><p className="eyebrow">MotionSense / Nhận diện</p><h1>Nhận diện</h1><p>Chọn đầu vào và xem xác suất của sáu hoạt động.</p></div><span className="badge">Random Forest · tại máy</span></div>
    {loading && <StatusPanel kind="loading" message="Đang tải dữ liệu và mô hình…" />}
    {listError && <StatusPanel kind="error" message={listError} onRetry={() => setRetry(n => n + 1)} />}
    {!loading && data && (!data.ready || !models.length) && <StatusPanel kind="setup" message="Dữ liệu hoặc mô hình chưa sẵn sàng. Hãy chạy Cai_dat.bat rồi thử lại." onRetry={() => setRetry(n => n + 1)} />}
    {data?.ready && models.length > 0 && <>
      <section className="card"><label htmlFor="recognition-model">Mô hình</label><select id="recognition-model" value={modelId} onChange={e => { clearResult(); setModelId(e.target.value); }}>
        {models.map(m => <option value={m.model_id} key={m.model_id}>{m.profile === 'full' ? 'Đầy đủ' : 'Rút gọn'} · {m.trees} cây · {m.model_id}</option>)}
      </select><p className="muted">{model?.feature_ids.length} đặc trưng · Seed {model?.seed} · Snapshot <span className="mono">{model?.dataset_id}</span></p>
      <div className="input-tabs" role="tablist" aria-label="Nguồn nhận diện">{(['sample', 'csv'] as const).map(t => <button key={t} id={`tab-${t}`} role="tab" aria-selected={tab === t} aria-controls={`panel-${t}`} tabIndex={tab === t ? 0 : -1} className={tab === t ? '' : 'secondary'} onClick={() => changeTab(t)} onKeyDown={e => {
        if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) { e.preventDefault(); const next = e.key === 'Home' ? 'sample' : e.key === 'End' ? 'csv' : tab === 'sample' ? 'csv' : 'sample'; changeTab(next); document.getElementById(`tab-${next}`)?.focus(); }
      }}>{t === 'sample' ? 'Mẫu có sẵn' : 'Tệp CSV'}</button>)}</div>
      <div role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`}>
        {tab === 'sample' ? <>
          <div className="recognition-input"><div><label htmlFor="recognition-subject">Người tham gia</label><select id="recognition-subject" value={subject} onChange={e => { clearResult(); setSubject(e.target.value); setOffset(0); }}>
            {data.subjects.map(s => <option key={`${s.split}:${s.subject_id}`} value={`${s.split}:${s.subject_id}`}>Người {s.subject_id} · {s.split === 'test' ? 'Kiểm tra' : 'Huấn luyện'} · {s.sample_count} mẫu</option>)}
          </select></div><div><label htmlFor="source-sample">Mẫu nguồn</label><select id="source-sample" value={sampleId} disabled={!samples?.items.length} onChange={e => { clearResult(); setSampleId(e.target.value); }}>
            {!samples && <option value="">Đang tải mẫu…</option>}{samples?.items.map(s => <option value={s.sample_id} key={s.sample_id}>{s.sample_id}</option>)}
          </select></div></div>
          <div className="pagination"><button className="secondary" disabled={!samples || offset === 0} onClick={() => { clearResult(); setOffset(n => n - 50); }}>Mẫu trước</button><span>{samples ? `${offset + 1}–${offset + samples.items.length} / ${samples.total} mẫu` : 'Đang tải mẫu…'}</span><button className="secondary" disabled={!samples || offset + 50 >= samples.total} onClick={() => { clearResult(); setOffset(n => n + 50); }}>Mẫu sau</button></div>
          {samples?.items.length === 0 && <StatusPanel kind="empty" message="Không có mẫu cho người tham gia này. Hãy chọn người khác." />}
          <p className="muted">Phát lại dữ liệu · {subject.startsWith('test:') ? 'Tập kiểm tra' : 'Tập huấn luyện'} · ID mẫu giữ nguyên dòng nguồn.</p>
        </> : <>
          <label htmlFor="recognition-file">Tệp đặc trưng</label><input id="recognition-file" type="file" accept=".csv,text/csv" onChange={e => { clearResult(); setFile(e.target.files?.[0] ?? null); }} />
          <p className="muted">Tối đa 16 MiB / 5.000 hàng. CSV đầy đủ dùng được với mô hình rút gọn; activity (nhãn thực tế) và subject_id chỉ là thông tin đối chiếu, không đưa vào mô hình.</p>
          <a className="button-link secondary" href={`/api/recognition/template?model_id=${encodeURIComponent(modelId)}`} download>Tải CSV mẫu</a>
        </>}
      </div><div className="session-controls"><button disabled={busy || !modelId || (tab === 'sample' ? !sampleId : !file)} onClick={() => void recognize()}>Nhận diện</button>{busy && <span role="status">Đang nhận diện…</span>}</div>
      </section>
      {error && <><StatusPanel kind="error" message={`${error.message}${error.details[0] ? ` ${error.details[0].message}` : ''} Sửa tệp hoặc chọn đầu vào khác rồi bấm Nhận diện.`} />{error.details.length > 0 && <div className="card table-scroll"><table aria-label="Lỗi CSV"><thead><tr><th scope="col">Dòng</th><th scope="col">Cột</th><th scope="col">Nguyên nhân</th></tr></thead><tbody>{error.details.map((d, i) => <tr key={i}><td>{d.row ?? '—'}</td><td>{d.column ?? '—'}</td><td>{d.message}</td></tr>)}</tbody></table></div>}</>}
      {prediction && <section data-testid="recognition-results">
        <div className="monitor-grid"><section className="card prediction-card" data-testid="recognition-prediction"><h2>Hoạt động dự đoán</h2><strong className="activity-name">{labels[prediction.predicted_label]}</strong><p className="mono sample-identity" data-testid="recognition-model">{prediction.model_id}</p><p>{sourceResult ? prediction.sample_id : `Tệp đặc trưng · dòng kết quả ${active + 1}`}</p><h3>Xác suất mô hình</h3><ProbabilityBars probabilities={prediction.probabilities} labels={labels} /></section>
          {sourceResult ? <section className="card"><h2>Tín hiệu mẫu nguồn</h2>{signalError ? <StatusPanel kind="error" message={`Không thể tải tín hiệu mẫu ${prediction.sample_id}. ${signalError} Bấm Thử lại để tải lại tín hiệu của mẫu này.`} onRetry={() => { setSignal(null); setSignalError(null); setSignalRetry(n => n + 1); }} /> : <SignalChart signal={signal?.sample_id === prediction.sample_id ? signal.signal : null} sampleId={prediction.sample_id} />}</section> : <section className="card"><h2>Đầu vào đặc trưng</h2><p>Tệp CSV không có tín hiệu thô để vẽ biểu đồ. Xác suất là đầu ra mô hình, không phải bảo đảm dự đoán đúng.</p></section>}
        </div>
        <details className="card"><summary>Đối chiếu nhãn thực tế</summary><p data-testid="recognition-actual">{prediction.actual_label === null ? 'Không có nhãn thực tế.' : `Nhãn thực tế: ${labels[prediction.actual_label]} · ${prediction.actual_label === prediction.predicted_label ? 'Đúng' : 'Sai'}`}</p></details>
        <section className="card"><h2>Kết quả nhận diện · {results.length} mẫu</h2><div className="table-scroll"><table aria-label="Kết quả nhận diện"><thead><tr><th scope="col">Mẫu / dòng</th><th scope="col">Hoạt động dự đoán</th><th scope="col">Chi tiết</th></tr></thead><tbody>{results.slice(resultOffset, resultOffset + 50).map((p, i) => <tr key={resultOffset + i}><th scope="row">{sourceResult ? p.sample_id : resultOffset + i + 1}</th><td>{labels[p.predicted_label]}</td><td><button className="secondary" aria-label={`Xem kết quả ${resultOffset + i + 1}`} aria-pressed={active === resultOffset + i} onClick={() => setActive(resultOffset + i)}>Xem xác suất</button></td></tr>)}</tbody></table></div>
          {results.length > 50 && <div className="pagination"><button className="secondary" disabled={resultOffset === 0} onClick={() => { setResultOffset(n => n - 50); setActive(resultOffset - 50); }}>Kết quả trước</button><span>{resultOffset + 1}–{Math.min(resultOffset + 50, results.length)} / {results.length}</span><button className="secondary" disabled={resultOffset + 50 >= results.length} onClick={() => { setResultOffset(n => n + 50); setActive(resultOffset + 50); }}>Kết quả sau</button></div>}
        </section>
      </section>}
    </>}
  </>;
}
