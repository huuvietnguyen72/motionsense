import { useEffect, useState } from 'react';
import type React from 'react';
import { api, isAbortError } from '../../api/client';
import type { DatasetInfo, ModelInfo, Sample, Session, Split } from '../../api/types';
import { StatusPanel } from '../../components/StatusPanel';
import { ProbabilityBars } from '../../components/ProbabilityBars';
import { SignalChart } from '../../components/SignalChart';
import { SessionHistory, sessionStatuses } from './SessionHistory';
import { sessionStorageKey, useSession } from './useSession';

const percent = (n: number) => new Intl.NumberFormat('vi-VN', { style: 'percent', maximumFractionDigits: 1 }).format(n);

export function MonitorPage(): React.JSX.Element {
  const [initialId] = useState(() => localStorage.getItem(sessionStorageKey));
  const state = useSession(initialId);
  const { session, rows, summary, busy, error } = state;
  const [data, setData] = useState<DatasetInfo | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [sourceError, setSourceError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [attempt, setAttempt] = useState(0);
  const [person, setPerson] = useState('');
  const [model, setModel] = useState('');
  const [sample, setSample] = useState<Sample | null>(null);
  const [signalError, setSignalError] = useState<string | null>(null);
  const [signalAttempt, setSignalAttempt] = useState(0);
  useEffect(() => {
    const abort = new AbortController();
    let alive = true;
    setLoading(true); setSourceError(null);
    async function load() {
      try {
        const info = await api<DatasetInfo>('/data', { signal: abort.signal });
        if (!alive) return;
        setData(info);
        const subject = info.subjects.find(s => s.split === 'test') ?? info.subjects[0];
        setPerson(subject ? `${subject.split}:${subject.subject_id}` : '');
        const result = await api<{ items: ModelInfo[] }>('/models', { signal: abort.signal });
        if (!alive) return;
        const ready = result.items.filter(m => m.status === 'ready' && m.dataset_id === info.dataset_id);
        setModels(ready);
        setModel((ready.find(m => m.profile === 'full') ?? ready[0])?.model_id ?? '');
      } catch (e) {
        if (alive && !isAbortError(e)) setSourceError(e instanceof Error ? e.message : 'Không thể tải nguồn. Hãy thử lại.');
      } finally { if (alive) setLoading(false); }
    }
    void load();
    return () => { alive = false; abort.abort(); };
  }, [attempt]);
  const prediction = session?.last_prediction;
  const sampleId = prediction?.sample_id ?? null;
  const datasetId = session?.dataset_id ?? null;
  useEffect(() => {
    const abort = new AbortController();
    let alive = true;
    setSample(null); setSignalError(null);
    if (sampleId) void api<Sample>(`/data/samples/${encodeURIComponent(sampleId)}`, { signal: abort.signal }).then(value => {
      if (!alive) return;
      if (value.sample_id !== sampleId || value.dataset_id !== datasetId) {
        setSignalError('Tín hiệu không thuộc dữ liệu của phiên. Hãy kiểm tra nguồn hoặc chạy Cai_dat.bat.');
      } else setSample(value);
    }).catch(e => { if (alive && !isAbortError(e)) setSignalError(e instanceof Error ? e.message : 'Không thể tải tín hiệu.'); });
    return () => { alive = false; abort.abort(); };
  }, [sampleId, datasetId, signalAttempt]);
  const labels = Object.fromEntries(data?.activities.map(a => [a.label_id, a.name_vi]) ?? []);
  const label = (id: number) => labels[id] ?? `Nhãn ${id}`;
  const canCreate = !loading && !sourceError && data?.ready && !!model && !!person && !busy;
  const controlDisabled = busy || !!error;
  const create = () => {
    if (!canCreate || !data?.dataset_id) return;
    const [split, subject] = person.split(':');
    state.create({ dataset_id: data.dataset_id, split: split as Split, subject_id: Number(subject), model_id: model });
  };
  const signal = sample?.sample_id === sampleId && sample.dataset_id === datasetId ? sample.signal : null;
  return <>
    <div className="page-heading"><div><p className="eyebrow">MotionSense / Phát lại dữ liệu</p><h1>Theo dõi</h1><p>Quan sát từng cửa sổ cảm biến và kết quả từ mô hình đã lưu.</p></div><span className="badge">Nguồn UCI HAR · tại máy</span></div>
    <section className="card monitor-selection" aria-label="Tạo phiên phát lại">
      <div><label htmlFor="monitor-person">Người tham gia</label><select id="monitor-person" value={person} onChange={e => setPerson(e.target.value)} disabled={loading || !data?.ready}>
        {!person && <option value="">Chưa có dữ liệu</option>}{data?.subjects.map(s => <option key={`${s.split}:${s.subject_id}`} value={`${s.split}:${s.subject_id}`}>Người {s.subject_id} · {s.split === 'test' ? 'Kiểm tra' : 'Huấn luyện'} ({s.split}) · {s.sample_count} mẫu</option>)}
      </select></div>
      <div><label htmlFor="monitor-model">Mô hình</label><select id="monitor-model" value={model} onChange={e => setModel(e.target.value)} disabled={loading || models.length === 0}>
        {!model && <option value="">Chưa có mô hình sẵn sàng</option>}{models.map(m => <option key={m.model_id} value={m.model_id}>{m.profile === 'full' ? 'Đầy đủ' : 'Rút gọn'} · {m.trees} cây · {m.model_id}</option>)}
      </select></div>
      <button disabled={!canCreate} onClick={create}>{session?.status === 'finished' ? 'Tạo phiên mới' : 'Tạo phiên'}</button>
      <p>Chọn người và mô hình cho phiên mới. Phiên đang mở giữ nguyên nguồn và phiên bản đã ghim.</p>
    </section>
    {loading && <StatusPanel kind="loading" message="Đang tải nguồn phát lại và mô hình tại máy." />}
    {sourceError && <StatusPanel kind="error" message={sourceError} onRetry={() => setAttempt(n => n + 1)} />}
    {!loading && data && !data.ready && <StatusPanel kind="setup" message="Dữ liệu chưa sẵn sàng. Hãy chạy Cai_dat.bat rồi tải lại nguồn. Lịch sử phiên vẫn xem và xuất được." onRetry={() => setAttempt(n => n + 1)} />}
    {!loading && !sourceError && data?.ready && models.length === 0 && <StatusPanel kind="setup" message="Chưa có mô hình sẵn sàng cho nguồn này. Hãy chạy Cai_dat.bat rồi tải lại nguồn; vẫn có thể mở lịch sử phiên." onRetry={() => setAttempt(n => n + 1)} />}
    {error && <div className="session-error"><StatusPanel kind="error" message={error} /><button disabled={busy} onClick={state.reconnect}>Kết nối lại</button></div>}
    {busy && <p className="session-sync" role="status">Đang đồng bộ phiên với dịch vụ tại máy…</p>}
    {!session && !busy && !error && <StatusPanel kind="empty" message="Chưa mở phiên. Chọn người và mô hình, sau đó bấm Tạo phiên; hoặc mở một phiên đã lưu bên dưới." />}
    {session && <>
      <div className="instrument-strip" aria-label="Trạng thái phiên"><div><span>Trạng thái</span><strong data-testid="session-status">{sessionStatuses[session.status]}</strong></div><div><span>Mẫu đã xử lý</span><strong><span data-testid="processed-count">{session.cursor}</span> / {session.total}</strong></div><div><span>Phân vùng đã ghim</span><strong>Người {session.subject_id} · {session.split}</strong></div><div><span>Nguồn</span><strong>Phát lại UCI HAR</strong></div></div>
      <div className="monitor-grid">
        <section className="card prediction-card" data-testid="prediction-card" aria-labelledby="prediction-heading"><p className="eyebrow">Kết quả từ mô hình</p><h2 id="prediction-heading">Hoạt động dự đoán</h2><strong className="activity-name" data-testid="prediction-label">{prediction ? label(prediction.predicted_label) : 'Chưa có kết quả'}</strong>
          <p>{prediction ? `Xác suất lớp dự đoán: ${percent(prediction.probabilities.find(p => p.label_id === prediction.predicted_label)?.value ?? 0)}` : 'Bắt đầu phiên để xử lý mẫu đầu tiên.'}</p>
          <p className="mono sample-identity" data-testid="prediction-sample">{sampleId ?? 'Chưa có mẫu'}</p>
          <h3>Xác suất mô hình</h3><ProbabilityBars probabilities={prediction?.probabilities ?? []} labels={labels} />
          <p className="muted">Xác suất không phải bảo đảm dự đoán đúng.</p>
        </section>
        <section className="card signal-card" aria-labelledby="signal-heading"><div className="section-heading"><h2 id="signal-heading">Gia tốc tổng X / Y / Z</h2><span>50 Hz · g</span></div><p>Thời gian tương đối trong cửa sổ, không phải bản ghi liên tục.</p>
          {signalError ? <StatusPanel kind="error" message={signalError} onRetry={() => setSignalAttempt(n => n + 1)} /> : <SignalChart signal={signal} sampleId={sampleId} />}
          <div className="session-controls">
            {session.status === 'ready' && <button disabled={controlDisabled} onClick={() => state.control('start')}>Bắt đầu</button>}
            {session.status === 'running' && <button disabled={controlDisabled} onClick={() => state.control('pause')}>Tạm dừng</button>}
            {session.status === 'paused' && <button disabled={controlDisabled} onClick={() => state.control('resume')}>Tiếp tục</button>}
            {session.status !== 'finished' && <button className="secondary" disabled={controlDisabled} onClick={() => state.control('finish')}>Kết thúc</button>}
            <div><label htmlFor="monitor-speed">Tốc độ phát lại</label><select id="monitor-speed" value={session.speed} disabled={controlDisabled || session.status === 'finished'} onChange={e => state.control('speed', Number(e.target.value) as Session['speed'])}><option value="0.5">0,5×</option><option value="1">1×</option><option value="2">2×</option></select></div>
          </div><p className="muted">1× = một mẫu mỗi giây phát lại. Máy chủ quyết định vị trí và nhịp xử lý.</p>
        </section>
      </div>
      <section className="card pinned-source" aria-label="Định danh phiên"><div><span>Phiên tại máy</span><strong className="mono" data-testid="session-id">{session.session_id}</strong></div><div><span>Mô hình đã ghim</span><strong className="mono" data-testid="pinned-model">{session.model_id}</strong></div><div><span>Snapshot dữ liệu</span><strong className="mono">{session.dataset_id}</strong></div></section>
      <section className="card" aria-labelledby="summary-heading"><div className="section-heading"><h2 id="summary-heading">{session.status === 'finished' ? 'Tổng kết phiên' : 'Phân bố hoạt động'}</h2><a className="export-link" href={`/api/sessions/${session.session_id}/export`}>Xuất CSV</a></div>
        {session.status === 'finished' && <p>{session.finish_reason === 'complete' ? 'Đã xử lý hết mẫu của người tham gia.' : 'Đã kết thúc theo yêu cầu; giữ toàn bộ mẫu đã xử lý.'}</p>}
        {summary ? <><p>Đã lưu <strong data-testid="summary-processed">{summary.processed}</strong> mẫu. Phân bố theo nhãn dự đoán và tỷ lệ mẫu; không suy ra thời lượng hoạt động.</p><div className="distribution">{summary.counts.map(c => <div key={c.label_id} data-testid={`distribution-${c.label_id}`}><span>{label(c.label_id)}</span><strong>{c.count} mẫu</strong><span>{summary.processed ? percent(c.count / summary.processed) : '—'}</span><div className="distribution-track"><span style={{ width: `${summary.processed ? c.count / summary.processed * 100 : 0}%` }} /></div></div>)}</div></> : <p role="status">Đang tải tổng kết từ nhật ký đã lưu…</p>}
      </section>
      <section className="card" aria-labelledby="rows-heading"><h2 id="rows-heading">Lịch sử nhận diện</h2><p>Thứ tự mẫu và thời điểm xử lý tại máy, lấy từ nhật ký phiên.</p>
        {rows.length === 0 ? <p role="status">Chưa có mẫu đã xử lý.</p> : <div className="table-scroll"><table aria-label="Lịch sử nhận diện"><thead><tr><th scope="col">Thứ tự</th><th scope="col">Mẫu nguồn</th><th scope="col">Dự đoán</th><th scope="col">Xác suất lớp</th><th scope="col">Xử lý tại máy</th></tr></thead><tbody>{rows.map(row => <tr key={row.ordinal}><th scope="row">{row.ordinal + 1}</th><td className="mono">{row.sample_id}</td><td>{label(row.prediction.predicted_label)}</td><td>{percent(row.prediction.probabilities.find(p => p.label_id === row.prediction.predicted_label)?.value ?? 0)}</td><td>{new Date(row.processed_at).toLocaleString('vi-VN')}</td></tr>)}</tbody></table></div>}
        <div className="pagination"><button className="secondary" disabled={busy || state.offset === 0} onClick={() => state.loadPage(state.offset - 50)}>Mẫu trước</button><span>{state.total ? `${state.offset + 1}–${Math.min(state.offset + 50, state.total)}` : '0'} / {state.total} mẫu</span><button className="secondary" disabled={busy || state.offset + 50 >= state.total} onClick={() => state.loadPage(state.offset + 50)}>Mẫu kế tiếp</button></div>
        <details className="actual-comparison"><summary>Đối chiếu nhãn thực tế</summary><p>Nhãn nguồn chỉ dùng đối chiếu, không đưa vào đầu vào hoặc thẻ dự đoán.</p>{summary && <p>{summary.correct} đúng / {summary.labeled} mẫu có nhãn · Accuracy phiên: {summary.session_accuracy === null ? 'Chưa có nhãn' : percent(summary.session_accuracy)}</p>}
          <div className="table-scroll"><table aria-label="Đối chiếu nhãn thực tế"><thead><tr><th scope="col">Mẫu</th><th scope="col">Dự đoán</th><th scope="col">Nhãn thực tế</th><th scope="col">Đối chiếu</th></tr></thead><tbody>{rows.map(row => <tr key={row.ordinal}><th scope="row" className="mono">{row.sample_id}</th><td>{label(row.prediction.predicted_label)}</td><td>{row.prediction.actual_label === null ? 'Không có nhãn' : label(row.prediction.actual_label)}</td><td>{row.prediction.actual_label === null ? '—' : row.prediction.actual_label === row.prediction.predicted_label ? 'Đúng' : 'Sai'}</td></tr>)}</tbody></table></div>
        </details>
      </section>
    </>}
    <SessionHistory onOpen={state.open} busy={busy} refreshKey={`${session?.session_id ?? ''}:${session?.status ?? ''}`} />
  </>;
}
