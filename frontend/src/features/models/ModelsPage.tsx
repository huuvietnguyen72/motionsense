import { useCallback, useEffect, useState } from 'react';
import type React from 'react';
import { api, isAbortError } from '../../api/client';
import type { DatasetInfo, ModelInfo } from '../../api/types';
import { StatusPanel } from '../../components/StatusPanel';
import { ConfusionMatrix } from '../../components/ConfusionMatrix';
import { useModelReport } from './useModelReport';
import { useTrainingJob } from './useTrainingJob';

const percent = (value: number) => new Intl.NumberFormat('vi-VN', { style: 'percent', maximumFractionDigits: 2 }).format(value);
const selectionKey = 'motionsense.analysis-model';
const jobNames = { queued: 'Đang chờ', running: 'Đang huấn luyện', succeeded: 'Hoàn tất', failed: 'Thất bại', interrupted: 'Gián đoạn' };
const stages: Record<string, string> = { queued: 'Đang chờ', validating: 'Kiểm tra dữ liệu', selecting_features: 'Chọn đặc trưng trên tập huấn luyện', fitting: 'Huấn luyện Random Forest', evaluating: 'Đánh giá mô hình', saving: 'Lưu phiên bản mô hình' };

export function ModelsPage(): React.JSX.Element {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [data, setData] = useState<DatasetInfo | null>(null);
  const [modelId, setModelId] = useState(() => localStorage.getItem(selectionKey) ?? '');
  const [fullId, setFullId] = useState('');
  const [reducedId, setReducedId] = useState('');
  const [all, setAll] = useState(false);
  const [profile, setProfile] = useState<'full' | 'reduced'>('full');
  const [trees, setTrees] = useState(50);
  const [retry, setRetry] = useState(0);
  const [refresh, setRefresh] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const onSuccess = useCallback(() => setRefresh(n => n + 1), []);
  const training = useTrainingJob(onSuccess);
  useEffect(() => {
    const abort = new AbortController();
    setLoading(true); setError(null);
    void Promise.all([api<{ items: ModelInfo[] }>('/models', { signal: abort.signal }), api<DatasetInfo>('/data', { signal: abort.signal })])
      .then(([list, info]) => {
        if (abort.signal.aborted) return;
        setModels(list.items); setData(info);
        const ready = list.items.filter(m => m.status === 'ready');
        const preferred = ready.find(m => m.profile === 'full' && m.dataset_id === info.dataset_id) ?? ready[0];
        setModelId(current => {
          const next = ready.some(m => m.model_id === current) ? current : preferred?.model_id ?? '';
          if (next) localStorage.setItem(selectionKey, next);
          return next;
        });
        setFullId(current => ready.some(m => m.model_id === current) ? current : ready.find(m => m.profile === 'full' && m.dataset_id === info.dataset_id)?.model_id ?? '');
        setReducedId(current => ready.some(m => m.model_id === current) ? current : ready.find(m => m.profile === 'reduced' && m.dataset_id === info.dataset_id)?.model_id ?? '');
      }).catch(e => { if (!abort.signal.aborted && !isAbortError(e)) setError(e.message); })
      .finally(() => { if (!abort.signal.aborted) setLoading(false); });
    return () => abort.abort();
  }, [retry, refresh]);
  const ready = models.filter(m => m.status === 'ready');
  const model = ready.find(m => m.model_id === modelId);
  const full = ready.find(m => m.model_id === fullId && m.profile === 'full');
  const reduced = ready.find(m => m.model_id === reducedId && m.profile === 'reduced');
  const same = Boolean(full && reduced && full.dataset_id === reduced.dataset_id);
  const snapshot = useModelReport(model?.model_id ?? '', retry);
  const fullSnapshot = useModelReport(same ? fullId : '', retry);
  const reducedSnapshot = useModelReport(same ? reducedId : '', retry);
  const report = snapshot?.report;
  const labels = Object.fromEntries(data?.activities.map(a => [a.label_id, a.name_vi]) ?? []);
  const features = [...(report?.feature_importance ?? [])].sort((a, b) => b.value - a.value || a.feature_id.localeCompare(b.feature_id)).slice(0, all ? undefined : model?.profile === 'reduced' ? 5 : 15);
  function selectModel(id: string) { setAll(false); setModelId(id); localStorage.setItem(selectionKey, id); }
  const options = (items: ModelInfo[]) => items.map(m => <option key={m.model_id} value={m.model_id}>{m.profile === 'full' ? 'Đầy đủ' : 'Rút gọn'} · {m.trees} cây · {m.model_id}</option>);

  return <>
    <div className="page-heading"><div><p className="eyebrow">MotionSense / Phân tích mô hình</p><h1>Phân tích mô hình</h1><p>Đánh giá trên người tham gia được giữ riêng khỏi tập huấn luyện.</p></div><span className="badge badge-ready">Phân chia theo người</span></div>
    {error && <StatusPanel kind="error" message={error} onRetry={() => setRetry(n => n + 1)} />}
    {loading && <StatusPanel kind="loading" message="Đang tải danh sách mô hình…" />}
    {!loading && !ready.length && <StatusPanel kind="setup" message="Chưa có mô hình sẵn sàng. Hãy thiết lập dữ liệu bằng Cai_dat.bat hoặc huấn luyện bên dưới." onRetry={() => setRetry(n => n + 1)} />}
    {ready.length > 0 && <section className="card"><label htmlFor="analysis-model">Mô hình phân tích</label><select id="analysis-model" value={modelId} onChange={e => selectModel(e.target.value)}>{options(ready)}</select>
      {model && <p className="muted">{model.feature_ids.length} đặc trưng · {model.trees} cây · Seed {model.seed} · {new Date(model.created_at).toLocaleString('vi-VN')}<br />Snapshot: <span className="mono">{model.dataset_id}</span></p>}
      {models.some(m => m.status === 'unavailable') && <p className="muted">{models.filter(m => m.status === 'unavailable').length} phiên bản không sẵn sàng; hãy kiểm tra dữ liệu / tệp mô hình.</p>}
    </section>}
    {snapshot?.error && <StatusPanel kind="error" message={snapshot.error} onRetry={() => setRetry(n => n + 1)} />}
    {model && !snapshot && <StatusPanel kind="loading" message="Đang tải báo cáo mô hình…" />}
    {report && model && <>
      <div className="instrument-strip report-identity"><div><span>Phiên bản báo cáo</span><strong className="mono" data-testid="report-model">{report.model_id}</strong></div><div><span>Cách chia</span><strong>UCI HAR · theo người</strong></div></div>
      <div className="metrics">{[['Accuracy kiểm tra', percent(report.accuracy), 'model-accuracy'], ['Macro F1', percent(report.macro_f1), 'model-f1'], ['Mẫu kiểm tra', new Intl.NumberFormat('vi-VN').format(report.test_count), 'model-test-count'], ['Số cây', String(model.trees), 'model-trees']].map(([name, value, id]) => <div className="metric" key={id}><span>{name}</span><strong data-testid={id}>{value}</strong></div>)}</div>
      <section className="card"><h2>Ma trận nhầm lẫn</h2><p data-testid="heldout-subjects">{model.dataset_id === data?.dataset_id ? `Người kiểm tra: ${data.subjects.filter(s => s.split === 'test').map(s => s.subject_id).join(', ')}` : 'Snapshot cũ: danh sách người kiểm tra không có trong dữ liệu hiện tại.'}</p><ConfusionMatrix labels={report.labels} matrix={report.confusion_matrix} labelNames={labels} /></section>
      <section className="card"><h2>Chỉ số theo lớp</h2><div className="table-scroll"><table aria-label="Chỉ số theo lớp"><thead><tr><th scope="col">Hoạt động</th><th scope="col">Precision</th><th scope="col">Recall</th><th scope="col">F1</th><th scope="col">Số mẫu</th></tr></thead><tbody>{report.per_class.map(c => <tr key={c.label_id}><th scope="row">{labels[c.label_id] ?? `Nhãn ${c.label_id}`}</th><td>{percent(c.precision)}</td><td>{percent(c.recall)}</td><td>{percent(c.f1)}</td><td>{c.support}</td></tr>)}</tbody></table></div></section>
      <section className="card"><div className="section-heading"><h2>Độ quan trọng đặc trưng</h2>{report.feature_importance.length > 15 && <button className="secondary" onClick={() => setAll(v => !v)}>{all ? 'Thu gọn đặc trưng' : 'Xem tất cả đặc trưng'}</button>}</div><p className="muted">Xếp hạng toàn mô hình; không giải thích quan hệ nhân quả hay từng dự đoán riêng lẻ.</p><div className="table-scroll feature-table"><table aria-label="Độ quan trọng đặc trưng"><thead><tr><th scope="col">ID</th><th scope="col">Mô tả nguồn</th><th scope="col">Độ quan trọng</th></tr></thead><tbody>{features.map(f => <tr key={f.feature_id}><th scope="row" className="mono">{f.feature_id}</th><td>{f.name}</td><td><meter min={0} max={1} value={f.value} aria-label={`Độ quan trọng ${f.feature_id}`} /> <span>{percent(f.value)}</span></td></tr>)}</tbody></table></div></section>
      <section className="card"><h2>Ước lượng trên tập huấn luyện</h2><div className="split-stats"><div><span>Accuracy huấn luyện</span><strong data-testid="train-accuracy">{percent(report.train_accuracy)}</strong></div><div><span>OOB</span><strong data-testid="oob-score">{report.oob_score === null ? 'Không có ước lượng OOB' : percent(report.oob_score)}</strong></div></div><p>Các giá trị này không thay thế đánh giá trên tập kiểm tra.</p>{report.oob_warning && <p role="note">{report.oob_warning}</p>}</section>
    </>}
    <section className="card"><h2>So sánh đầy đủ / rút gọn</h2><p>Cùng snapshot và phân chia UCI theo người. Huấn luyện lại không tạo tập đánh giá độc lập mới.</p><div className="recognition-input"><div><label htmlFor="compare-full">Mô hình đầy đủ để so sánh</label><select id="compare-full" value={fullId} onChange={e => setFullId(e.target.value)}><option value="">Chọn mô hình</option>{options(ready.filter(m => m.profile === 'full'))}</select></div><div><label htmlFor="compare-reduced">Mô hình rút gọn để so sánh</label><select id="compare-reduced" value={reducedId} onChange={e => setReducedId(e.target.value)}><option value="">Chọn mô hình</option>{options(ready.filter(m => m.profile === 'reduced'))}</select></div></div>
      {full && reduced && !same && <StatusPanel kind="error" message="Chỉ so sánh hai mô hình cùng snapshot dữ liệu. Hãy chọn lại phiên bản." />}
      {(!full || !reduced) && <p className="muted">Chọn một mô hình đầy đủ và một mô hình rút gọn để so sánh.</p>}
      {same && (fullSnapshot?.error || reducedSnapshot?.error) && <StatusPanel kind="error" message={fullSnapshot?.error ?? reducedSnapshot!.error!} onRetry={() => setRetry(n => n + 1)} />}
      {same && !fullSnapshot?.error && !reducedSnapshot?.error && (!fullSnapshot?.report || !reducedSnapshot?.report) && <p role="status">Đang tải báo cáo so sánh…</p>}
      {same && fullSnapshot?.report && reducedSnapshot?.report && <><p className="mono dataset-identity">{full!.dataset_id}</p><div className="table-scroll"><table aria-label="So sánh mô hình" className="comparison-table"><thead><tr><th scope="col">Phiên bản / cấu hình</th><th scope="col">Cây / seed</th><th scope="col">Đặc trưng</th><th scope="col">Accuracy kiểm tra</th><th scope="col">Macro F1</th><th scope="col">Mẫu kiểm tra</th></tr></thead><tbody>{[[full!, fullSnapshot.report], [reduced!, reducedSnapshot.report]].map(([m, r]) => {
        const info = m as ModelInfo; const values = r as NonNullable<typeof report>;
        return <tr key={info.model_id}><th scope="row"><span>{info.profile === 'full' ? 'Đầy đủ' : 'Rút gọn'}</span><span className="history-time mono">{info.model_id}</span></th><td>{info.trees} / {info.seed}</td><td>{info.feature_ids.length}</td><td>{percent(values.accuracy)}</td><td>{percent(values.macro_f1)}</td><td>{values.test_count}</td></tr>;
      })}</tbody></table></div></>}
    </section>
    <section className="card"><h2>Huấn luyện phiên bản mới</h2><p>Chọn đặc trưng chỉ trên tập huấn luyện. Mô hình mới được lưu thành phiên bản riêng.</p><form onSubmit={e => { e.preventDefault(); void training.submit(profile, trees); }}><div className="recognition-input"><div><label htmlFor="train-profile">Cấu hình huấn luyện</label><select id="train-profile" value={profile} disabled={training.busy} onChange={e => setProfile(e.target.value as 'full' | 'reduced')}><option value="full">Đầy đủ · 561 đặc trưng</option><option value="reduced">Rút gọn · 5 đặc trưng</option></select></div><div><label htmlFor="train-trees">Số cây</label><select id="train-trees" value={trees} disabled={training.busy} onChange={e => setTrees(Number(e.target.value))}>{[25, 50, 100, 150].map(n => <option key={n}>{n}</option>)}</select></div></div><div className="session-controls"><button disabled={training.busy || !data?.ready}>{training.job?.status === 'failed' || training.job?.status === 'interrupted' ? 'Huấn luyện lại' : 'Huấn luyện'}</button></div></form>
      {training.error && <StatusPanel kind="error" message={training.error} onRetry={training.jobId ? training.reconnect : undefined} />}
      {training.submitting && <p role="status">Đang gửi yêu cầu huấn luyện…</p>}
      {training.missing && <button className="secondary" onClick={training.dismissMissing}>Bỏ tác vụ không tồn tại</button>}
      {training.jobId && <div className="job-panel" data-testid="job-panel"><p className="mono" data-testid="job-id">{training.jobId}</p><strong role="status" data-testid="job-status">{training.job ? jobNames[training.job.status] : 'Đang đọc tác vụ…'}</strong>{training.job && <p>{stages[training.job.stage] ?? 'Đang xử lý tác vụ'}</p>}
        {training.job?.error && <StatusPanel kind="error" message={training.job.error.error.message} />}
        {training.job?.status === 'succeeded' && <><p>Đã lưu mô hình <span className="mono">{training.job.model_id}</span>. Phiên theo dõi vẫn giữ mô hình đã ghim.</p><button className="secondary" disabled={!ready.some(m => m.model_id === training.job?.model_id)} onClick={() => selectModel(training.job!.model_id!)}>Chọn mô hình mới</button></>}
      </div>}
    </section>
  </>;
}
