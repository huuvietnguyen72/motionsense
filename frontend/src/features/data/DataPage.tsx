import { useEffect, useState } from 'react';
import type React from 'react';
import { api, isAbortError } from '../../api/client';
import type { DatasetInfo, FeatureInfo, ModelInfo } from '../../api/types';
import { StatusPanel } from '../../components/StatusPanel';

type Load<T> = { kind: 'loading' } | { kind: 'error'; message: string } | { kind: 'ready'; value: T };
const format = (number: number) => new Intl.NumberFormat('vi-VN').format(number);
const errorMessage = (error: unknown) => error instanceof Error ? error.message : 'Không thể tải dữ liệu. Hãy thử lại.';

export function DataPage(): React.JSX.Element {
  const [attempt, setAttempt] = useState(0);
  const [dataset, setDataset] = useState<Load<DatasetInfo>>({ kind: 'loading' });
  const [features, setFeatures] = useState<Load<FeatureInfo[]>>({ kind: 'loading' });
  const [models, setModels] = useState<Load<ModelInfo[]>>({ kind: 'loading' });
  const [modelId, setModelId] = useState('');
  const [query, setQuery] = useState('');
  const retry = () => setAttempt(value => value + 1);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    const options = { signal: controller.signal };
    setDataset({ kind: 'loading' });
    setFeatures({ kind: 'loading' });
    setModels({ kind: 'loading' });
    setModelId('');
    async function load() {
      try {
        const info = await api<DatasetInfo>('/data', options);
        if (!active) return;
        setDataset({ kind: 'ready', value: info });
        if (!info.ready) return;
        void api<{ items: FeatureInfo[] }>('/data/features', options).then(result => {
          if (active) setFeatures({ kind: 'ready', value: result.items });
        }).catch(error => {
          if (active && !isAbortError(error)) setFeatures({ kind: 'error', message: errorMessage(error) });
        });
        void api<{ items: ModelInfo[] }>('/models', options).then(result => {
          if (!active) return;
          const ready = result.items.filter(model => model.status === 'ready' && model.dataset_id === info.dataset_id);
          setModels({ kind: 'ready', value: ready });
          setModelId((ready.find(model => model.profile === 'full') ?? ready[0])?.model_id ?? '');
        }).catch(error => {
          if (active && !isAbortError(error)) setModels({ kind: 'error', message: errorMessage(error) });
        });
      } catch (error) {
        if (active && !isAbortError(error)) setDataset({ kind: 'error', message: errorMessage(error) });
      }
    }
    void load();
    return () => { active = false; controller.abort(); };
  }, [attempt]);

  const info = dataset.kind === 'ready' ? dataset.value : null;
  const filtered = features.kind === 'ready' ? features.value.filter(feature =>
    `${feature.feature_id} ${feature.name}`.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase())) : [];
  const model = models.kind === 'ready' ? models.value.find(item => item.model_id === modelId) : undefined;
  return (
    <>
      <div className="page-heading"><div><p className="eyebrow">MotionSense / Nguồn dữ liệu</p><h1>Dữ liệu</h1>
        <p>Hiểu nguồn cảm biến, cách chia mẫu và lược đồ đặc trưng trước khi nhận diện.</p></div>
        {info && <span className={`badge ${info.ready ? 'badge-ready' : ''}`}>{info.ready ? 'Dữ liệu sẵn sàng' : 'Chưa thiết lập'}</span>}
      </div>
      {dataset.kind === 'loading' && <StatusPanel kind="loading" message="Đang tải thông tin dữ liệu từ dịch vụ tại máy." />}
      {dataset.kind === 'error' && <StatusPanel kind="error" message={dataset.message} onRetry={retry} />}
      {info && <>
        <section className="card source-card" aria-labelledby="source-heading">
          <div><p className="eyebrow">Nguồn nghiên cứu</p><h2 id="source-heading">UCI HAR · Hoạt động từ điện thoại</h2>
            <p>Human Activity Recognition Using Smartphones — Anguita, D., Ghio, A., Oneto, L., Parra, X. và Reyes-Ortiz, J. L. (2013).</p>
            <p>Gia tốc kế và con quay hồi chuyển trên Samsung Galaxy S II.</p>
          </div>
          <div className="source-links"><a href={info.source_url} target="_blank" rel="noreferrer">Nguồn UCI HAR</a>
            <span>DOI <a href="https://doi.org/10.24432/C54S4K" target="_blank" rel="noreferrer">10.24432/C54S4K</a></span>
            <span>Giấy phép <strong data-testid="dataset-license">{info.license}</strong></span>
          </div>
        </section>
        {!info.ready ? <StatusPanel kind="setup" message="Dữ liệu chưa sẵn sàng hoặc tệp đã lưu không hợp lệ. Hãy chạy Cai_dat.bat để chuẩn bị dữ liệu, sau đó khởi động lại MotionSense." onRetry={retry} /> : <>
          <section className="metrics" aria-label="Thống kê dữ liệu">
            <div className="metric"><span>Tổng số mẫu</span><strong data-testid="dataset-total">{format(info.split_counts.train + info.split_counts.test)}</strong><small>Cửa sổ cảm biến</small></div>
            <div className="metric"><span>Người tham gia</span><strong data-testid="dataset-subjects">{format(new Set(info.subjects.map(s => s.subject_id)).size)}</strong><small>Chia tập theo người</small></div>
            <div className="metric"><span>Đặc trưng mỗi mẫu</span><strong data-testid="dataset-features">{format(info.feature_count)}</strong><small>Mã cột duy nhất</small></div>
            <div className="metric"><span>Hoạt động</span><strong>{format(info.activities.length)}</strong><small>Nhãn gốc từ nguồn</small></div>
          </section>
          <div className="data-grid">
            <section className="card" aria-labelledby="split-heading"><h2 id="split-heading">Phân chia theo người</h2>
              <p>Giữ nguyên tập train/test chính thức của UCI. Người tham gia ở hai tập không giao nhau.</p>
              <div className="split-stats"><div><span>Huấn luyện · train</span><strong data-testid="dataset-train">{format(info.split_counts.train)}</strong><small>{format(info.subjects.filter(s => s.split === 'train').length)} người</small></div>
                <div><span>Kiểm tra · test</span><strong data-testid="dataset-test">{format(info.split_counts.test)}</strong><small>{format(info.subjects.filter(s => s.split === 'test').length)} người</small></div></div>
              <details><summary>Xem người tham gia và số mẫu</summary><div className="table-scroll"><table aria-label="Người tham gia"><thead><tr><th scope="col">Mã người</th><th scope="col">Phân vùng</th><th scope="col">Số mẫu</th></tr></thead>
                <tbody>{info.subjects.map(s => <tr key={`${s.split}:${s.subject_id}`}><th scope="row">{s.subject_id}</th><td>{s.split}</td><td>{format(s.sample_count)}</td></tr>)}</tbody></table></div></details>
            </section>
            <section className="card" aria-labelledby="activities-heading"><h2 id="activities-heading">Hoạt động trong dữ liệu</h2>
              <table aria-label="Hoạt động trong dữ liệu"><thead><tr><th scope="col">Hoạt động</th><th scope="col">Nhãn</th><th scope="col">Số mẫu</th></tr></thead><tbody>
                {info.activities.map(activity => <tr key={activity.label_id}><th scope="row">{activity.name_vi}</th><td className="mono">{activity.label_id}</td><td>{format(activity.count)}</td></tr>)}
              </tbody></table>
            </section>
          </div>
          <section className="card window-card" aria-labelledby="window-heading"><div><p className="eyebrow">Cách đọc tín hiệu</p><h2 id="window-heading">Một mẫu là một cửa sổ cảm biến</h2></div>
            <dl className="window-facts"><div><dt>Tần số lấy mẫu</dt><dd>50 Hz</dd></div><div><dt>Mỗi trục X / Y / Z</dt><dd>128 điểm</dd></div><div><dt>Độ dài cửa sổ</dt><dd>2,56 giây</dd></div><div><dt>Chồng lấn</dt><dd>50%</dd></div></dl>
            <p>Trục thời gian là thời gian tương đối trong cửa sổ: 0–2,54 giây, bước 0,02 giây; gia tốc tổng có đơn vị g. Các cửa sổ chồng lấn, không phải bản ghi liên tục.</p>
            <p>Phát lại dữ liệu theo thứ tự mẫu của từng người. Thống kê theo số mẫu và tỷ lệ mẫu; không suy ra thời lượng hoạt động bằng cách cộng 2,56 giây cho từng cửa sổ.</p>
          </section>
          <div className="data-grid feature-grid">
            <section className="card" aria-labelledby="features-heading"><div className="section-heading"><h2 id="features-heading">Lược đồ đặc trưng</h2><span className="mono">f001…f561</span></div>
              <p>ID giữ thứ tự cột gốc. Tên mô tả có thể trùng nhau; dùng ID khi nhập CSV.</p>
              {features.kind === 'loading' ? <StatusPanel kind="loading" message="Đang tải danh sách đặc trưng." /> : features.kind === 'error' ? <StatusPanel kind="error" message={features.message} onRetry={retry} /> : <>
                <label htmlFor="feature-search">Tìm đặc trưng theo ID hoặc tên</label>
                <input id="feature-search" type="search" value={query} onChange={event => setQuery(event.target.value)} placeholder="Ví dụ: f001 hoặc tBodyAcc" />
                <p className="result-count" aria-live="polite">{format(filtered.length)} / {format(features.value.length)} đặc trưng</p>
                {filtered.length === 0 ? <StatusPanel kind="empty" message="Không tìm thấy đặc trưng. Thử ID hoặc tên khác, hoặc xóa nội dung tìm kiếm." /> : <div className="table-scroll feature-table" tabIndex={0} role="region" aria-label="Bảng đặc trưng cuộn được"><table aria-label="Danh sách đặc trưng"><thead><tr><th scope="col">ID cột</th><th scope="col">Tên trong nguồn</th></tr></thead><tbody>
                  {filtered.map(feature => <tr key={feature.feature_id}><td className="mono">{feature.feature_id}</td><td>{feature.name}</td></tr>)}
                </tbody></table></div>}
              </>}
            </section>
            <section className="card template-card" aria-labelledby="template-heading"><p className="eyebrow">Nhập dữ liệu</p><h2 id="template-heading">CSV theo mô hình</h2>
              <p>Tệp gồm các ID đặc trưng mà phiên bản mô hình yêu cầu và một hàng mẫu hợp lệ. Các giá trị đầu vào phải là số hữu hạn.</p>
              {models.kind === 'loading' ? <StatusPanel kind="loading" message="Đang tải mô hình sẵn sàng." /> : models.kind === 'error' ? <StatusPanel kind="error" message={models.message} onRetry={retry} /> : models.value.length === 0 ? <StatusPanel kind="setup" message="Chưa có mô hình sẵn sàng cho dữ liệu này. Hãy chạy Cai_dat.bat và khởi động lại MotionSense để tải CSV mẫu." /> : <>
                <label htmlFor="template-model">Mô hình cho tệp CSV</label><select id="template-model" value={modelId} onChange={event => setModelId(event.target.value)}>
                  {models.value.map(item => <option key={item.model_id} value={item.model_id}>{item.profile === 'full' ? 'Đầy đủ' : 'Rút gọn'} · {item.trees} cây · {item.model_id}</option>)}
                </select>
                {model && <><dl className="model-info"><div><dt>Số cột đặc trưng</dt><dd>{format(model.feature_ids.length)}</dd></div><div><dt>Phiên bản mô hình</dt><dd className="mono">{model.model_id}</dd></div></dl>
                  <a className="button-link" href={`/api/recognition/template?model_id=${encodeURIComponent(model.model_id)}`} download>Tải CSV mẫu</a></>}
              </>}
              <p className="muted">Mã người và nhãn thực tế không phải đặc trưng đầu vào của mô hình.</p>
            </section>
          </div>
          <p className="dataset-identity">Phiên bản dữ liệu <span className="mono">{info.dataset_id}</span></p>
        </>}
      </>}
    </>
  );
}
