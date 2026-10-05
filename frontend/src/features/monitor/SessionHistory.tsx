import { useEffect, useState } from 'react';
import type React from 'react';
import { api, isAbortError } from '../../api/client';
import type { Page, Session } from '../../api/types';
import { StatusPanel } from '../../components/StatusPanel';

export const sessionStatuses = { ready: 'Sẵn sàng', running: 'Đang chạy', paused: 'Tạm dừng', finished: 'Đã kết thúc' };

export function SessionHistory({ onOpen, refreshKey, busy = false }: { onOpen: (id: string) => void; refreshKey?: string; busy?: boolean }): React.JSX.Element {
  const [offset, setOffset] = useState(0);
  const [page, setPage] = useState<Page<Session> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    const abort = new AbortController();
    let alive = true;
    setLoading(true); setError(null);
    void api<Page<Session>>(`/sessions?offset=${offset}&limit=50`, { signal: abort.signal }).then(value => {
      if (alive) { setPage(value); setLoading(false); }
    }).catch(e => { if (alive && !isAbortError(e)) { setError(e instanceof Error ? e.message : 'Không thể tải lịch sử.'); setLoading(false); } });
    return () => { alive = false; abort.abort(); };
  }, [offset, refreshKey, attempt]);
  return <section className="card" aria-labelledby="history-heading"><div className="section-heading"><h2 id="history-heading">Phiên đã lưu</h2><button className="secondary" disabled={loading} onClick={() => setAttempt(n => n + 1)}>Làm mới lịch sử</button></div>
    <p>Mở kết quả tại máy, kể cả khi mô hình hoặc nguồn dữ liệu không còn sẵn sàng.</p>
    {error && <StatusPanel kind="error" message={error} onRetry={() => setAttempt(n => n + 1)} />}
    {loading && <p role="status">Đang tải lịch sử phiên…</p>}
    {page && !loading && (page.total === 0 ? <p role="status">Chưa có phiên đã lưu. Chọn người và mô hình để tạo phiên đầu tiên.</p> : <>
      <div className="table-scroll"><table aria-label="Phiên đã lưu"><thead><tr><th scope="col">Người / phân vùng</th><th scope="col">Mô hình ghim</th><th scope="col">Trạng thái</th><th scope="col">Mẫu đã xử lý</th><th scope="col">Mở phiên</th></tr></thead><tbody>{page.items.map(s => <tr key={s.session_id}>
        <th scope="row">Người {s.subject_id} · {s.split}<small className="history-time">{new Date(s.created_at).toLocaleString('vi-VN')}</small></th>
        <td className="mono history-model">{s.model_id}</td><td>{sessionStatuses[s.status]}</td><td>{s.cursor} / {s.total}</td>
        <td><button className="secondary" aria-label={`Mở phiên ${s.session_id}`} disabled={busy} onClick={() => onOpen(s.session_id)}>Mở</button></td>
      </tr>)}</tbody></table></div>
      <div className="pagination"><button className="secondary" disabled={offset === 0 || loading} onClick={() => setOffset(n => Math.max(0, n - 50))}>Phiên trước</button><span>{offset + 1}–{Math.min(offset + 50, page.total)} / {page.total} phiên</span><button className="secondary" disabled={offset + 50 >= page.total || loading} onClick={() => setOffset(n => n + 50)}>Phiên kế tiếp</button></div>
    </>)}
  </section>;
}
