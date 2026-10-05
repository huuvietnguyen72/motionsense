import type React from 'react';

interface Props {
  kind: 'loading' | 'empty' | 'error' | 'setup';
  message: string;
  onRetry?: () => void;
}

export function StatusPanel({ kind, message, onRetry }: Props): React.JSX.Element {
  const titles = { loading: 'Đang tải', empty: 'Chưa có dữ liệu', error: 'Chưa thể hoàn tất', setup: 'Cần thiết lập' };
  return (
    <div className={`status-panel status-${kind}`} role={kind === 'error' ? 'alert' : 'status'} aria-busy={kind === 'loading'}>
      <span className="status-marker" aria-hidden="true">{kind === 'loading' ? '…' : kind === 'error' ? '!' : 'i'}</span>
      <div><strong>{titles[kind]}</strong><p>{message}</p>
        {onRetry && <button type="button" onClick={onRetry}>Thử lại</button>}
      </div>
    </div>
  );
}
