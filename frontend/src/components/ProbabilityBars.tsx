import type React from 'react';
import type { Probability } from '../api/types';

export function ProbabilityBars({ probabilities, labels }: { probabilities: Probability[]; labels: Record<number, string> }): React.JSX.Element {
  return <div className="probabilities" data-testid="probabilities">
    {probabilities.length === 0 ? <p>Chưa có xác suất mô hình.</p> : probabilities.map(p => <div className="probability-row" key={p.label_id}>
      <label htmlFor={`probability-${p.label_id}`}>{labels[p.label_id] ?? `Nhãn ${p.label_id}`}</label>
      <meter id={`probability-${p.label_id}`} data-testid={`probability-${p.label_id}`} min={0} max={1} value={p.value}>{p.value}</meter>
      <span>{new Intl.NumberFormat('vi-VN', { style: 'percent', maximumFractionDigits: 1 }).format(p.value)}</span>
    </div>)}
  </div>;
}
