import type React from 'react';
import type { Signal } from '../api/types';

export function SignalChart({ signal, sampleId }: { signal: Signal | null; sampleId: string | null }): React.JSX.Element {
  if (!signal || !sampleId) return <div className="signal-placeholder" role="status">{sampleId ? 'Đang tải cửa sổ tín hiệu đúng mẫu…' : 'Tín hiệu xuất hiện sau mẫu dự đoán đầu tiên.'}</div>;
  const min = Math.min(...signal.x, ...signal.y, ...signal.z);
  const max = Math.max(...signal.x, ...signal.y, ...signal.z);
  const margin = Math.max((max - min) * 0.1, 0.05);
  const low = min - margin, high = max + margin;
  const x = (t: number) => 48 + t / 2.54 * 624;
  const y = (v: number) => 190 - (v - low) / (high - low) * 166;
  return <div className="signal-chart" data-testid="signal-chart" data-sample-id={sampleId}>
    <svg viewBox="0 0 704 232" role="img" aria-label={`Gia tốc tổng X, Y, Z của mẫu ${sampleId}; 128 điểm mỗi trục, đơn vị g, thời gian 0 đến 2,54 giây`}>
      {[low, (low + high) / 2, high].map(v => <g key={v}><line x1={48} x2={672} y1={y(v)} y2={y(v)} className="signal-grid" /><text x={40} y={y(v) + 4} textAnchor="end">{v.toFixed(2)}</text></g>)}
      <text x={16} y={14}>g</text>
      {[0, 1.28, 2.54].map(t => <text key={t} x={x(t)} y={214} textAnchor="middle">{String(t).replace('.', ',')} s</text>)}
      {(['x', 'y', 'z'] as const).map(axis => <polyline key={axis} className={`trace trace-${axis}`} data-values={JSON.stringify(signal[axis])}
        points={signal[axis].map((v, i) => `${x(signal.time_seconds[i])},${y(v)}`).join(' ')} />)}
    </svg>
    <div className="signal-legend"><span className="axis-x">X</span><span className="axis-y">Y</span><span className="axis-z">Z</span><span className="mono">{sampleId}</span><span>0–2,54 s · 128 điểm / trục</span></div>
    <details><summary>Số liệu tín hiệu của cửa sổ</summary><div className="table-scroll signal-values"><table aria-label="Số liệu gia tốc tổng"><thead><tr><th scope="col">Thời gian (s)</th><th scope="col">X (g)</th><th scope="col">Y (g)</th><th scope="col">Z (g)</th></tr></thead><tbody>
      {signal.time_seconds.map((t, i) => <tr key={i}><th scope="row">{t.toFixed(2)}</th><td>{signal.x[i]}</td><td>{signal.y[i]}</td><td>{signal.z[i]}</td></tr>)}
    </tbody></table></div></details>
  </div>;
}
