import type React from 'react';

export function ConfusionMatrix({ labels, matrix, labelNames }: { labels: number[]; matrix: number[][]; labelNames: Record<number, string> }): React.JSX.Element {
  const max = Math.max(1, ...matrix.flat());
  return <div className="table-scroll"><table aria-label="Ma trận nhầm lẫn" className="confusion-matrix">
    <caption>Hàng: thực tế · Cột: dự đoán</caption>
    <thead><tr><th scope="col">Hoạt động</th>{labels.map(id => <th scope="col" key={id}>{labelNames[id] ?? `Nhãn ${id}`}</th>)}</tr></thead>
    <tbody>{matrix.map((row, r) => <tr key={labels[r]}><th scope="row">{labelNames[labels[r]] ?? `Nhãn ${labels[r]}`}</th>{row.map((value, c) => <td key={labels[c]} style={{ backgroundColor: `rgba(18, 107, 99, ${0.04 + 0.25 * value / max})` }}>{value}</td>)}</tr>)}</tbody>
  </table></div>;
}
