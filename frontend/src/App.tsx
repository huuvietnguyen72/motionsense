import { useEffect, useState } from 'react';
import type React from 'react';
import { DataPage } from './features/data/DataPage';
import { MonitorPage } from './features/monitor/MonitorPage';
import { RecognitionPage } from './features/recognition/RecognitionPage';
import { ModelsPage } from './features/models/ModelsPage';

const routes = [
  { hash: '#theo-doi', name: 'Theo dõi', mark: '↗' },
  { hash: '#nhan-dien', name: 'Nhận diện', mark: '◎' },
  { hash: '#mo-hinh', name: 'Phân tích mô hình', mark: '▥' },
  { hash: '#du-lieu', name: 'Dữ liệu', mark: '▤' },
];
const currentRoute = () => routes.find(route => route.hash === window.location.hash) ?? routes[0];

export function App(): React.JSX.Element {
  const [route, setRoute] = useState(currentRoute);
  useEffect(() => {
    const onHashChange = () => setRoute(currentRoute());
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content" onClick={event => {
        event.preventDefault();
        document.getElementById('main-content')?.focus();
      }}>Đến nội dung chính</a>
      <aside className="sidebar">
        <header className="brand"><span className="brand-symbol" aria-hidden="true">M</span>
          <div><strong>MotionSense</strong><p>Theo dõi và phân tích hoạt động</p></div>
        </header>
        <nav aria-label="Điều hướng chính">
          {routes.map(item => <a key={item.hash} href={item.hash} aria-current={item.hash === route.hash ? 'page' : undefined}>
            <span aria-hidden="true">{item.mark}</span>{item.name}
          </a>)}
        </nav>
        <footer className="sidebar-footer"><span className="local-dot" aria-hidden="true" />Ứng dụng tại máy
          <p>Nguồn: Phát lại dữ liệu UCI HAR</p>
        </footer>
      </aside>
      <main id="main-content" tabIndex={-1}>
        {route.hash === '#du-lieu' ? <DataPage /> : route.hash === '#theo-doi' ? <MonitorPage /> : route.hash === '#nhan-dien' ? <RecognitionPage /> : <ModelsPage />}
      </main>
    </div>
  );
}
