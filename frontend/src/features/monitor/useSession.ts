import { useCallback, useEffect, useRef, useState } from 'react';
import { api, isAbortError } from '../../api/client';
import type { Page, Session, SessionRow, SessionSummary, Split } from '../../api/types';

// Document identity, never persisted. Reconnection rotates it to force attach/pause.
let clientId = crypto.randomUUID();
export const sessionStorageKey = 'motionsense.session_id';
type CreateRequest = { dataset_id: string; split: Split; subject_id: number; model_id: string };
type Action = 'start' | 'pause' | 'resume' | 'finish' | 'speed';
type View = { session: Session | null; rows: SessionRow[]; summary: SessionSummary | null;
  offset: number; total: number; error: string | null; busy: boolean };
type Commands = { open: (id: string, reconnect?: boolean) => void; create: (body: CreateRequest) => void;
  control: (action: Action, speed?: Session['speed']) => void; page: (offset: number) => void };
const empty: View = { session: null, rows: [], summary: null, offset: 0, total: 0, error: null, busy: false };

export function useSession(sessionId: string | null) {
  const [view, setView] = useState<View>(empty);
  const commands = useRef<Commands | null>(null);

  useEffect(() => {
    const abort = new AbortController();
    let alive = true, version = 0, blocked = false, manual = false, attached = false, mayRun = false;
    let selected = sessionId, session: Session | null = null, offset = 0;
    let queue = Promise.resolve();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const valid = (v: number) => alive && v === version;
    const stop = () => { clearTimeout(timer); timer = undefined; };
    const call = <T,>(path: string, body?: unknown) => api<T>(path, {
      signal: abort.signal, ...(body === undefined ? {} : { method: 'POST', body: JSON.stringify(body) }),
    });
    const failure = (error: unknown, v: number) => {
      if (!valid(v) || isAbortError(error)) return;
      blocked = true;
      stop();
      setView(old => ({ ...old, error: error instanceof Error ? error.message : 'Không thể đồng bộ phiên. Hãy kết nối lại.' }));
    };
    const accept = (next: Session, v: number, row?: SessionRow | null) => {
      if (!valid(v) || next.session_id !== selected) return;
      session = next;
      mayRun = next.status === 'running';
      setView(old => {
        const rows = row && row.ordinal >= offset && row.ordinal < offset + 50
          ? [...old.rows.filter(item => item.ordinal !== row.ordinal), row].sort((a, b) => a.ordinal - b.ordinal)
          : old.rows;
        return { ...old, session: next, rows, total: next.cursor };
      });
    };
    const readSummary = async (id: string, v: number) => {
      const summary = await call<SessionSummary>(`/sessions/${id}/summary`);
      if (valid(v)) setView(old => ({ ...old, summary }));
    };
    const readRows = async (id: string, v: number) => {
      const rows = await call<Page<SessionRow>>(`/sessions/${id}/rows?offset=${offset}&limit=50`);
      if (valid(v)) setView(old => ({ ...old, rows: rows.items, offset: rows.offset, total: rows.total }));
    };
    const sync = async (id: string, v: number) => {
      const next = await call<Session>(`/sessions/${id}`);
      accept(next, v);
      await readRows(id, v);
      await readSummary(id, v);
    };
    // All session requests, including controls and pagination, share one queue.
    const enqueue = (work: () => Promise<void>, v: number) => {
      queue = queue.then(async () => {
        if (valid(v)) await work();
      }).catch(error => failure(error, v));
      return queue;
    };
    const schedule = () => {
      stop();
      if (!alive || blocked || manual || !attached || session?.status !== 'running') return;
      const v = version;
      timer = setTimeout(() => {
        void enqueue(async () => {
          if (manual || blocked || session?.status !== 'running') return;
          const current = session;
          const result = await call<{ session: Session; row: SessionRow | null }>(`/sessions/${current.session_id}/advance`,
            { client_id: clientId, expected_cursor: current.cursor });
          accept(result.session, v, result.row);
          if (result.row || result.session.status !== current.status) await readSummary(current.session_id, v);
        }, v).then(() => { if (valid(v)) schedule(); });
      }, 250);
    };
    const perform = (work: (v: number) => Promise<void>, changeSelection = false) => {
      if (!alive || (manual && !changeSelection)) return;
      stop();
      manual = true;
      const v = changeSelection ? ++version : version;
      // A manual read is not reconnection: keep the failure visible while
      // blocked so pagination cannot expose silently ineffective controls.
      setView(old => ({ ...old, busy: true }));
      void enqueue(() => work(v), v).then(() => {
        if (!valid(v)) return;
        manual = false;
        setView(old => ({ ...old, busy: false }));
        schedule();
      });
    };
    const pauseOld = async () => {
      if (attached && session?.status === 'running') {
        await call<Session>(`/sessions/${session.session_id}/control`, { client_id: clientId, action: 'pause' });
      }
      attached = false;
      mayRun = false;
    };
    const attach = async (id: string, v: number) => {
      const restored = await call<Session>(`/sessions/${id}/attach`, { client_id: clientId });
      if (!valid(v)) return;
      attached = true;
      // The same document may open its own running history entry, or return
      // before the disposal pause arrives. Attach alone only pauses NEW owners.
      if (restored.status === 'running') {
        await call<Session>(`/sessions/${id}/control`, { client_id: clientId, action: 'pause' });
      }
      await sync(id, v);
      if (valid(v)) {
        // Restore both internal connectivity and its visible recovery state
        // only after attachment AND snapshot/rows/summary synchronization.
        blocked = false;
        setView(old => ({ ...old, error: null }));
      }
    };
    const open = (id: string, reconnect = false) => {
      const oldId = selected;
      selected = id;
      offset = 0;
      setView(old => ({ ...empty, error: old.error, busy: true }));
      localStorage.setItem(sessionStorageKey, id);
      perform(async v => {
        if (oldId !== id) await pauseOld();
        if (reconnect) clientId = crypto.randomUUID();
        session = null;
        await attach(id, v);
      }, true);
    };
    const control = (action: Action, speed?: Session['speed']) => {
      if (!session || blocked || !attached) return;
      perform(async v => {
        const id = session!.session_id;
        mayRun = mayRun || action === 'start' || action === 'resume';
        const next = await call<Session>(`/sessions/${id}/control`, { client_id: clientId, action, ...(speed === undefined ? {} : { speed }) });
        accept(next, v);
        await sync(id, v);
      });
    };
    commands.current = {
      open, control,
      create: body => perform(async v => {
        const next = await call<Session>('/sessions', body);
        if (!valid(v)) return;
        selected = next.session_id;
        session = null;
        attached = false;
        offset = 0;
        setView(old => ({ ...empty, error: old.error, busy: true }));
        localStorage.setItem(sessionStorageKey, selected);
        await attach(selected, v);
      }),
      page: nextOffset => {
        if (!session) return;
        perform(async v => {
          offset = Math.max(0, nextOffset);
          await readRows(session!.session_id, v);
        });
      },
    };
    // Defer restoration one microtask so StrictMode's discarded setup does no I/O.
    if (sessionId) void Promise.resolve().then(() => { if (alive) open(sessionId); });
    const visibility = () => {
      if (document.visibilityState !== 'hidden') return;
      stop();
      // Queue behind any in-flight inference/control; never resume on tab return.
      const v = version;
      void enqueue(async () => {
        if (session?.status !== 'running' || !attached || blocked) return;
        const next = await call<Session>(`/sessions/${session.session_id}/control`, { client_id: clientId, action: 'pause' });
        accept(next, v);
        await sync(next.session_id, v);
      }, v);
    };
    let leaving = false;
    const leave = () => {
      if (leaving) return;
      leaving = true;
      stop();
      if (attached && session && mayRun) {
        // Unload cannot await the normal queue. The server serializes this with
        // a final inference; its watchdog remains the network-failure fallback.
        void fetch(`/api/sessions/${session.session_id}/control`, {
          method: 'POST', keepalive: true, headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ client_id: clientId, action: 'pause' }),
        }).catch(() => undefined);
      }
    };
    document.addEventListener('visibilitychange', visibility);
    window.addEventListener('pagehide', leave);
    return () => {
      alive = false;
      leave();
      abort.abort();
      document.removeEventListener('visibilitychange', visibility);
      window.removeEventListener('pagehide', leave);
      commands.current = null;
    };
  }, [sessionId]);

  const open = useCallback((id: string) => commands.current?.open(id), []);
  const create = useCallback((body: CreateRequest) => commands.current?.create(body), []);
  const control = useCallback((action: Action, speed?: Session['speed']) => commands.current?.control(action, speed), []);
  const loadPage = useCallback((offset: number) => commands.current?.page(offset), []);
  const reconnect = () => {
    const id = view.session?.session_id ?? localStorage.getItem(sessionStorageKey);
    if (id) commands.current?.open(id, true);
  };
  return { ...view, open, create, control, loadPage, reconnect };
}
