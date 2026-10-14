// History: every finished session, newest first; open one for its alert
// timeline and report, and have the LLM rewrite the report if wanted.

import { useEffect, useState } from "react";
import { api, type SessionMeta } from "../api";
import { Banner } from "../App";
import { Alerts, Markdown, type AlertItem } from "../components/Session";

interface Full extends SessionMeta { log: { alerts: AlertItem[] } | null }

export default function History({ editor }: { editor: boolean }) {
  const [sessions, setSessions] = useState<SessionMeta[] | null>(null);
  const [open, setOpen] = useState<Full | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api<SessionMeta[]>("/api/sessions").then(setSessions).catch((e) => setErr(e.message));
  }, []);

  const show = (id: string) => {
    setErr("");
    api<Full>(`/api/sessions/${encodeURIComponent(id)}`).then(setOpen).catch((e) => setErr(e.message));
  };

  const rewrite = async () => {
    if (!open) return;
    setBusy(true); setErr("");
    try {
      const r = await api<{ report: string }>(`/api/sessions/${encodeURIComponent(open.id)}/report`, { body: {} });
      setOpen({ ...open, report: r.report });
    } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  };

  return (
    <div className="history">
      <section>
        <h2>Sessions</h2>
        {sessions === null ? <p className="muted">Loading…</p>
          : sessions.length === 0 ? <p className="muted">No sessions yet. Start one or open a replay.</p>
          : (
            <table>
              <thead><tr><th>When</th><th>Procedure</th><th>Source</th><th>Progress</th><th>Alerts</th><th>Outcome</th></tr></thead>
              <tbody>
                {sessions.map((s) => (
                  <tr key={s.id} className={open?.id === s.id ? "on" : ""} onClick={() => show(s.id)}>
                    <td>{(s.ended_at ?? "").replace("T", " ").replace("+00:00", " UTC")}</td>
                    <td>{s.procedure}</td>
                    <td className="muted">{s.source ?? ""}</td>
                    <td>{s.steps_done ?? "—"}</td>
                    <td className={s.red ? "bad" : ""}>{s.alert_count}</td>
                    <td>{s.outcome ?? ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
      </section>
      {err && <Banner kind="bad">{err}</Banner>}
      {open && (
        <section className="detail">
          <h2>{open.procedure} · {open.id}</h2>
          <div className="cols">
            <div>
              <h2>Alert timeline</h2>
              <Alerts alerts={[...(open.log?.alerts ?? [])].reverse()} />
            </div>
            <div>
              <h2>Report</h2>
              {open.report ? <Markdown text={open.report} /> : <p className="muted">No report.</p>}
              {editor && (
                <button disabled={busy} onClick={rewrite}>{busy ? "Writing…" : "Rewrite with the LLM"}</button>
              )}
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
