// The live session: start a replay (no hardware) or watch the edge laptop's
// session; checklist or rules, legality panel or board, alerts, then the report.

import { Suspense, lazy, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { api, socketUrl, type Msg, type Row, type Rule, type SpecMeta, type Tower } from "../api";
import { Banner } from "../App";
import { Ask } from "../components/Ask";
import { verdict, type Flash, type Mode } from "../jenga";
import { Alerts, Board, Checklist, Markdown, Rules, TowerPanel, isBoard, type AlertItem } from "../components/Session";


const Tower3D = lazy(() => import("../components/Tower3D"));
interface View {
  procedure: string; kind: string; source: string; t: number; progress: string;
  checklist: Row[]; rules: Rule[]; next: string | null; tower?: Tower; frame?: string;
  alerts: AlertItem[]; perception: "ok" | "lost"; end?: { summary: string; report: string };
  error?: string;
}

const EMPTY: View = {
  procedure: "", kind: "", source: "", t: 0, progress: "", checklist: [], rules: [], next: null,
  alerts: [], perception: "ok",
};

function reduce(v: View, m: Msg): View {
  switch (m.type) {
    case "hello":
      return { ...EMPTY, procedure: m.procedure, kind: m.kind, source: m.source,
               checklist: m.checklist, rules: m.rules };
    case "state":
      return { ...v, t: m.t, progress: m.progress, checklist: m.checklist, next: m.next,
               tower: m.tower ?? v.tower, perception: m.perception ?? "ok" };
    case "alert":
      return { ...v, alerts: [...v.alerts, m] };
    case "frame":
      return { ...v, frame: m.jpeg };
    case "end":
      return { ...v, end: { summary: m.summary, report: m.report } };
    case "error":
      return { ...v, error: m.message };
  }
}

export default function Live() {
  const [view, dispatch] = useReducer(reduce, EMPTY);
  const [connected, setConnected] = useState(false);
  const [edge, setEdge] = useState<{ online: boolean; name: string; running: boolean } | null>(null);
  const [scenarios, setScenarios] = useState<string[]>([]);
  const [specs, setSpecs] = useState<SpecMeta[]>([]);
  const [scenario, setScenario] = useState("");
  const [speed, setSpeed] = useState("2");
  const [noise, setNoise] = useState("");
  const [cam, setCam] = useState({ spec: "", video: "", reference: "" });
  const [game, setGame] = useState("http://127.0.0.1:8000");
  const [err, setErr] = useState("");
  const [seed, setSeed] = useState<{ q: string; n: number }>();
  const [mode, setMode] = useState<Mode>("legal");
  const [flat, setFlat] = useState(false);
  const [toast, setToast] = useState<AlertItem | null>(null);
  const ask = (q: string) => setSeed((p) => ({ q, n: (p?.n ?? 0) + 1 }));
  const explain = (a: AlertItem) =>
    ask(`Explain the ${a.code.replaceAll("_", " ")} alert at ${a.t.toFixed(1)} s: what happened and what should have been done?`);

  // The latest alert: a toast over the stage, and the block it names lit up in 3D.
  const last = view.alerts[view.alerts.length - 1];
  const flash = useMemo<Flash | undefined>(() => {
    if (!last) return undefined;
    const m = /took from L(\d+)s(\d+), placed at L(\d+)s(\d+)/.exec(last.text);
    return m ? { from: [+m[1], +m[2]], to: [+m[3], +m[4]], key: view.alerts.length }
             : { key: view.alerts.length };
  }, [view.alerts.length]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!last || view.end) return;
    setToast(last);
    const t = window.setTimeout(() => setToast(null), 5000);
    return () => clearTimeout(t);
  }, [view.alerts.length]); // eslint-disable-line react-hooks/exhaustive-deps
  const counts = useMemo(() => {
    const c = { safe: 0, topples: 0, blocked: 0 };
    view.tower?.rows.forEach((r, L) => r.forEach((v, s) => { if (v) c[verdict(view.tower!, L, s)]++; }));
    return c;
  }, [view.tower]);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    let ws: WebSocket | null = null;
    let retry: number | undefined;
    const connect = async () => {
      ws = new WebSocket(await socketUrl());
      ws.onopen = () => setConnected(true);
      ws.onmessage = (e) => { const m = JSON.parse(e.data) as Msg; if (m.v === 1) dispatch(m); };
      ws.onclose = () => {
        setConnected(false);
        if (alive.current) retry = window.setTimeout(connect, 1500);
      };
    };
    connect();
    api<string[]>("/api/scenarios").then((s) => { setScenarios(s); setScenario(s[0] ?? ""); }).catch(() => {});
    api<SpecMeta[]>("/api/specs").then((s) => {
      setSpecs(s); setCam((c) => ({ ...c, spec: s.find((x) => x.kind === "dag")?.name ?? "" }));
    }).catch(() => {});
    const poll = () => api<typeof edge>("/api/edge").then(setEdge).catch(() => {});
    poll();
    const timer = window.setInterval(poll, 5000);
    return () => { alive.current = false; ws?.close(); clearTimeout(retry); clearInterval(timer); };
  }, []);

  const run = (path: string, body: unknown) => {
    setErr("");
    api(path, { body }).catch((e) => setErr(e.message));
  };

  const fromEdge = view.source.startsWith("edge:");
  const rulebook = view.rules.length > 0;
  return (
    <div className="live">
      {!connected && <Banner kind="warn">Not connected to the server; reconnecting…</Banner>}
      {fromEdge && edge && !edge.online && !view.end &&
        <Banner kind="bad">Edge node offline: the laptop running the cameras has stopped reporting.</Banner>}
      {view.perception === "lost" && !view.end &&
        <Banner kind="warn">Calibration lost (markers hidden): alerts are paused until they are back in view.</Banner>}
      {view.error && <Banner kind="bad">{view.error}</Banner>}
      {view.source.startsWith("game:") && !view.end &&
        <Banner kind="warn">Following the game server: pulls come from its MARK taps and finder (with your corrections);
          where each block was put back is not observed, so it is assumed legal.</Banner>}

      <section className="controls">
        <h2>Session</h2>
        <label>Replay a scenario (no hardware)</label>
        <select value={scenario} onChange={(e) => setScenario(e.target.value)}>
          {scenarios.map((s) => <option key={s}>{s}</option>)}
        </select>
        <label>Speed</label>
        <select value={speed} onChange={(e) => setSpeed(e.target.value)}>
          <option value="1">1× (real time)</option><option value="2">2×</option>
          <option value="5">5×</option><option value="0">instant</option>
        </select>
        <label>Detector noise <span className="muted">(optional)</span></label>
        <input value={noise} onChange={(e) => setNoise(e.target.value)} placeholder="e.g. drop=0.1,ghost=0.05" />
        <div className="row">
          <button className="primary" onClick={() => run("/api/replay", { scenario, speed: parseFloat(speed), noise })}>
            Replay
          </button>
          <button onClick={() => run("/api/stop", {})}>Stop</button>
        </div>

        <h2 className="gap">Real game <span className="tag">phones</span></h2>
        <p className="muted">
          Follow the game server: both phones, a MARK tap after each pull, then left / middle / right.
          Its confirmed pulls drive the tower, the rules and the LLM here.
        </p>
        <label>Game server</label>
        <input value={game} onChange={(e) => setGame(e.target.value)} placeholder="http://127.0.0.1:8000" />
        <div className="row">
          <button className="primary" onClick={() => run("/api/follow", { url: game.trim() })}>Follow</button>
          <button onClick={() => run("/api/stop", {})}>Stop</button>
        </div>

        <h2 className="gap">Edge laptop</h2>
        <p className="muted">
          {edge?.online ? <>● <b>{edge.name}</b> is connected. Sessions it starts appear here.</>
            : <>No edge connected. On the demo laptop run <code>python -m thirdeye.server.edge --cloud {location.origin} …</code></>}
        </p>

        <details>
          <summary>Camera on this server (local runs)</summary>
          <label>Spec</label>
          <select value={cam.spec} onChange={(e) => setCam({ ...cam, spec: e.target.value })}>
            {specs.filter((s) => s.kind === "dag").map((s) => <option key={s.name}>{s.name}</option>)}
          </select>
          <label>Video file (repo path; blank = camera 0)</label>
          <input value={cam.video} onChange={(e) => setCam({ ...cam, video: e.target.value })} placeholder="data/raw/jigsaw_R001.mp4" />
          <label>Box picture (jigsaw)</label>
          <input value={cam.reference} onChange={(e) => setCam({ ...cam, reference: e.target.value })} placeholder="data/box.jpg" />
          <div className="row">
            <button onClick={() => run("/api/camera", { spec: cam.spec, camera: 0, video: cam.video || null,
              reference: cam.reference || null })}>Start</button>
          </div>
        </details>
        {err && <p className="err">{err}</p>}
      </section>

      <section className="view">
        <h2>Live view</h2>
        {view.procedure ? (
          <div className="status">
            <span className={`pill ${view.end ? "" : view.source.startsWith("replay") ? "replay" : "live"}`}>
              <i />{view.end ? "Ended" : view.source.startsWith("replay") ? "Replay"
                : view.source.startsWith("game:") ? "Live · game server" : "Live"}
            </span>
            <span className="stat-chip"><b>{view.procedure}</b>{view.kind === "constraints" ? "rulebook" : "steps"}</span>
            <span className="stat-chip"><b>{view.t.toFixed(1)} s</b>time</span>
            {view.progress && <span className="stat-chip"><b>{view.progress.replace(/ moves$/, "")}</b>
              {view.progress.endsWith("moves") ? "moves" : "steps"}</span>}
            <span className={`stat-chip ${view.alerts.length ? "bad" : "ok"}`}><b>{view.alerts.length}</b>alerts</span>
            {view.tower && <span className="stat-chip"><b>{view.tower.top_complete_layer}</b>top complete layer</span>}
          </div>
        ) : <p className="muted">Pick a scenario and press Replay, or start a session on the edge laptop.</p>}
        <div className="stage">
          {view.frame ? <img src={`data:image/jpeg;base64,${view.frame}`} alt="camera" />
            : view.tower ? (flat ? <TowerPanel t={view.tower} />
              : <Suspense fallback={<TowerPanel t={view.tower} />}>
                  <Tower3D tower={view.tower} mode={mode} flash={flash} onPick={(L, sl, kind) => ask(kind === "ghost"
                    ? `Can the next block go at layer ${L}, slot ${sl}? Why?`
                    : `Can I take the block at layer ${L}, slot ${sl}? Is it allowed, and would the tower stay up?`)} />
                </Suspense>)
            : isBoard(view.checklist) ? <Board rows={view.checklist} next={view.next} />
            : view.kind === "constraints" ? <p className="muted">Waiting for the first tower reading…</p>
            : view.procedure ? <p className="muted">No board view for this procedure; follow the checklist.</p> : null}
          {toast && (
            <div className={`toast ${toast.severity}`}>
              <div><b>{toast.code.replaceAll("_", " ")}</b> · {toast.t.toFixed(1)} s</div>
              <div>{toast.text}</div>
              <button className="link" onClick={() => { explain(toast); setToast(null); }}>Explain with the LLM →</button>
            </div>
          )}
        </div>
        {view.tower && !view.frame && (
          <div className="stage-bar">
            <div className="seg">
              <button className={!flat ? "on" : ""} onClick={() => setFlat(false)}>3D</button>
              <button className={flat ? "on" : ""} onClick={() => setFlat(true)}>2D</button>
            </div>
            {!flat && (
              <div className="seg">
                <button className={mode === "legal" ? "on" : ""} onClick={() => setMode("legal")}>Where it can go</button>
                <button className={mode === "safe" ? "on" : ""} onClick={() => setMode("safe")}>Safe pulls</button>
              </div>
            )}
            {!flat && (mode === "legal" ? (
              <span className="key"><i className="k-ghost" />next block may go · <i className="k-tcl" />top complete layer · drag to turn, click a block to ask</span>
            ) : (
              <span className="key">
                <i className="k-ok" />safe {counts.safe} · <i className="k-warn" />would topple {counts.topples} · <i className="k-idle" />not allowed {counts.blocked}
              </span>
            ))}
          </div>
        )}
      </section>

      <section className="side">
        <h2>Alerts</h2>
        <Alerts alerts={view.alerts} onExplain={explain} />
        <h2 className="gap">Ask the referee <span className="tag">LLM</span></h2>
        <Ask ready={!!view.procedure} seed={seed} />
        <h2 className="gap">{rulebook ? "Rules in force" : "Checklist"}</h2>
        {rulebook ? <Rules rules={view.rules} /> : <Checklist rows={view.checklist} />}
      </section>

      {view.end && (
        <section className="report">
          <h2>Report</h2>
          <pre className="summary">{view.end.summary}</pre>
          <Markdown text={view.end.report} />
          <p className="muted">Saved to <a href="#/history">History</a>.</p>
        </section>
      )}
    </div>
  );
}
