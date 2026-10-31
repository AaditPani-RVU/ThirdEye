// The game console: everything the game server's own viewer (iphone_saved.py) has, in this UI.
// Two live camera pictures with the red / green / amber overlay, START / MARK / STOP, the left-middle-right
// box that names the pulled layer, setup and marker status, training-data capture, tower settings,
// Re-detect, coverage check, rotate, and the NO MOVES LEFT / GAME OVER banner.
// Everything goes through /game/... on this server (gameproxy.py); the game server decides everything.

import { useEffect, useRef, useState } from "react";
import { API, api } from "../api";

const G = (p: string) => `${API}/game/${p}`;
const post = (p: string, body?: unknown) =>
  fetch(G(p), { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) }).catch(() => {});

interface Labels { w: number; h: number; items: [string, number, number, number, string][] }
interface Overlay { green: number; amber?: number; red: number }
interface GameStatus {
  cv: boolean; markers: number[]; android_markers: number[];
  preview_age: number | null; android_age: number | null; preview_fps: number;
  capturing: boolean; saved: number; marks: number; marks_pending?: number; tracking: string;
  upload_rate: number; upload_age: number | null; files: number;
  tower: { layers: number; removed: number[][] };
  overlay: Overlay | null; android_overlay: Overlay | null;
  dataset?: { on?: boolean; frames?: number; crops?: number; total_crops?: number };
  last_label: { status?: string; message?: string; confident?: boolean } | null;
  label_stats?: { moves: number; wrong: number };
  autotower?: { status?: string; note?: string };
  tracker_msg: string;
  height_info?: { lidar_layers: number; top_mm: number } | null;
  tower_fit?: { shift_mm: number[]; tilt_deg: number; twist_deg: number; height_mm: number; layers: number;
                layers_float: number; inliers: number; rms_mm: number; converged: boolean } | null;
  labels?: { iphone?: Labels; android?: Labels };
  no_moves?: boolean;
  game_over?: { reason: string } | null;
}

const LEGEND: [string, string][] = [
  ["white: layer, counted from the bottom", "#fff"],
  ["yellow digit: slot of layers 1,3,5,7,9", "#ffd700"],
  ["cyan digit: slot of layers 2,4,6,8,10", "#00dcff"],
];

// The picture is turned by the viewer, so its labels are drawn here, upright, at the turned positions.
function drawTo(cv: HTMLCanvasElement, img: HTMLImageElement, rot: number, lab?: Labels) {
  const x = cv.getContext("2d");
  if (!x) return;
  const turned = rot % 180 !== 0;
  const sw = turned ? img.naturalHeight : img.naturalWidth, sh = turned ? img.naturalWidth : img.naturalHeight;
  if (cv.width !== sw || cv.height !== sh) { cv.width = sw; cv.height = sh; }
  x.save();
  x.translate(cv.width / 2, cv.height / 2);
  x.rotate((rot * Math.PI) / 180);
  x.drawImage(img, -img.naturalWidth / 2, -img.naturalHeight / 2);
  x.restore();
  const k = img.naturalWidth / 960;
  const a = (rot * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
  if (lab && lab.w === img.naturalWidth && lab.h === img.naturalHeight) {
    x.lineJoin = "round";
    for (const [t, px, py, sc, col] of lab.items) {
      const dx = px - lab.w / 2, dy = py - lab.h / 2;
      const X = cv.width / 2 + dx * c - dy * s, Y = cv.height / 2 + dx * s + dy * c;
      x.font = `bold ${Math.round(sc * 34 * k)}px sans-serif`;
      x.lineWidth = 4 * k; x.strokeStyle = "#000"; x.strokeText(t, X, Y);
      x.fillStyle = col; x.fillText(t, X, Y);
    }
  }
  const f = Math.round(15 * k), lh = Math.round(22 * k);
  x.font = `${f}px sans-serif`;
  x.fillStyle = "#000"; x.fillRect(4 * k, cv.height - 4 * k - lh * LEGEND.length - 6 * k, 330 * k, lh * LEGEND.length + 6 * k);
  LEGEND.forEach(([t, col], i) => { x.fillStyle = col; x.fillText(t, 10 * k, cv.height - 10 * k - lh * (LEGEND.length - 1 - i)); });
}

function Feed({ path, gap, rot, labels }: { path: string; gap: number; rot: number; labels?: Labels }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const rotRef = useRef(rot);
  rotRef.current = rot;
  const labRef = useRef(labels);
  labRef.current = labels;
  useEffect(() => {
    let dead = false;
    let timer = 0;
    const next = (ms: number) => { if (!dead) timer = window.setTimeout(load, ms); };
    const load = () => {
      const im = new Image();
      im.onload = () => { if (ref.current) drawTo(ref.current, im, rotRef.current, labRef.current); next(gap); };
      im.onerror = () => next(600);
      im.src = `${G(path)}?${Date.now()}`;
    };
    load();
    return () => { dead = true; clearTimeout(timer); };
  }, [path, gap]);
  return <canvas ref={ref} width={960} height={540} className="g-canvas" />;
}

function markerTag(live: boolean, ids: number[]): [string, string] {
  if (!live) return ["", "no signal"];
  return [ids.length >= 3 ? "ok" : ids.length === 2 ? "warn" : "bad", `markers ${ids.length ? ids.join(", ") : "none"}`];
}

const saved = (k: string, d: number) => {
  try { return Number(localStorage.getItem(k) ?? d); } catch { return d; }
};

export default function Game() {
  const [s, setS] = useState<GameStatus | null>(null);
  const [offline, setOffline] = useState("");
  const [rotA, setRotA] = useState(() => saved("rotA", 90));
  const [rotB, setRotB] = useState(() => saved("rotB", 0));
  const [layers, setLayers] = useState("10");
  const [removed, setRemoved] = useState("");
  const [coverage, setCoverage] = useState("");
  const [judging, setJudging] = useState("");
  const lbl = useRef<HTMLInputElement>(null);
  const tlRef = useRef<HTMLInputElement>(null);
  const trRef = useRef<HTMLInputElement>(null);

  // --- status, 2.5 times a second -------------------------------------------------------
  useEffect(() => {
    let dead = false;
    let timer = 0;
    const poll = async () => {
      try {
        const r = await fetch(G("status.json"), { cache: "no-store" });
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail ?? `HTTP ${r.status}`);
        const j = (await r.json()) as GameStatus;
        if (dead) return;
        setS(j);
        setOffline("");
        if (document.activeElement !== tlRef.current && document.activeElement !== trRef.current) {
          setLayers(String(j.tower.layers));
          setRemoved(j.tower.removed.map((b) => `L${b[0] + 1}S${b[1]}`).join(" "));
        }
      } catch (e) {
        if (!dead) setOffline((e as Error).message);
      }
      if (!dead) timer = window.setTimeout(poll, 400);
    };
    poll();
    return () => { dead = true; clearTimeout(timer); };
  }, []);

  const rotate = (which: "a" | "b") => {
    if (which === "a") setRotA((r) => { const n = (r + 90) % 360; try { localStorage.setItem("rotA", String(n)); } catch { /* private mode */ } return n; });
    else setRotB((r) => { const n = (r + 90) % 360; try { localStorage.setItem("rotB", String(n)); } catch { /* private mode */ } return n; });
  };
  const cmd = (name: string) => {
    post(`cmd/${name}`);
    if (name === "mark") setTimeout(() => lbl.current?.focus(), 150);
  };

  // Space = MARK, R / T = rotate (not while typing in a box)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement | null)?.tagName === "INPUT") return;
      if (e.key === "r" || e.key === "R") rotate("a");
      else if (e.key === "t" || e.key === "T") rotate("b");
      else if (e.code === "Space") { e.preventDefault(); cmd("mark"); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const apply = () => {
    const rem = [...removed.matchAll(/L\s*(\d+)\s*S\s*(\d)/gi)].map((m) => [Number(m[1]) - 1, Number(m[2])]);
    post("cmd/tower", { layers: Number(layers), removed: rem });
  };
  const showCoverage = async () => {
    setCoverage("measuring...");
    try { setCoverage(await (await fetch(G("coverage.txt"))).text()); } catch { setCoverage("could not reach the game server"); }
  };
  const judge = async () => {
    setJudging("starting...");
    try {
      const info = await api<{ url: string }>("/game_info");
      await api("/api/follow", { body: { url: info.url } });
      location.hash = "#/live";
    } catch (e) { setJudging((e as Error).message); }
  };

  if (!s) {
    return (
      <div className="g-offline">
        <h2>Game console</h2>
        <p>{offline ? <>Cannot reach the game server: <code>{offline}</code></> : "Connecting to the game server..."}</p>
        <p className="muted">Start it with <code>.venv/bin/python iphone_saved.py</code> in the lidar-capture folder
          (this page talks to it on port 8000).</p>
      </div>
    );
  }

  const live = s.preview_age !== null && s.preview_age < 2;
  const aLive = s.android_age !== null && s.android_age < 10;
  const rec: [string, string] = !live ? ["idle", "no phone preview"] : s.capturing ? ["bad", "● RECORDING"] : ["ok", "ready – not recording"];
  const mk: [string, string] = !s.cv ? ["idle", "marker check off (run with .venv)"] : !live ? ["idle", "markers: –"]
    : s.markers.length >= 3 ? ["ok", `${s.markers.length} / 4 markers`]
    : s.markers.length === 2 ? ["warn", "2 / 4 markers – borderline"] : ["bad", `${s.markers.length} / 4 markers – move phone`];
  const am: [string, string] = aLive
    ? [s.android_markers.length >= 3 ? "ok" : s.android_markers.length === 2 ? "warn" : "bad",
       `markers ${s.android_markers.join(", ") || "none"}`]
    : ["idle", "not connected"];
  const [aCls, aTxt] = markerTag(live, s.markers);
  const [bCls, bTxt] = markerTag(aLive, s.android_markers);
  const dataOn = !!s.dataset?.on;
  const ll = s.last_label, st = s.label_stats;
  const labelCls = ll ? (ll.status === "error" ? "bad" : ll.status === "done" ? (ll.confident ? "ok" : "warn") : "idle")
    : s.marks_pending ? "warn" : "idle";
  const labelTxt = ll ? `${ll.message ?? ll.status ?? ""}${st?.moves ? `   [${st.moves - st.wrong}/${st.moves} right so far; no correction counts as right]` : ""}`
    : s.marks_pending ? "move marked: type left / middle / right" : "no label yet";
  const ov = s.overlay ?? s.android_overlay;
  const au = s.autotower ?? {};
  const tm = s.tracker_msg ?? "";
  const tmCls = /could not|error|COLLAPSED/.test(tm) ? "bad" : tm.startsWith("ready") || tm.startsWith("move") ? "ok" : "idle";

  return (
    <div className="g-wrap">
      {(s.game_over || s.no_moves) && (
        <div className="g-alert">
          {s.game_over ? `GAME OVER — ${s.game_over.reason}` : "NO MOVES LEFT — every remaining block would topple the tower"}
        </div>
      )}

      <div className="g-view">
        <div className="g-pane">
          <div className="g-plabel">iPhone · LiDAR <span className={`g-tag ${aCls}`}>{aTxt}</span></div>
          <Feed path="preview.jpg" gap={10} rot={rotA} labels={s.labels?.iphone} />
        </div>
        <div className="g-pane">
          <div className="g-plabel">Android <span className={`g-tag ${bCls}`}>{bTxt}</span></div>
          <Feed path="android.jpg" gap={120} rot={rotB} labels={s.labels?.android} />
        </div>
      </div>

      <aside className="g-side">
        <h3>Capture</h3>
        <div className={`g-chip ${rec[0]}`}>{rec[1]}</div>
        <button className="g-btn" onClick={() => cmd("start")}>START recording</button>
        <button className="g-btn g-mark" onClick={() => cmd("mark")}>MARK move done &nbsp;(Space)</button>
        <button className="g-btn" onClick={() => cmd("stop")}>STOP recording</button>
        <p className="g-hint">Control everything from here. Don't touch the phone.</p>

        <h3 className="g-warnh">After MARK: which block did you pull?</h3>
        <p className="g-hint">After MARK, type <b>left</b>, <b>middle</b> or <b>right</b> (as in the iPhone picture), then Enter.
          Add a number to say the layer yourself, e.g. <b>middle 5</b>. A bare number corrects the last answer;
          type <b>fell</b> if the tower fell. Needs data capture ON.</p>
        <input ref={lbl} className="g-input" placeholder="left / middle / right  [layer]" autoComplete="off"
          onKeyDown={(e) => {
            if (e.key !== "Enter") return;
            const v = e.currentTarget.value.trim();
            if (v) { post("cmd/label", { text: v }); e.currentTarget.value = ""; }
          }} />
        <div className={`g-chip small ${labelCls}`}>{labelTxt}</div>
        <div className="g-line"><span>Frames saved</span><b>{s.saved}</b></div>
        <div className="g-line"><span>Moves marked</span><b>{s.marks}</b></div>

        <h3>Setup</h3>
        <div className={`g-chip ${mk[0]}`}>{mk[1]}</div>
        <div className="g-line"><span>Marker IDs</span><span>{s.markers.length ? s.markers.join(", ") : "–"}</span></div>
        <div className="g-line"><span>Phone tracking</span><span>{s.tracking}</span></div>

        <h3>Camera 2 (Android)</h3>
        <div className={`g-chip ${am[0]}`}>{am[1]}</div>

        <h3>Training data</h3>
        <button className={`g-btn ${dataOn ? "g-stop" : "g-blue"}`} onClick={() => post(`cmd/dataset/${dataOn ? "off" : "on"}`)}>
          {dataOn ? "■ Stop data capture" : "Start data capture"}
        </button>
        <p className="g-hint">
          {dataOn ? `RECORDING DATA · this run: ${s.dataset?.frames ?? 0} frames, ${s.dataset?.crops ?? 0} crops · all runs: ${s.dataset?.total_crops ?? 0} crops`
            : `off · ${s.dataset?.total_crops ?? 0} crops saved so far. Records both cameras and cuts out every block face.`}
        </p>

        <h3>Tower (red / green / amber)</h3>
        <div className="g-line"><span>Layers</span>
          <input ref={tlRef} className="g-small" type="number" min={1} max={18} value={layers} onChange={(e) => setLayers(e.target.value)} /></div>
        <div className="g-line"><span>Removed</span>
          <input ref={trRef} className="g-small wide" placeholder="e.g. L5S1, L7S0" value={removed} onChange={(e) => setRemoved(e.target.value)} /></div>
        <button className="g-btn" onClick={apply}>Apply</button>
        <p className="g-hint">{ov
          ? `green ${ov.green}${ov.amber ? `  ·  amber (loose) ${ov.amber}` : ""}  ·  red ${ov.red}${s.height_info ? `  ·  LiDAR height ≈ ${s.height_info.lidar_layers} layers` : ""}`
          : "overlay needs 2+ markers in view"}</p>
        <p className="g-hint">auto-detect: {au.status ?? "–"}{au.note ? ` · ${au.note}` : ""}</p>
        {s.tower_fit && (
          <>
            <div className={`g-chip small ${s.tower_fit.converged && s.tower_fit.inliers >= 0.55 ? "ok" : "warn"}`}>
              {s.tower_fit.converged ? "overlay fitted to the real tower (LiDAR)" : "fitting the overlay to the tower…"}
            </div>
            <p className="g-hint">
              LiDAR measures {s.tower_fit.height_mm.toFixed(0)} mm ≈ {s.tower_fit.layers_float.toFixed(1)} layers
              · {(s.tower_fit.inliers * 100).toFixed(0)}% of the tower's points fit within 4 mm
              · frame tilt corrected {s.tower_fit.tilt_deg.toFixed(1)}°, shift ({s.tower_fit.shift_mm.map((x) => x.toFixed(0)).join(", ")}) mm
            </p>
            {(Math.abs(s.tower_fit.layers_float - Math.round(s.tower_fit.layers_float)) <= 0.4
              ? [Math.round(s.tower_fit.layers_float)]
              : [Math.floor(s.tower_fit.layers_float), Math.ceil(s.tower_fit.layers_float)]   // halfway: let the player pick
            ).map((n) => (
              <button key={n} className="g-btn g-blue" disabled={n === s.tower.layers}
                onClick={() => post("cmd/tower", { layers: n, removed: s.tower.removed })}>
                Use {n} layers (from the LiDAR height)
              </button>
            ))}
          </>
        )}
        <button className="g-btn" onClick={() => cmd("redetect")}>Re-detect tower (after rebuilding)</button>
        <button className="g-btn g-blue" onClick={showCoverage}>Check coverage (which blocks can I judge?)</button>
        {coverage && <pre className="g-pre">{coverage}</pre>}
        <div className={`g-chip small ${tmCls}`}>{tm || "tracker idle"}</div>
        <p className="g-hint">L = layer from the bottom (1 = bottom), S = slot 0, 1 or 2. Green = can be pulled, amber = loose, red = no.</p>

        <h3>Third Eye referee</h3>
        <button className="g-btn g-primary" onClick={judge}>Judge this game with Third Eye rules ▸</button>
        {judging && <p className="g-hint">{judging}</p>}
        <p className="g-hint">Opens the Live tab: his rule checker and 3D tower follow the pulls confirmed here, and the referee can answer questions.</p>

        <h3>Link</h3>
        <div className="g-line"><span>Preview</span><span>{live ? `${s.preview_fps.toFixed(1)} fps` : "–"}</span></div>
        <div className="g-line"><span>Uploads</span><span>{s.upload_rate.toFixed(1)} files/s ({s.files} total)</span></div>
        <div className="g-line"><span>Last upload</span><span>{s.upload_age === null ? "–" : `${s.upload_age.toFixed(1)} s ago`}</span></div>
        <button className="g-btn" onClick={() => rotate("a")}>Rotate iPhone view (R)</button>
        <button className="g-btn" onClick={() => rotate("b")}>Rotate Android view (T)</button>
        <p className="g-hint">Green = at least 3 of 4 markers visible. Need 2+ for the tower pose; 4 is best. Keep the phones still while recording.</p>
      </aside>
    </div>
  );
}
