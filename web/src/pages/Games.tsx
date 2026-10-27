// Real recorded Jenga games: for every pull the player tapped, the tapped column
// before and after, the layer we found, the teammate's finder's answer and the
// confirmed layer. Data from scripts/make_demo_data.py, served at /demo.

import { useEffect, useMemo, useState } from "react";
import { API } from "../api";
import { Banner } from "../App";

interface Face { layer: number; before: string; after: string; z: number }
interface Pull {
  session: string; seq: number; pos: string; truth: number; corrected: boolean;
  teammate: number; teammate_confident: boolean; ours: number | null; camera: string | null;
  column: Face[];
}
interface Data {
  summary: { pulls: number; ours_top1: number; teammate_top1: number; sessions: number };
  pulls: Pull[];
}
type Filter = "all" | "right" | "wrong";

const pct = (x: number) => `${Math.round(x * 100)} %`;
const clock = (session: string) => session.replace(/^live_\d{8}_(\d\d)(\d\d)\d\d$/, "$1:$2");

export default function Games() {
  const [data, setData] = useState<Data | null>(null);
  const [err, setErr] = useState("");
  const [filter, setFilter] = useState<Filter>("all");

  useEffect(() => {
    fetch(`${API}/demo/pulls.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`))))
      .then(setData)
      .catch(() => setErr("No real-game data on this server (run scripts/make_demo_data.py)."));
  }, []);

  const shown = useMemo(() => (data?.pulls ?? []).filter((p) =>
    filter === "all" || (filter === "right") === (p.ours === p.truth)), [data, filter]);

  if (err) return <Banner kind="warn">{err}</Banner>;
  if (!data) return <p className="loading">Loading…</p>;
  const s = data.summary;
  // from the pulls themselves, so the rounding matches the Overview page
  const exact = (who: "ours" | "teammate") =>
    data.pulls.filter((p) => p[who] === p.truth).length / Math.max(data.pulls.length, 1);
  return (
    <div className="games">
      <section className="games-head">
        <div>
          <h2>Real games · which layer was pulled?</h2>
          <p className="lede small">
            {s.sessions} games recorded with an iPhone and an Android phone. After each pull the player
            taps once and says <i>left</i>, <i>middle</i> or <i>right</i>; the finder works out the layer
            by comparing every block face in that column before and after.
          </p>
        </div>
        <div className="scores">
          <div className="stat ok"><div className="value">{pct(exact("ours"))}</div><div className="label">ours, exact layer</div></div>
          <div className="stat"><div className="value">{pct(exact("teammate"))}</div><div className="label">teammate's finder*</div></div>
          <div className="stat"><div className="value">{s.pulls}</div><div className="label">pulls</div></div>
        </div>
      </section>
      <p className="muted foot">
        * The truth is the player's typed correction, or the teammate's answer if the player let it stand,
        so the teammate's number is optimistic. Layers count from 1 at the bottom.
      </p>

      <div className="seg" role="tablist">
        {(["all", "right", "wrong"] as Filter[]).map((f) => (
          <button key={f} className={filter === f ? "on" : ""} onClick={() => setFilter(f)}>
            {f === "all" ? `All ${data.pulls.length}` : f === "right" ? "We got it" : "We missed"}
          </button>
        ))}
      </div>

      <div className="pulls">
        {shown.map((p) => <PullCard key={`${p.session}-${p.seq}`} p={p} />)}
      </div>
    </div>
  );
}

function PullCard({ p }: { p: Pull }) {
  const right = p.ours === p.truth;
  const top = Math.max(...p.column.map((f) => f.z), 1);
  return (
    <section className={`pull ${right ? "right" : "wrong"}`}>
      <header>
        <b>Game {clock(p.session)} · pull {p.seq}</b>
        <span className="muted">{p.pos} column · {p.camera ?? "no camera"}</span>
      </header>
      <div className="verdicts">
        <span className="chip truth">Layer {p.truth}{p.corrected && <small> typed</small>}</span>
        <span className={`chip ${right ? "ok" : "bad"}`}>Ours {p.ours ?? "–"} {right ? "✓" : "✗"}</span>
        <span className={`chip ${p.teammate === p.truth ? "ok" : "bad"} soft`}>
          Teammate {p.teammate} {p.teammate === p.truth ? "✓" : "✗"}
        </span>
      </div>
      <div className="column">
        <div className="colhead"><span /><span>before</span><span>after</span><span>change</span></div>
        {p.column.map((f) => (
          <div key={f.layer} className={`face${f.layer === p.truth ? " is-truth" : ""}${f.layer === p.ours ? " is-ours" : ""}`}>
            <span className="n">{f.layer}</span>
            <img src={`${API}/demo/${f.before}`} alt={`layer ${f.layer} before`} loading="lazy" />
            <img src={`${API}/demo/${f.after}`} alt={`layer ${f.layer} after`} loading="lazy" />
            <span className="bar"><i style={{ width: `${Math.max(0, f.z) / top * 100}%` }} /></span>
          </div>
        ))}
      </div>
    </section>
  );
}
