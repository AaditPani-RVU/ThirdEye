// What Third Eye is, how it works, and where it stands -- the page to open first in a demo.
// The numbers are the committed evaluation results (eval/results, scripts/eval_*.py);
// they are quoted here, not recomputed.

const STAGES: [string, string][] = [
  ["Manual", "A rulebook or instruction sheet, in plain text"],
  ["LLM compiler", "Reads it into a procedure spec; validated, one repair round"],
  ["Cameras", "Mat markers fix the pose; depth and hands per frame"],
  ["World state", "Board of pieces, or the tower as a layer × slot lattice"],
  ["Checker", "Deterministic: every rule is checked on every move"],
  ["Alerts + report", "As it happens, then a report checked against the log"],
];

interface Stat { value: string; label: string; note: string; tone?: "ok" | "warn" }
const STATS: Stat[] = [
  { value: "15 / 16", label: "manuals compiled", tone: "ok",
    note: "Groq gpt-oss-120b. Step accuracy 1.00, Jenga rule accuracy 1.00. A local 8B model failed 9 of 16." },
  { value: "95 %", label: "report claims backed by the log", tone: "ok",
    note: "4 of 80 claims unsupported across 16 sessions (target ≤ 5 %)." },
  { value: "73 %", label: "pulled layer found, real games", tone: "ok",
    note: "49 pulls, 6 recorded games, colour only. 86 % in the top two. See Real games." },
  { value: "3 / 10", label: "pulls caught from depth, one camera", tone: "warn",
    note: "No false alarms at that setting, but one corner camera sees a pulled block as a 15–35 mm dip. Built for two." },
];

export default function Overview() {
  return (
    <div className="overview">
      <section className="hero">
        <p className="kicker">Third Eye</p>
        <h2 className="headline">Watches a physical procedure and flags mistakes as they happen.</h2>
        <p className="lede">
          The rules are not hand-coded: an LLM reads them from the manual, and a deterministic checker
          applies them to what the cameras see. Shown here on <b>Jenga</b> (rules about which blocks
          may be pulled) and <b>jigsaw</b> assembly (order, position, rotation).
        </p>
        <div className="cta">
          <a className="btn primary" href="#/live">Run a replay</a>
          <a className="btn" href="#/games">Real games</a>
          <a className="btn" href="#/procedures">Compiled rules</a>
        </div>
      </section>

      <section>
        <h2>How it works</h2>
        <ol className="flow">
          {STAGES.map(([name, what], i) => (
            <li key={name} className={i === 1 || i === 4 ? "key" : ""}>
              <span className="step">{i + 1}</span>
              <b>{name}</b>
              <span className="what">{what}</span>
            </li>
          ))}
        </ol>
      </section>

      <section>
        <h2>Where it stands</h2>
        <div className="stats">
          {STATS.map((s) => (
            <div key={s.label} className={`stat ${s.tone ?? ""}`}>
              <div className="value">{s.value}</div>
              <div className="label">{s.label}</div>
              <p className="note">{s.note}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="two">
        <div>
          <h2>Works today</h2>
          <ul className="ticks">
            <li>Rule compiler for Jenga, jigsaw and LEGO manuals, with validation and repair</li>
            <li>Checker and alerts on every scripted scenario, live in this page</li>
            <li>Layer finder on real recordings (iPhone + Android, colour crops)</li>
            <li>Session reports, with every claim checked against the log</li>
          </ul>
        </div>
        <div>
          <h2>Next</h2>
          <ul className="todo">
            <li>Two-camera depth recordings of full games, to score the depth path properly</li>
            <li>Feed the layer finder's answer into the live Jenga checker</li>
            <li>Re-pose a moving camera every frame</li>
          </ul>
        </div>
      </section>
    </div>
  );
}
