// The pieces of a session view: tower legality panel, jigsaw board, checklist,
// rules, alerts and the report. Shared by the live page and history.

import type { Row, Rule, Tower as TowerT } from "../api";

export function TowerPanel({ t }: { t: TowerT }) {
  const legal = new Set(t.legal_slots.map((s) => s.join(",")));
  const height = Math.max(t.rows.length, ...t.legal_slots.map((s) => s[0] + 1));
  const layers = Array.from({ length: height }, (_, L) => L).reverse();
  return (
    <>
      <div className="tower">
        {layers.map((L) => {
          const row = t.rows[L] || [0, 0, 0];
          return (
            <div key={L} className={`layer ${L === t.top_complete_layer ? "tcl" : ""}`}>
              <span className="n">{L}</span>
              {row.map((v, s) => (
                <span key={s} className={`blk ${v ? "" : legal.has(`${L},${s}`) ? "legal" : "empty"}`} />
              ))}
            </div>
          );
        })}
      </div>
      <p className="legend">
        layer {t.top_complete_layer} is the top complete layer: take from below it · green outline:
        where the block may go
      </p>
    </>
  );
}

const CELL = /^r(\d+)c(\d+)$/;

export function isBoard(rows: Row[]): boolean {
  return rows.length > 0 && rows.every((r) => CELL.test(r.id));
}

export function Board({ rows, next }: { rows: Row[]; next: string | null }) {
  const pos = rows.map((r) => r.id.match(CELL)!.slice(1).map(Number));
  const nr = Math.max(...pos.map((p) => p[0])) + 1;
  const nc = Math.max(...pos.map((p) => p[1])) + 1;
  const byCell = Object.fromEntries(rows.map((r) => [r.id, r]));
  const cells = [];
  for (let r = 0; r < nr; r++)
    for (let c = 0; c < nc; c++) {
      const id = `r${r}c${c}`;
      cells.push(<div key={id} className={`cell ${byCell[id]?.status ?? ""} ${next === id ? "next" : ""}`}>{id}</div>);
    }
  return (
    <>
      <div className="board" style={{ gridTemplateColumns: `repeat(${nc}, minmax(34px, 64px))` }}>{cells}</div>
      <p className="legend">green placed · red flagged · dashed next</p>
    </>
  );
}

const MARKS: Record<string, string> = { done: "✓", error: "!", active: "~", pending: "·" };

export function Checklist({ rows }: { rows: Row[] }) {
  return (
    <ol className="check">
      {rows.map((r) => (
        <li key={r.id} className={r.status}>
          <span className="mark">{MARKS[r.status]}</span>
          <span className="id">{r.id}</span>
          <span>{r.instruction}</span>
        </li>
      ))}
    </ol>
  );
}

export function Rules({ rules }: { rules: Rule[] }) {
  return (
    <ol className="check">
      {rules.map((r) => (
        <li key={r.id}><span className="mark">§</span><span className="id">{r.id}</span><span>{r.text}</span></li>
      ))}
    </ol>
  );
}

export interface AlertItem { t: number; code: string; severity: string; text: string; target: string }

export function Alerts({ alerts }: { alerts: AlertItem[] }) {
  if (!alerts.length) return <p className="muted">None yet.</p>;
  return (
    <div className="alerts">
      {[...alerts].reverse().map((a, i) => (
        <div key={i} className={`alert ${a.severity}`}>
          <div className="top">
            <span className="code">
              {a.code.replaceAll("_", " ")}{a.target && a.target !== "move" ? ` · ${a.target}` : ""}
            </span>
            <span>{a.t.toFixed(1)}s</span>
          </div>
          {a.text}
        </div>
      ))}
    </div>
  );
}

/** The report's small Markdown subset: headings, bullets, bold, rules. */
export function Markdown({ text }: { text: string }) {
  const out: React.ReactNode[] = [];
  let bullets: React.ReactNode[] = [];
  const inline = (s: string) =>
    s.split(/\*\*(.+?)\*\*/g).map((part, i) => (i % 2 ? <b key={i}>{part}</b> : part));
  const flush = () => {
    if (bullets.length) out.push(<ul key={`u${out.length}`}>{bullets}</ul>);
    bullets = [];
  };
  text.split("\n").forEach((l, i) => {
    if (l.startsWith("- ")) { bullets.push(<li key={i}>{inline(l.slice(2))}</li>); return; }
    flush();
    if (l.startsWith("# ")) out.push(<h3 key={i}>{inline(l.slice(2))}</h3>);
    else if (l.startsWith("## ")) out.push(<h4 key={i}>{inline(l.slice(3))}</h4>);
    else if (l === "---") out.push(<hr key={i} />);
    else if (l.trim()) out.push(<p key={i}>{inline(l)}</p>);
  });
  flush();
  return <div className="md">{out}</div>;
}
