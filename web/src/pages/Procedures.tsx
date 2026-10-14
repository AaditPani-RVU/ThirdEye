// Procedures: pick a stored spec, or paste instructions and compile one with the
// LLM; view and edit the JSON, re-validate, save (FR-CMP-9, FR-UI-6).

import { useEffect, useState } from "react";
import { ApiError, api, type Problem, type SpecMeta } from "../api";
import { Banner } from "../App";

type Kind = "jigsaw" | "constraints" | "dag";
const KINDS: [Kind, string][] = [
  ["jigsaw", "Jigsaw instructions (ordered phases)"],
  ["constraints", "Game rulebook (Jenga: rules, no order)"],
  ["dag", "Build manual (LEGO, parked)"],
];

export default function Procedures({ editor }: { editor: boolean }) {
  const [specs, setSpecs] = useState<SpecMeta[]>([]);
  const [name, setName] = useState("");
  const [json, setJson] = useState("");
  const [problems, setProblems] = useState<Problem[] | null>(null);
  const [msg, setMsg] = useState<{ kind: "bad" | "info" | "warn"; text: string } | null>(null);
  const [text, setText] = useState("");
  const [kind, setKind] = useState<Kind>("jigsaw");
  const [cols, setCols] = useState(4);
  const [rows, setRows] = useState(3);
  const [model, setModel] = useState("");
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState<{ problems: string[]; raw?: string } | null>(null);

  const refresh = () => api<SpecMeta[]>("/api/specs").then(setSpecs).catch((e) => setMsg({ kind: "bad", text: e.message }));
  useEffect(() => { refresh(); }, []);

  const open = async (n: string) => {
    setMsg(null); setProblems(null); setFailed(null);
    const spec = await api<object>(`/api/specs/${encodeURIComponent(n)}`);
    setName(n); setJson(JSON.stringify(spec, null, 2));
  };

  const parse = (): object | null => {
    try { return JSON.parse(json); } catch (e) {
      setMsg({ kind: "bad", text: `Not valid JSON: ${(e as Error).message}` }); return null;
    }
  };

  const validate = async () => {
    const spec = parse(); if (!spec) return;
    const r = await api<{ valid: boolean; problems: Problem[] }>("/api/validate", { body: { spec } });
    setProblems(r.problems);
    setMsg(r.valid ? { kind: "info", text: "Valid." } : { kind: "bad", text: "The spec does not validate." });
  };

  const save = async () => {
    const spec = parse(); if (!spec || !name) { if (!name) setMsg({ kind: "bad", text: "Give the spec a name." }); return; }
    try {
      const r = await api<{ problems: Problem[] }>(`/api/specs/${encodeURIComponent(name)}`, { method: "PUT", body: { spec } });
      setProblems(r.problems); setMsg({ kind: "info", text: `Saved as ${name}.` }); refresh();
    } catch (e) {
      const d = (e as ApiError).detail as { problems?: Problem[] } | undefined;
      if (d?.problems) setProblems(d.problems);
      setMsg({ kind: "bad", text: (e as Error).message });
    }
  };

  const compile = async () => {
    setBusy(true); setMsg(null); setFailed(null); setProblems(null);
    try {
      const r = await api<{ spec: { procedure?: string }; warnings: string[] }>("/api/compile", {
        body: { text, kind, model: model || null, ...(kind === "jigsaw" ? { cols, rows } : {}) },
      });
      setJson(JSON.stringify(r.spec, null, 2));
      setName((n) => n || r.spec.procedure || "compiled");
      setMsg({ kind: r.warnings.length ? "warn" : "info",
               text: `Compiled and validated.${r.warnings.length ? ` Warnings: ${r.warnings.join("; ")}` : ""} Review it, then Save.` });
    } catch (e) {
      const d = (e as ApiError).detail as { problems?: string[]; raw?: string } | undefined;
      setFailed({ problems: d?.problems ?? [], raw: d?.raw });
      setMsg({ kind: "bad", text: `Compile failed: ${(e as Error).message}` });
    } finally { setBusy(false); }
  };

  return (
    <div className="procedures">
      <section>
        <h2>Specs</h2>
        <ul className="list">
          {specs.map((s) => (
            <li key={s.name} className={s.name === name ? "on" : ""}>
              <button className="link" onClick={() => open(s.name)}>{s.name}</button>
              <span className="muted"> {s.kind === "constraints" ? "rulebook" : s.state_model}
                {s.model ? ` · ${s.model}` : ""}{s.created_by && s.created_by !== "repository" ? ` · ${s.created_by}` : ""}</span>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h2>Compile from instructions</h2>
        {!editor && <p className="muted">Viewers can read specs; compiling and saving need an editor account.</p>}
        <label>What is it?</label>
        <select value={kind} onChange={(e) => setKind(e.target.value as Kind)}>
          {KINDS.map(([k, l]) => <option key={k} value={k}>{l}</option>)}
        </select>
        {kind === "jigsaw" && (
          <div className="row">
            <label>Columns<input type="number" min={2} max={40} value={cols} onChange={(e) => setCols(+e.target.value)} /></label>
            <label>Rows<input type="number" min={2} max={40} value={rows} onChange={(e) => setRows(+e.target.value)} /></label>
          </div>
        )}
        <label>Instructions or rulebook (plain text)</label>
        <textarea rows={10} value={text} onChange={(e) => setText(e.target.value)}
          placeholder="Put in the four corners, then the edge pieces, then fill the interior." />
        <label>Model <span className="muted">(blank = server default)</span></label>
        <input value={model} onChange={(e) => setModel(e.target.value)} placeholder="groq:openai/gpt-oss-120b" />
        <div className="row">
          <button className="primary" disabled={!editor || busy || !text.trim()} onClick={compile}>
            {busy ? "Compiling…" : "Compile"}
          </button>
        </div>
        {failed && (
          <div className="failed">
            <p>The LLM's spec still failed validation after one repair round:</p>
            <ul>{failed.problems.map((p, i) => <li key={i}>{p}</li>)}</ul>
            <div className="row">
              <button disabled={busy} onClick={compile}>Retry</button>
              {failed.raw && <button onClick={() => { setJson(failed.raw!); setFailed(null); }}>Edit by hand</button>}
            </div>
          </div>
        )}
      </section>

      <section className="editor">
        <h2>Spec</h2>
        {msg && <Banner kind={msg.kind}>{msg.text}</Banner>}
        <label>Name</label>
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="jigsaw_6x4" />
        <textarea className="json" rows={24} value={json} spellCheck={false} onChange={(e) => setJson(e.target.value)}
          placeholder="Open a spec on the left, or compile one." />
        <div className="row">
          <button onClick={validate} disabled={!json}>Validate</button>
          <button className="primary" onClick={save} disabled={!editor || !json}>Save</button>
        </div>
        {problems && problems.length > 0 && (
          <ul className="problems">
            {problems.map((p, i) => (
              <li key={i} className={p.severity}><b>{p.severity}</b> {p.code}{p.target && ` [${p.target}]`}: {p.message}</li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
