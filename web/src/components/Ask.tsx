// Ask the referee: a question about the live session, answered by the LLM from
// what the cameras and the checker produced (thirdeye/explain/ask.py).

import { useEffect, useState } from "react";
import { api } from "../api";

interface Answer {
  question: string; answer: string; cites: string[]; dropped: string[];
  context: string; author: string; seconds: number;
}

const SUGGESTED = [
  "Which blocks can I still take safely?",
  "Why was the last move flagged?",
  "Where can the next block go?",
];

/** `seed` asks a question from outside (an alert's Explain button); `n` makes each click new. */
export function Ask({ ready, seed }: { ready: boolean; seed?: { q: string; n: number } }) {
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [answers, setAnswers] = useState<Answer[]>([]);

  const send = async (question: string) => {
    if (!question.trim() || busy) return;
    setQ(question); setBusy(true); setErr("");
    try {
      const a = await api<Answer>("/api/ask", { body: { question } });
      setAnswers((xs) => [a, ...xs].slice(0, 5));
      setQ("");
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => { if (seed) send(seed.q); }, [seed?.n]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="ask">
      <p className="muted">
        The LLM answers from what the cameras saw and the checker decided. It never sees the video,
        and every answer cites the rules and alerts it rests on.
      </p>
      <form onSubmit={(e) => { e.preventDefault(); send(q); }} className="row">
        <input value={q} onChange={(e) => setQ(e.target.value)} disabled={!ready || busy}
               placeholder={ready ? "Ask about this game…" : "Start a session first"} />
        <button className="primary" disabled={!ready || busy || !q.trim()}>{busy ? "Thinking…" : "Ask"}</button>
      </form>
      <div className="chips">
        {SUGGESTED.map((s) => (
          <button key={s} className="chip" disabled={!ready || busy} onClick={() => send(s)}>{s}</button>
        ))}
      </div>
      {err && <p className="err">{err}</p>}
      {answers.map((a, i) => (
        <div key={answers.length - i} className="answer">
          <div className="q">{a.question}</div>
          <p>{a.answer}</p>
          <div className="chips">
            {a.cites.map((c) => <span key={c} className="chip ok" title="found in the session">✓ {c}</span>)}
            {a.dropped.map((c) => (
              <span key={c} className="chip bad" title="not in the session: dropped">✗ {c}</span>
            ))}
            {!a.cites.length && !a.dropped.length && <span className="chip soft">from the tower state</span>}
          </div>
          <details>
            <summary className="muted">What the LLM was given · {a.author} · {a.seconds.toFixed(1)} s</summary>
            <pre className="summary">{a.context}</pre>
          </details>
        </div>
      ))}
    </div>
  );
}
