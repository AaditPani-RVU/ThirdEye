// A phone streaming its camera here (the capture app, thirdeye/server/phone.py):
// the latest frame with the mat markers it found, and START / STOP / MARK.

import { useEffect, useState } from "react";
import { API, api } from "../api";
import { Banner } from "../App";

interface Phone {
  live: boolean; age: number | null; fps: number; markers: number[]; size: [number, number] | null;
  recording: boolean; folder: string | null; saved: number; marks: number;
}

export default function Camera() {
  const [st, setSt] = useState<Phone | null>(null);
  const [off, setOff] = useState(false);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const poll = () => api<Phone>("/api/phone").then((s) => { setSt(s); setOff(false); })
      .catch(() => setOff(true));
    poll();
    const status = window.setInterval(poll, 1000);
    const frame = window.setInterval(() => setTick((t) => t + 1), 200);
    return () => { clearInterval(status); clearInterval(frame); };
  }, []);

  const cmd = (c: string) => fetch(`${API}/cmd/${c}`, { method: "POST" }).catch(() => {});
  const host = location.hostname.endsWith(".ts.net") ? "100.122.110.15" : location.hostname;

  if (off) return <Banner kind="warn">Phone camera is not available on this server (sign-in is on).</Banner>;
  const ok = (st?.markers.length ?? 0) >= 2;
  return (
    <div className="camera">
      <section className="feed">
        <div className="feedhead">
          <span className={`dot ${st?.live ? "on" : ""}`} />
          <b>{st?.live ? "Phone connected" : "Waiting for the phone"}</b>
          {st?.live && <span className="muted">{st.fps.toFixed(0)} fps{st.size && ` · ${st.size[0]}×${st.size[1]}`}</span>}
          {st?.recording && <span className="rec">● REC</span>}
        </div>
        <div className="frame">
          {st?.live
            ? <img src={`${API}/api/phone/frame.jpg?t=${tick}`} alt="phone camera" />
            : <div className="empty">
                <p>In the capture app on the phone, type</p>
                <code className="big">{host}</code>
                <p className="muted">Tailscale must be on on the phone. Frames appear here as soon as it connects.</p>
              </div>}
        </div>
      </section>

      <section className="side">
        <h2>Mat markers</h2>
        <p className={`markers ${ok ? "ok" : "warn"}`}>
          {st?.markers.length ? st.markers.map((m) => <span key={m}>{m}</span>) : <i>none in view</i>}
        </p>
        <p className="muted">{ok ? "Enough to fix the tower's pose." : "Need at least two of the four mat markers in view."}</p>

        <h2 className="gap">Recording</h2>
        <div className="row">
          {st?.recording
            ? <button onClick={() => cmd("stop")}>Stop</button>
            : <button className="primary" onClick={() => cmd("start")} disabled={!st?.live}>Start</button>}
          <button onClick={() => cmd("mark")} disabled={!st?.recording}>Mark pull</button>
        </div>
        <dl className="facts">
          <dt>Frames saved</dt><dd>{st?.saved ?? 0}</dd>
          <dt>Pulls marked</dt><dd>{st?.marks ?? 0}</dd>
          {st?.folder && <><dt>Folder</dt><dd><code>data/phone/{st.folder}</code></dd></>}
        </dl>
        <p className="muted">The phone's own START / STOP / MARK buttons do the same.</p>
      </section>
    </div>
  );
}
