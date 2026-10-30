// Sign in -> Overview -> Game (the game server's console) -> Live (replay or edge) -> Real games -> Procedures (pick, compile, edit) -> History.
// The page only draws what the server sends; nothing about a session is decided here.

import { useEffect, useState } from "react";
import type { User } from "firebase/auth";
import { api, type Me } from "./api";
import { firebaseConfigured, signIn, signOut, watchUser } from "./auth";
import Live from "./pages/Live";
import Procedures from "./pages/Procedures";
import History from "./pages/History";
import Overview from "./pages/Overview";
import Games from "./pages/Games";
import Camera from "./pages/Camera";
import Game from "./pages/Game";

type Tab = "overview" | "game" | "live" | "camera" | "games" | "procedures" | "history";
const TABS: [Tab, string][] = [["overview", "Overview"], ["game", "Game"], ["live", "Live"], ["camera", "Camera"], ["games", "Real games"],
  ["procedures", "Procedures"], ["history", "History"]];

function tabFromHash(): Tab {
  const t = location.hash.replace(/^#\/?/, "").split("/")[0];
  return (TABS.map(([k]) => k) as string[]).includes(t) ? (t as Tab) : "overview";
}

export default function App() {
  const [config, setConfig] = useState<{ auth: boolean } | null>(null);
  const [configError, setConfigError] = useState("");
  const [fbUser, setFbUser] = useState<User | null | undefined>(undefined);
  const [me, setMe] = useState<Me | null>(null);
  const [tab, setTab] = useState<Tab>(tabFromHash);

  useEffect(() => {
    api<{ auth: boolean }>("/api/config").then(setConfig)
      .catch((e) => setConfigError(`Cannot reach the server: ${e.message}`));
    const onHash = () => setTab(tabFromHash());
    window.addEventListener("hashchange", onHash);
    const unwatch = watchUser(setFbUser);
    return () => { window.removeEventListener("hashchange", onHash); unwatch(); };
  }, []);

  const needsSignIn = config?.auth && !fbUser;
  useEffect(() => {
    if (!config || needsSignIn || fbUser === undefined && firebaseConfigured) return;
    api<Me>("/api/me").then(setMe).catch(() => setMe(null));
  }, [config, needsSignIn, fbUser]);

  if (configError) return <Banner kind="bad">{configError}</Banner>;
  if (!config || (firebaseConfigured && fbUser === undefined)) return <p className="loading">Loading…</p>;
  if (config.auth && !firebaseConfigured)
    return <Banner kind="bad">The server asks for sign-in, but this build has no Firebase config.</Banner>;
  if (needsSignIn) return <Login />;

  return (
    <>
      <header>
        <h1><span className="eye">●</span> Third Eye</h1>
        <nav>
          {TABS.map(([k, label]) => (
            <a key={k} href={`#/${k}`} className={tab === k ? "on" : ""}>{label}</a>
          ))}
        </nav>
        <span className="who">
          {me && me.uid !== "local" && <>{me.email}{!me.editor && " (viewer)"} ·{" "}
            <button className="link" onClick={() => signOut()}>sign out</button></>}
        </span>
      </header>
      <main>
        {tab === "overview" && <Overview />}
        {tab === "game" && <Game />}
        {tab === "live" && <Live />}
        {tab === "camera" && <Camera />}
        {tab === "games" && <Games />}
        {tab === "procedures" && <Procedures editor={me?.editor ?? false} />}
        {tab === "history" && <History editor={me?.editor ?? false} />}
      </main>
    </>
  );
}

function Login() {
  const [err, setErr] = useState("");
  return (
    <div className="login">
      <h1><span className="eye">●</span> Third Eye</h1>
      <p>Catches mistakes in a procedure as they happen, with rules an LLM reads from the manual.</p>
      <button className="primary" onClick={() => signIn().catch((e) => setErr(e.message))}>
        Sign in with Google
      </button>
      {err && <p className="err">{err}</p>}
    </div>
  );
}

export function Banner({ kind, children }: { kind: "bad" | "warn" | "info"; children: React.ReactNode }) {
  return <div className={`banner ${kind}`}>{children}</div>;
}
