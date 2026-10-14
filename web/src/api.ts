// The server's HTTP API and WebSocket, with the signed-in user's token attached.
// Types mirror thirdeye/server/protocol.py (v1) and thirdeye/server/app.py.

import { idToken } from "./auth";

export const API = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") || "";

export type Status = "pending" | "active" | "done" | "error";
export interface Row { id: string; status: Status; instruction: string }
export interface Rule { id: string; text: string; code: string }
export interface Tower {
  rows: number[][];
  top_layer: number;
  top_complete_layer: number;
  legal_slots: [number, number][];
}
export type Msg =
  | { v: 1; type: "hello"; procedure: string; kind: string; source: string; checklist: Row[]; rules: Rule[] }
  | { v: 1; type: "state"; t: number; checklist: Row[]; next: string | null; progress: string;
      tower?: Tower; perception?: "ok" | "lost" }
  | { v: 1; type: "alert"; t: number; code: string; severity: "alert" | "warning"; text: string;
      target: string; rule: string }
  | { v: 1; type: "frame"; t: number; jpeg: string }
  | { v: 1; type: "end"; summary: string; report: string; outcome: string | null }
  | { v: 1; type: "error"; message: string };

export interface SpecMeta {
  name: string; procedure?: string; kind?: string; state_model?: string; model?: string;
  created_by?: string; valid?: boolean; updated_at?: string;
}
export interface Problem { severity: string; code: string; target: string; message: string }
export interface SessionMeta {
  id: string; procedure: string; kind: string; spec?: string; source?: string; run?: string;
  started_at?: string; ended_at?: string; steps_done?: string | null; alert_count: number;
  red: number; outcome?: string | null; report?: string; created_by?: string;
}
export interface Me { uid: string; email: string; name: string; editor: boolean }

export class ApiError extends Error {
  constructor(public status: number, public detail: unknown) {
    super(typeof detail === "string" ? detail
      : (detail as { message?: string })?.message ?? `HTTP ${status}`);
  }
}

export async function api<T>(path: string, init: { method?: string; body?: unknown } = {}): Promise<T> {
  const token = await idToken();
  const res = await fetch(API + path, {
    method: init.method ?? (init.body === undefined ? "GET" : "POST"),
    headers: {
      ...(init.body === undefined ? {} : { "Content-Type": "application/json" }),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: init.body === undefined ? undefined : JSON.stringify(init.body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(res.status, (data as { detail?: unknown }).detail ?? res.statusText);
  return data as T;
}

export async function socketUrl(): Promise<string> {
  const base = API || location.origin;
  const url = new URL("/ws", base);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  const token = await idToken();
  if (token) url.searchParams.set("token", token);
  return url.toString();
}
