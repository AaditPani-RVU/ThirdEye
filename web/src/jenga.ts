// What the Jenga rules and the stability rule say about a block; shared by the
// 3D tower and the live page (kept apart so three.js loads only with the 3D view).

import type { Tower } from "./api";

export type Mode = "legal" | "safe";
export type Verdict = "safe" | "topples" | "blocked";
export interface Flash { from?: [number, number]; to?: [number, number]; key: number }

/** What the rules (take only below the top complete layer) and stability say about pulling L,s. */
export function verdict(t: Tower, L: number, s: number): Verdict {
  if (L >= t.top_complete_layer) return "blocked";
  const r = [...(t.rows[L] ?? [0, 0, 0])];
  r[s] = 0;
  return r[1] || (r[0] && r[2]) ? "safe" : "topples";
}

export const VERDICT_TEXT: Record<Verdict, string> = {
  safe: "Safe to take",
  topples: "Allowed by the rules, but the layer would topple: it needs its middle block or both sides",
  blocked: "Not allowed: take only from below the top complete layer",
};
