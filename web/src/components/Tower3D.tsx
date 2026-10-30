// The Jenga tower in 3D, from the occupancy the checker is judging (protocol `tower`).
// Blocks drop in when placed and slide out when pulled; the mat shows the tape
// square and the four markers as on the real board. Hover a block for what the
// rules and the stability rule say about it; click to ask the LLM.

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Tower } from "../api";
import { VERDICT_TEXT, verdict, type Flash, type Mode } from "../jenga";

// A Jenga block in mm (75 x 25 x 15), a hair smaller so the seams show.
const LEN = 75, WID = 25, HGT = 15;
const MARKER_C = 80, MARKER_MM = 40;   // marker centres and size, as on the board (tower_points.py)

const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";

function place(L: number, s: number): { pos: THREE.Vector3; rotY: number; out: THREE.Vector3 } {
  const off = (s - 1) * WID, y = L * HGT + HGT / 2;
  return L % 2 === 0
    ? { pos: new THREE.Vector3(0, y, off), rotY: 0, out: new THREE.Vector3(1, 0, 0) }
    : { pos: new THREE.Vector3(off, y, 0), rotY: Math.PI / 2, out: new THREE.Vector3(0, 0, 1) };
}

/** A soft wood grain, drawn once. */
function grain(): THREE.Texture {
  const c = document.createElement("canvas");
  c.width = 256; c.height = 64;
  const g = c.getContext("2d")!;
  g.fillStyle = "#fff"; g.fillRect(0, 0, 256, 64);
  for (let i = 0; i < 26; i++) {
    g.strokeStyle = `rgba(90,60,30,${0.05 + Math.random() * 0.1})`;
    g.lineWidth = 1 + Math.random() * 2;
    g.beginPath();
    const y0 = Math.random() * 64;
    for (let x = 0; x <= 256; x += 16) g.lineTo(x, y0 + Math.sin(x / 40 + i) * 3);
    g.stroke();
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/** A marker-like pattern (decorative: the real ids are on the paper). */
function markerTexture(id: number): THREE.Texture {
  const c = document.createElement("canvas");
  c.width = c.height = 120;
  const g = c.getContext("2d")!;
  g.fillStyle = "#fff"; g.fillRect(0, 0, 120, 120);
  g.fillStyle = "#111"; g.fillRect(10, 10, 100, 100);
  g.fillStyle = "#fff";
  for (let i = 0; i < 16; i++) {
    if (((id * 2654435761 + i * 40503) >>> (i % 7)) & 1) g.fillRect(26 + (i % 4) * 17, 26 + Math.floor(i / 4) * 17, 17, 17);
  }
  return new THREE.CanvasTexture(c);
}

function label(text: string, color: string, bold = false): THREE.Sprite {
  const c = document.createElement("canvas");
  c.width = 96; c.height = 48;
  const g = c.getContext("2d")!;
  g.font = `${bold ? 700 : 500} 30px system-ui, sans-serif`;
  g.fillStyle = color; g.textAlign = "right"; g.textBaseline = "middle";
  g.fillText(text, 92, 26);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthWrite: false }));
  s.scale.set(16, 8, 1);
  return s;
}

interface Tween { t0: number; dur: number; step: (k: number) => void; done?: () => void }

interface Scene3 {
  renderer: THREE.WebGLRenderer; scene: THREE.Scene; camera: THREE.PerspectiveCamera;
  controls: OrbitControls; tower: THREE.Group; ghosts: THREE.Group; labels: THREE.Group;
  marks: THREE.Group; blocks: Map<string, THREE.Mesh>; tweens: Tween[]; geo: THREE.BoxGeometry;
  edges: THREE.EdgesGeometry; wood: THREE.Texture; first: boolean; height: number;
}

const ease = (k: number) => 1 - Math.pow(1 - k, 3);

/** Aim at the middle of the tower from a three-quarter view, far enough to see all of it and the board. */
function frame(st: Scene3): void {
  const h = Math.max(st.height, 6) * HGT;
  const fov = (st.camera.fov * Math.PI) / 180;
  const dist = Math.max((h / 0.72) / (2 * Math.tan(fov / 2)), 620) / Math.min(1, st.camera.aspect);
  st.controls.target.set(0, h / 2, 0);
  st.camera.position.copy(new THREE.Vector3(0.62, 0.42, 0.66).normalize().multiplyScalar(dist)).add(st.controls.target);
}

export default function Tower3D({ tower, mode, flash, onPick }: {
  tower: Tower; mode: Mode; flash?: Flash; onPick?: (L: number, s: number, kind: "block" | "ghost") => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const s3 = useRef<Scene3 | null>(null);
  const towerRef = useRef(tower);
  towerRef.current = tower;
  const pickRef = useRef(onPick);
  pickRef.current = onPick;
  const [tip, setTip] = useState<{ x: number; y: number; title: string; text: string; tone: string } | null>(null);
  const [spin, setSpin] = useState(true);

  // --- one-time setup -------------------------------------------------------------------
  useEffect(() => {
    const el = host.current!;
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    el.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, 1, 1, 5000);
    camera.position.set(260, 260, 320);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotate = !still;
    controls.autoRotateSpeed = 0.8;
    controls.minDistance = 150; controls.maxDistance = 2000;
    controls.maxPolarAngle = Math.PI * 0.49;
    controls.addEventListener("start", () => { controls.autoRotate = false; setSpin(false); });

    scene.add(new THREE.HemisphereLight(0xffffff, 0x8a7560, 2.1));
    const sun = new THREE.DirectionalLight(0xffffff, 1.8);
    sun.position.set(90, 700, 220);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -260, right: 260, top: 260, bottom: -260, near: 10, far: 1600 });
    sun.shadow.radius = 6;
    scene.add(sun);

    // The board: cardboard, the tape square, the four markers.
    const mat = new THREE.Mesh(new THREE.BoxGeometry(300, 4, 300),
      new THREE.MeshStandardMaterial({ color: 0xc4a07a, roughness: 0.95 }));
    mat.position.y = -2; mat.receiveShadow = true;
    scene.add(mat);
    const tape = new THREE.MeshStandardMaterial({ color: 0xeee6d0, roughness: 0.9 });
    const half = LEN / 2 + 9;
    for (const [w, d, x, z] of [[LEN + 36, 18, 0, half], [LEN + 36, 18, 0, -half], [18, LEN, half, 0], [18, LEN, -half, 0]]) {
      const strip = new THREE.Mesh(new THREE.BoxGeometry(w, 0.6, d), tape);
      strip.position.set(x, 0.3, z); strip.receiveShadow = true;
      scene.add(strip);
    }
    [[-1, 1], [-1, -1], [1, -1], [1, 1]].forEach(([sx, sz], id) => {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(MARKER_MM, MARKER_MM),
        new THREE.MeshStandardMaterial({ map: markerTexture(id), roughness: 0.8 }));
      m.rotation.x = -Math.PI / 2;
      m.position.set(sx * MARKER_C, 0.4, sz * MARKER_C);
      m.receiveShadow = true;
      scene.add(m);
    });

    const groups = [new THREE.Group(), new THREE.Group(), new THREE.Group(), new THREE.Group()];
    groups.forEach((g) => scene.add(g));
    const [towerG, ghosts, labels, marks] = groups;
    const geo = new THREE.BoxGeometry(LEN - 0.6, HGT - 0.6, WID - 0.8);
    s3.current = {
      renderer, scene, camera, controls, tower: towerG, ghosts, labels, marks,
      blocks: new Map(), tweens: [], geo, edges: new THREE.EdgesGeometry(geo), wood: grain(),
      first: true, height: 0,
    };

    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / Math.max(h, 1);
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();

    // Hover and click.
    const ray = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    let down: { x: number; y: number } | null = null;
    const hit = (e: PointerEvent) => {
      const r = renderer.domElement.getBoundingClientRect();
      ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      ray.setFromCamera(ndc, camera);
      const found = ray.intersectObjects([...towerG.children, ...ghosts.children], false)
        .find((i) => i.object.userData.L !== undefined && !i.object.userData.leaving);
      return found ? { obj: found.object, x: e.clientX - r.left, y: e.clientY - r.top } : null;
    };
    const move = (e: PointerEvent) => {
      const h = hit(e);
      renderer.domElement.style.cursor = h ? "pointer" : "grab";
      if (!h) { setTip(null); return; }
      const { L, s, kind } = h.obj.userData as { L: number; s: number; kind: "block" | "ghost" };
      if (kind === "ghost") {
        setTip({ x: h.x, y: h.y, title: `Layer ${L} · slot ${s}`, text: "The next block may go here", tone: "ok" });
      } else {
        const v = verdict(towerRef.current, L, s);
        setTip({ x: h.x, y: h.y, title: `Layer ${L} · slot ${s}`, text: VERDICT_TEXT[v],
                 tone: v === "safe" ? "ok" : v === "topples" ? "warn" : "idle" });
      }
    };
    const press = (e: PointerEvent) => { down = { x: e.clientX, y: e.clientY }; };
    const release = (e: PointerEvent) => {
      if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) return;
      const h = hit(e);
      if (h) pickRef.current?.(h.obj.userData.L, h.obj.userData.s, h.obj.userData.kind);
    };
    const leave = () => setTip(null);
    renderer.domElement.addEventListener("pointermove", move);
    renderer.domElement.addEventListener("pointerdown", press);
    renderer.domElement.addEventListener("pointerup", release);
    renderer.domElement.addEventListener("pointerleave", leave);

    let raf = 0;
    const loop = (now: number) => {
      const st = s3.current!;
      st.tweens = st.tweens.filter((tw) => {
        const k = Math.min(Math.max((now - tw.t0) / tw.dur, 0), 1);
        if (now >= tw.t0) tw.step(ease(k));
        if (k >= 1) { tw.done?.(); return false; }
        return true;
      });
      const pulse = 0.5 + 0.5 * Math.sin(now / 280);
      st.ghosts.children.forEach((g) => {
        const m = (g as THREE.Mesh).material as THREE.MeshBasicMaterial;
        if (m.opacity !== undefined) m.opacity = 0.12 + 0.18 * pulse;
      });
      st.marks.children.forEach((g) => {
        const m = (g as THREE.Mesh).material as THREE.MeshStandardMaterial;
        if (m.emissiveIntensity !== undefined) m.emissiveIntensity = 0.4 + 0.6 * pulse;
      });
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      scene.traverse((o) => {
        const m = o as THREE.Mesh;
        m.geometry?.dispose();
        const mats = Array.isArray(m.material) ? m.material : m.material ? [m.material] : [];
        mats.forEach((x) => { (x as THREE.MeshStandardMaterial).map?.dispose(); x.dispose(); });
      });
      el.removeChild(renderer.domElement);
      s3.current = null;
    };
  }, []);

  // --- the tower, on every reading --------------------------------------------------------
  useEffect(() => {
    const st = s3.current;
    if (!st) return;
    const now = performance.now();
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const wood = new THREE.Color(css("--slot"));
    const tone = { safe: new THREE.Color(css("--ok")), topples: new THREE.Color(css("--warn")),
                   blocked: new THREE.Color(css("--idle")) };
    const want = new Set<string>();
    tower.rows.forEach((row, L) => row.forEach((v, s) => { if (v) want.add(`${L},${s}`); }));

    for (const key of want) {
      if (st.blocks.has(key)) continue;
      const [L, s] = key.split(",").map(Number);
      const { pos, rotY } = place(L, s);
      const m = new THREE.Mesh(st.geo, new THREE.MeshStandardMaterial({
        map: st.wood, roughness: 0.75, transparent: true, emissive: new THREE.Color(css("--bad")), emissiveIntensity: 0 }));
      m.castShadow = true; m.receiveShadow = true;
      m.rotation.y = rotY;
      m.userData = { L, s, kind: "block", tint: 0.92 + Math.random() * 0.12 };
      const edge = new THREE.LineSegments(st.edges, new THREE.LineBasicMaterial({ color: 0x3b2a18, transparent: true, opacity: 0.25 }));
      m.add(edge);
      st.tower.add(m);
      st.blocks.set(key, m);
      if (still) { m.position.copy(pos); continue; }
      const from = pos.clone().add(new THREE.Vector3(0, st.first ? 120 + L * 4 : 70, 0));
      m.position.copy(from);
      (m.material as THREE.Material).opacity = 0;
      st.tweens.push({ t0: now + (st.first ? L * 45 + s * 12 : 0), dur: st.first ? 520 : 420, step: (k) => {
        m.position.lerpVectors(from, pos, k);
        (m.material as THREE.Material).opacity = Math.min(1, k * 2);
      } });
    }
    for (const [key, m] of st.blocks) {
      if (want.has(key)) continue;
      st.blocks.delete(key);
      m.userData.leaving = true;
      const { out } = place(m.userData.L, m.userData.s);
      const from = m.position.clone(), to = from.clone().addScaledVector(out, 110);
      const done = () => { st.tower.remove(m); (m.material as THREE.Material).dispose(); };
      if (still) { done(); continue; }
      st.tweens.push({ t0: now, dur: 600, step: (k) => {
        m.position.lerpVectors(from, to, k);
        (m.material as THREE.Material).opacity = 1 - k;
      }, done });
    }

    // Colours: wood, or the safe-pull verdict.
    for (const m of st.blocks.values()) {
      const mt = m.material as THREE.MeshStandardMaterial;
      const v = verdict(tower, m.userData.L, m.userData.s);
      mt.color.copy(mode === "safe" ? tone[v] : wood).multiplyScalar(m.userData.tint);
      if (mode === "safe") mt.color.lerp(wood, 0.25);
    }

    // Where the next block may go: pulsing green ghosts.
    st.ghosts.children.forEach((g) => ((g as THREE.Mesh).material as THREE.Material).dispose());
    st.ghosts.clear();
    const ok = new THREE.Color(css("--ok"));
    for (const [L, s] of tower.legal_slots) {
      const { pos, rotY } = place(L, s);
      const g = new THREE.Mesh(st.geo, new THREE.MeshBasicMaterial({ color: ok, transparent: true, opacity: 0.2, depthWrite: false }));
      g.position.copy(pos); g.rotation.y = rotY;
      g.userData = { L, s, kind: "ghost" };
      g.add(new THREE.LineSegments(st.edges, new THREE.LineBasicMaterial({ color: ok })));
      st.ghosts.add(g);
    }

    // Layer numbers; the top complete layer in the accent colour.
    const height = Math.max(tower.rows.length, ...tower.legal_slots.map(([L]) => L + 1));
    st.labels.children.forEach((c) => ((c as THREE.Sprite).material.map?.dispose(), (c as THREE.Sprite).material.dispose()));
    st.labels.clear();
    const muted = css("--muted"), accent = css("--accent");
    for (let L = 0; L < height; L++) {
      const tcl = L === tower.top_complete_layer;
      if (!tcl && L % 2 && height > 12) continue;
      const sp = label(String(L), tcl ? accent : muted, tcl);
      sp.position.set(-LEN / 2 - 22, L * HGT + HGT / 2, LEN / 2 + 22);
      st.labels.add(sp);
    }

    if (height !== st.height) {
      st.height = height;
      if (st.first) frame(st);
      else st.controls.target.set(0, (height * HGT) / 2, 0);
    }
    st.first = false;
  }, [tower, mode]);

  // --- an alert: the block that moved glows red; where it came from, a red outline ------------
  useEffect(() => {
    const st = s3.current;
    if (!st || !flash) return;
    st.marks.clear();
    const now = performance.now();
    const bad = new THREE.Color(css("--bad"));
    if (flash.from) {
      const { pos, rotY } = place(...flash.from);
      const g = new THREE.LineSegments(st.edges, new THREE.LineBasicMaterial({ color: bad }));
      g.position.copy(pos); g.rotation.y = rotY; g.scale.setScalar(1.04);
      st.marks.add(g);
    }
    const target = flash.to && st.blocks.get(flash.to.join(","));
    const mt = target?.material as THREE.MeshStandardMaterial | undefined;
    st.tweens.push({ t0: now, dur: 3200, step: (k) => {
      if (mt) mt.emissiveIntensity = (1 - k) * (0.35 + 0.35 * Math.sin(k * 30));
    }, done: () => { st.marks.clear(); if (mt) mt.emissiveIntensity = 0; } });
  }, [flash?.key]); // eslint-disable-line react-hooks/exhaustive-deps

  const s = s3.current;
  return (
    <div className="tower3d">
      <div ref={host} className="canvas" />
      {tip && (
        <div className={`tip ${tip.tone}`} style={{ left: tip.x + 14, top: tip.y + 14 }}>
          <b>{tip.title}</b><span>{tip.text}</span>
          {onPick && <small>Click to ask the LLM</small>}
        </div>
      )}
      <div className="hud">
        <button className={spin ? "on" : ""} title="Turn slowly"
                onClick={() => { if (s) { s.controls.autoRotate = !spin; setSpin(!spin); } }}>⟳</button>
        <button title="Reset the view" onClick={() => {
          if (s) frame(s);
        }}>⌂</button>
      </div>
    </div>
  );
}
