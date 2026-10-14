"""Where specs and sessions are kept: local files, or Firestore + Cloud Storage.

    THIRDEYE_STORE=local      (default) JSON under eval/store/, for a laptop
    THIRDEYE_STORE=firestore  Firestore for metadata, GCS for the documents;
                              THIRDEYE_BUCKET names the bucket

The split follows the CS3235 requirements doc (section 9): small, queried
records -- spec metadata, session summaries, each alert -- go to Firestore;
whole documents -- spec JSON and session logs -- go to Cloud Storage, with the
Firestore record holding their `gs://` URI. Only text is stored: no video ever
reaches either (NFR-PRIV-1).

    specs/{name}                {procedure, kind, state_model, compiler_version, model,
                                 created_by, uri, valid, updated_at}
    sessions/{id}               {spec, source, started_at, ended_at, steps_done,
                                 alert_count, outcome, uri, report, created_by}
    sessions/{id}/alerts/{n}    {t, code, severity, rule, text, target}

Both stores pass the same tests; the Firestore one is exercised against an
in-memory stand-in for the two Google clients.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from thirdeye.sessionlog import SessionRecord

ROOT = Path(__file__).resolve().parents[2]

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


def safe_name(name: str) -> str:
    """Names become file and document ids, so they are held to a plain alphabet."""
    if not _NAME.match(name) or ".." in name:
        raise ValueError(f"bad name {name!r}: letters, digits, _ . - only, up to 80")
    return name


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def session_summary(record: SessionRecord) -> dict[str, Any]:
    done = sum(r.status == "done" for r in record.checklist)
    return {
        "procedure": record.procedure,
        "kind": record.kind,
        "spec": record.meta.get("spec"),
        "source": record.meta.get("source"),
        "run": record.meta.get("run"),
        "started_at": record.meta.get("started_at"),
        "ended_at": record.meta.get("ended_at") or now(),
        "steps_done": (f"{done}/{len(record.checklist)}"
                       if record.checklist and record.kind == "dag" else None),
        "alert_count": len(record.alerts),
        "red": len(record.red()),
        "outcome": record.outcome,
    }


def spec_summary(spec: dict[str, Any]) -> dict[str, Any]:
    prov = spec.get("provenance") or {}
    return {
        "procedure": spec.get("procedure"),
        "kind": spec.get("kind"),
        "state_model": spec.get("state_model"),
        "compiler_version": prov.get("compiler_version"),
        "model": prov.get("model"),
    }


class Store(Protocol):
    def list_specs(self) -> list[dict[str, Any]]: ...
    def get_spec(self, name: str) -> dict[str, Any] | None: ...
    def put_spec(self, name: str, spec: dict[str, Any], by: str, valid: bool) -> None: ...
    def list_sessions(self, limit: int = 100) -> list[dict[str, Any]]: ...
    def get_session(self, sid: str) -> dict[str, Any] | None: ...
    def put_session(self, sid: str, record: SessionRecord, report: str, by: str) -> None: ...
    def set_report(self, sid: str, report: str) -> None: ...


# --- local files ------------------------------------------------------------------------


class LocalStore:
    """`root/specs/<name>.json`, `root/sessions/<id>.json` (log) + `<id>.meta.json`.

    Seeded with the repository's own `specs/` (read-only): a spec saved under
    the same name shadows it.
    """

    def __init__(self, root: Path, seed_specs: Path | None = ROOT / "specs"):
        self.root = Path(root)
        self.seed = seed_specs
        (self.root / "specs").mkdir(parents=True, exist_ok=True)
        (self.root / "sessions").mkdir(parents=True, exist_ok=True)

    def _spec_paths(self) -> dict[str, Path]:
        paths = {}
        if self.seed is not None and self.seed.exists():
            paths.update({p.stem: p for p in self.seed.glob("*.json")})
        paths.update({p.stem: p for p in (self.root / "specs").glob("*.json")
                      if not p.name.endswith(".meta.json")})
        return paths

    def list_specs(self) -> list[dict[str, Any]]:
        out = []
        for name, path in sorted(self._spec_paths().items()):
            try:
                spec = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            meta_path = path.with_suffix(".meta.json")
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
                "created_by": "repository", "valid": True}
            out.append({"name": name, **spec_summary(spec), **meta})
        return out

    def get_spec(self, name: str) -> dict[str, Any] | None:
        path = self._spec_paths().get(safe_name(name))
        return json.loads(path.read_text()) if path else None

    def put_spec(self, name: str, spec: dict[str, Any], by: str, valid: bool) -> None:
        path = self.root / "specs" / f"{safe_name(name)}.json"
        path.write_text(json.dumps(spec, indent=2) + "\n")
        path.with_suffix(".meta.json").write_text(json.dumps(
            {"created_by": by, "valid": valid, "updated_at": now()}))

    def list_sessions(self, limit: int = 100) -> list[dict[str, Any]]:
        metas = []
        for p in (self.root / "sessions").glob("*.meta.json"):
            metas.append({"id": p.name.removesuffix(".meta.json"), **json.loads(p.read_text())})
        metas.sort(key=lambda m: m.get("ended_at") or "", reverse=True)
        return metas[:limit]

    def get_session(self, sid: str) -> dict[str, Any] | None:
        base = self.root / "sessions" / safe_name(sid)
        meta = base.with_name(f"{sid}.meta.json")
        if not meta.exists():
            return None
        log = base.with_name(f"{sid}.json")
        return {"id": sid, **json.loads(meta.read_text()),
                "log": json.loads(log.read_text()) if log.exists() else None}

    def put_session(self, sid: str, record: SessionRecord, report: str, by: str) -> None:
        base = self.root / "sessions" / safe_name(sid)
        record.write(base.with_name(f"{sid}.json"))
        base.with_name(f"{sid}.meta.json").write_text(json.dumps(
            {**session_summary(record), "report": report, "created_by": by}))

    def set_report(self, sid: str, report: str) -> None:
        meta = self.root / "sessions" / f"{safe_name(sid)}.meta.json"
        data = json.loads(meta.read_text())
        data["report"] = report
        meta.write_text(json.dumps(data))


# --- Firestore + Cloud Storage ---------------------------------------------------------


class CloudStore:
    """Firestore for records, a GCS bucket for documents.

    `db` and `bucket` are a `google.cloud.firestore.Client` and a
    `google.cloud.storage.Bucket`; injected so the tests can stand them in.
    """

    def __init__(self, db: Any, bucket: Any):
        self.db = db
        self.bucket = bucket

    @classmethod
    def from_env(cls) -> CloudStore:  # pragma: no cover - needs Google credentials
        from google.cloud import firestore, storage

        bucket = os.environ.get("THIRDEYE_BUCKET")
        if not bucket:
            raise RuntimeError("THIRDEYE_STORE=firestore needs THIRDEYE_BUCKET")
        return cls(firestore.Client(), storage.Client().bucket(bucket))

    def _put_blob(self, path: str, data: dict[str, Any]) -> str:
        self.bucket.blob(path).upload_from_string(json.dumps(data, indent=1),
                                                  content_type="application/json")
        return f"gs://{self.bucket.name}/{path}"

    def _get_blob(self, path: str) -> dict[str, Any] | None:
        blob = self.bucket.blob(path)
        if not blob.exists():
            return None
        return json.loads(blob.download_as_text())

    def list_specs(self) -> list[dict[str, Any]]:
        return sorted(({"name": d.id, **d.to_dict()} for d in self.db.collection("specs").stream()),
                      key=lambda s: s["name"])

    def get_spec(self, name: str) -> dict[str, Any] | None:
        return self._get_blob(f"specs/{safe_name(name)}.json")

    def put_spec(self, name: str, spec: dict[str, Any], by: str, valid: bool) -> None:
        uri = self._put_blob(f"specs/{safe_name(name)}.json", spec)
        self.db.collection("specs").document(name).set(
            {**spec_summary(spec), "created_by": by, "uri": uri, "valid": valid,
             "updated_at": now()})

    def list_sessions(self, limit: int = 100) -> list[dict[str, Any]]:
        docs = [{"id": d.id, **d.to_dict()} for d in self.db.collection("sessions").stream()]
        docs.sort(key=lambda m: m.get("ended_at") or "", reverse=True)
        return docs[:limit]

    def get_session(self, sid: str) -> dict[str, Any] | None:
        doc = self.db.collection("sessions").document(safe_name(sid)).get()
        if not doc.exists:
            return None
        return {"id": sid, **doc.to_dict(), "log": self._get_blob(f"sessions/{sid}.json")}

    def put_session(self, sid: str, record: SessionRecord, report: str, by: str) -> None:
        uri = self._put_blob(f"sessions/{safe_name(sid)}.json", record.to_json())
        ref = self.db.collection("sessions").document(sid)
        ref.set({**session_summary(record), "uri": uri, "report": report, "created_by": by})
        alerts = ref.collection("alerts")
        for n, a in enumerate(record.alerts):
            alerts.document(f"{n:04d}").set({
                "t": a.t, "code": a.code, "severity": a.severity, "rule": a.rule,
                "text": a.text, "target": a.target})

    def set_report(self, sid: str, report: str) -> None:
        self.db.collection("sessions").document(safe_name(sid)).update({"report": report})


def store_from_env(local_root: Path | None = None) -> Store:
    kind = os.environ.get("THIRDEYE_STORE", "local")
    if kind == "firestore":
        return CloudStore.from_env()
    if kind != "local":
        raise RuntimeError(f"THIRDEYE_STORE must be local or firestore, not {kind!r}")
    return LocalStore(local_root or ROOT / "eval" / "store")
