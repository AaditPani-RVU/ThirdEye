"""Who may call the API (CS3235 requirements doc, section 9: authentication).

    THIRDEYE_AUTH=none       (default) no checks -- a laptop on localhost
    THIRDEYE_AUTH=firebase   browsers send a Firebase ID token (Google sign-in);
                             FIREBASE_PROJECT names the project the token is for;
                             THIRDEYE_ALLOWED_EMAILS (comma- or semicolon-separated, optional):
                             listed accounts may compile and save specs; any
                             other signed-in Google account may only watch,
                             replay and read history. Empty: everyone may edit.

The edge laptop is not a person: it authenticates with a per-device token,
THIRDEYE_EDGE_TOKEN, held in Secret Manager on the cloud side and in the
laptop's untracked `.env`. It is compared in constant time.

Tokens arrive as `Authorization: Bearer <token>`, or as `?token=` on the
WebSocket, which browsers cannot give headers.
"""

from __future__ import annotations

import hmac
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Request, WebSocket


@dataclass(frozen=True)
class User:
    uid: str
    email: str = ""
    name: str = ""
    #: May spend LLM calls and change stored specs.
    editor: bool = True

    @property
    def label(self) -> str:
        return self.email or self.uid


LOCAL = User(uid="local", email="local", name="local user")

Verifier = Callable[[str], dict[str, Any]]


def _firebase_verifier(project: str) -> Verifier:  # pragma: no cover - network
    from google.auth.transport import requests
    from google.oauth2 import id_token

    session = requests.Request()

    def verify(token: str) -> dict[str, Any]:
        return id_token.verify_firebase_token(token, session, audience=project)

    return verify


@dataclass
class Auth:
    mode: str = "none"
    verify: Verifier | None = None
    allowed: frozenset[str] = frozenset()
    edge_token: str = ""

    @classmethod
    def from_env(cls) -> Auth:
        mode = os.environ.get("THIRDEYE_AUTH", "none")
        # Commas or semicolons: gcloud's --set-env-vars splits on commas.
        raw = os.environ.get("THIRDEYE_ALLOWED_EMAILS", "").replace(";", ",")
        allowed = frozenset(e.strip().lower() for e in raw.split(",") if e.strip())
        edge = os.environ.get("THIRDEYE_EDGE_TOKEN", "")
        if mode == "none":
            return cls(edge_token=edge)
        if mode != "firebase":
            raise RuntimeError(f"THIRDEYE_AUTH must be none or firebase, not {mode!r}")
        project = os.environ.get("FIREBASE_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise RuntimeError("THIRDEYE_AUTH=firebase needs FIREBASE_PROJECT")
        return cls(mode, _firebase_verifier(project), allowed, edge)

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def user_from_token(self, token: str | None) -> User:
        if not self.enabled:
            return LOCAL
        if not token:
            raise HTTPException(401, "sign in first")
        assert self.verify is not None
        try:
            claims = self.verify(token)
        except Exception as exc:  # any verification failure is the same answer
            raise HTTPException(401, "invalid or expired sign-in") from exc
        email = str(claims.get("email", "")).lower()
        listed = bool(claims.get("email_verified")) and email in self.allowed
        return User(claims.get("uid") or claims.get("sub", ""), email, claims.get("name", ""),
                    editor=not self.allowed or listed)

    def edge_ok(self, token: str | None) -> bool:
        if not self.edge_token:
            return not self.enabled
        if not token:
            return False
        return hmac.compare_digest(token.encode(), self.edge_token.encode())


def bearer(request: Request | WebSocket) -> str | None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.query_params.get("token")
