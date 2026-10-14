"""The LLM backends, chosen by one string (SW-8, NFR-COST-1).

    groq:openai/gpt-oss-120b      Groq (default; key GROQ_API_KEY)
    gemini:gemini-2.5-flash       Google Gemini (key GEMINI_API_KEY)
    claude:claude-sonnet-5        Anthropic (key ANTHROPIC_API_KEY, the `llm` extra)
    ollama:qwen3:8b               a local model through Ollama, no key, offline

Every backend fills the same one-method protocol the compiler uses
(`compile.LLM`): system prompt, user text and a JSON schema in, JSON text out.
Groq, Gemini and Ollama also take `images=` (JPEG bytes) for a vision model --
used only by the end-to-end VLM baseline, never by the system itself.
Each is asked for schema-constrained output in its own API's terms; the
validator downstream does not trust any of them more than the others.

Groq, Gemini and Ollama are plain HTTPS/JSON calls through the standard
library, so they need no SDK. Keys come from the environment or an untracked
`.env` at the repository root, never from code or config (NFR-PRIV-2).
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = "groq:openai/gpt-oss-120b"

#: Bare model names from before providers were explicit keep working.
_PREFIXES = {"claude-": "claude", "gemini-": "gemini"}


class LLMError(RuntimeError):
    """The provider failed or stopped early; not a problem with the spec."""


def load_env(path: Path = ROOT / ".env") -> None:
    """Read KEY=value lines from `.env` into the environment, never overriding it."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _key(name: str) -> str:
    load_env()
    value = os.environ.get(name)
    if not value:
        raise LLMError(f"{name} is not set (environment or .env)")
    return value


def _post(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> Any:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "thirdeye", **headers},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise LLMError(f"{url}: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"{url}: {exc.reason}") from exc


def _wait_s(message: str) -> float | None:
    """The wait a rate-limit message asks for ("try again in 13.9s"), if any."""
    if "429" not in message:
        return None
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", message)
    return (int(m.group(1) or 0) * 60 + float(m.group(2))) if m else 20.0


def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


@dataclass
class GroqLLM:
    """Groq's OpenAI-compatible chat API with strict JSON-schema output.

    The free tier allows a few thousand tokens a minute, so a rate-limit
    answer is waited out (as long as Groq asks, a few times) rather than
    counted as a failed compile.
    """

    model: str = "openai/gpt-oss-120b"
    max_tokens: int = 16000
    timeout: float = 180.0
    post: Any = _post
    retries: int = 6
    sleep: Any = time.sleep

    def complete(self, system: str, user: str, schema: dict[str, Any] | None,
                 images: list[bytes] | None = None) -> str:
        for attempt in range(self.retries + 1):
            try:
                return self._complete(system, user, schema, images or [])
            except LLMError as exc:
                wait = _wait_s(str(exc))
                if wait is None and "does not match the expected schema" in str(exc) \
                        and attempt < 2:
                    wait = 0.0   # a bad generation on Groq's side: ask again, once or twice
                if wait is None or attempt == self.retries:
                    raise
                self.sleep(wait + 1.0)
        raise AssertionError("unreachable")

    def _complete(self, system: str, user: str, schema: dict[str, Any] | None,
                  images: list[bytes]) -> str:
        content: Any = user
        if images:
            content = [{"type": "text", "text": user}] + [
                {"type": "image_url", "image_url": {"url": _data_url(img)}} for img in images]
        data = self.post(
            "https://api.groq.com/openai/v1/chat/completions",
            {
                "model": self.model,
                "max_completion_tokens": self.max_tokens,
                "temperature": 0,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": content}],
                # Strict schema decoding is offered for some models only; the
                # rest follow the schema best-effort, and the validator decides.
                "response_format": {"type": "json_object"} if schema is None else {
                    "type": "json_schema", "json_schema": {
                        "name": "output", "schema": schema,
                        "strict": self.model.startswith("openai/gpt-oss")}},
            },
            {"Authorization": f"Bearer {_key('GROQ_API_KEY')}"},
            self.timeout,
        )
        choice = data["choices"][0]
        if choice.get("finish_reason") not in ("stop", None):
            raise LLMError(f"the model stopped early: {choice.get('finish_reason')}")
        return choice["message"]["content"]


@dataclass
class GeminiLLM:
    """Gemini's generateContent with a JSON-schema response."""

    model: str = "gemini-2.5-flash"
    max_tokens: int = 16000
    timeout: float = 180.0
    post: Any = _post

    def complete(self, system: str, user: str, schema: dict[str, Any] | None,
                 images: list[bytes] | None = None) -> str:
        parts: list[dict[str, Any]] = [{"text": user}] + [
            {"inline_data": {"mime_type": "image/jpeg",
                             "data": base64.b64encode(img).decode()}} for img in images or []]
        data = self.post(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent",
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": self.max_tokens,
                    "responseMimeType": "application/json",
                    **({"responseJsonSchema": schema} if schema is not None else {}),
                },
            },
            {"x-goog-api-key": _key("GEMINI_API_KEY")},
            self.timeout,
        )
        candidate = (data.get("candidates") or [{}])[0]
        if candidate.get("finishReason") not in ("STOP", None):
            raise LLMError(f"the model stopped early: {candidate.get('finishReason')}")
        return "".join(p.get("text", "") for p in candidate["content"]["parts"])


@dataclass
class OllamaLLM:
    """A local model through Ollama's chat API, held to the schema by `format`."""

    model: str = "qwen3:8b"
    max_tokens: int = 16000
    timeout: float = 900.0
    host: str = ""
    post: Any = _post

    def complete(self, system: str, user: str, schema: dict[str, Any] | None,
                 images: list[bytes] | None = None) -> str:
        host = self.host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        if not host.startswith("http"):
            host = f"http://{host}"
        body = {
                "model": self.model,
                "stream": False,
                "think": False,
                "format": schema if schema is not None else "json",
                "options": {"temperature": 0, "num_predict": self.max_tokens,
                            "num_ctx": 16384},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user,
                              **({"images": [base64.b64encode(i).decode() for i in images]}
                                 if images else {})}],
        }
        try:
            data = self.post(f"{host}/api/chat", body, {}, self.timeout)
        except LLMError as exc:
            if "thinking" not in str(exc):
                raise
            # Models without a thinking mode reject the switch that turns it off.
            body = {k: v for k, v in body.items() if k != "think"}
            data = self.post(f"{host}/api/chat", body, {}, self.timeout)
        if data.get("done_reason") == "length":
            raise LLMError("the model stopped early: length")
        return data["message"]["content"]


def make_llm(name: str | None = None, max_tokens: int = 16000):
    """`provider:model` (or a bare Claude/Gemini model name) -> a backend."""
    from thirdeye.compiler.compile import ClaudeLLM

    name = name or os.environ.get("THIRDEYE_LLM") or DEFAULT_MODEL
    provider, sep, model = name.partition(":")
    if not sep:
        provider = next((p for pre, p in _PREFIXES.items() if name.startswith(pre)), "")
        model = name
    backends = {"groq": GroqLLM, "gemini": GeminiLLM, "ollama": OllamaLLM, "claude": ClaudeLLM}
    if provider not in backends:
        raise ValueError(f"unknown LLM {name!r}: use one of "
                         f"{', '.join(f'{p}:<model>' for p in backends)}")
    llm = backends[provider](model=model, max_tokens=max_tokens)
    llm.model = model
    llm.name = f"{provider}:{model}"
    return llm
