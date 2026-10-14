"""LLM backends: each asks for schema-constrained JSON in its own API's terms.

No network: every backend takes its HTTP call as a parameter.
"""

from __future__ import annotations

import pytest

from thirdeye.compiler.compile import ClaudeLLM
from thirdeye.compiler.llm import (
    GeminiLLM,
    GroqLLM,
    LLMError,
    OllamaLLM,
    load_env,
    make_llm,
)

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}}


class Recorder:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[tuple[str, dict, dict]] = []

    def __call__(self, url, body, headers, timeout):
        self.calls.append((url, body, headers))
        out = self.responses.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gk")
    monkeypatch.setenv("GEMINI_API_KEY", "mk")
    monkeypatch.delenv("THIRDEYE_LLM", raising=False)


def test_groq_asks_for_strict_json_schema() -> None:
    post = Recorder({"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]})
    assert GroqLLM(post=post).complete("sys", "user", SCHEMA) == "{}"
    url, body, headers = post.calls[0]
    assert url.endswith("/chat/completions")
    assert headers["Authorization"] == "Bearer gk"
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] == SCHEMA
    assert body["messages"][0] == {"role": "system", "content": "sys"}


def test_groq_cut_off_is_an_error() -> None:
    post = Recorder({"choices": [{"finish_reason": "length", "message": {"content": "{"}}]})
    with pytest.raises(LLMError, match="length"):
        GroqLLM(post=post).complete("s", "u", SCHEMA)


def test_gemini_asks_for_a_json_schema_response() -> None:
    post = Recorder({"candidates": [{"finishReason": "STOP",
                                     "content": {"parts": [{"text": '{"a":'}, {"text": '"x"}'}]}}]})
    assert GeminiLLM(post=post).complete("sys", "user", SCHEMA) == '{"a":"x"}'
    url, body, headers = post.calls[0]
    assert "gemini-2.5-flash:generateContent" in url
    assert headers["x-goog-api-key"] == "mk"
    assert body["generationConfig"]["responseJsonSchema"] == SCHEMA


def test_ollama_passes_the_schema_as_format_and_retries_without_think() -> None:
    post = Recorder(LLMError('HTTP 400: "gemma3:4b" does not support thinking'),
                    {"done_reason": "stop", "message": {"content": "{}"}})
    assert OllamaLLM(model="gemma3:4b", post=post).complete("s", "u", SCHEMA) == "{}"
    assert post.calls[0][1]["think"] is False
    assert "think" not in post.calls[1][1]
    assert post.calls[1][1]["format"] == SCHEMA


def test_a_missing_key_says_which(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("GROQ_API_KEY")
    monkeypatch.setattr("thirdeye.compiler.llm.ROOT", tmp_path)
    monkeypatch.setattr("thirdeye.compiler.llm.load_env", lambda: None)
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        GroqLLM(post=Recorder()).complete("s", "u", SCHEMA)


def test_env_file_never_overrides_the_environment(monkeypatch, tmp_path) -> None:
    env = tmp_path / ".env"
    env.write_text("# keys\nGROQ_API_KEY=from_file\nNEW_KEY='quoted'\n")
    monkeypatch.delenv("NEW_KEY", raising=False)
    load_env(env)
    import os

    assert os.environ["GROQ_API_KEY"] == "gk"
    assert os.environ["NEW_KEY"] == "quoted"


@pytest.mark.parametrize("name, cls, model", [
    ("groq:openai/gpt-oss-120b", GroqLLM, "openai/gpt-oss-120b"),
    ("ollama:qwen3:8b", OllamaLLM, "qwen3:8b"),
    ("gemini:gemini-2.5-pro", GeminiLLM, "gemini-2.5-pro"),
    ("claude-sonnet-5", ClaudeLLM, "claude-sonnet-5"),
])
def test_make_llm_parses_provider_and_model(name, cls, model) -> None:
    llm = make_llm(name)
    assert isinstance(llm, cls) and llm.model == model
    assert llm.name.endswith(model)


def test_make_llm_default_and_env(monkeypatch) -> None:
    assert isinstance(make_llm(), GroqLLM)
    monkeypatch.setenv("THIRDEYE_LLM", "ollama:gemma3:4b")
    assert make_llm().name == "ollama:gemma3:4b"
    with pytest.raises(ValueError, match="unknown LLM"):
        make_llm("mystery:model")


def test_groq_waits_out_a_rate_limit() -> None:
    limited = LLMError("HTTP 429: Rate limit reached ... Please try again in 1m2.5s.")
    post = Recorder(limited, {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]})
    waits = []
    assert GroqLLM(post=post, sleep=waits.append).complete("s", "u", SCHEMA) == "{}"
    assert waits == [63.5]


def test_groq_gives_up_on_other_errors() -> None:
    post = Recorder(LLMError("HTTP 400: bad schema"))
    with pytest.raises(LLMError, match="400"):
        GroqLLM(post=post, sleep=lambda s: None).complete("s", "u", SCHEMA)


def test_no_schema_falls_back_to_plain_json_mode() -> None:
    ok = {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}
    post = Recorder(ok)
    GroqLLM(post=post).complete("s", "u", None)
    assert post.calls[0][1]["response_format"] == {"type": "json_object"}
    post = Recorder({"done_reason": "stop", "message": {"content": "{}"}})
    OllamaLLM(post=post).complete("s", "u", None)
    assert post.calls[0][1]["format"] == "json"
