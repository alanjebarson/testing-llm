"""Chat-completions wrapper for OpenAI and Ollama (via Ollama's OpenAI-compatible API).

Pacing to an RPM limit, retry with backoff on 429, finish-reason capture.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

OLLAMA_DEFAULT_URL = "http://localhost:11434/v1"


@dataclass
class LLMResult:
    text: str
    finish_reason: str  # stop, length, content_filter, ... or ERROR
    latency_s: float
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class RateLimiter:
    """Sleeps only as long as needed to keep calls at or under `rpm` (0 = no pacing)."""

    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._last = 0.0

    def wait(self) -> None:
        remaining = self._last + self.interval - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        self._last = time.monotonic()


class ChatClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        requests_per_minute: float = 0,
        timeout_seconds: float = 30,
        max_retries: int = 3,
        backoff_base_seconds: float = 10,
        provider: str = "openai",
    ):
        from openai import OpenAI

        # SDK retries are off so retry/backoff behaviour is controlled (and logged) here.
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds, max_retries=0)
        self.provider = provider
        self.base_url = base_url
        self._limiter = RateLimiter(requests_per_minute)
        self._retries = max_retries
        self._backoff = backoff_base_seconds

    def check_model(self, model: str) -> None:
        """Fail fast with a helpful message if a local Ollama server/model isn't available."""
        if self.provider != "ollama":
            return
        import openai

        try:
            available = [m.id for m in self._client.models.list()]
        except openai.APIConnectionError:
            raise SystemExit(
                f"Can't reach Ollama at {self.base_url}. Start it with `ollama serve` "
                "(or `sudo systemctl start ollama`)."
            ) from None
        if model not in available and f"{model}:latest" not in available:
            raise SystemExit(
                f"Model '{model}' isn't pulled in Ollama. Run `ollama pull {model}`. "
                f"Available: {', '.join(available) or 'none'}"
            )

    @staticmethod
    def _request(model, prompt, generation, system_instruction, response_schema) -> dict[str, Any]:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        # Drop unset params: OpenAI reasoning models (o-series, gpt-5) reject temperature/top_p.
        kwargs: dict[str, Any] = {k: v for k, v in generation.items() if v is not None}
        kwargs.update(model=model, messages=messages)
        if response_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "strict": True, "schema": response_schema},
            }
        return kwargs

    def generate(
        self,
        model: str,
        prompt: str,
        generation: dict,
        system_instruction: str | None = None,
        response_schema: dict | None = None,
    ) -> LLMResult:
        import openai

        kwargs = self._request(model, prompt, generation, system_instruction, response_schema)
        last_error = ""
        for attempt in range(self._retries):
            self._limiter.wait()
            start = time.monotonic()
            try:
                resp = self._client.chat.completions.create(**kwargs)
                choice = resp.choices[0]
                msg = choice.message
                # A model refusal (structured outputs) comes back in .refusal, not .content.
                text = (msg.content or getattr(msg, "refusal", None) or "").strip()
                return LLMResult(text, choice.finish_reason or "unknown", time.monotonic() - start)
            except openai.RateLimitError as e:
                last_error = f"429: {str(e)[:150]}"
                if "insufficient_quota" in str(e):
                    break  # out of credit: retrying won't help
                wait = self._backoff * (2**attempt)
                print(f"   rate limited, waiting {wait}s (attempt {attempt + 1}/{self._retries})")
                time.sleep(wait)
            except (openai.AuthenticationError, openai.BadRequestError, openai.NotFoundError) as e:
                last_error = f"{e.status_code}: {str(e)[:150]}"
                break  # bad key / bad params / unknown model: retrying won't help
            except openai.OpenAIError as e:  # 5xx, timeouts, connection errors
                last_error = str(e)[:150]
                time.sleep(3)
        return LLMResult("", "ERROR", 0.0, error=last_error or "retries exhausted")


def make_client(role_cfg: dict, retry_cfg: dict) -> ChatClient:
    """Build the client for the actor or judge section of settings.yaml."""
    provider = role_cfg.get("provider", "openai").lower()
    common = dict(
        requests_per_minute=role_cfg.get("requests_per_minute", 0),
        timeout_seconds=role_cfg.get("timeout_seconds", 30),
        max_retries=retry_cfg["max_retries"],
        backoff_base_seconds=retry_cfg["backoff_base_seconds"],
        provider=provider,
    )
    if provider == "ollama":
        # Ollama ignores the key, but the SDK requires one.
        return ChatClient("ollama", base_url=role_cfg.get("base_url") or OLLAMA_DEFAULT_URL, **common)
    if provider == "openai":
        from .config import get_api_key

        return ChatClient(get_api_key(), base_url=role_cfg.get("base_url"), **common)
    raise ValueError(f"Unknown provider {provider!r} (use 'ollama' or 'openai')")
