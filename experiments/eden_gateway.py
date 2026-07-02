"""Shared Eden gateway adapter for ARC research experiments.

This module provides a small OpenAI-compatible client wrapper that can be used
by experiment scripts when the repo is pointed at a shared Eden gateway.
It is intentionally stdlib-only so it can be imported from any experiment
directory without introducing new runtime dependencies.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_CLIENT_NAME = "arc-principle-validation"


class EdenGatewayError(RuntimeError):
    """Raised when the Eden gateway is misconfigured or unavailable."""


@dataclass(frozen=True)
class EdenGatewayConfig:
    """Configuration for the shared Eden gateway."""

    api_base_url: str
    api_key: str
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    client_name: str = DEFAULT_CLIENT_NAME


def _normalize_gateway_url(raw_url: str) -> str:
    raw_url = (raw_url or "").strip()
    if not raw_url:
        raise EdenGatewayError("EDEN_GATEWAY_URL is empty.")
    if not raw_url.startswith(("http://", "https://")):
        raw_url = f"https://{raw_url}"

    raw_url = raw_url.rstrip("/")
    if raw_url.endswith("/openai/v1") or raw_url.endswith("/v1"):
        return raw_url
    return f"{raw_url}/openai/v1"


def load_eden_gateway_config(
    env: Optional[Mapping[str, str]] = None,
) -> Optional[EdenGatewayConfig]:
    """Load Eden gateway config from environment variables.

    Returns None when Eden is not configured at all. If only one of the two
    required environment variables is present, raises an explicit error so
    active experiment paths do not silently fall back to a raw provider key.
    """

    env = env or os.environ
    raw_url = (env.get("EDEN_GATEWAY_URL") or "").strip()
    api_key = (env.get("EDEN_GATEWAY_API_KEY") or "").strip()

    if not raw_url and not api_key:
        return None

    missing = []
    if not raw_url:
        missing.append("EDEN_GATEWAY_URL")
    if not api_key:
        missing.append("EDEN_GATEWAY_API_KEY")
    if missing:
        raise EdenGatewayError(
            "Eden gateway is partially configured. Set both EDEN_GATEWAY_URL "
            f"and EDEN_GATEWAY_API_KEY before using active experiment paths. "
            f"Missing: {', '.join(missing)}."
        )

    return EdenGatewayConfig(api_base_url=_normalize_gateway_url(raw_url), api_key=api_key)


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def extract_chat_text(response: Any) -> str:
    """Extract text from an OpenAI-style chat completion response."""

    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")

    if not choices:
        return ""

    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None and isinstance(choice, dict):
        message = choice.get("message")

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if content:
        return _coerce_text(content)

    text = getattr(choice, "text", None)
    if text is None and isinstance(choice, dict):
        text = choice.get("text")
    return _coerce_text(text)


def extract_responses_text(response: Any) -> str:
    """Extract text from an OpenAI Responses API payload."""

    output_text = getattr(response, "output_text", None)
    if output_text is None and isinstance(response, dict):
        output_text = response.get("output_text")
    if output_text:
        return _coerce_text(output_text)

    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")

    if not output:
        return ""

    collected = []
    for item in output:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content")
        if not content:
            continue
        for part in content:
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if text:
                collected.append(_coerce_text(text))

    return "\n".join(collected)


class EdenGatewayOpenAIClient:
    """Minimal OpenAI-compatible wrapper that talks to Eden."""

    def __init__(self, config: EdenGatewayConfig):
        self.config = config
        self.chat = _ChatAPI(self)
        self.responses = _ResponsesAPI(self)

    def _request_json(self, path: str, payload: Mapping[str, Any]) -> Any:
        url = f"{self.config.api_base_url.rstrip('/')}/{path.lstrip('/')}"
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-Eden-Client": self.config.client_name,
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = ""
            message = f"Eden gateway request failed with HTTP {exc.code}."
            if detail:
                message = f"{message} {detail}"
            raise EdenGatewayError(message) from exc
        except URLError as exc:
            raise EdenGatewayError(f"Eden gateway request failed: {exc.reason}") from exc

        if not response_body:
            return {}

        return _to_namespace(json.loads(response_body))

    def chat_completions(self, **payload: Any) -> Any:
        return self._request_json("chat/completions", payload)

    def responses_api(self, **payload: Any) -> Any:
        return self._request_json("responses", payload)

    def generate_text(
        self,
        prompt: str,
        *,
        model: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {"model": model, "messages": messages}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        response = self.chat_completions(**payload)
        return extract_chat_text(response)


class _ChatAPI:
    def __init__(self, client: EdenGatewayOpenAIClient):
        self.completions = _ChatCompletionsAPI(client)


class _ChatCompletionsAPI:
    def __init__(self, client: EdenGatewayOpenAIClient):
        self._client = client

    def create(self, **payload: Any) -> Any:
        return self._client.chat_completions(**payload)


class _ResponsesAPI:
    def __init__(self, client: EdenGatewayOpenAIClient):
        self._client = client

    def create(self, **payload: Any) -> Any:
        return self._client.responses_api(**payload)
