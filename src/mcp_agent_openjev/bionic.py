"""Bionic (LM Studio) client for the Open Responses API (`/v1/responses`).

This is the only LM Studio surface that returns logprobs: `/v1/completions` and `/v1/chat/completions` return
`logprobs: null` and cap `top_logprobs` at 20. The Open Responses endpoint returns `message.output_text.logprobs` with an
uncapped `top_logprobs` list, so one request yields every candidate letter's log-probability at the answer position.

Two request settings are load-bearing:

- `reasoning.effort: "none"` -- the model then emits exactly one token (the answer letter) instead of reasoning first.
- `temperature: 1.0`, `top_p: 1.0`, `frequency_penalty: 0`, `presence_penalty: 0` -- Bionic otherwise defaults to
  temperature 0.8 / top_p 0.95 / frequency_penalty 1.1, which reshapes the logits and makes the same prompt score
  differently from run to run.
"""

from __future__ import annotations

import requests

from mcp_agent_openjev.config import Config

TOP_LOGPROBS = 64  # comfortably covers the 52 single-letter option labels


class BionicError(RuntimeError):
    """Bionic returned an error or an unparseable readout."""


class BionicClient:
    """One-token readout over Bionic's `/v1/responses` endpoint."""

    def __init__(self, config: Config):
        self.config = config
        self._url = config.responses_url
        self._headers = {"Content-Type": "application/json"}
        if config.api_key:
            self._headers["Authorization"] = f"Bearer {config.api_key}"

    def top_logprobs(self, body: str) -> tuple[dict[str, float], int]:
        """POST the prompt body; return ({token text: logprob} at the answer position, prompt token count)."""
        payload = {
            "model": self.config.model,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": body}]}],
            "include": ["message.output_text.logprobs"],
            "top_logprobs": TOP_LOGPROBS,
            "reasoning": {"effort": "none"},
            "temperature": 1.0,
            "top_p": 1.0,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        try:
            resp = requests.post(self._url, headers=self._headers, json=payload, timeout=self.config.timeout)
        except requests.RequestException as exc:
            raise BionicError(f"Bionic unreachable: {type(exc).__name__}: {exc}") from exc
        if resp.status_code != 200:
            raise BionicError(f"Bionic HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise BionicError(f"Bionic returned non-JSON: {resp.text[:200]}") from exc

        top: dict[str, float] = {}
        for item in data.get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text" and c.get("logprobs"):
                        for entry in c["logprobs"][0].get("top_logprobs", []):
                            if entry.get("logprob") is not None:
                                top[entry.get("token", "")] = float(entry["logprob"])
        if not top:
            raise BionicError("Bionic returned no top_logprobs at the answer position (is the model in readout mode?)")
        tokens = int(data.get("usage", {}).get("input_tokens", 0))
        return top, tokens
