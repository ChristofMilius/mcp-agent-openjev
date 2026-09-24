"""Unit tests for the chat-scores JSON path and stock-backend dispatch."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from openjev_router.client import DecisionClient
from openjev_router.config import Config


def _make_client(**cfg) -> DecisionClient:
    base = {
        "base_url": "http://localhost:1234/v1",
        "model": "test/model",
        "temperature": 1.3,
    }
    base.update(cfg)
    return DecisionClient(Config(**base))


def test_scores_method_on_any_chat_endpoint() -> None:
    client = _make_client(method="scores")
    resp = MagicMock()
    resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {"scores": {"billing": 9.0, "tech_support": 1.0, "UNKNOWN": 0.1}}
                    ),
                }
            }
        ]
    }
    resp.raise_for_status = MagicMock()
    with patch("openjev_router.client.requests.post", return_value=resp) as mock_post:
        decision = client.decide_choice(
            state={"ticket": "overcharged"}, candidates=["billing", "tech_support"]
        )
    assert decision.value == "billing"
    assert not decision.abstained
    payload = mock_post.call_args.kwargs["json"]
    assert "logprobs" not in payload
    assert "reasoning_effort" in payload


def test_scores_json_code_fence_stripped() -> None:
    client = _make_client(method="scores")
    resp = MagicMock()
    resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '```json\n{"scores": {"x": 7.0, "y": 1.0}}\n```',
                }
            }
        ]
    }
    resp.raise_for_status = MagicMock()
    with patch("openjev_router.client.requests.post", return_value=resp):
        decision = client.decide_choice(state={"a": 1}, candidates=["x", "y"])
    assert decision.value == "x"


def test_backend_ollama_reuses_stock_client() -> None:
    client = _make_client(base_url="http://localhost:11434", backend="ollama")
    assert client._legacy is not None


def test_backend_legacy_hits_completions_endpoint() -> None:
    client = _make_client(base_url="http://localhost:8000/v1", backend="legacy")
    resp = MagicMock()
    resp.json.return_value = {
        "choices": [
            {
                "text": "A",
                "logprobs": {"top_logprobs": [{"A": -0.1, "B": -3.0, "UNKNOWN": -5.0}]},
            }
        ]
    }
    resp.raise_for_status = MagicMock()
    with patch("openjev_router.client.requests.post", return_value=resp) as mock_post:
        decision = client.decide_choice(state={"q": "x"}, candidates=["a", "b"])
    assert decision.value == "a"
    assert "/completions" in mock_post.call_args.args[0]
    assert mock_post.call_args.kwargs["json"]["max_tokens"] == 1


def test_backend_auto_resolves_to_chat_for_lmstudio() -> None:
    client = _make_client(base_url="http://localhost:1234/v1")
    assert client.backend == "chat"
    assert client._legacy is None


def test_backend_auto_resolves_to_ollama_for_11434() -> None:
    client = _make_client(base_url="http://localhost:11434")
    assert client.backend == "ollama"
    assert client._legacy is not None
