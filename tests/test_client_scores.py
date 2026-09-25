"""Unit tests for the chat-scores JSON path and stock-backend dispatch."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from mcp_agent_openjev.client import DecisionClient, InvalidTierError
from mcp_agent_openjev.config import Config


def _make_client(**cfg) -> DecisionClient:
    base = {
        "base_url": "http://localhost:1234/v1",
        "model": "test/model",
        "temperature": 1.3,
    }
    base.update(cfg)
    return DecisionClient(Config(**base))


def _scores_response(scores: dict[str, float]) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": json.dumps({"scores": scores})}}]
    }
    resp.raise_for_status = MagicMock()
    return resp


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
    with patch("mcp_agent_openjev.client.requests.post", return_value=resp) as mock_post:
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
    with patch("mcp_agent_openjev.client.requests.post", return_value=resp):
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
    with patch("mcp_agent_openjev.client.requests.post", return_value=resp) as mock_post:
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


# -- tier validation ---------------------------------------------------------
# A malformed tier used to fall through to `tier.get("label") or
# tier.get("value") or index`, so e.g. {"$text": "low"} silently became the label
# "0" and was rendered into the prompt as "0. 0", leaving the model to rank
# meaningless numeric labels. Malformed tiers now raise.


def test_tier_dict_without_label_raises() -> None:
    client = _make_client(method="scores")
    with pytest.raises(InvalidTierError, match="without 'label' or 'value'"):
        client.decide_score(state={"a": 1}, tiers=[{"$text": "low"}, {"$text": "high"}])


def test_tier_score_must_match_position() -> None:
    client = _make_client(method="scores")
    with pytest.raises(InvalidTierError, match="does not match the tier's position"):
        client.decide_score(
            state={"a": 1}, tiers=[{"label": "low", "score": 10}, {"label": "high", "score": 1}]
        )


def test_tier_score_must_be_numeric() -> None:
    client = _make_client(method="scores")
    with pytest.raises(InvalidTierError, match="must be a number"):
        client.decide_score(state={"a": 1}, tiers=[{"label": "low", "score": "zero"}])


def test_duplicate_tier_labels_raise() -> None:
    client = _make_client(method="scores")
    with pytest.raises(InvalidTierError, match="duplicate tier label"):
        client.decide_score(state={"a": 1}, tiers=["low", "high", {"label": "low"}])


def test_malformed_tier_raises_before_any_request() -> None:
    client = _make_client(method="scores")
    with patch("mcp_agent_openjev.client.requests.post") as mock_post:
        with pytest.raises(InvalidTierError):
            client.decide_score(state={"a": 1}, tiers=[{"$text": "low"}])
    mock_post.assert_not_called()


def test_tier_accepts_value_alias_and_matching_score() -> None:
    client = _make_client(method="scores")
    resp = _scores_response({"low": 0.0, "high": 9.0, "UNKNOWN": 0.0})
    with patch("mcp_agent_openjev.client.requests.post", return_value=resp):
        decision = client.decide_score(
            state={"a": 1},
            tiers=[
                {"value": "low", "score": 0},
                {"label": "high", "score": 1},
            ],
        )
    assert decision.level_probabilities["high"] > decision.level_probabilities["low"]


def test_level_probabilities_are_keyed_by_label() -> None:
    client = _make_client(method="scores")
    resp = _scores_response({"low": 1.0, "medium": 5.0, "high": 0.0, "UNKNOWN": 0.0})
    with patch("mcp_agent_openjev.client.requests.post", return_value=resp):
        decision = client.decide_score(state={"a": 1}, tiers=["low", "medium", "high"])
    assert set(decision.level_probabilities) == {"low", "medium", "high", "UNKNOWN"}


def test_expected_score_uses_ordinal_positions() -> None:
    client = _make_client(method="scores")
    resp = _scores_response({"low": 0.0, "medium": 5.0, "high": 0.0, "UNKNOWN": 0.0})
    with patch("mcp_agent_openjev.client.requests.post", return_value=resp):
        decision = client.decide_score(state={"a": 1}, tiers=["low", "medium", "high"])
    probs = decision.level_probabilities
    expected = probs["low"] * 0 + probs["medium"] * 1 + probs["high"] * 2
    assert decision.expected_score == pytest.approx(expected)
