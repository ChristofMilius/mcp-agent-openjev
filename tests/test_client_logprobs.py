"""Unit tests for the chat-logprobs backend path (mocked wire responses)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from mcp_agent_openjev.client import DecisionClient, NoLogprobsError
from mcp_agent_openjev.config import Config


def _logprobs_response(letter_probs: dict[str, float]) -> MagicMock:
    best = max(letter_probs, key=lambda tok: letter_probs[tok])
    best_logprob = letter_probs[best]
    top = [{"token": tok, "logprob": prob, "bytes": None} for tok, prob in letter_probs.items()]
    resp = MagicMock()
    resp.json.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": best},
                "logprobs": {
                    "content": [{"token": best, "logprob": best_logprob, "top_logprobs": top}]
                },
                "finish_reason": "stop",
            }
        ]
    }
    resp.raise_for_status = MagicMock()
    return resp


def _make_client(**cfg) -> DecisionClient:
    base = {
        "base_url": "http://localhost:1234/v1",
        "model": "test/model",
        "temperature": 1.3,
        "abstain_threshold": "0.45",
    }
    base.update(cfg)
    return DecisionClient(Config(**base))


def test_choice_from_logprobs() -> None:
    client = _make_client()
    resp = _logprobs_response({"A": -0.1, "B": -3.2, "C": -5.6, "D": -3.0})
    with patch("mcp_agent_openjev.client.requests.post", return_value=resp) as mock_post:
        decision = client.decide_choice(
            state={"ticket": "unauthorized login"},
            candidates=["billing", "tech_support", "security"],
        )
    assert decision.value == "billing"
    assert not decision.abstained
    assert decision.tentative_value is None
    assert abs(sum(decision.probabilities.values()) - 1.0) < 1e-9
    assert 0.0 <= decision.confidence <= 1.0
    payload = mock_post.call_args.kwargs["json"]
    assert payload["logprobs"] is True
    assert payload["top_logprobs"] == 20
    assert payload["reasoning_effort"] == "none"
    assert payload["max_tokens"] == 1


def test_abstention_contract() -> None:
    client = _make_client(abstain_threshold="0.50")
    # UNKNOWN = C; a real (non -100) UNKNOWN logprob keeps the winner below 0.50
    resp = _logprobs_response({"A": -0.2, "B": -0.3, "C": -0.4})
    with patch("mcp_agent_openjev.client.requests.post", return_value=resp):
        decision = client.decide_choice(
            state={"query": "random"},
            candidates=["card_arrival", "change_pin"],
        )
    assert decision.abstained is True
    assert decision.value == "UNKNOWN"
    assert decision.tentative_value == "card_arrival"
    assert "UNKNOWN" in decision.probabilities


def test_auto_threshold_scales_with_candidates() -> None:
    client = _make_client(abstain_threshold="auto")
    resp = _logprobs_response({"A": -0.2, "B": -0.3, "C": -0.4, "D": -0.5})
    with patch("mcp_agent_openjev.client.requests.post", return_value=resp):
        decision = client.decide_choice(state={"q": "x"}, candidates=["a", "b", "c"])
    # 4 options incl. UNKNOWN -> threshold 1.25 / 4 = 0.3125
    assert decision.abstained is True  # max calibrated below 0.3125


def test_letter_logprob_variants_matched() -> None:
    client = _make_client()
    resp = _logprobs_response({" A": -1.1, "B": -0.1, "a": -5.0, "C": -3.3})
    with patch("mcp_agent_openjev.client.requests.post", return_value=resp):
        decision = client.decide_choice(
            state={"ticket": "printer broken"},
            candidates=["billing", "tech_support"],
        )
    assert decision.value == "tech_support"
    assert decision.probabilities["billing"] < decision.probabilities["tech_support"]


def test_prompt_includes_unknown_when_abstain_allowed() -> None:
    client = _make_client()
    resp = _logprobs_response({"A": -0.1, "B": -3.0})
    with patch("mcp_agent_openjev.client.requests.post", return_value=resp) as mock_post:
        client.decide_choice(state={"a": 1}, candidates=["x", "y"], allow_abstain=False)
        prompt = mock_post.call_args.kwargs["json"]["messages"][0]["content"]
        assert "UNKNOWN" not in prompt
        client.decide_choice(state={"a": 1}, candidates=["x", "y"], allow_abstain=True)
        prompt = mock_post.call_args.kwargs["json"]["messages"][0]["content"]
        assert "UNKNOWN" in prompt


def test_fallback_to_scores_when_no_logprobs() -> None:
    client = _make_client()
    no_logprobs = MagicMock()
    no_logprobs.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "A"}, "logprobs": None}]
    }
    no_logprobs.raise_for_status = MagicMock()
    scores_resp = MagicMock()
    scores_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"scores": {"x": 8.0, "y": 2.0, "UNKNOWN": 0.5}}),
                }
            }
        ]
    }
    scores_resp.raise_for_status = MagicMock()
    with patch(
        "mcp_agent_openjev.client.requests.post",
        side_effect=[no_logprobs, scores_resp],
    ) as mock_post:
        decision = client.decide_choice(state={"a": 1}, candidates=["x", "y"])
    assert decision.value == "x"
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[1].kwargs["json"]["max_tokens"] == 1024


def test_logprobs_forced_raises_without_logprobs() -> None:
    client = _make_client(method="logprobs")
    resp = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "A"}, "logprobs": None}]
    }
    resp.raise_for_status = MagicMock()
    with patch("mcp_agent_openjev.client.requests.post", return_value=resp):
        with pytest.raises(NoLogprobsError):
            client.decide_choice(state={"a": 1}, candidates=["x", "y"])


def test_noul_judgment() -> None:
    client = _make_client()
    resp = _logprobs_response({"A": -0.1, "B": -3.0})  # TRUE ranked above FALSE
    with patch("mcp_agent_openjev.client.requests.post", return_value=resp):
        decision = client.decide_noul(state={"url": "http://x"}, assertion="the url uses http")
    assert decision.value is True
    assert decision.confidence >= 0.5
    assert 0.0 <= decision.probability_true <= 1.0


def test_score_expected_value() -> None:
    client = _make_client()
    resp = _logprobs_response({"A": -100.0, "B": -100.0, "C": -0.01})  # tier C ~deterministic
    with patch("mcp_agent_openjev.client.requests.post", return_value=resp):
        decision = client.decide_score(
            state={"desc": "data leak"},
            tiers=["low", "medium", "high"],
        )
    assert decision.expected_score == pytest.approx(2.0, abs=0.05)
    assert decision.level_probabilities["high"] > 0.9
    assert 0.0 <= decision.confidence <= 1.0


def test_score_with_explicit_weights() -> None:
    client = _make_client()
    resp = _logprobs_response({"A": -0.01, "B": -100.0, "C": -100.0})  # low is deterministic
    with patch("mcp_agent_openjev.client.requests.post", return_value=resp):
        decision = client.decide_score(
            state={"desc": "printer"},
            tiers=[
                {"label": "low", "score": 0},
                {"label": "high", "score": 1},
                {"label": "critical", "score": 2},
            ],
        )
    assert decision.expected_score == pytest.approx(0.0, abs=0.05)
    assert decision.level_probabilities["low"] > 0.9


def test_too_many_options_rejected() -> None:
    client = _make_client()
    from mcp_agent_openjev.client import TooManyOptionsError

    with pytest.raises(TooManyOptionsError):
        client.decide_choice(state={"a": 1}, candidates=[f"opt_{i}" for i in range(27)])
