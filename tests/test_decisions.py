"""Unit tests for the OpenJev decision client over Bionic (LM Studio)."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from mcp_agent_openjev.client import DecisionClient, InvalidTierError, TooManyOptionsError
from mcp_agent_openjev.config import Config
from mcp_agent_openjev.openjev import OpenJevReadoutError


def _client(**cfg: Any) -> DecisionClient:
    base: dict[str, Any] = {"model": "openjev"}
    base.update(cfg)
    return DecisionClient(Config(**base))


def _mock(c: DecisionClient, top: dict[str, float], tokens: int = 69):
    return patch.object(c.client, "top_logprobs", return_value=(top, tokens))


# -- choice -----------------------------------------------------------------


def test_choice_calibrates_and_picks_winner() -> None:
    c = _client()
    top = {"A": -0.01, "B": -4.0, "C": -7.0, "D": -8.0}
    with _mock(c, top):
        d = c.decide_choice(state={"s": 1}, candidates=["a", "b", "c", "d"])
    assert d.value == "a"
    assert d.probabilities["a"] > 0.9
    assert not d.abstained
    assert d.confidence > 0.9
    assert d.raw_logits["a"] == pytest.approx(-0.01, abs=1e-6)


def test_choice_abstains_on_low_confidence_and_keeps_tentative() -> None:
    c = _client()
    top = {"A": -0.5, "B": -0.6}
    with _mock(c, top):
        d = c.decide_choice(state={"s": 1}, candidates=["a", "b"])
    assert d.abstained
    assert d.value == "UNKNOWN"
    assert d.tentative_value == "a"


def test_choice_without_abstain_keeps_the_winner() -> None:
    c = _client()
    top = {"A": -0.01, "B": -4.0}
    with _mock(c, top):
        d = c.decide_choice(state={"s": 1}, candidates=["a", "b"], allow_abstain=False)
    assert not d.abstained
    assert d.value == "a"
    assert "UNKNOWN" not in d.probabilities


def test_choice_missing_letter_raises() -> None:
    c = _client()
    top = {"A": -0.01, "B": -4.0}  # C, D, E absent
    with _mock(c, top):
        with pytest.raises(OpenJevReadoutError):
            c.decide_choice(state={"s": 1}, candidates=["a", "b", "c"])


def test_choice_rejects_more_than_52_candidates() -> None:
    c = _client()
    with pytest.raises(TooManyOptionsError):
        c.decide_choice(state={"s": 1}, candidates=[f"o{i}" for i in range(53)], allow_abstain=False)


# -- noul -------------------------------------------------------------------


def test_noul_true_when_true_dominates() -> None:
    c = _client()
    top = {"A": -0.01, "B": -4.0}
    with _mock(c, top):
        d = c.decide_noul(state={"s": 1}, assertion="the sky is blue")
    assert d.value is True
    assert d.probability_true > 0.9
    assert not d.abstained


def test_noul_false_when_false_dominates() -> None:
    c = _client()
    top = {"A": -4.0, "B": -0.01}
    with _mock(c, top):
        d = c.decide_noul(state={"s": 1}, assertion="the sky is green")
    assert d.value is False
    assert d.probability_true < 0.1


# -- score ------------------------------------------------------------------


def test_score_expected_value_uses_ordinal_positions() -> None:
    c = _client()
    top = {"A": -0.01, "B": -4.0, "C": -7.0, "D": -8.0, "E": -9.0}
    with _mock(c, top):
        d = c.decide_score(state={"s": 1}, tiers=["low", "medium", "high"])
    assert d.level_probabilities["low"] > 0.9
    assert d.expected_score == pytest.approx(0.0, abs=0.05)
    assert d.tier_weights == {"low": 0.0, "medium": 1.0, "high": 2.0}


def test_score_tier_value_alias_accepted() -> None:
    c = _client()
    top = {"A": -7.0, "B": -0.01, "C": -4.0}  # low, high, UNKNOWN
    with _mock(c, top):
        d = c.decide_score(state={"s": 1}, tiers=[{"value": "low"}, {"label": "high", "score": 1}])
    assert d.level_probabilities["high"] > d.level_probabilities["low"]


# -- tier validation --------------------------------------------------------


def test_tier_dict_without_label_raises() -> None:
    c = _client()
    with pytest.raises(InvalidTierError, match="without 'label' or 'value'"):
        c.decide_score(state={"a": 1}, tiers=[{"$text": "low"}, {"$text": "high"}])


def test_tier_score_must_match_position() -> None:
    c = _client()
    with pytest.raises(InvalidTierError, match="does not match the tier's position"):
        c.decide_score(state={"a": 1}, tiers=[{"label": "low", "score": 10}, {"label": "high", "score": 1}])


def test_tier_score_must_be_numeric() -> None:
    c = _client()
    with pytest.raises(InvalidTierError, match="must be a number"):
        c.decide_score(state={"a": 1}, tiers=[{"label": "low", "score": "zero"}])


def test_duplicate_tier_labels_raise() -> None:
    c = _client()
    with pytest.raises(InvalidTierError, match="duplicate tier label"):
        c.decide_score(state={"a": 1}, tiers=["low", "high", {"label": "low"}])
