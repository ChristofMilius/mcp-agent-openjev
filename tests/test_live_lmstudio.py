"""Live smoke tests against a running backend (skipped when none is reachable).

Run only when an OpenAI-compatible backend is up — LM Studio on
http://localhost:1234/v1 by default. Configure via the same JEV_* env vars the
service uses.
"""

from __future__ import annotations

import pytest
import requests

from mcp_agent_openjev.client import DecisionClient
from mcp_agent_openjev.config import Config


def _backend_reachable(cfg: Config) -> bool:
    try:
        return requests.get(cfg.models_url, timeout=5).status_code == 200
    except Exception:
        return False


cfg = Config.from_env()
ENDPOINT_UP = _backend_reachable(cfg)

pytestmark = pytest.mark.skipif(
    not ENDPOINT_UP, reason="no reachable decision backend on localhost"
)


def test_live_choice() -> None:
    client = DecisionClient(cfg)
    decision = client.decide_choice(
        state={"ticket": "I noticed an unauthorized login attempt from an unknown IP."},
        candidates=["billing", "tech_support", "security"],
        criteria={
            "billing": "payments",
            "tech_support": "technical faults",
            "security": "unauthorized access",
        },
    )
    assert decision.value in decision.probabilities
    assert abs(sum(decision.probabilities.values()) - 1.0) < 1e-6
    assert 0.0 <= decision.confidence <= 1.0


def test_live_noul() -> None:
    client = DecisionClient(cfg)
    decision = client.decide_noul(
        state={"text": "The sky is blue during a clear day."},
        assertion="The statement is about the weather.",
    )
    assert 0.0 <= decision.probability_true <= 1.0


def test_live_score() -> None:
    client = DecisionClient(cfg)
    decision = client.decide_score(
        state={"text": "Customer PII was exposed in a public bucket for 3 days."},
        tiers=["low", "medium", "high", "critical"],
    )
    assert 1.0 <= decision.expected_score <= 4.0
    assert abs(sum(decision.level_probabilities.values()) - 1.0) < 1e-6
