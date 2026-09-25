"""OpenJevPro harness glue for the chat-logprobs backend.

OpenJev's stock benchmark harness ships three engines — TypeSafe Jev (commercial
API), OpenJevProEngine (wraps the *stock* client, Ollama/legacy only) and
DirectStructuredEngine (Ollama /api/chat). None can reach LM Studio, so this
adapter plugs our ``DecisionClient`` into the stock ``OpenJevProHarness``
unchanged
"""

from __future__ import annotations

import time
from typing import Any

from openjevpro.harness import BaseDecisionEngine

from mcp_agent_openjev.client import DecisionClient

__all__ = ["OpenJevRouterEngine"]


class OpenJevRouterEngine(BaseDecisionEngine):
    """Stock-harness engine backed by the chat-logprobs DecisionClient."""

    def __init__(self, client: DecisionClient, name: str = "OpenJev Router (LM Studio)"):
        self.name = name
        self.client = client

    def evaluate_choice(
        self,
        state: dict[str, Any],
        candidates: list[str],
        criteria: dict[str, str],
    ) -> dict[str, Any]:
        t0 = time.time()
        decision = self.client.decide_choice(
            state=state,
            candidates=candidates,
            criteria=criteria,
            allow_abstain=True,
        )
        return {
            "choice": decision.value,
            "confidence": decision.confidence,
            "probabilities": decision.probabilities,
            "abstained": decision.abstained,
            "latency_ms": (time.time() - t0) * 1000,
        }
