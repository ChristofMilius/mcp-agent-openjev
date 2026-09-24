"""Run OpenJev's benchmark harness against LM Studio via openjev-router.

Reads a small labeled intent-routing dataset, drives every item through
``OpenJevProHarness`` (concurrent engine evaluation, resumable checkpoints)
and prints the accuracy/latency summary.

Usage:
    uv run python examples/benchmark_harness.py
"""

from __future__ import annotations

import json

from openjevpro.harness import OpenJevProHarness

from openjev_router.client import DecisionClient
from openjev_router.config import Config
from openjev_router.harness import OpenJevRouterEngine

DATASET = [
    {"text": "I was charged twice for my subscription this month", "category": "billing"},
    {"text": "Unauthorized login from an unknown IP address", "category": "security"},
    {"text": "My printer stopped feeding paper after the update", "category": "tech_support"},
    {"text": "How do I export my transaction history as CSV?", "category": "billing"},
    {"text": "Can someone review my recent phishing report submission?", "category": "security"},
    {"text": "The web app returns a 500 error on the dashboard", "category": "tech_support"},
]

CANDIDATES = ["billing", "tech_support", "security"]

CRITERIA = {
    "billing": "payments, charges, invoices, transactions",
    "tech_support": "technical faults, software/hardware failures",
    "security": "unauthorized access, phishing, account compromise",
}


def main() -> None:
    cfg = Config.from_env()
    client = DecisionClient(cfg)
    engine = OpenJevRouterEngine(client)

    harness = OpenJevProHarness(engines=[engine])
    result = harness.run(
        dataset=DATASET,
        candidates=CANDIDATES,
        criteria=CRITERIA,
        checkpoint_file="harness_checkpoint.json",
        max_workers=4,
    )

    print("\n=== summary ===")
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
