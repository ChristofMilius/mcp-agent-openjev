"""Live example: rate an incident description across severity tiers."""

from __future__ import annotations

from typing import Any

from mcp_agent_openjev.client import DecisionClient
from mcp_agent_openjev.config import Config

INCIDENTS = [
    {
        "id": "INC-101",
        "text": "Sensitive customer records were exposed in a public S3 bucket for 3 days.",
    },
    {
        "id": "INC-102",
        "text": "The print queue on the 2nd floor is stuck again; nobody printed since this morning.",
    },
]


def main() -> int:
    client = DecisionClient(Config.from_env())
    tiers: list[dict[str, Any]] = [
        {"label": "low", "score": 1, "value": "Cosmetic or localized annoyance"},
        {"label": "medium", "score": 2, "value": "Partial degradation, no data exposure"},
        {"label": "high", "score": 3, "value": "Data exposure or broad outage"},
        {"label": "critical", "score": 4, "value": "Confirmed breach, service is down"},
    ]
    print(f"backend={client.backend} model={client.config.model}\n")
    for incident in INCIDENTS:
        decision = client.decide_score(state=incident, tiers=tiers)
        print(f"{incident['id']}: {incident['text']}")
        print(
            f"  -> expected severity {decision.expected_score:.2f}"
            f"  (confidence {decision.confidence:.2%}"
            f"{'  [ABSTAINED]' if decision.abstained else ''})"
        )
        for tier, prob in sorted(decision.level_probabilities.items(), key=lambda kv: -kv[1]):
            print(f"      {tier:<12} {prob:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
