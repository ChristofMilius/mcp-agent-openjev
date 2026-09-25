"""Live example: route a support ticket to a department with abstention."""

from __future__ import annotations

from mcp_agent_openjev.client import DecisionClient
from mcp_agent_openjev.config import Config

TICKET_ROUTES = ["billing", "tech_support", "security", "card_lost"]

CASES = [
    {
        "ticket_id": "TCK-4821",
        "text": "I noticed an unauthorized login attempt from an unknown IP address.",
    },
    {
        "ticket_id": "TCK-4822",
        "text": "How do I locate my card? It was dispatched two weeks ago.",
    },
]


def main() -> int:
    client = DecisionClient(Config.from_env())
    criteria = {
        "billing": "Charges, invoices, payment methods, fees",
        "tech_support": "Technical faults, errors, account access",
        "security": "Unauthorized access, suspicious activity, fraud",
        "card_lost": "Lost or stolen cards and replacement requests",
    }
    print(f"backend={client.backend} model={client.config.model}\n")
    for case in CASES:
        decision = client.decide_choice(state=case, candidates=TICKET_ROUTES, criteria=criteria)
        print(f"{case['ticket_id']}: {case['text']}")
        print(
            f"  -> {decision.value}  (confidence {decision.confidence:.2%}"
            f"{'  [ABSTAINED] tentative=' + str(decision.tentative_value) if decision.abstained else ''})"
        )
        for option, prob in sorted(decision.probabilities.items(), key=lambda kv: -kv[1]):
            print(f"      {option:<15} {prob:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
