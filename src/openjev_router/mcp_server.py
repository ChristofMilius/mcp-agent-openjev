"""MCP server exposing OpenJev decisions as callable agent tools."""

from __future__ import annotations

import json
from typing import Any

from mcp.server.mcpserver import MCPServer

from openjev_router.client import DecisionClient
from openjev_router.config import Config

SERVER_NAME = "openjev-router"

INSTRUCTIONS = """\
Local typed probabilistic decision service backed by OpenJev + logprob calibration.

Decisions are System-1 primitives over an OpenAI-compatible chat endpoint
(default: LM Studio):

- decide_choice — multi-class categorical decision over a candidate set, with a
  calibrated probability distribution, confidence, and optional abstention
  (value normalizes to UNKNOWN when uncertain).
- decide_noul   — binary truth judgment (TRUE/FALSE) with P(true).
- decide_score  — ordinal evaluation across ordered tiers: expected score plus
  probability mass per tier.
- decision_status — connection health and configured backend/model.

All decisions return calibrated posterior probabilities (temperature-scaled
softmax over candidate logits) and never invent options outside the supplied set.
""".strip()


def _tool_error(tool: str, exc: Exception) -> str:
    return json.dumps(
        {"status": "error", "tool": tool, "error": f"{type(exc).__name__}: {exc}"},
        indent=2,
        ensure_ascii=False,
    )


def create_server(config: Config | None = None) -> MCPServer:
    """Build an MCPServer with the decision tool surface registered."""
    cfg = config or Config.from_env()
    client = DecisionClient(cfg)
    server = MCPServer(name=SERVER_NAME, instructions=INSTRUCTIONS)

    @server.tool()
    def decision_status() -> str:
        """Report the configured backend, model, and endpoint reachability."""
        try:
            import requests

            resp = requests.get(cfg.models_url, timeout=cfg.timeout)
            reachable = resp.status_code == 200
            count = len(resp.json().get("data") or []) if reachable else 0
            info = {
                "status": "ok" if reachable else "unreachable",
                "backend": client.backend,
                "method": cfg.method,
                "model": cfg.model,
                "base_url": cfg.base_url,
                "endpoint_reachable": reachable,
                "models_served": count,
            }
        except Exception as exc:
            info = {
                "status": "unreachable",
                "backend": client.backend,
                "method": cfg.method,
                "model": cfg.model,
                "base_url": cfg.base_url,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return json.dumps(info, indent=2, ensure_ascii=False)

    @server.tool()
    def decide_choice(
        state: dict[str, Any],
        candidates: list[str],
        criteria: str | dict[str, str] = "",
        allow_abstain: bool = True,
    ) -> str:
        """Select the single best candidate for the state, with calibrated probabilities.

        state: dict of context to evaluate. candidates: the allowed options (strings).
        criteria: how to judge them (plain string or mapping). allow_abstain: when the
        best confidence is too low, value becomes UNKNOWN and tentative_value keeps the
        raw argmax. Returns ChoiceDecision JSON.
        """
        try:
            decision = client.decide_choice(
                state=state, candidates=candidates, criteria=criteria, allow_abstain=allow_abstain
            )
            return json.dumps(decision.model_dump(), indent=2, ensure_ascii=False)
        except Exception as exc:
            return _tool_error("decide_choice", exc)

    @server.tool()
    def decide_noul(state: dict[str, Any], assertion: str) -> str:
        """Judge whether a binary assertion is TRUE or FALSE for the given state."""
        try:
            decision = client.decide_noul(state=state, assertion=assertion)
            return json.dumps(decision.model_dump(), indent=2, ensure_ascii=False)
        except Exception as exc:
            return _tool_error("decide_noul", exc)

    @server.tool()
    def decide_score(
        state: dict[str, Any],
        tiers: list[str | dict[str, Any]],
        criteria: str | dict[str, str] = "",
        allow_abstain: bool = True,
    ) -> str:
        """Rate the state across ordered tiers (e.g. severity levels).

        tiers: ordered list of strings, or dicts with 'label'/'value' and an optional
        'score' weight. Returns ScoreDecision JSON with expected_score and per-tier
        probability mass.
        """
        try:
            decision = client.decide_score(
                state=state, tiers=tiers, criteria=criteria, allow_abstain=allow_abstain
            )
            return json.dumps(decision.model_dump(), indent=2, ensure_ascii=False)
        except Exception as exc:
            return _tool_error("decide_score", exc)

    return server


def run(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8030) -> None:
    """Run the decision MCP server.

    transport: "stdio" (default), "sse", or "streamable-http".
    """
    server = create_server()
    if transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(transport=transport, host=host, port=port)


__all__ = ["SERVER_NAME", "create_server", "run"]
