"""mcp-agent-openjev — local typed probabilistic decision service.

Wraps OpenJev (Choice / Noul / Score with calibrated probabilities and
abstention) behind an OpenAI-compatible chat endpoint — primarily LM Studio's
`/v1/chat/completions` logprobs, which OpenJev's stock Ollama / legacy
`/completions` clients cannot reach.
"""

from __future__ import annotations

__version__ = "0.1.0"


def main() -> int:
    """Console entry point (`mcp-agent-openjev`). Dispatches to the CLI."""
    from mcp_agent_openjev.cli import main as _cli_main

    return _cli_main()


__all__ = ["__version__", "main"]
