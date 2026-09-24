"""openjev-router CLI — one-shot decisions, HTTP service, and MCP server."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from openjev_router import __version__
from openjev_router.config import Config


def _parse_state(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {"text": raw}


def _dump(decision: Any) -> None:
    print(json.dumps(decision.model_dump(), indent=2, ensure_ascii=False))


def _cmd_doctor(args) -> int:
    from openjev_router.client import DecisionClient

    print(f"openjev-router {__version__} - doctor\n")
    try:
        cfg = Config.from_env()
    except Exception as exc:
        print(f"[config] {type(exc).__name__}: {exc}")
        return 1

    print(f"  base_url  : {cfg.base_url}")
    print(f"  model     : {cfg.model}")
    print(f"  backend   : {cfg.backend}  -> resolved {cfg.resolve_backend()}")
    print(f"  method    : {cfg.method}")
    print(f"  temperature: {cfg.temperature}")
    print(f"  threshold : {cfg.abstain_threshold}")
    print(f"  reasoning : {'disabled' if cfg.disable_reasoning else 'enabled'}")

    import requests

    try:
        resp = requests.get(cfg.models_url, timeout=cfg.timeout)
        data = resp.json().get("data") or []
        print(f"\n  endpoint  : reachable ({resp.status_code}), {len(data)} models served")
        for m in data:
            print(f"    - {m.get('id')}")
    except Exception as exc:
        print(f"\n  endpoint  : unreachable -> {type(exc).__name__}: {exc}")

    client = DecisionClient(cfg)
    print(f"\n  client    : backend={client.backend}")
    return 0


def _cmd_choice(args) -> int:
    from openjev_router.client import DecisionClient

    cfg = Config.from_env().with_overrides(
        model=args.model, temperature=args.temperature, abstain_threshold=args.abstain_threshold
    )
    decision = DecisionClient(cfg).decide_choice(
        state=_parse_state(args.state),
        candidates=args.candidate,
        criteria=args.criteria,
        allow_abstain=not args.no_abstain,
    )
    _dump(decision)
    return 0


def _cmd_noul(args) -> int:
    from openjev_router.client import DecisionClient

    cfg = Config.from_env().with_overrides(model=args.model, temperature=args.temperature)
    decision = DecisionClient(cfg).decide_noul(
        state=_parse_state(args.state), assertion=args.assertion
    )
    _dump(decision)
    return 0


def _cmd_score(args) -> int:
    from openjev_router.client import DecisionClient

    cfg = Config.from_env().with_overrides(
        model=args.model, temperature=args.temperature, abstain_threshold=args.abstain_threshold
    )
    decision = DecisionClient(cfg).decide_score(
        state=_parse_state(args.state),
        tiers=args.tier,
        criteria=args.criteria,
        allow_abstain=not args.no_abstain,
    )
    _dump(decision)
    return 0


def _cmd_serve(args) -> int:
    from openjev_router.mcp_server import run

    transport = "streamable-http" if args.http else "stdio"
    try:
        run(transport=transport, host=args.host, port=args.port)
    except KeyboardInterrupt:
        pass
    return 0


def _cmd_http(args) -> int:
    import uvicorn

    from openjev_router.http_app import create_app

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def _add_decision_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("state", help="state/context text, or a JSON object to evaluate")
    parser.add_argument(
        "-c", "--candidate", action="append", default=[], help="candidate option (repeatable)"
    )
    parser.add_argument("--criteria", default="", help="evaluation criteria")
    parser.add_argument("--model", default=None, help="override the configured backend model")
    parser.add_argument(
        "--temperature", type=float, default=None, help="override calibration temperature"
    )
    parser.add_argument(
        "--no-abstain",
        action="store_true",
        help="disable UNKNOWN abstention (choices always resolve)",
    )
    parser.add_argument(
        "--abstain-threshold",
        default=None,
        help="override abstention threshold (float) or 'auto' (choice/score only)",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="openjev-router",
        description=(
            "Typed probabilistic decisions (Choice/Noul/Score) over OpenJev + a local "
            "OpenAI-compatible chat endpoint."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("doctor", help="show config, backend, and endpoint reachability")
    p.set_defaults(func=_cmd_doctor)

    p = sub.add_parser("choice", help="one-shot categorical choice")
    _add_decision_args(p)
    p.set_defaults(func=_cmd_choice)

    p = sub.add_parser("noul", help="one-shot binary truth judgment")
    p.add_argument("state", help="state/context text, or a JSON object")
    p.add_argument("assertion", help="the assertion to judge TRUE or FALSE")
    p.add_argument("--model", default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.set_defaults(func=_cmd_noul)

    p = sub.add_parser("score", help="one-shot ordinal tier evaluation")
    p.add_argument("state", help="state/context text, or a JSON object")
    p.add_argument("-t", "--tier", action="append", default=[], help="ordered tier (repeatable)")
    p.add_argument("--criteria", default="")
    p.add_argument("--model", default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--no-abstain", action="store_true")
    p.add_argument(
        "--abstain-threshold", default=None, help="override abstention threshold (float) or 'auto'"
    )
    p.set_defaults(func=_cmd_score)

    p = sub.add_parser("serve", help="run the MCP server (stdio by default)")
    p.add_argument("--http", action="store_true", help="use streamable-http transport instead")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8030)
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("http", help="run the FastAPI HTTP decision service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8377)
    p.add_argument("--log-level", default="warning")
    p.set_defaults(func=_cmd_http)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"openjev-router: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
