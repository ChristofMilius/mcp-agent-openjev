"""FastAPI HTTP service exposing OpenJev decisions over the wire."""

from __future__ import annotations

from typing import Any

import requests as _requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mcp_agent_openjev.client import DecisionClient, DecisionError
from mcp_agent_openjev.config import Config
from mcp_agent_openjev.schemas import ChoiceDecision, NoulDecision, ScoreDecision

APP_NAME = "mcp-agent-openjev"


class ChoiceRequest(BaseModel):
    state: dict[str, Any] = Field(description="Incoming state / context to evaluate")
    candidates: list[str] = Field(
        min_length=1, description="Candidate options (or an enum's values)"
    )
    criteria: str | dict[str, str] = ""
    allow_abstain: bool = True
    model: str | None = None
    temperature: float | None = None
    abstain_threshold: float | str | None = None


class NoulRequest(BaseModel):
    state: dict[str, Any]
    assertion: str
    model: str | None = None


class ScoreRequest(BaseModel):
    state: dict[str, Any]
    tiers: list[str | dict[str, Any]] = Field(
        min_length=1,
        description=(
            "Ordered tiers: label strings, or dicts with 'label' (or 'value') and an "
            "optional 'score' equal to the tier's position. Malformed tiers are rejected "
            "with 400."
        ),
    )
    criteria: str | dict[str, str] = ""
    allow_abstain: bool = True
    model: str | None = None
    temperature: float | None = None
    abstain_threshold: float | str | None = None


def create_app(config: Config | None = None) -> FastAPI:
    cfg = config or Config.from_env()
    client = DecisionClient(cfg)
    app = FastAPI(title=APP_NAME)

    def _client_for(req) -> DecisionClient:
        if req.model is None and req.temperature is None and req.abstain_threshold is None:
            return client
        overrides = {
            "model": getattr(req, "model", None),
            "temperature": getattr(req, "temperature", None),
            "abstain_threshold": getattr(req, "abstain_threshold", None),
        }
        specific = cfg.with_overrides(**overrides)
        return DecisionClient(specific)

    @app.get("/health")
    def health() -> dict[str, Any]:
        try:
            models_resp = _requests.get(cfg.models_url, timeout=cfg.timeout)
            models_reachable = models_resp.status_code == 200
            model_count = len(models_resp.json().get("data") or []) if models_reachable else 0
        except Exception:
            models_reachable, model_count = False, 0
        return {
            "status": "ok",
            "app": APP_NAME,
            "backend": client.backend,
            "method": "scores" if cfg.method == "scores" else f"{cfg.method} (logprobs preferred)",
            "model": cfg.model,
            "base_url": cfg.base_url,
            "endpoint_reachable": models_reachable,
            "models_served": model_count,
        }

    @app.post("/v1/decide/choice", response_model=ChoiceDecision)
    def decide_choice(req: ChoiceRequest) -> ChoiceDecision:
        try:
            return _client_for(req).decide_choice(
                state=req.state,
                candidates=req.candidates,
                criteria=req.criteria,
                allow_abstain=req.allow_abstain,
            )
        except DecisionError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/v1/decide/noul", response_model=NoulDecision)
    def decide_noul(req: NoulRequest) -> NoulDecision:
        try:
            return _client_for(req).decide_noul(state=req.state, assertion=req.assertion)
        except DecisionError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/v1/decide/score", response_model=ScoreDecision)
    def decide_score(req: ScoreRequest) -> ScoreDecision:
        try:
            return _client_for(req).decide_score(
                state=req.state,
                tiers=req.tiers,
                criteria=req.criteria,
                allow_abstain=req.allow_abstain,
            )
        except DecisionError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    return app


__all__ = ["create_app"]
