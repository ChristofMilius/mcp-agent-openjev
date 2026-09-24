"""Environment-driven configuration for openjev_router."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "google/gemma-4-26b-a4b-qat"

BACKENDS = ("auto", "chat", "ollama", "legacy")
METHODS = ("auto", "logprobs", "scores")


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None else value


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_flag(name: str, default: bool) -> bool:
    raw = _env(name).strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean flag, got {raw!r}")


def _env_threshold(name: str, default: str) -> str:
    raw = _env(name).strip().lower()
    if not raw:
        return default
    if raw == "auto":
        return "auto"
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be 'auto' or a float, got {raw!r}") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
    return str(value)


def _load_dotenv() -> None:
    """Load `.env` from the current working directory if present."""
    load_dotenv(Path.cwd() / ".env", override=False)


@dataclass(frozen=True)
class Config:
    """Immutable connection + calibration settings for a DecisionClient."""

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str = ""
    backend: str = "auto"
    method: str = "auto"
    temperature: float = 1.3
    abstain_threshold: str = "0.45"
    disable_reasoning: bool = True
    timeout: float = 60.0
    top_logprobs: int = 20

    @classmethod
    def from_env(cls) -> Config:
        _load_dotenv()
        backend = _env("JEV_BACKEND", "auto").strip().lower()
        if backend not in BACKENDS:
            raise ValueError(f"JEV_BACKEND must be one of {BACKENDS}, got {backend!r}")
        method = _env("JEV_METHOD", "auto").strip().lower()
        if method not in METHODS:
            raise ValueError(f"JEV_METHOD must be one of {METHODS}, got {method!r}")
        api_key = _env("JEV_API_KEY") or _env("LM_STUDIO_API_KEY")
        return cls(
            base_url=_env("JEV_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            model=_env("JEV_MODEL", DEFAULT_MODEL).strip(),
            api_key=api_key.strip(),
            backend=backend,
            method=method,
            temperature=_env_float("JEV_TEMPERATURE", 1.3),
            abstain_threshold=_env_threshold("JEV_ABSTAIN_THRESHOLD", "0.45"),
            disable_reasoning=_env_flag("JEV_DISABLE_REASONING", True),
            timeout=_env_float("JEV_TIMEOUT", 60.0),
            top_logprobs=int(_env_float("JEV_TOP_LOGPROBS", 20.0)),
        )

    def resolve_backend(self) -> str:
        """Map the `auto` backend onto a concrete backend name."""
        if self.backend != "auto":
            return self.backend
        return "ollama" if ":11434" in self.base_url else "chat"

    def with_overrides(self, **kwargs) -> Config:
        """Return a copy with per-request values applied (None values are ignored)."""
        clean = {k: v for k, v in kwargs.items() if v is not None}
        if "abstain_threshold" in clean:
            clean["abstain_threshold"] = str(clean["abstain_threshold"])
        if "temperature" in clean:
            clean["temperature"] = float(clean["temperature"])
        if "model" in clean:
            clean["model"] = str(clean["model"])
        return replace(self, **clean)

    @property
    def decisions_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    @property
    def models_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/models"):
            return base
        return f"{base}/models"

    @property
    def threshold(self) -> float | str:
        """Abstention threshold as a float, or the string `auto`."""
        return "auto" if self.abstain_threshold == "auto" else float(self.abstain_threshold)


__all__ = ["Config", "BACKENDS", "METHODS"]
