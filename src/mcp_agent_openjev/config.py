"""Environment-driven configuration for mcp_agent_openjev.

The decision path is a single LM Studio-compatible client: the OpenJev prompt goes to Bionic's `/v1/responses` endpoint,
which is the one LM Studio surface that returns logprobs. The released helper's READOUT_* variables still work.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_BASE_URL = "http://127.0.0.1:1234"
DEFAULT_MODEL = "openjev"


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


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


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
    """Immutable connection + calibration settings for the OpenJev client."""

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str = ""
    timeout: float = 120.0
    temperature: float = 0.85
    noul_t: float = 1.829074
    noul_bias: float = 0.0
    perms: int = 1
    abstain_threshold: str = "auto"

    @classmethod
    def from_env(cls) -> Config:
        _load_dotenv()
        api_key = _env("JEV_API_KEY") or _env("LM_STUDIO_API_KEY")
        return cls(
            base_url=_env("JEV_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            model=_env("JEV_MODEL", DEFAULT_MODEL).strip(),
            api_key=api_key.strip(),
            timeout=_env_float("JEV_TIMEOUT", 120.0),
            temperature=_env_float("JEV_TEMPERATURE", _env_float("READOUT_T", 0.85)),
            noul_t=_env_float("JEV_NOUL_T", _env_float("READOUT_NOUL_T", 1.829074)),
            noul_bias=_env_float("JEV_NOUL_BIAS", _env_float("READOUT_NOUL_BIAS", 0.0)),
            perms=_env_int("JEV_PERMS", _env_int("READOUT_PERMS", 1)),
            abstain_threshold=_env_threshold("JEV_ABSTAIN_THRESHOLD", "auto"),
        )

    def with_overrides(self, **kwargs) -> Config:
        """Return a copy with per-request values applied (None values are ignored)."""
        clean = {k: v for k, v in kwargs.items() if v is not None}
        if "abstain_threshold" in clean:
            clean["abstain_threshold"] = str(clean["abstain_threshold"])
        for k in ("temperature", "noul_t", "noul_bias"):
            if k in clean:
                clean[k] = float(clean[k])
        if "perms" in clean:
            clean["perms"] = int(clean["perms"])
        return replace(self, **clean)

    @property
    def responses_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/responses"

    @property
    def models_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/models"

    @property
    def threshold(self) -> float | str:
        """Abstention threshold as a float, or the string `auto`."""
        return "auto" if self.abstain_threshold == "auto" else float(self.abstain_threshold)


__all__ = ["Config"]
