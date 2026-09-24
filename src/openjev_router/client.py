"""Typed probabilistic decision client backed by OpenJev.

Two layers:

* ``DecisionClient`` — the public surface. ``decide_choice`` / ``decide_noul`` /
  ``decide_score`` return OpenJev's pydantic decision schemas with temperature
  calibrated probabilities and abstention.

* Wire backends — ``chat`` (our logprobs/scores path over
  ``/v1/chat/completions``, the only endpoint LM Studio exports) and the two
  stock openjevpro backends (``ollama`` via ``/api/chat``, ``legacy`` via
  ``/v1/completions``).

The chat backend is the reason this project exists: OpenJev's built-in clients
target Ollama and the legacy completions endpoint, neither of which LM Studio
serves. The letter-scoring algorithm is preserved, only the wire protocol is
changed to return ``choices[0].logprobs.content``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from enum import Enum
from typing import Any

import requests
from openjevpro.calibrator import TemperatureCalibrator
from openjevpro.client import OpenJevProClient as _StockClient

from openjev_router.config import Config
from openjev_router.schemas import ChoiceDecision, NoulDecision, ScoreDecision

Criteria = str | dict[str, str]


class DecisionError(Exception):
    """Base error for decision failures, carrying a user-facing message."""


class NoLogprobsError(DecisionError):
    """The backend answered without `logprobs.content`, so calibration is impossible."""


class TooManyOptionsError(DecisionError):
    """More than 26 candidates cannot be expressed as single letter tokens."""


def _criteria_text(criteria: Criteria) -> str:
    if isinstance(criteria, dict):
        return "\n".join(f"- {k}: {v}" for k, v in criteria.items())
    return str(criteria).strip()


def _normalize_options(candidates: type[Enum] | list[str]) -> list[str]:
    if isinstance(candidates, type) and issubclass(candidates, Enum):
        return [e.value for e in candidates]
    return [str(o) for o in candidates]


def _effective_threshold(n_options: int, configured: float | str) -> float:
    """Auto threshold scales inversely with candidate count (1.25 / N)."""
    if configured == "auto":
        return 1.25 / max(n_options, 1)
    return float(configured)


class _ChatCompleter:
    """Thin POST wrapper over an OpenAI-compatible chat completions endpoint."""

    def __init__(self, config: Config):
        self.config = config
        self._headers = {"Content-Type": "application/json"}
        if config.api_key:
            self._headers["Authorization"] = f"Bearer {config.api_key}"

    def _payload(
        self, content: str, *, max_tokens: int, temperature: float, logprobs: bool
    ) -> dict:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = self.config.top_logprobs
        if self.config.disable_reasoning:
            payload["reasoning_effort"] = "none"
        return payload

    def complete(
        self, content: str, *, max_tokens: int, temperature: float, logprobs: bool
    ) -> dict:
        resp = requests.post(
            self.config.decisions_url,
            headers=self._headers,
            json=self._payload(
                content, max_tokens=max_tokens, temperature=temperature, logprobs=logprobs
            ),
            timeout=self.config.timeout,
        )
        resp.raise_for_status()
        return resp.json()


def _strip_code_fence(content: str) -> str:
    if "```" not in content:
        return content.strip()
    for part in content.split("```"):
        block = part.strip()
        if block.startswith("json"):
            block = block[4:].strip()
        if block.startswith("{") and block.endswith("}"):
            return block
    return content.strip()


class DecisionClient:
    """Calibrated typed decision client over a configurable backend."""

    def __init__(self, config: Config):
        if config.temperature <= 0:
            raise ValueError("Temperature must be positive.")
        self.config = config
        self.backend = config.resolve_backend()
        self.calibrator = TemperatureCalibrator(temperature=config.temperature)
        self._legacy: _StockClient | None = None

        if self.backend in ("ollama", "legacy"):
            self._legacy = _StockClient(
                base_url=config.base_url,
                api_key=config.api_key,
                model=config.model,
                temperature_scaling=config.temperature,
                abstain_threshold=config.threshold,
                backend=self.backend,
            )

    # -- public surface ----------------------------------------------------

    def decide_choice(
        self,
        state: dict[str, Any],
        candidates: type[Enum] | list[str],
        criteria: Criteria = "",
        allow_abstain: bool = True,
    ) -> ChoiceDecision:
        """Multi-class categorical decision with calibrated probabilities."""
        if self._legacy is not None:
            return self._legacy.decide_choice(
                state=state,
                candidates=candidates,
                criteria=criteria,
                allow_abstain=allow_abstain,
            )

        options = _normalize_options(candidates)
        if allow_abstain and "UNKNOWN" not in options:
            options.append("UNKNOWN")
        logits = self._score_options(state, options, criteria)
        return self._to_choice(logits, options, allow_abstain)

    def decide_noul(self, state: dict[str, Any], assertion: str) -> NoulDecision:
        """Binary truth judgment: is the assertion TRUE or FALSE for the state?"""
        decision = self.decide_choice(
            state=state,
            candidates=["TRUE", "FALSE"],
            criteria=f"Evaluate whether the following assertion is strictly TRUE or FALSE: {assertion}",
            allow_abstain=False,
        )
        p_true = decision.probabilities.get("TRUE", 0.5)
        value = p_true >= 0.5
        conf = p_true if value else (1.0 - p_true)
        return NoulDecision(
            value=value,
            probability_true=p_true,
            confidence=conf,
            abstained=decision.abstained,
        )

    def decide_score(
        self,
        state: dict[str, Any],
        tiers: Sequence[str | dict[str, Any]],
        criteria: Criteria = "",
        allow_abstain: bool = True,
    ) -> ScoreDecision:
        """Ordinal evaluation: probability mass across tiers + expected score."""
        labels: list[str] = []
        scores: dict[str, float] = {}
        for index, tier in enumerate(tiers):
            if isinstance(tier, dict):
                label = str(tier.get("label") or tier.get("value") or index)
                weight = float(tier.get("score", index))
            else:
                label, weight = str(tier), float(index)
            labels.append(label)
            scores[label] = weight

        decision = self.decide_choice(
            state=state, candidates=labels, criteria=criteria, allow_abstain=allow_abstain
        )
        weight = dict(scores)
        if "UNKNOWN" in decision.probabilities and "UNKNOWN" not in weight:
            weight["UNKNOWN"] = 0.0
        expected = sum(p * weight.get(label, 0.0) for label, p in decision.probabilities.items())
        return ScoreDecision(
            expected_score=expected,
            level_probabilities=decision.probabilities,
            confidence=decision.confidence,
            abstained=decision.abstained,
        )

    # -- chat-backend scoring ----------------------------------------------

    def _score_options(
        self, state: dict[str, Any], options: list[str], criteria: Criteria
    ) -> dict[str, float]:
        if len(options) > 26:
            raise TooManyOptionsError(
                f"{len(options)} candidates exceed the 26 single-letter token limit; "
                "split the decision into a hierarchy or drop allow_abstain."
            )
        letters = [chr(65 + i) for i in range(len(options))]
        letter_to_option = dict(zip(letters, options))
        prompt = self._build_letter_prompt(state, options, criteria)

        if self.config.method in ("auto", "logprobs"):
            try:
                return self._score_logprobs(prompt, letter_to_option)
            except NoLogprobsError:
                if self.config.method == "logprobs":
                    raise
            except requests.HTTPError:
                if self.config.method == "logprobs":
                    raise
        return self._score_json(prompt, options)

    def _build_letter_prompt(
        self, state: dict[str, Any], options: list[str], criteria: Criteria
    ) -> str:
        lines = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)]
        return (
            f"Given the following state:\n{json.dumps(state, ensure_ascii=False, indent=2)}\n\n"
            f"Evaluation criteria:\n{_criteria_text(criteria)}\n\n"
            f"Select the single best option from the list below:\n" + "\n".join(lines) + "\n\n"
            "Reply with ONLY the option letter (e.g. A, B, C):"
        )

    def _score_logprobs(self, prompt: str, letter_to_option: dict[str, str]) -> dict[str, float]:
        data = self._chat.complete(prompt, max_tokens=1, temperature=0.0, logprobs=True)
        choice = data["choices"][0]
        content = (choice.get("message") or {}).get("content") or ""
        if not content:
            raise NoLogprobsError(
                "backend returned an empty message (reasoning may have consumed the token budget); "
                "set JEV_DISABLE_REASONING=1 or use method=scores."
            )
        logprob_block = (choice.get("logprobs") or {}).get("content") or []
        if not logprob_block:
            raise NoLogprobsError("backend returned no logprobs.content for the completion.")
        token_probs = {
            entry.get("token"): float(entry.get("logprob", -100.0))
            for entry in (logprob_block[0].get("top_logprobs") or [])
        }
        logits: dict[str, float] = {}
        for letter, option in letter_to_option.items():
            for attempt in (letter, f" {letter}", letter.lower(), f" {letter.lower()}"):
                if attempt in token_probs:
                    logits[option] = token_probs[attempt]
                    break
            else:
                logits[option] = -100.0
        return logits

    def _score_json(self, prompt: str, options: list[str]) -> dict[str, float]:
        schema = json.dumps({opt: 0.0 for opt in options})
        content = (
            f"{prompt}\n\n"
            f"Rate the relative likelihood (0.0 to 10.0) of each option being the single correct choice.\n"
            f"Output ONLY a valid JSON object matching this schema:\n"
            f'{{"scores": {schema}}}'
        )
        data = self._chat.complete(content, max_tokens=1024, temperature=0.0, logprobs=False)
        raw = (data["choices"][0].get("message") or {}).get("content") or ""
        parsed = json.loads(_strip_code_fence(raw))
        scores = parsed.get("scores", {})
        return {opt: float(scores.get(opt, 0.0)) for opt in options}

    # -- decision assembly --------------------------------------------------

    def _to_choice(
        self, logits: dict[str, float], options: list[str], allow_abstain: bool
    ) -> ChoiceDecision:
        probabilities = self.calibrator.calibrate(logits)
        best = max(probabilities, key=probabilities.get)
        confidence = probabilities[best]
        effective_threshold = _effective_threshold(len(options), self.config.threshold)
        abstained = (allow_abstain and best == "UNKNOWN") or confidence < effective_threshold
        value = "UNKNOWN" if abstained else best
        tentative = best if (abstained and best != "UNKNOWN") else None
        return ChoiceDecision(
            value=value,
            probabilities=probabilities,
            confidence=confidence,
            abstained=abstained,
            tentative_value=tentative,
            raw_logits=logits,
        )

    @property
    def _chat(self) -> _ChatCompleter:
        completer = getattr(self, "_completer", None)
        if completer is None:
            completer = _ChatCompleter(self.config)
            object.__setattr__(self, "_completer", completer)
        return completer


__all__ = ["DecisionClient", "DecisionError", "NoLogprobsError", "TooManyOptionsError"]
